// ===========================================================================
//  GEMM Optimization Journey on RTX 3090 (GA102, sm_86)
//  Following "Programming Massively Parallel Processors" (Cudabook)
//
//  C[N][N] = A[N][N] * B[N][N]   (row-major, FP32)
//
//  Build:  make
//  Run:    ./gemm 1024          # run all kernels
//          ./gemm 4096 2        # run only kernel id 2
// ===========================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cfloat>
#include <vector>
#include <cuda_runtime.h>

#ifdef USE_CUBLAS
#include <cublas_v2.h>
#endif

// ---------------------------------------------------------------------------
// Hardware constants (RTX 3090 / GA102)
// ---------------------------------------------------------------------------
static constexpr double PEAK_FP32_GFLOPS = 35600.0;  // 10496 cores * 2 * 1.695 GHz
static constexpr double PEAK_BW_GBPS     = 936.0;    // GDDR6X 384-bit @ 19.5 Gbps

// Verify against CPU golden reference only when N <= this.
#ifndef VERIFY_MAX_N
#define VERIFY_MAX_N 1024
#endif

// ---------------------------------------------------------------------------
// Tunables.  MUST be defined before any kernel that uses them -- a macro is
// expanded at its point of use, so defining BLOCK_DIM further down the file
// makes it invisible to everything above.
// ---------------------------------------------------------------------------
#ifndef BLOCK_DIM
#define BLOCK_DIM 16   // 16x16 = 256 thr/block -> 6 blocks/SM -> 100% occupancy.
                       // Try 32 (=1024 thr) and explain the occupancy drop.
#endif

#ifndef TILE
#define TILE BLOCK_DIM      // one thread <-> one shared-memory element
#endif

// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "[CUDA ERROR] %s:%d  '%s'\n  -> %s\n",             \
                    __FILE__, __LINE__, #call, cudaGetErrorString(err_));      \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Catches launch-config errors AND async errors from inside the kernel.
#define CUDA_CHECK_KERNEL()                                                    \
    do {                                                                       \
        CUDA_CHECK(cudaGetLastError());                                        \
        CUDA_CHECK(cudaDeviceSynchronize());                                   \
    } while (0)

// Usable from both host launchers and device code.
__host__ __device__ static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// ===========================================================================
//  STAGE 1 : Naive Global-Memory GEMM  (Cudabook Ch.4)
//
//  Each thread computes exactly ONE element of C.
//  Arithmetic Intensity = 0.25 FLOP/Byte  ->  DRAM-bound ceiling ~234 GFLOP/s
// ===========================================================================

// --- v1: row from .y, col from .x --------------------------------------
// Expected: B and C accesses are coalesced, A access is a warp broadcast.
__global__ void gemm_naive_v1(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int n)
{
    // Version A
    // TODO(you): map this thread to (row, col) using blockIdx.y/.x
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO(you): guard the boundary  (row < n && col < n)
    if (row < n && col < n) {
        float sum = 0.0f;
    // TODO(you): accumulate  sum += A[row*n + k] * B[k*n + col]  for k in [0,n)
        for (int k=0; k<n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
    // TODO(you): write C[row*n + col] = sum
        C[row * n + col] = sum;
    }
}

// --- v2: row from .x, col from .y  (the "trap") ------------------------
// Numerically identical to v1, but A-read and C-write become strided.
__global__ void gemm_naive_v2(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int n)
{
    // Version B
    // TODO(you): same math as v1, but swap which of blockIdx.x/.y feeds row/col
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    // TODO(you): guard the boundary  (row < n && col < n)
    if (row < n && col < n) {
        float sum = 0.0f;
    // TODO(you): accumulate  sum += A[row*n + k] * B[k*n + col]  for k in [0,n)
        for (int k=0; k<n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
    // TODO(you): write C[row*n + col] = sum
        C[row * n + col] = sum;
    }
}

// ===========================================================================
//  STAGE 2 : Shared-Memory Tiling   (Cudabook Ch.5)
//
//  Same output-per-thread as Stage 1 (one C element), but the k-loop is
//  blocked: each iteration stages a TILE x TILE sub-block of A and of B into
//  shared memory, then every thread in the block reuses them TILE times.
//
//  Block-level AI is UNCHANGED at TILE/4 = 4 FLOP/Byte -- L1 was already
//  giving us that. What changes is that the reuse is now GUARANTEED instead
//  of being at the mercy of L1 eviction. Expect ~1.5-2x, not 10x.
// ===========================================================================

__global__ void gemm_tiled_v1(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int n)
{
    // TODO(you): __shared__ float As[TILE][TILE], Bs[TILE][TILE];
    __shared__ float As[TILE][TILE], Bs[TILE][TILE];
    // TODO(you): this thread's OUTPUT coordinates (row, col) -- same mapping
    //            as gemm_naive_v1 (row from .y, col from .x).
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    for (int t = 0; t < ceil_div(n, TILE); ++t) {
    //     (a) cooperative load: each thread brings in exactly ONE element of A
    //         and ONE of B.
    //           - the A sub-block and the B sub-block travel along k in
    //             DIFFERENT directions -- their index expressions differ.
    //           - out of range? store 0.0f. Do NOT skip the store: a stale
    //             value from iteration t-1 would silently corrupt the sum.
        As[threadIdx.y][threadIdx.x] = (row < n && (t * TILE + threadIdx.x) < n) ? A[row * n + (t * TILE + threadIdx.x)] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (col < n && (t * TILE + threadIdx.y) < n) ? B[(t * TILE + threadIdx.y) * n + col] : 0.0f;
    //     (b) barrier #1   (RAW: everybody wrote before anybody reads)
        __syncthreads();
    //     (c) for (int kk = 0; kk < TILE; ++kk) sum += As[..][..] * Bs[..][..];
        for (int kk=0; kk<TILE; ++kk) {
            sum += As[threadIdx.y][kk] * Bs[kk][threadIdx.x];
        }
    //     (d) barrier #2   (WAR: everybody read before anybody overwrites)
        __syncthreads();
    }
    //
    // TODO(you): guarded write-back  
    if (row < n && col < n) C[row * n + col] = sum;
    // !! TRAP !! __syncthreads() must be reached by EVERY thread of the block.
    // Never put the k-loop inside `if (row < n && col < n) { ... }` -- the
    // edge threads would skip the barrier and the block hangs (or worse,
    // silently produces garbage). Guard the LOADS and the STORE, never the
    // BARRIERS.
}

// ===========================================================================
//  STAGE 3 : Thread-Level (Register) Tiling   (Cudabook Ch.6)
//
//  Stage 2 hit a shared-memory wall: 2 LDS per FFMA caps the kernel at
//  17.8 TB/s * 0.25 FLOP/B = 4.45 TFLOP/s no matter how good the DRAM side is.
//
//  Fix: give each thread a TM x TN micro-tile of C held entirely in registers.
//  Per k-step a thread now loads TM + TN floats from smem and issues TM * TN
//  FFMAs, so the ratio goes from 0.5 to (TM*TN)/(TM+TN) = 4 at 8x8, which
//  lifts the shared-memory ceiling to ~35.6 TFLOP/s = FP32 peak.
//
//  Block computes BM x BN of C, marching over k in chunks of BK.
//    threads/block = (BM/TM) * (BN/TN) = 16 * 16 = 256
//    smem          = (BM*BK + BK*BN) * 4 B = 8 KB  (of 100 KB/SM)
//    A-loads/thread per k-chunk = BM*BK / 256 = 4   (same for B)
// ===========================================================================

// Individually overridable, e.g. nvcc -DBK=16. Note the cooperative-load
// mapping in the kernels assumes BM*BK == BK*BN == 4 * THREADS_PER_BLOCK,
// so changing BK means reworking the load index derivation too.
#ifndef BM
#define BM 128      // block tile rows    (C)
#endif
#ifndef BN
#define BN 128      // block tile cols    (C)
#endif
#ifndef BK
#define BK 8        // k-chunk depth
#endif
#ifndef TM
#define TM 8        // per-thread micro-tile rows
#endif
#ifndef TN
#define TN 8        // per-thread micro-tile cols
#endif
#define THREADS_PER_BLOCK ((BM / TM) * (BN / TN))   // 256

__global__ void gemm_regtile_v1(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int n)
{
    // Launched with a 1-D block of THREADS_PER_BLOCK threads.
    //
    // TODO(you): __shared__ float As[...], Bs[BK][BN];
    //     DESIGN QUESTION -- should As be [BM][BK] or [BK][BM]?
    //     In the inner loop every thread reads a COLUMN of the A-tile
    //     (fixed k, TM different rows). Which layout makes that a
    //     contiguous, bank-conflict-free read? Decide before you write it.
    __shared__ float As[BM][BK], Bs[BK][BN];
    // TODO(you): this thread's position inside the 16x16 grid of micro-tiles:
    const int tRow = threadIdx.x / (BN / TN);
    const int tCol = threadIdx.x % (BN / TN);
    //     -> this thread owns C rows [blockIdx.y*BM + tRow*TM, +TM)
    //                       C cols [blockIdx.x*BN + tCol*TN, +TN)
    //
    // TODO(you): float acc[TM][TN] = {0.0f};        // MUST stay in registers
    //            float regA[TM], regB[TN];
    float acc[TM][TN] = {0.0f};
    float regA[TM], regB[TN];
    for (int k0 = 0; k0 < n; k0 += BK) {
    //   (a) cooperative load of A[.. , k0..k0+BK) and B[k0..k0+BK, ..] into
    //       smem. 256 threads, 1024 elements each -> 4 elements per thread.
    //       Derive a SEPARATE (row, col) mapping for loading; it has nothing
    //       to do with tRow/tCol. Keep global reads coalesced. Zero-fill OOB.
        for (int i=0; i<4; ++i) {
            int idx = threadIdx.x + i * 256;
            int row_a = idx / BK;
            int col_a = idx % BK;
            int global_row_a = blockIdx.y * BM + row_a;
            int global_col_a = k0 + col_a;
            As[row_a][col_a] = (global_row_a < n && global_col_a < n) ? A[global_row_a * n + global_col_a] : 0.0f;
            int row_b = idx / BN;
            int col_b = idx % BN;
            int global_row_b = k0 + row_b;
            int global_col_b = blockIdx.x * BN + col_b;
            Bs[row_b][col_b] = (global_row_b < n && global_col_b < n) ? B[global_row_b * n + global_col_b] : 0.0f;
        }
    //   (b) __syncthreads();                         // RAW
        __syncthreads();
    //   (c) #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
    //           #pragma unroll
            for (int i=0; i<TM; ++i) { // TM smem loads
                regA[i] = As[tRow * TM + i][kk];
            }        
    //           #pragma unroll
            for (int j=0; j<TN; ++j) { // TN smem loads
                regB[j] = Bs[kk][tCol * TN + j];
            }
    //           #pragma unroll
            for (int i=0; i<TM; ++i) {
    //             #pragma unroll
                for (int j=0; j<TN; ++j) { // TM*TN FFMA
                    acc[i][j] += regA[i] * regB[j];
                }
            }
        }
    //
    //   (d) __syncthreads();                         // WAR
        __syncthreads();
    }
    //
    // TODO(you): write back acc[i][j] with bounds guards.
    for (int i=0; i<TM; ++i) {
        for (int j=0; j<TN; ++j) {
            int global_row = blockIdx.y * BM + tRow * TM + i;
            int global_col = blockIdx.x * BN + tCol * TN + j; 
            if (global_row < n && global_col < n) {
                C[global_row * n + global_col] = acc[i][j];
            }
        }
    }   
    // !! CHECK `make resource` !! Every #pragma unroll above is load-bearing.
    // If ptxas reports ANY "bytes lmem" or "spill", acc[][] fell out of the
    // register file into DRAM and the whole design is void.
}

// ===========================================================================
//  STAGE 4a : Transposed As + vectorized shared loads
//
//  ncu on gemm_regtile_v1 (N=2048):
//      shared-ld instructions   41,943,040
//      wavefronts              134,218,687   -> 3.20 per instruction
//      bank conflicts           67,108,864   -> exactly 2.00x oversubscribed
//      DRAM throughput                5.2%   -> DRAM is NOT the problem
//
//  Per warp per kk the LDS pipe needs 32 wavefronts while the FFMA pipe needs
//  only 16 cycles, so the kernel is capped at 35.6 * 16/32 = 17.8 TFLOP/s.
//
//  v1 stored As as [BM][BK], which makes regA[i] = As[tRow*TM+i][kk] stride by
//  BK -- the compiler cannot vectorize it (it DID vectorize regB, which is why
//  we measured 10 LDS/kk instead of 16), and the two tRow values in a warp
//  land 64 words apart == same bank == 2-way conflict.
//
//  Goal here: regA 16 -> 8 wavefronts, lifting the ceiling to ~23.7 TFLOP/s.
//  Keep gemm_regtile_v1 untouched so both run side by side.
// ===========================================================================

// TODO(you): pick the pad.  A transposed STORE writes As[k][row] for 4
// consecutive k, i.e. addresses stride by the row pitch. With pitch = BM = 128
// words, 128 % 32 == 0, so every k lands in the SAME bank -- and across a warp
// the two halves collide 2-way while banks 16..31 sit idle.
// Padding the pitch breaks that alignment. Work out on paper which pad value
// makes BOTH of these conflict-free, then set it:
//   (i)  the transposed store   As[k0_local + c][row]
//   (ii) the vectorized read    As[kk][tRow*TM + i], i = 0..TM-1
#ifndef AS_PAD
#define AS_PAD 4        // <-- 0 is deliberately wrong. Derive the right value.
#endif

__global__ void gemm_vec_v1(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int n)
{
    // TODO(you): __shared__ float As[BK][BM + AS_PAD];   // TRANSPOSED
    //            __shared__ float Bs[BK][BN];            // unchanged
    __shared__ float As[BK][BM + AS_PAD], Bs[BK][BN];
    // TODO(you): same tRow / tCol micro-tile mapping as v1.
    const int tRow = threadIdx.x / (BN / TN);
    const int tCol = threadIdx.x % (BN / TN);
    const int tid = threadIdx.x;

    float acc[TM][TN] = {0.0f};
    float regA[TM], regB[TN];
    for (int k0 = 0; k0 < n; k0 += BK) {
    // ---- (a) global -> shared, now with float4 ----------------------------
    // The A tile is BM*BK = 128*8 = 1024 floats = 256 float4s, and there are
    // exactly 256 threads, so each thread moves ONE float4 (was: 4 scalars in
    // a loop). Same for B.
    //   TODO: derive the float4 mapping for A. Each row of the A tile is BK=8
    //         floats = 2 float4s, so 128 rows * 2 = 256. Which (row, half)
    //         does thread tid own?
    //   TODO: the A store is TRANSPOSED -- one float4 read becomes 4 scalar
    //         stores As[k+0..3][row]. There is no way around the scatter; the
    //         pad is what keeps it conflict-free.
    //   TODO: B needs no transpose, so its float4 can be stored as a float4.
    //   TODO: alignment. reinterpret_cast<const float4*> requires the address
    //         to be 16-byte aligned, which needs n % 4 == 0. Guard it: fall
    //         back to the scalar path (or just bail to v1) otherwise, and
    //         still zero-fill out-of-range elements.
        if (n % 4 == 0) {
            int row_a = tid / 2;
            int vec_a = tid % 2;
            int global_row_a = blockIdx.y * BM + row_a;
            int global_col_a = k0 + vec_a * 4;
            
            float4 val_a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (global_row_a < n && global_col_a + 3 < n) {
                val_a = *reinterpret_cast<const float4*>(&A[global_row_a * n + global_col_a]);
            }
            else if (global_row_a < n) {
                if (global_col_a + 0 < n) val_a.x = A[global_row_a * n + global_col_a + 0];
                if (global_col_a + 1 < n) val_a.y = A[global_row_a * n + global_col_a + 1];
                if (global_col_a + 2 < n) val_a.z = A[global_row_a * n + global_col_a + 2];
                if (global_col_a + 3 < n) val_a.w = A[global_row_a * n + global_col_a + 3];
            }

            int smem_col_a = row_a;
            int smem_row_a_base = vec_a * 4;
            As[smem_row_a_base + 0][smem_col_a] = val_a.x;
            As[smem_row_a_base + 1][smem_col_a] = val_a.y;
            As[smem_row_a_base + 2][smem_col_a] = val_a.z;
            As[smem_row_a_base + 3][smem_col_a] = val_a.w;

            // 2. 載入 B 並直接寫入 Bs
            int row_b = tid / 32;
            int col_b_vec = tid % 32;
            int global_row_b = k0 + row_b;
            int global_col_b = blockIdx.x * BN + col_b_vec * 4;

            float4 val_b = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (global_row_b < n && global_col_b + 3 < n) {
                val_b = *reinterpret_cast<const float4*>(&B[global_row_b * n + global_col_b]);
            }
            else if (global_row_b < n) {
                if (global_col_b + 0 < n) val_b.x = B[global_row_b * n + global_col_b + 0];
                if (global_col_b + 1 < n) val_b.y = B[global_row_b * n + global_col_b + 1];
                if (global_col_b + 2 < n) val_b.z = B[global_row_b * n + global_col_b + 2];
                if (global_col_b + 3 < n) val_b.w = B[global_row_b * n + global_col_b + 3];
            }

            reinterpret_cast<float4*>(&Bs[row_b][col_b_vec * 4])[0] = val_b;
        }
        else {
            for (int i=0; i<4; ++i) {
                // ---- (a) global -> shared (暫時保留 scalar 確保正確，或後續改 float4) ----
                int idx = threadIdx.x + i * 256;
                int row_a = idx / BK;
                int col_a = idx % BK;
                int global_row_a = blockIdx.y * BM + row_a;
                int global_col_a = k0 + col_a;
                // 注意這裡：As 是 [BK][BM + AS_PAD]，所以寫入時要轉置為 [col_a][row_a]
                As[col_a][row_a] = (global_row_a < n && global_col_a < n) ? A[global_row_a * n + global_col_a] : 0.0f;
                int row_b = idx / BN;
                int col_b = idx % BN;
                int global_row_b = k0 + row_b;
                int global_col_b = blockIdx.x * BN + col_b;
                Bs[row_b][col_b] = (global_row_b < n && global_col_b < n) ? B[global_row_b * n + global_col_b] : 0.0f;
            }
        }
    // ---- (b) __syncthreads() ----------------------------------------------
        __syncthreads();
    // ---- (c) inner loop ---------------------------------------------------
    //   #pragma unroll  on EVERY loop below -- write them out this time, v1
    //   only worked because nvcc happened to auto-unroll at -O3.
    //     for (kk < BK) {
    //         regA[0..TM) <- As[kk][tRow*TM ..]    // now contiguous -> float4
    //         regB[0..TN) <- Bs[kk][tCol*TN ..]    // already float4 in v1
    //         TM*TN FFMAs into acc[][]
    //     }
        for (int kk = 0; kk < BK; ++kk) {
            #pragma unroll
            for (int i=0; i<TM; ++i) { // TM smem loads
                regA[i] = As[kk][tRow * TM + i]; // 讀取轉置後的 As
            }        
            #pragma unroll
            for (int j=0; j<TN; ++j) { // TN smem loads
                regB[j] = Bs[kk][tCol * TN + j];
            }
            #pragma unroll
            for (int i=0; i<TM; ++i) {
                #pragma unroll
                for (int j=0; j<TN; ++j) { // TM*TN FFMA
                    acc[i][j] += regA[i] * regB[j];
                }
            }
        }
    // ---- (d) __syncthreads() ----------------------------------------------
        __syncthreads();
    }
    // ---- write back -------------------------------------------------------
    //   acc[i][0..TN) is contiguous in C -> 2 float4 stores per row of the
    //   micro-tile, when alignment and bounds allow.
    for (int i=0; i<TM; ++i) {
        for (int j=0; j<TN; ++j) {
            int global_row = blockIdx.y * BM + tRow * TM + i;
            int global_col = blockIdx.x * BN + tCol * TN + j; 
            if (global_row < n && global_col < n) {
                C[global_row * n + global_col] = acc[i][j];
            }
        }
    }   
    // CHECKLIST, in this order -- do not skip ahead to GFLOP/s:
    //   1. make resource            -> spill MUST still be 0
    //   2. make run N=1024          -> [PASS]
    //   3. make run N=1000          -> [PASS]  (odd n, exercises the fallback)
    //   4. make bank KERNEL=gemm_vec_v1  -> wavefronts/inst should drop 3.20 -> ~2.4
    //   5. make run N=4096          -> only now look at throughput
}

// ===========================================================================
//  STAGE 4b : split the B micro-tile to kill regB's bank conflict
//
//  ncu on gemm_vec_v1 (N=2048, AS_PAD=4):
//      shared-ld instructions   16,777,216   (4 per warp per kk)
//      wavefronts               83,886,846   (20.0 per warp per kk)
//      bank conflicts           33,554,432   -- ALL of it is regB
//
//  Wavefront budget per warp per kk:
//      regA  2 x LDS.128 @ 2 (broadcast: threads 0-15 share an address) =  4
//      regB  2 x LDS.128 @ 8 (2-way conflict)                          = 16
//                                                                  total 20
//  against an FFMA demand of 64 inst / 4 per cycle = 16 cycles. LDS still wins.
//
//  WHY regB collides: an LDS.128 is served 8 threads per cycle (8 x 16 B =
//  128 B). In one such phase tCol runs 0..7, so the word offsets are
//  tCol*8 = 0, 8, 16, 24, 32, 40, 48, 56. Each thread claims 4 consecutive
//  banks, so tCol=0 takes banks 0-3 and tCol=4 takes banks (32 mod 32)=0-3
//  as well -- different addresses, same banks, 2-way conflict.
//
//  FIX: make one phase cover 32 CONSECUTIVE words instead of spanning 64.
//  Each thread still owns TN=8 columns, but they come from two halves of BN:
//        regB[0..3] <- Bs[kk][tCol*4]
//        regB[4..7] <- Bs[kk][BN/2 + tCol*4]
//  Now a phase reads words 0,4,8,...,28 -> all 32 banks exactly once.
//
//  Expected: regB 16 -> 8 wavefronts, total 20 -> 12, which drops BELOW the
//  FFMA demand of 16. The LDS pipe stops being the limiter for the first time
//  in this project.
// ===========================================================================

__global__ void gemm_vec_v2(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int n)
{
    // Copied verbatim from gemm_vec_v1. Everything below is unchanged except
    // the two blocks marked "TODO(you) 4b" -- the regB load and the write-back.
    __shared__ float As[BK][BM + AS_PAD], Bs[BK][BN];
    const int tRow = threadIdx.x / (BN / TN);
    const int tCol = threadIdx.x % (BN / TN);
    const int tid = threadIdx.x;

    float acc[TM][TN] = {0.0f};
    float regA[TM], regB[TN];
    for (int k0 = 0; k0 < n; k0 += BK) {
        if (n % 4 == 0) {
            int row_a = tid / 2;
            int vec_a = tid % 2;
            int global_row_a = blockIdx.y * BM + row_a;
            int global_col_a = k0 + vec_a * 4;

            float4 val_a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (global_row_a < n && global_col_a + 3 < n) {
                val_a = *reinterpret_cast<const float4*>(&A[global_row_a * n + global_col_a]);
            }
            else if (global_row_a < n) {
                if (global_col_a + 0 < n) val_a.x = A[global_row_a * n + global_col_a + 0];
                if (global_col_a + 1 < n) val_a.y = A[global_row_a * n + global_col_a + 1];
                if (global_col_a + 2 < n) val_a.z = A[global_row_a * n + global_col_a + 2];
                if (global_col_a + 3 < n) val_a.w = A[global_row_a * n + global_col_a + 3];
            }

            int smem_col_a = row_a;
            int smem_row_a_base = vec_a * 4;
            As[smem_row_a_base + 0][smem_col_a] = val_a.x;
            As[smem_row_a_base + 1][smem_col_a] = val_a.y;
            As[smem_row_a_base + 2][smem_col_a] = val_a.z;
            As[smem_row_a_base + 3][smem_col_a] = val_a.w;

            int row_b = tid / 32;
            int col_b_vec = tid % 32;
            int global_row_b = k0 + row_b;
            int global_col_b = blockIdx.x * BN + col_b_vec * 4;

            float4 val_b = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (global_row_b < n && global_col_b + 3 < n) {
                val_b = *reinterpret_cast<const float4*>(&B[global_row_b * n + global_col_b]);
            }
            else if (global_row_b < n) {
                if (global_col_b + 0 < n) val_b.x = B[global_row_b * n + global_col_b + 0];
                if (global_col_b + 1 < n) val_b.y = B[global_row_b * n + global_col_b + 1];
                if (global_col_b + 2 < n) val_b.z = B[global_row_b * n + global_col_b + 2];
                if (global_col_b + 3 < n) val_b.w = B[global_row_b * n + global_col_b + 3];
            }

            reinterpret_cast<float4*>(&Bs[row_b][col_b_vec * 4])[0] = val_b;
        }
        else {
            for (int i=0; i<4; ++i) {
                int idx = threadIdx.x + i * 256;
                int row_a = idx / BK;
                int col_a = idx % BK;
                int global_row_a = blockIdx.y * BM + row_a;
                int global_col_a = k0 + col_a;
                As[col_a][row_a] = (global_row_a < n && global_col_a < n) ? A[global_row_a * n + global_col_a] : 0.0f;
                int row_b = idx / BN;
                int col_b = idx % BN;
                int global_row_b = k0 + row_b;
                int global_col_b = blockIdx.x * BN + col_b;
                Bs[row_b][col_b] = (global_row_b < n && global_col_b < n) ? B[global_row_b * n + global_col_b] : 0.0f;
            }
        }
        __syncthreads();
        for (int kk = 0; kk < BK; ++kk) {
            #pragma unroll
            for (int i=0; i<TM; ++i) { // TM smem loads
                regA[i] = As[kk][tRow * TM + i]; // 讀取轉置後的 As
            }

            // ================= TODO(you) 4b : CHANGE #1 =================
            // v1 read 8 contiguous columns starting at tCol*TN, which makes an
            // LDS.128 phase span 64 words and collide 2-way.
            // Split it into two halves of BN so a phase covers 32 consecutive
            // words:   regB[0..3] <- Bs[kk][tCol*4]
            //          regB[4..7] <- Bs[kk][BN/2 + tCol*4]
            #pragma unroll
            for (int j=0; j<4; ++j) { // TN smem loads
                regB[j] = Bs[kk][tCol * 4 + j];        // <-- replace this
                regB[j + 4] = Bs[kk][BN / 2 + tCol * 4 + j];
            }
            // ============================================================

            #pragma unroll
            for (int i=0; i<TM; ++i) {
                #pragma unroll
                for (int j=0; j<TN; ++j) { // TM*TN FFMA
                    acc[i][j] += regA[i] * regB[j];
                }
            }
        }
        __syncthreads();
    }

    // ==================== TODO(you) 4b : CHANGE #2 ====================
    // acc[i][j] no longer maps to 8 contiguous columns of C. Derive the new
    // column for each j and make this agree with CHANGE #1:
    //     j = 0..3  ->  blockIdx.x*BN + ?
    //     j = 4..7  ->  blockIdx.x*BN + ?
    // Leaving v1's mapping here compiles, runs, reports a plausible GFLOP/s,
    // and is WRONG. Check N=1024 and N=1000 before looking at throughput.
    for (int i=0; i<TM; ++i) {
        for (int j=0; j<TN; ++j) {
            int global_row = blockIdx.y * BM + tRow * TM + i;
            int global_col = (j < 4) ? blockIdx.x * BN + tCol * 4 + j : blockIdx.x * BN + BN / 2 + tCol * 4 + (j - 4);   // <-- fix this
            if (global_row < n && global_col < n) {
                C[global_row * n + global_col] = acc[i][j];
            }
        }
    }
    // ==================================================================
}

// ===========================================================================
//  STAGE 4c : register-level double buffering (software pipelining)
//
//  ncu on gemm_vec_v2 (N=2048) -- bank conflicts are gone, so what is left:
//      shared-ld bank conflicts            0
//      shared-st bank conflicts            0
//      wavefronts/warp/kk               12.0   (LDS 12 < FFMA 16: not the cap)
//      sm__pipe_fma_cycles_active      52.84%  <- the FMA pipe idles half the time
//      DRAM throughput                  4.97%
//    warp issue stalls (% of warp-active cycles):
//      not_selected                    25.75%  <- healthy: spare warps exist
//      short_scoreboard                18.87%  <- waiting on shared memory
//      long_scoreboard                 16.65%  <- waiting on global memory
//      barrier                          6.72%
//
//  The bus is idle AND warps are waiting: this is a LATENCY problem, not a
//  bandwidth one. Prefetching the kk+1 fragments while the kk FFMAs run lets
//  the FFMAs cover the shared-memory latency.
//
//  Scope: this kernel only pipelines the smem->register hop (short_scoreboard).
//  The global->smem hop (long_scoreboard + barrier) needs a ping-pong smem
//  buffer and a reordered k-loop; that is left as future work.
//
//  Realistic expectation: 10-20%, i.e. ~18-19 TFLOP/s. Speedup estimates in
//  this project have run optimistic twice (Stage 2 and Stage 4b), so do not
//  budget for more.
// ===========================================================================

__global__ void gemm_db_v1(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int n)
{
    // Copied verbatim from gemm_vec_v2 (Stage 4b, zero bank conflicts).
    __shared__ float As[BK][BM + AS_PAD], Bs[BK][BN];
    const int tRow = threadIdx.x / (BN / TN);
    const int tCol = threadIdx.x % (BN / TN);
    const int tid = threadIdx.x;

    float acc[TM][TN] = {0.0f};
    // float regA[TM], regB[TN];
    for (int k0 = 0; k0 < n; k0 += BK) {
        if (n % 4 == 0) {
            int row_a = tid / 2;
            int vec_a = tid % 2;
            int global_row_a = blockIdx.y * BM + row_a;
            int global_col_a = k0 + vec_a * 4;

            float4 val_a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (global_row_a < n && global_col_a + 3 < n) {
                val_a = *reinterpret_cast<const float4*>(&A[global_row_a * n + global_col_a]);
            }
            else if (global_row_a < n) {
                if (global_col_a + 0 < n) val_a.x = A[global_row_a * n + global_col_a + 0];
                if (global_col_a + 1 < n) val_a.y = A[global_row_a * n + global_col_a + 1];
                if (global_col_a + 2 < n) val_a.z = A[global_row_a * n + global_col_a + 2];
                if (global_col_a + 3 < n) val_a.w = A[global_row_a * n + global_col_a + 3];
            }

            int smem_col_a = row_a;
            int smem_row_a_base = vec_a * 4;
            As[smem_row_a_base + 0][smem_col_a] = val_a.x;
            As[smem_row_a_base + 1][smem_col_a] = val_a.y;
            As[smem_row_a_base + 2][smem_col_a] = val_a.z;
            As[smem_row_a_base + 3][smem_col_a] = val_a.w;

            int row_b = tid / 32;
            int col_b_vec = tid % 32;
            int global_row_b = k0 + row_b;
            int global_col_b = blockIdx.x * BN + col_b_vec * 4;

            float4 val_b = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (global_row_b < n && global_col_b + 3 < n) {
                val_b = *reinterpret_cast<const float4*>(&B[global_row_b * n + global_col_b]);
            }
            else if (global_row_b < n) {
                if (global_col_b + 0 < n) val_b.x = B[global_row_b * n + global_col_b + 0];
                if (global_col_b + 1 < n) val_b.y = B[global_row_b * n + global_col_b + 1];
                if (global_col_b + 2 < n) val_b.z = B[global_row_b * n + global_col_b + 2];
                if (global_col_b + 3 < n) val_b.w = B[global_row_b * n + global_col_b + 3];
            }

            reinterpret_cast<float4*>(&Bs[row_b][col_b_vec * 4])[0] = val_b;
        }
        else {
            for (int i=0; i<4; ++i) {
                int idx = threadIdx.x + i * 256;
                int row_a = idx / BK;
                int col_a = idx % BK;
                int global_row_a = blockIdx.y * BM + row_a;
                int global_col_a = k0 + col_a;
                As[col_a][row_a] = (global_row_a < n && global_col_a < n) ? A[global_row_a * n + global_col_a] : 0.0f;
                int row_b = idx / BN;
                int col_b = idx % BN;
                int global_row_b = k0 + row_b;
                int global_col_b = blockIdx.x * BN + col_b;
                Bs[row_b][col_b] = (global_row_b < n && global_col_b < n) ? B[global_row_b * n + global_col_b] : 0.0f;
            }
        }
        __syncthreads();
        // ============== TODO(you) 4c : RESTRUCTURE THIS LOOP ==============
        // Today every kk iteration is: issue 4 LDS, WAIT for them, then do 64
        // FFMAs that depend on them. That wait is the 18.87% short_scoreboard.
        //
        // Software pipelining breaks the dependency by one iteration:
        //
        //   PROLOGUE   load the kk = 0 fragments
        //   for (kk = 0; kk < BK; ++kk) {
        //       if (kk + 1 < BK)  issue the loads for kk+1   <-- no wait here
        //       64 FFMAs using the kk fragments              <-- covers the wait
        //       the kk+1 fragments become current
        //   }
        //
        // Two ways to hold two generations of fragments:
        //   (A) float regA[2][TM], regB[2][TN], indexed by (kk & 1).
        //       REQUIRES the kk loop to be fully unrolled, or (kk & 1) is not a
        //       compile-time constant and BOTH arrays go to local memory.
        //       Note 4b has no #pragma unroll on this loop -- nvcc did it for
        //       you. For 4c, write it explicitly. Do not rely on luck.
        //   (B) separate cur/next sets with an explicit copy at the end. More
        //       registers live at once, but no indexing question.
        //
        // !! THE REAL CONSTRAINT !! You are at 118 registers and 2 blocks/SM.
        // sm_86 gives 65536 registers per SM, so 2 blocks x 256 threads caps
        // you at 65536/(2*256) = 128 registers. Budget: 10.
        //     double-buffer regA AND regB  -> +16 -> 134 -> 1 block -> 16.7%
        //     double-buffer only one       -> + 8 -> 126 -> 2 blocks -> 33.3%
        // Decide which, and predict the outcome, BEFORE you measure. Then run
        // make resource FIRST -- if registers cross 128 or spill appears, the
        // occupancy you lose may cost more than the latency you hid.
        // ==================================================================
        // Prologue：先載入 kk = 0 的 fragments
        float regA_cur[TM], regB_cur[TN];
        #pragma unroll
        for (int i=0; i<TM; ++i) {
            regA_cur[i] = As[0][tRow * TM + i];
        }
        #pragma unroll
        for (int j=0; j<4; ++j) {
            regB_cur[j]     = Bs[0][tCol * 4 + j];
            regB_cur[j + 4] = Bs[0][BN / 2 + tCol * 4 + j];
        }

        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            // 如果還有下一輪，預先發出 kk + 1 的載入（不等待，讓下面的 FFMAs 掩蓋延遲）
            float regA_next[TM], regB_next[TN];
            if (kk + 1 < BK) {
                #pragma unroll
                for (int i=0; i<TM; ++i) {
                    regA_next[i] = As[kk + 1][tRow * TM + i];
                }
                #pragma unroll
                for (int j=0; j<4; ++j) {
                    regB_next[j]     = Bs[kk + 1][tCol * 4 + j];
                    regB_next[j + 4] = Bs[kk + 1][BN / 2 + tCol * 4 + j];
                }
            }

            // 使用當前（cur）的 fragments 進行運算
            #pragma unroll
            for (int i=0; i<TM; ++i) {
                #pragma unroll
                for (int j=0; j<TN; ++j) {
                    acc[i][j] += regA_cur[i] * regB_cur[j];
                }
            }

            // 將 next 推進成為 cur，準備進入下一輪
            if (kk + 1 < BK) {
                #pragma unroll
                for (int i=0; i<TM; ++i) {
                    regA_cur[i] = regA_next[i];
                }
                #pragma unroll
                for (int j=0; j<TN; ++j) {
                    regB_cur[j] = regB_next[j];
                }
            }
        }
        __syncthreads();
    }

    for (int i=0; i<TM; ++i) {
        for (int j=0; j<TN; ++j) {
            int global_row = blockIdx.y * BM + tRow * TM + i;
            int global_col = (j < 4) ? blockIdx.x * BN + tCol * 4 + j : blockIdx.x * BN + BN / 2 + tCol * 4 + (j - 4);
            if (global_row < n && global_col < n) {
                C[global_row * n + global_col] = acc[i][j];
            }
        }
    }
}

// ===========================================================================
//  Future work : shared-memory ping-pong, wider BK, cp.async (sm_80+)
// ===========================================================================

// ---------------------------------------------------------------------------
// Launch wrappers  (one per kernel, so the benchmark loop stays generic)
// ---------------------------------------------------------------------------
static void launch_naive_v1(const float* A, const float* B, float* C, int n) {
    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid(ceil_div(n, BLOCK_DIM), ceil_div(n, BLOCK_DIM));
    gemm_naive_v1<<<grid, block>>>(A, B, C, n);
}

static void launch_naive_v2(const float* A, const float* B, float* C, int n) {
    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid(ceil_div(n, BLOCK_DIM), ceil_div(n, BLOCK_DIM));
    gemm_naive_v2<<<grid, block>>>(A, B, C, n);
}

static void launch_tiled_v1(const float* A, const float* B, float* C, int n) {
    dim3 block(TILE, TILE);
    dim3 grid(ceil_div(n, TILE), ceil_div(n, TILE));
    gemm_tiled_v1<<<grid, block>>>(A, B, C, n);
}

static void launch_regtile_v1(const float* A, const float* B, float* C, int n) {
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(ceil_div(n, BN), ceil_div(n, BM));
    gemm_regtile_v1<<<grid, block>>>(A, B, C, n);
}

static void launch_vec_v1(const float* A, const float* B, float* C, int n) {
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(ceil_div(n, BN), ceil_div(n, BM));
    gemm_vec_v1<<<grid, block>>>(A, B, C, n);
}

static void launch_vec_v2(const float* A, const float* B, float* C, int n) {
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(ceil_div(n, BN), ceil_div(n, BM));
    gemm_vec_v2<<<grid, block>>>(A, B, C, n);
}

static void launch_db_v1(const float* A, const float* B, float* C, int n) {
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(ceil_div(n, BN), ceil_div(n, BM));
    gemm_db_v1<<<grid, block>>>(A, B, C, n);
}

#ifdef USE_CUBLAS
static cublasHandle_t g_cublas = nullptr;
static void launch_cublas(const float* A, const float* B, float* C, int n) {
    const float alpha = 1.0f, beta = 0.0f;
    // cuBLAS is column-major. Our data is row-major.
    // Trick: C^T = B^T * A^T  -> just swap the A/B arguments.
    cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n, &alpha, B, n, A, n, &beta, C, n);
}
#endif

struct KernelEntry {
    const char* name;
    void (*launch)(const float*, const float*, float*, int);
};

static const KernelEntry g_kernels[] = {
    { "1. naive  (row=y, col=x)  [coalesced]",   launch_naive_v1 },
    { "2. naive  (row=x, col=y)  [strided]",     launch_naive_v2 },
    { "3. tiled  (shared memory)",               launch_tiled_v1 },
    { "4. regtile (128x128 blk, 8x8/thread)",    launch_regtile_v1 },
    { "5. vec     (As^T + float4 smem loads)",   launch_vec_v1     },
    { "6. vec2    (+ split regB, no conflicts)",  launch_vec_v2     },
    { "7. db      (+ register double buffering)", launch_db_v1      },
#ifdef USE_CUBLAS
    { "0. cuBLAS SGEMM           [reference]",   launch_cublas   },
#endif
};
static const int g_num_kernels = sizeof(g_kernels) / sizeof(g_kernels[0]);

// ---------------------------------------------------------------------------
// CPU golden reference.  ikj loop order = cache-friendly, accumulate in double.
// ---------------------------------------------------------------------------
static void gemm_cpu_reference(const float* A, const float* B,
                               double* C, int n)
{
    memset(C, 0, sizeof(double) * (size_t)n * n);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            const double a = A[(size_t)i * n + k];
            const float* brow = B + (size_t)k * n;
            double* crow = C + (size_t)i * n;
            for (int j = 0; j < n; ++j) crow[j] += a * brow[j];
        }
    }
}

// Verify a float GEMM result against a double reference.
//
// NOTE: per-element relative error is the WRONG metric here. A dot product of
// n random signed terms can cancel down to ~0, which makes |err|/|ref| explode
// even for a perfectly good kernel. Use instead:
//   * normwise relative error : max|err| / max|ref|      (scale-invariant)
//   * backward error          : max|err| / (n*eps*|A||B|)  -- should be O(1)
static bool verify(const float* gpu, const double* ref, int n,
                   double max_a, double max_b, double tol)
{
    double max_abs = 0.0, ref_norm = 0.0;
    int bad_i = -1, bad_j = -1;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            size_t idx = (size_t)i * n + j;
            double r = ref[idx];
            if (fabs(r) > ref_norm) ref_norm = fabs(r);
            double abs_e = fabs((double)gpu[idx] - r);
            if (abs_e > max_abs) { max_abs = abs_e; bad_i = i; bad_j = j; }
        }
    }
    const double rel      = max_abs / (ref_norm + 1e-30);
    const double expected = (double)n * FLT_EPSILON * max_a * max_b;
    const double backward = max_abs / (expected + 1e-30);
    const bool   ok       = (rel <= tol);

    printf("      verify : max_abs=%.3e  normwise_rel=%.3e  backward=%.2f  %s\n",
           max_abs, rel, backward, ok ? "[PASS]" : "[FAIL]");
    if (!ok) {
        size_t idx = (size_t)bad_i * n + bad_j;
        printf("               worst at C[%d][%d]: gpu=%.8f ref=%.8f\n",
               bad_i, bad_j, gpu[idx], ref[idx]);
    }
    return ok;
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    const int n        = (argc > 1) ? atoi(argv[1]) : 1024;
    const int only_id  = (argc > 2) ? atoi(argv[2]) : -1;   // 1-based, -1 = all

    if (n <= 0) { fprintf(stderr, "usage: %s <N> [kernel_id]\n", argv[0]); return 1; }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    const size_t elems = (size_t)n * n;
    const size_t bytes = elems * sizeof(float);
    const double gflop = 2.0 * (double)n * n * n / 1e9;

    printf("================================================================\n");
    printf("  GPU      : %s  (sm_%d%d, %d SMs)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("  Problem  : C[%d x %d] = A * B   (FP32, row-major)\n", n, n);
    printf("  Workload : %.2f GFLOP,  %.1f MB/matrix\n", gflop, bytes / 1048576.0);
    printf("  Roofline : peak %.1f TFLOP/s | DRAM %.0f GB/s | ridge %.0f FLOP/B\n",
           PEAK_FP32_GFLOPS / 1000.0, PEAK_BW_GBPS,
           PEAK_FP32_GFLOPS / PEAK_BW_GBPS);
    printf("================================================================\n");

    // ---- host buffers ----
    float* hA = (float*)malloc(bytes);
    float* hB = (float*)malloc(bytes);
    float* hC = (float*)malloc(bytes);
    if (!hA || !hB || !hC) { fprintf(stderr, "host alloc failed\n"); return 1; }

    srand(42);                                    // deterministic input
    double max_a = 0.0, max_b = 0.0;
    for (size_t i = 0; i < elems; ++i) {
        hA[i] = (float)rand() / RAND_MAX - 0.5f;  // [-0.5, 0.5)
        hB[i] = (float)rand() / RAND_MAX - 0.5f;
        if (fabs(hA[i]) > max_a) max_a = fabs(hA[i]);
        if (fabs(hB[i]) > max_b) max_b = fabs(hB[i]);
    }

    // ---- golden reference (small N only) ----
    const bool do_verify = (n <= VERIFY_MAX_N);
    double* hRef = nullptr;
    if (do_verify) {
        printf("\n[CPU] computing golden reference (N=%d)...\n", n);
        hRef = (double*)malloc(sizeof(double) * elems);
        gemm_cpu_reference(hA, hB, hRef, n);
        printf("[CPU] done.\n");
    } else {
        printf("\n[CPU] N=%d > %d, skipping verification (perf mode).\n",
               n, VERIFY_MAX_N);
    }

    // ---- device buffers ----
    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

#ifdef USE_CUBLAS
    cublasCreate(&g_cublas);
#endif

    cudaEvent_t beg, end;
    CUDA_CHECK(cudaEventCreate(&beg));
    CUDA_CHECK(cudaEventCreate(&end));

    const int warmup = 2;
    const int iters  = (n <= 512) ? 20 : 5;
    double best_gflops = 0.0;

    printf("\n%-42s %10s %12s %8s\n", "kernel", "time(ms)", "GFLOP/s", "%peak");
    printf("----------------------------------------------------------------------------\n");

    for (int id = 0; id < g_num_kernels; ++id) {
        if (only_id > 0 && (id + 1) != only_id) continue;

        CUDA_CHECK(cudaMemset(dC, 0, bytes));

        // --- warmup (also surfaces launch errors before we time anything) ---
        for (int w = 0; w < warmup; ++w) g_kernels[id].launch(dA, dB, dC, n);
        CUDA_CHECK_KERNEL();

        // --- timed ---
        CUDA_CHECK(cudaEventRecord(beg));
        for (int it = 0; it < iters; ++it) g_kernels[id].launch(dA, dB, dC, n);
        CUDA_CHECK(cudaEventRecord(end));
        CUDA_CHECK(cudaEventSynchronize(end));

        float ms_total = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms_total, beg, end));
        const double ms     = ms_total / iters;
        const double gflops = gflop / (ms * 1e-3);
        if (gflops > best_gflops) best_gflops = gflops;

        printf("%-42s %10.3f %12.1f %7.2f%%\n", g_kernels[id].name,
               ms, gflops, 100.0 * gflops / PEAK_FP32_GFLOPS);

        if (do_verify) {
            CUDA_CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));
            verify(hC, hRef, n, max_a, max_b, 1e-5);
        }
    }

    printf("----------------------------------------------------------------------------\n");
    printf("  best: %.1f GFLOP/s  (%.2f%% of peak, %.2fx the 234 GFLOP/s naive-DRAM bound)\n",
           best_gflops, 100.0 * best_gflops / PEAK_FP32_GFLOPS, best_gflops / 234.0);

#ifdef USE_CUBLAS
    cublasDestroy(g_cublas);
#endif
    CUDA_CHECK(cudaEventDestroy(beg));
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    free(hA); free(hB); free(hC); free(hRef);
    return 0;
}
