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
//  STAGE 2 : Shared-Memory Tiling      (Cudabook Ch.5)   -- later
//  STAGE 3 : Thread-Level / Register Tiling (Ch.6)       -- later
//  STAGE 4 : Unrolling / Vectorized LD / Double Buffer   -- later
// ===========================================================================

// ---------------------------------------------------------------------------
// Launch wrappers  (one per kernel, so the benchmark loop stays generic)
// ---------------------------------------------------------------------------
#ifndef BLOCK_DIM
#define BLOCK_DIM 16   // 16x16 = 256 thr/block -> 6 blocks/SM -> 100% occupancy.
                       // Try 32 (=1024 thr) and explain the occupancy drop.
#endif

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

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
