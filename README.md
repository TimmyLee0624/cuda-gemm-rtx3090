# CUDA SGEMM on RTX 3090 — from 2.3 to 19.0 TFLOP/s (74.8% of cuBLAS)

<!-- TODO: one paragraph. What this repo is, and the one thing that makes it
     different from the other 500 "CUDA matmul tutorial" repos: every step was
     PREDICTED from a roofline model before it was measured, and the times the
     prediction was wrong are written down. Keep it to 3-4 sentences. -->

A from-scratch FP32 GEMM (`C = A × B`, row-major, square) optimized in seven
stages on a single RTX 3090, following the structure of *Programming Massively
Parallel Processors*. Every kernel is verified against a CPU reference and
profiled with Nsight Compute.

```
Hardware   NVIDIA GeForce RTX 3090 (GA102, sm_86, 82 SMs)
           FP32 peak    35.6 TFLOP/s
           DRAM         936 GB/s        -> ridge point 38 FLOP/Byte
           Shared mem   17.8 TB/s       (82 SM × 128 B/clk × 1.695 GHz)
Toolchain  CUDA 12.9, nvcc -O3 -arch=sm_86 -lineinfo
```

## Results

All figures at `N = 4096`, single kernel launch averaged over 5 iterations.

| # | Kernel | Time (ms) | GFLOP/s | % peak | % cuBLAS | Bottleneck it removed |
|---|--------|----------:|--------:|-------:|---------:|---|
| 1 | Naive, coalesced | 60.39 | 2276 | 6.39% | 9.0% | — (baseline) |
| 2 | Naive, strided (anti-pattern) | 230.84 | 595 | 1.67% | 2.3% | — (shows what coalescing is worth) |
| 3 | Shared-memory tiling | 45.38 | 3029 | 8.51% | 11.9% | best-effort L1 reuse |
| 4 | Register tiling 128×128, 8×8/thread | 10.38 | 13243 | 37.20% | 52.3% | DRAM bandwidth |
| 5 | Transposed `As` + `float4` shared loads | 8.55 | 16077 | 45.16% | 63.4% | shared-memory instruction count |
| 6 | Split `regB` micro-tile | 8.29 | 16589 | 46.60% | 65.5% | shared-memory bank conflicts |
| 7 | Register double buffering | **7.25** | **18962** | **53.26%** | **74.8%** | shared-memory **latency** |
| — | cuBLAS SGEMM (reference ceiling) | 5.42 | 25347 | 71.20% | 100% | — |

**8.3× over the naive baseline, 31.8× over the uncoalesced variant.**

The rightmost column is the point: every stage removed a *different*, measured
bottleneck. The remaining one is global-memory latency
(`long_scoreboard` 23.26%), which needs a shared-memory ping-pong buffer.

Correctness is checked at `N = 1024` and `N = 1000` (deliberately not a
multiple of the tile size) against a double-precision CPU reference.

## Build & run

```bash
make                      # or: make CUBLAS=1  to include the cuBLAS reference
make run N=4096           # always rebuilds first
make run N=1024           # N <= 1024 also runs CPU verification
make resource             # per-kernel register / shared-mem / spill report
make bank KERNEL=gemm_vec_v1 ID=5    # shared-memory bank-conflict counters
```

## Method

<!-- TODO: explain the loop you actually followed. Roughly:
       1. derive an upper bound from a roofline model
       2. write down the predicted number BEFORE running anything
       3. measure
       4. reconcile - and when they disagree, fix the MODEL, not just the code
     Two or three sentences. This section is the reason the repo exists. -->

### Verifying a float GEMM

<!-- TODO: explain why per-element relative error is the wrong metric here
     (catastrophic cancellation in a length-N dot product of signed randoms),
     and what the harness uses instead:
       normwise relative error   max|err| / max|ref|
       backward error            max|err| / (n · eps · max|A| · max|B|)
     State what a healthy backward error looks like (O(1)) and what it looked
     like when a kernel was genuinely broken (~5 x 10^5). -->

## Stage 1 — Naive baseline

<!-- TODO: arithmetic intensity of one-element-per-thread (0.25 FLOP/Byte),
     the 234 GFLOP/s DRAM bound it implies, and why the measured 2259 GFLOP/s
     is 9.5x HIGHER than that bound. -->

**Coalescing.** The only difference between kernels 1 and 2 is which of
`blockIdx.x`/`.y` feeds `row` and which feeds `col`. Same result, 3.8× apart.

<!-- TODO: the per-matrix table — which access is coalesced, which is a warp
     broadcast, which is strided, in each variant. Note that the victims are
     A and C, not B, and that A in the fast variant is a BROADCAST, not a
     coalesced read. -->

**The ratio is a function of `BLOCK_DIM`, not a constant.** Counting 32-byte
sectors per warp predicts it:

| `BLOCK_DIM` | warp shape | v1 sectors | v2 sectors | predicted | measured |
|---|---|---|---|---|---|
| 8×8 | 8 col × 4 row | 5 | 9 | 1.80× | 1.78× |
| 16×16 | 16 col × 2 row | 4 | 17 | 4.25× | 4.24× |
| 32×32 | 32 col × 1 row | 5 | 33 | 6.60× | 7.77× |

## Stage 2 — Shared-memory tiling

<!-- TODO: the key counter-intuitive result. Block-level AI is TILE/4, and a
     naive kernel with a BD×BD block ALREADY gets AI = BD/4 for free from L1.
     So tiling does NOT raise arithmetic intensity - it converts best-effort
     L1 reuse into guaranteed staging. Hence 1.29x, not 10x. -->

| `BLOCK_DIM` | AI = BD/4 | roofline | measured (naive) | achieved |
|---|---|---|---|---|
| 8 | 2 | 1872 | 1783 | 95% |
| 16 | 4 | 3744 | 2220 | 59% |
| 32 | 8 | 7488 | 2158 | 29% |

<!-- TODO: read the last column out loud - L1 stops sustaining the reuse as
     the tile grows. THAT is the argument for shared memory, and it is why
     Stage 2 raised the 16×16 case from 59% to 76% of the same roofline. -->

## Stage 3 — Register tiling

<!-- TODO: derive 128×128 / 8×8 from two independent directions:
       (a) DRAM roofline:   AI = TILE/4, ridge point 38  ->  TILE ~= 152 -> 128
       (b) shared-mem roofline: 1 output/thread issues 2 LDS per FFMA, capping
           the kernel at 17.8 TB/s × 0.25 = 4.45 TFLOP/s. At TM=TN=8 it is
           16 LDS per 64 FFMA -> AI_smem = 2.0 -> ceiling 35.6 TFLOP/s = peak.
     Both land on 8×8. Say that explicitly. -->

**Occupancy fell from ~100% to 30% while throughput rose 4.4×.**

<!-- TODO: explain. 117 registers/thread limits the SM to 2 blocks. The reason
     it does not matter: occupancy exists to hide latency with TLP, and 64
     independent FFMAs per thread hide it with ILP instead. This is the single
     best story in the repo - do not bury it. -->

## Stage 4a — Transposed `As` and vectorized shared loads

Nsight Compute on Stage 3 (`N = 2048`):

```
shared-ld instructions    41,943,040     (10 per warp per kk)
wavefronts               134,218,687     ( 3.20 per instruction)
bank conflicts            67,108,864     ( exactly 2.00x oversubscribed)
DRAM throughput                5.24%     <- DRAM is no longer the problem
```

<!-- TODO: the wavefront budget argument. Per warp per kk the LDS pipe needed
     32 wavefronts against an FFMA demand of 16 cycles, capping the kernel at
     35.6 × 16/32 = 17.8 TFLOP/s. Measured 13.3 = 75% of that. -->

**`As[BM][BK]` was the mistake.** <!-- TODO: it makes regA stride by BK, which
     (a) prevents the compiler from vectorizing it - note that it DID vectorize
     regB, which is how 10 LDS/kk instead of 16 gives the 41.9M figure - and
     (b) puts the two tRow values of a warp 64 words apart == same bank. -->

**Choosing the pad.** Two constraints have to hold at once:

| | requirement |
|---|---|
| `float4` alignment on the `regA` read | `pitch ≡ 0 (mod 4)` |
| transposed store hits disjoint banks | `4·pitch ≡ 16 (mod 32)` → `pitch ≡ 4 (mod 8)` |

`pitch = BM + AS_PAD` and `BM = 128 ≡ 0 (mod 8)`, so **`AS_PAD = 4`**.

<!-- TODO: the failure worth writing down. "Pad with a prime" is folklore that
     targets bank conflicts alone; with an odd pitch, kk*pitch is not 4-word
     aligned, float4 silently degrades to 8 scalar LDS.32, and the kernel
     measures 11936 GFLOP/s instead of 14635. Give both numbers. -->

| | LDS inst | wavefronts/warp/kk | N=4096 |
|---|---|---|---|
| Stage 3, `As[BM][BK]` | 41,943,040 | 32 | 13267 |
| `AS_PAD = 7` | 41,943,040 | 24 | 11936 |
| `AS_PAD = 4` | **16,777,216** | **20** | **16053** |

<!-- TODO: note that regA came out at 4 wavefronts rather than the predicted 8,
     because threads 0-15 of a warp share one address and the hardware
     broadcasts. The implementation beat the model here. -->

## Stage 4b — Splitting the B micro-tile

<!-- TODO after implementing. Cover:
     - why an LDS.128 phase of 8 threads at stride 8 words collides 2-way
       (tCol and tCol+4 are 32 words = one full bank rotation apart)
     - the fix: regB[0..3] from Bs[kk][tCol*4], regB[4..7] from
       Bs[kk][BN/2 + tCol*4], so a phase covers 32 consecutive words
     - what it does to acc[][]'s column mapping and the write-back
     - measured wavefronts and throughput -->

## Stage 4c — Register double buffering

Once the bank conflicts were gone, shared-memory *bandwidth* was no longer the
cap (12 wavefronts/warp/kk against an FFMA demand of 16) but the FMA pipe still
idled half the time. The remaining cost was the LDS→FFMA *dependency*: issue 4
loads, wait, then run 64 FFMAs that need them.

Software pipelining moves the loads one iteration ahead, so the kk FFMAs cover
the kk+1 loads' latency.

| Stall (% of warp-active cycles) | Stage 4b | Stage 4c |
|---|---:|---:|
| `short_scoreboard` (shared memory) | 18.83% | **4.65%** |
| `long_scoreboard` (global memory) | 17.55% | 23.26% |
| `barrier` | 6.75% | 8.45% |
| `not_selected` (warp ready, not picked) | 25.34% | 32.37% |
| `sm__pipe_fma_cycles_active` | 52.19% | **57.92%** |

<!-- TODO: two things worth explaining here.
     (1) Why the OTHER stalls rose in percentage terms while the kernel got
         14.5% faster - these are shares of warp-active cycles, and removing
         one stall shrinks the denominator. not_selected rising is good news.
     (2) The register cliff. Double-buffering BOTH regA and regB costs +16
         registers -> 134 -> 136 allocated -> 1 block/SM -> occupancy halves.
         Only one of them could be buffered. The kernel landed on exactly 128
         registers, which is precisely 65536/(2*256) - one register more and
         occupancy would have dropped 50%. State that this is a discrete cliff
         and that `make resource` is the only thing standing between you and
         silently falling off it. -->

## Where the model was wrong

<!-- TODO: this section is the point of the repo. Be specific and honest.
     Candidates, all of which actually happened:
       - AI stated as TILE/2, actually TILE/4
       - Stage 2 predicted 1.5-2x, measured 1.29x
       - v1/v2 ratio argued as a fixed 8x; it is a function of BLOCK_DIM
       - L2 capacity hypothesis for the ratio: falsified by sweeping N
       - AS_PAD=7 made things slower, and why
       - a leftover unconditional copy loop made Stage 4a slower than Stage 3
         while still passing every correctness check
     For each: what was predicted, what was measured, what the model was
     missing. One or two lines each. -->

## Future work

<!-- TODO: and say why NOT for the ones you are skipping. Notably:
     - double buffering: DRAM sits under 6%, so hiding global latency is not
       where the time goes. Revisit only after the LDS pipe is balanced.
     - Tensor Cores: on GA102 the TF32 tensor path has the SAME 35.6 TFLOP/s
       peak as the FP32 CUDA cores (unlike GA100, where TF32 is 8x FP32), so
       WMMA/TF32 buys no throughput here while costing 13 mantissa bits.
       Verify this against the GA102 whitepaper before publishing the claim.
       FP16/BF16 in with FP32 accumulate is ~2x, but that is HGEMM - a
       different project with a different verification story. -->

## Repository layout

```
gemm.cu      all six kernels, CPU reference, verification, timing harness
Makefile     build, resource report, bank-conflict profiling
results/     raw benchmark output per stage
```

Each stage is tagged: `git checkout stage1` … `stage4c`.
