# CUDA SGEMM on RTX 3090 — from 2.3 to 19.0 TFLOP/s (74.8% of cuBLAS)

This project implements and progressively optimizes FP32 GEMM kernels from scratch on an NVIDIA RTX 3090, following the CUDA optimization techniques introduced in *Programming Massively Parallel Processors*. Before each optimization, we identify the performance bottleneck and use a roofline model to derive theoretical performance bounds and predict potential improvements. When measurements deviate from predictions, we investigate the underlying causes using Nsight Compute, refine the performance model, and document where and why our predictions were wrong. Through seven kernel implementations, performance improves from 2.3 to 19.0 TFLOP/s, achieving an 8.3× speedup over the naive baseline and 74.8% of cuBLAS SGEMM performance.

The workload is square, row-major FP32 GEMM (`C = A × B`). Every kernel is verified against a CPU reference and profiled with Nsight Compute.

```
Hardware   NVIDIA GeForce RTX 3090 (GA102, sm_86, 82 SMs)
           FP32 peak    35.6 TFLOP/s
           DRAM         936 GB/s        -> ridge point 38 FLOP/Byte
           Shared mem   17.8 TB/s       (82 SM × 128 B/clk × 1.695 GHz)
Toolchain  CUDA 12.9, nvcc -O3 -arch=sm_86 -lineinfo
```

## Results

All figures at `N = 4096`, single kernel launch averaged over 5 iterations.

| # | Kernel | Time (ms) | GFLOP/s | % peak | % cuBLAS | Optimization target |
|---|--------|----------:|--------:|-------:|---------:|---|
| 1 | Naive, coalesced | 60.39 | 2276 | 6.39% | 9.0% | — (baseline) |
| 2 | Naive, strided (anti-pattern) | 230.84 | 595 | 1.67% | 2.3% | — (shows what coalescing is worth) |
| 3 | Shared-memory tiling | 45.38 | 3029 | 8.51% | 11.9% | best-effort L1 reuse |
| 4 | Register tiling 128×128, 8×8/thread | 10.38 | 13243 | 37.20% | 52.3% | DRAM bandwidth |
| 5 | Transposed `As` + `float4` shared loads | 8.55 | 16077 | 45.16% | 63.4% | shared-memory instruction count |
| 6 | Split `regB` micro-tile | 8.29 | 16589 | 46.60% | 65.5% | shared-memory bank conflicts |
| 7 | Register double buffering | **7.25** | **18962** | **53.26%** | **74.8%** | shared-memory **latency** |
| — | cuBLAS SGEMM (reference) | 5.42 | 25347 | 71.20% | 100% | — |

**8.3× over the naive baseline, 31.8× over the uncoalesced variant.**

The profiling results motivate further investigation of global-memory latency and synchronization overhead. Shared-memory double buffering is a potential next step, but its profitability remains to be experimentally validated.

The main results table and the individual controlled experiments below report separate measurements; their throughput values are retained as recorded.

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

Before each optimization, we derive a theoretical performance bound from the roofline model using arithmetic intensity, memory bandwidth, and peak compute throughput, and record our predictions before running any experiments. We then benchmark and profile the kernel with Nsight Compute to compare measured performance against these predictions. When measurements disagree with predictions, we investigate missing factors such as cache reuse, shared-memory bank conflicts, and instruction dependencies, refining the performance model rather than merely modifying the kernel.

### Verifying a float GEMM

Per-element relative error is unreliable for this GEMM workload because cancellation in length-N dot products of signed random inputs can produce reference values close to zero, amplifying even small absolute errors.

Instead, we use a double-precision CPU reference and two error metrics:

- **Normwise relative error:** `max|error| / max|reference|`
- **Backward-error indicator:** `max|error| / (N · ε · max|A| · max|B|)`, where ε is the FP32 machine epsilon.

The normwise relative error provides a matrix-level correctness criterion without being dominated by individual near-zero reference values. The backward-error indicator normalizes the observed error by an estimated floating-point error scale that accounts for the dot-product length, numerical precision, and input magnitudes. Healthy kernels exhibit values of O(1) or smaller, whereas a genuinely broken kernel produced a value of approximately 5 × 10⁵, far beyond the expected rounding-error scale.

## Stage 1 — Naive baseline

In the naive one-element-per-thread GEMM, each thread computes a length-N dot product. Each iteration loads two FP32 operands (8 bytes) and performs one FMA (2 FLOPs), yielding an arithmetic intensity of approximately 0.25 FLOP/Byte under a no-reuse assumption, neglecting the final output store. With the RTX 3090's DRAM bandwidth of 936 GB/s, this gives a predicted performance bound of 234 GFLOP/s.

However, the measured throughput reaches 2259 GFLOP/s, approximately 9.65× higher than the original prediction. This does not violate the roofline model; rather, it exposes an incorrect assumption that every per-thread global-memory load generates corresponding DRAM traffic.

Even without explicit shared-memory tiling, warp-level broadcast, memory coalescing, and L1/L2 cache reuse can reduce redundant memory traffic. Consequently, the original model overestimates DRAM traffic and underestimates the effective DRAM arithmetic intensity. This discrepancy highlights the importance of distinguishing per-thread memory requests from actual DRAM traffic when constructing a performance model.

### Memory coalescing

The two naive GEMM kernels perform identical arithmetic but differ in how `threadIdx.x` and `threadIdx.y` map to the output matrix's row and column indices. Since CUDA warps are formed with threadIdx.x varying fastest, this mapping determines the memory access patterns of warp-level instructions on row-major matrices.

For BLOCK_DIM = 32:

| Matrix | Version 1 (Coalesced) | Version 2 (Strided) |
|---|---|---|
| A load | Broadcast (1 sector) | Strided (32 sectors) |
| B load | Coalesced (4 sectors) | Broadcast (1 sector) |
| C store | Coalesced (4 sectors) | Strided (32 sectors) |

The victims of the unfavorable thread mapping are A and C, not B. Importantly, the A load in Version 1 is a warp broadcast rather than a contiguous coalesced read.

The performance gap is not a fixed 8× factor; it depends on the block dimension. For a square thread block of dimension D ∈ {8, 16, 32}, assuming aligned accesses and fully active warps, the number of 32-byte load sectors per warp per k iteration is:

- Version 1: S_v1 = 32/D + D/8
- Version 2: S_v2 = D + 1
- Predicted speedup: S_v2 / S_v1

The final C store is excluded from this first-order model because it occurs only once per thread, whereas A and B are loaded during every k iteration.

The sector model closely predicts the measured speedups for the 8×8 and 16×16 configurations and captures the overall trend. The larger discrepancy for 32×32 highlights the limitations of modeling execution time solely through warp-level sector requests. Additional factors, including cache behavior, memory latency, occupancy, and warp scheduling, may contribute to this discrepancy and require further profiling to establish their individual effects.

| `BLOCK_DIM` | warp shape | v1 sectors | v2 sectors | predicted | measured |
|---|---|---|---|---|---|
| 8×8 | 8 col × 4 row | 5 | 9 | 1.80× | 1.78× |
| 16×16 | 16 col × 2 row | 4 | 17 | 4.25× | 4.24× |
| 32×32 | 32 col × 1 row | 5 | 33 | 6.60× | 7.77× |

## Stage 2 — Shared-memory tiling

Shared-memory tiling explicitly stages input tiles through cooperative loading, allowing threads within a block to reuse data without relying on hardware cache retention.

For a square tile of dimension T, each phase loads two T × T FP32 input tiles, requiring 8T² bytes, and performs 2T³ FLOPs. Ignoring the final C store, the ideal arithmetic intensity is therefore:

$$
AI = \frac{2T^3}{8T^2} = \frac{T}{4}\ \text{FLOP/Byte}
$$

The counterintuitive observation is that a naive GEMM kernel with a D × D thread block has the same ideal block-level arithmetic intensity under perfect intra-block reuse. At each k iteration, the block performs 2D² FLOPs while requiring only D distinct elements from A and D distinct elements from B:

$$
AI = \frac{2D^2}{8D} = \frac{D}{4}\ \text{FLOP/Byte}
$$

When D = T, both kernels have the same ideal block-level arithmetic intensity. Shared-memory tiling does not introduce new mathematical reuse opportunities; instead, it converts best-effort hardware cache reuse into explicit, software-controlled staging while reducing redundant global load instructions.

| `BLOCK_DIM` | AI (FLOP/Byte) | Modeled roofline (GFLOP/s) | Naive (GFLOP/s) | Achieved |
|---|---|---|---|---|
| 8 | 2 | 1872 | 1783 | 95% |
| 16 | 4 | 3744 | 2220 | 59% |
| 32 | 8 | 7488 | 2158 | 29% |

As the block size increases from 8×8 to 32×32, the ideal block-level arithmetic intensity rises from 2 to 8 FLOP/Byte, yet the naive kernel achieves progressively smaller fractions of the corresponding roofline: 95%, 59%, and 29%. Increasing potential data reuse does not guarantee that hardware caches can sustain it; cache retention, warp scheduling, occupancy, and memory instruction overhead may all limit realized performance.

Shared-memory tiling converts best-effort cache reuse into explicitly managed intra-block reuse through cooperative loading and synchronization, while reducing redundant global load operations. For the 16×16 configuration, tiling improves the achieved fraction of the same ideal roofline from 59% to 76%, yielding a 1.29× speedup without increasing the ideal block-level arithmetic intensity.

## Stage 3 — Register tiling

The register-tiling configuration is motivated by two levels of the roofline model.

**DRAM roofline.** The RTX 3090 has a DRAM ridge point of approximately 38 FLOP/Byte (35.6 TFLOP/s / 936 GB/s). Since the ideal arithmetic intensity of a square block tile is T/4, the target tile dimension is approximately 152. We select a 128×128 block tile as a practical design candidate, yielding an ideal arithmetic intensity of 32 FLOP/Byte and a DRAM roofline of 29.95 TFLOP/s.

**Shared-memory roofline.** Under a simplified thread-level operand traffic model, a one-output-per-thread kernel requires two FP32 shared-memory operands per FMA, yielding an arithmetic intensity of 0.25 FLOP/Byte and a modeled throughput ceiling of 4.45 TFLOP/s at 17.8 TB/s. An 8×8 register micro-tile instead performs 64 FMAs using only 16 FP32 operands per k-step, increasing logical shared-memory arithmetic intensity to 2 FLOP/Byte. This reaches the FP32 compute peak of 35.6 TFLOP/s in the simplified shared-memory roofline model.

Combining these choices gives (128/8) × (128/8) = 256 threads per block. With BK = 8, the two input tiles require 8 KiB of shared memory. This configuration improves data reuse at both memory hierarchy levels while remaining feasible under CUDA resource constraints. The roofline models motivate the configuration but do not establish its optimality; actual performance also depends on register allocation, occupancy, bank conflicts, and instruction throughput.

### Why lower occupancy does not imply lower performance

Register tiling reduces occupancy from nearly 100% to approximately 30%, yet improves throughput from 3.03 to 13.24 TFLOP/s, a 4.37× speedup.

The kernel uses 117 registers per thread. Accounting for the register allocation granularity of 256 registers per warp, the effective allocation is 120 registers per thread, or 30,720 registers per 256-thread block. With 65,536 registers per SM, only two blocks can reside concurrently, giving 16 resident warps and a theoretical occupancy of 33.3%.

Occupancy provides thread-level parallelism (TLP) for latency hiding, but it is not the only mechanism available. The 8×8 register micro-tile exposes 64 independent accumulator dependency chains per thread, providing substantial instruction-level parallelism (ILP). Independent FMA instructions can be scheduled without waiting for previous updates to other accumulators.

Register tiling also improves operand reuse, increasing logical shared-memory arithmetic intensity from 0.25 to 2 FLOP/Byte. The larger block tile increases ideal DRAM arithmetic intensity from 4 to 32 FLOP/Byte.

These improvements can outweigh the cost of reduced occupancy. The result demonstrates that occupancy is a means of hiding latency rather than a performance objective by itself. However, occupancy and throughput measurements alone cannot isolate the individual contributions of ILP, data reuse, and other execution effects.

## Stage 4a — Transposed `As` and vectorized shared loads

Nsight Compute on Stage 3 (`N = 2048`):

```
shared-ld instructions    41,943,040     (10 per warp per kk)
wavefronts               134,218,687     ( 3.20 per instruction)
bank conflicts            67,108,864     ( exactly 2.00x oversubscribed)
DRAM throughput                5.24%     <- Low DRAM bandwidth utilization
```

### Shared memory wavefront budget

Although register tiling increases the logical shared-memory arithmetic intensity to 2 FLOP/Byte, the original roofline model does not account for the additional service demand caused by bank conflicts.

At N = 2048, Nsight Compute reports 41,943,040 shared load instructions and 134,218,687 wavefronts, corresponding to 10 LDS instructions and approximately 32 wavefronts per warp per kk iteration.

Each thread performs 64 FMAs per kk iteration, giving 2,048 FMAs per warp. With 128 FP32 execution lanes per SM, the equivalent FP32 service demand is 16 cycles.

Assuming a shared-memory service rate of one wavefront per cycle and ideal overlap between the shared-memory and FP32 pipelines, the LDS service demand of 32 cycles exceeds the FFMA demand of 16 cycles. The resulting throughput ceiling is therefore:

$$
35.6 \times \frac{16}{32} = 17.8\ \text{TFLOP/s}
$$

The measured throughput of approximately 13.3 TFLOP/s reaches 75% of this refined ceiling.

This explains why logical operand arithmetic intensity alone can overestimate achievable performance: bank conflicts consume additional shared-memory service capacity without increasing useful operand traffic. The wavefront budget provides a more detailed performance model, although instruction issue limits, dependencies, synchronization, and memory latency remain unmodeled.

### Transposed A tile and vectorization

The original `As[BM][BK]` layout is poorly suited to the outer-product access pattern of register tiling.

For a fixed kk iteration, consecutive regA operands are separated by BK = 8 words, whereas regB operands are contiguous. Consequently, the compiler vectorizes the eight regB loads into two LDS.128 instructions, while regA requires eight LDS.32 instructions. This results in ten shared load instructions per warp per kk iteration.

The original layout also introduces bank conflicts. Each warp contains two tRow groups of 16 threads. Their regA addresses are separated by TM × BK = 64 words, mapping two distinct addresses to the same shared-memory bank. Although each group benefits from broadcast, the two groups incur a 2-way bank conflict.

Transposing the A tile to `As[BK][BM + AS_PAD]` reduces the regA operand stride from eight words to one word and the address separation between the two tRow groups from 64 words to eight words. With appropriate alignment, the compiler can vectorize the regA loads into two LDS.128 instructions.

The total shared load instruction count therefore decreases from ten to four per warp per kk iteration. At N = 2048, the measured dynamic instruction count drops from 41,943,040 to 16,777,216, a 60% reduction.

However, improving the consumer read layout introduces a new producer-side constraint: contiguous global-memory operands must be scattered into different shared-memory rows. This can introduce store bank conflicts, making padding necessary to satisfy both vector alignment and bank-mapping requirements.

### Padding selection and the failed prime-padding experiment

The shared-memory layout must satisfy two constraints simultaneously: consumer vector alignment and producer bank mapping.

To enable 128-bit regA loads across all kk iterations, the pitch must satisfy pitch ≡ 0 (mod 4). Meanwhile, the transposed producer stores require the even and odd lanes of a warp to access disjoint sets of 16 banks. This requires 4 × pitch ≡ 16 (mod 32), or equivalently, pitch ≡ 4 (mod 8).

Since pitch = BM + AS_PAD and BM = 128, the valid padding values satisfy AS_PAD ≡ 4 (mod 8). We select the smallest non-negative solution, AS_PAD = 4, which adds only 128 bytes of shared memory per block.

The initial experiment with AS_PAD = 7 demonstrates the limitation of padding heuristics that consider bank conflicts alone. Although this configuration reduces shared-load wavefronts from 32 to 24 per warp per kk iteration, its pitch of 135 words does not guarantee 16-byte vector alignment. Consequently, the dynamic shared load instruction count remains at 41,943,040, and the expected vectorization benefit is not realized.

At N = 4096, Pad 7 achieves 11,936 GFLOP/s, approximately 10% slower than the original register-tiled kernel at 13,267 GFLOP/s. In contrast, Pad 4 reduces the dynamic shared load instruction count to 16,777,216 and achieves 16,053 GFLOP/s.

This failed experiment demonstrates that reducing shared-memory wavefronts alone does not guarantee higher kernel throughput. Effective layout optimization must jointly consider bank mapping, vector alignment, instruction count, and resource usage. Further profiling is required to isolate the complete cause of the Pad 7 performance regression.

| Configuration | LDS instructions (N=2048) | Wavefronts/warp/kk | GFLOP/s (N=4096) |
|---|---|---|---|
| Stage 3, `As[BM][BK]` | 41,943,040 | 32 | 13267 |
| `AS_PAD = 7` | 41,943,040 | 24 | 11936 |
| `AS_PAD = 4` | **16,777,216** | **20** | **16053** |

### Broadcast-aware shared-memory model

The original model predicted eight wavefronts for the two vectorized regA loads, assuming that all 32 lanes requested distinct 16-byte operands. Under this assumption, each LDS.128 instruction requires four wavefronts.

However, the thread mapping assigns only two distinct tRow values to each warp, with 16 lanes sharing each value. Since regA addresses depend on tRow rather than tCol, all lanes within each group request identical operands.

Shared-memory broadcast explains why the regA loads require only four wavefronts instead of the predicted eight. Together with the 16 wavefronts attributed to regB, the total shared-load service demand is 20 wavefronts per warp per kk iteration, rather than the predicted 24.

Under the simplified wavefront budget model, this revises the theoretical throughput ceiling from 23.73 to 28.48 TFLOP/s, an increase of 20%. This represents a correction to the performance model rather than an independently measured broadcast speedup.

The discrepancy demonstrates that logical operand bytes and vector width alone are insufficient to estimate shared-memory service demand. A more accurate model must account for warp-level address sharing, broadcast, bank mapping, and architecture-specific vector request decomposition.

With regB now accounting for 80% of the modeled shared-load wavefronts, the next optimization targets the B micro-tile mapping.

## Stage 4b — Splitting the B micro-tile

In Stage 4a, each thread loads eight consecutive B operands using `Bs[kk][tCol * 8 + j]`. Although this mapping supports two LDS.128 instructions, adjacent threads have a vector stride of eight words. Within an eight-lane service phase, tCol and tCol + 4 access distinct addresses separated by 32 words, corresponding to one full bank rotation. This produces a two-way bank conflict, resulting in 16 wavefronts for the two regB loads.

To eliminate these conflicts, we split the eight output columns into two groups of four:

```cpp
#pragma unroll
for (int j = 0; j < 4; ++j) {
    regB[j] = Bs[kk][tCol * 4 + j];
    regB[j + 4] = Bs[kk][BN / 2 + tCol * 4 + j];
}
```

The new mapping reduces the inter-thread vector stride from eight to four words. Each eight-lane phase now accesses 32 consecutive words distributed across all 32 banks, eliminating bank conflicts while preserving 16-byte vector alignment.

Instead of computing one contiguous 8×8 micro-tile, each thread computes two separated 8×4 sub-tiles. The number of FMAs, logical operands, and accumulators remains unchanged.

The first group covers columns 0–63, while the second covers columns 64–127. Together, they form a bijective mapping that preserves unique output ownership. The accumulator computation remains unchanged, but the global-memory writeback must follow the new column mapping to preserve numerical correctness.

The recorded Nsight Compute results at N = 2048 report zero shared-memory load and store bank conflicts. The regB service demand decreases from 16 to eight wavefronts, reducing the total shared-load wavefront count from 20 to 12 per warp per kk iteration, a 40% reduction.

At N = 4096, measured throughput improves from 16.077 to 16.589 TFLOP/s, corresponding to a modest 1.032× speedup.

The limited improvement highlights the distinction between memory service throughput and dependency latency. Although shared-load service demand is now below the equivalent FP32 service demand of 16 cycles, dependent FFMAs may still wait for their operands. This motivates the next optimization: software pipelining through register double buffering to overlap operand loading with computation.

## Stage 4c — Register double buffering

| Profiling metric (%) | Stage 4b | Stage 4c |
|---|---:|---:|
| `short_scoreboard` (shared memory) | 18.83% | **4.65%** |
| `long_scoreboard` (global memory) | 17.55% | 23.26% |
| `barrier` | 6.75% | 8.45% |
| `not_selected` (warp ready, not picked) | 25.34% | 32.37% |
| `sm__pipe_fma_cycles_active` | 52.19% | **57.92%** |

After eliminating shared-memory bank conflicts, the shared-load service demand decreases to 12 wavefronts per warp per kk iteration, below the equivalent FP32 service demand of 16 cycles. However, sufficient shared-memory service capacity does not eliminate the load-to-use dependency latency between LDS and FFMA instructions.

Software pipelining moves operand loads one iteration ahead, allowing the 64 FFMAs of the current iteration to overlap with operand loading for the next iteration. Separate current and next operand buffers prevent premature overwriting while preserving the original K-dimension accumulation order.

Measured throughput increases from 16.589 to 18.962 TFLOP/s, corresponding to a 1.143× speedup. Short scoreboard stalls decrease from 18.83% to 4.65%, while FMA pipeline activity increases from 52.19% to 57.92%. These observations are consistent with improved operand readiness and latency hiding.

The FMA activity metric is not a warp-stall percentage and should be interpreted using its own denominator.

**Interpreting stall percentages.** Although long scoreboard, barrier, and not-selected percentages increase, their absolute costs do not necessarily increase. These metrics represent shares of warp-active cycles, and reducing one stall category can change the denominator and the relative contributions of other categories. The increase in not-selected stalls is also consistent with more warps becoming eligible for instruction issue. However, absolute counters and scheduler metrics are required to establish the corresponding changes in absolute stall costs.

**The register allocation cliff.** Software pipelining increases register pressure by potentially extending the overlapping live ranges of operands from consecutive iterations. Stage 4b uses 118 registers per thread. With 65,536 registers per SM and 256 threads per block, maintaining two resident blocks requires no more than:

$$
\frac{65536}{2 \times 256} = 128\ \text{registers/thread}
$$

Under a simplified additive resource model, double-buffering one eight-element operand fragment increases the estimated register count to 126, whereas buffering both A and B increases it to 134. Accounting for register allocation granularity, 134 registers correspond to an effective allocation of 136 registers per thread. This reduces the number of resident blocks from two to one and halves theoretical occupancy from 33.3% to 16.7%.

This is a discrete resource allocation cliff rather than a linear performance cost. The recorded final kernel uses exactly 128 registers per thread, placing it at the boundary for maintaining two resident blocks.

The current source implementation contains both A and B current/next operand buffers. Therefore, the actual physical register allocation must be verified using the corresponding compiler output rather than inferred solely from source-level variables. Running `make resource` after each scheduling modification is essential for detecting excessive register allocation and spilling before their costs outweigh the benefits of latency hiding.

## Where the model was wrong

- **Arithmetic intensity:** Initially stated as TILE/2, whereas accounting for both FP32 input tiles gives TILE/4. The original formula overestimated ideal block-level arithmetic intensity by a factor of two.

- **Shared-memory tiling:** Stage 2 was predicted to achieve a 1.5–2× speedup but delivered only 1.29×. The model underestimated the data reuse already provided by hardware caches in the naive kernel.

- **Coalescing ratio:** The v1/v2 performance ratio was initially assumed to be a fixed 8×. Measurements instead gave 1.78×, 4.24×, and 7.77× for BLOCK_DIM values of 8, 16, and 32. The corrected sector model accounts for warp geometry and both input access patterns.

- **L2 capacity hypothesis:** L2 cache capacity was initially proposed as the main explanation for the v1/v2 performance gap, but an N sweep did not support this hypothesis. Cache capacity alone was insufficient to explain the observed difference.

- **Shared-memory padding:** AS_PAD=7 reduced shared-load wavefronts from 32 to 24 but decreased throughput from 13,267 to 11,936 GFLOP/s. The original model ignored vector alignment; AS_PAD=4 satisfies both producer-side bank-mapping and consumer-side vector-alignment constraints.

- **Redundant copy loop:** A leftover unconditional copy loop made Stage 4a slower than Stage 3 while passing every correctness check. Semantic correctness does not guarantee profitability; generated instructions and measured performance must be validated independently.

## Future work

### Shared-memory double buffering

The current register double-buffering implementation pipelines the shared-to-register loads within the inner `kk` loop, but does not overlap global-to-shared memory transfers across outer `k0` iterations.

Earlier profiling showed DRAM throughput below 6% of peak, motivating our initial focus on the shared-memory pipeline. However, low bandwidth utilization does not rule out memory-latency bottlenecks. After register pipelining reduces short-scoreboard stalls from 18.83% to 4.65%, the increased relative shares of long-scoreboard and barrier stalls motivate further investigation of global-memory latency and synchronization overhead.

A natural next step is shared-memory ping-pong buffering, potentially using Ampere's `cp.async` instructions to overlap loading the next tile with computation on the current tile.

The transposed A-tile layout introduces an additional constraint: a contiguous 16-byte asynchronous copy cannot directly implement the existing scattered shared-memory stores. Possible alternatives include smaller asynchronous copies, an intermediate staging buffer, or a redesigned shared-memory layout.

Resource usage must also be considered. Stage 4c already uses 128 registers per thread, placing it at the register allocation threshold for two resident blocks per SM. Additional register pressure may halve theoretical occupancy. Consequently, further pipelining should be evaluated through resource reports, correctness tests, and controlled performance measurements rather than assumed to be profitable.

### Tensor Cores and mixed precision

This project deliberately retains FP32 CUDA Core computation to preserve a consistent numerical and performance baseline across all optimization stages.

According to [NVIDIA's GA102 Architecture Whitepaper](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf), the RTX 3090 provides approximately 35.6 TFLOP/s of dense TF32 Tensor Core throughput, equal to its FP32 CUDA Core peak. Although Tensor Core instructions may still improve realized execution efficiency, TF32 does not increase the theoretical dense compute peak on this device and reduces input precision by 13 fraction bits.

FP16 and BF16 Tensor Core operations with FP32 accumulation provide approximately 71 TFLOP/s of dense compute throughput. Storing inputs in 16-bit formats can also reduce memory traffic.

However, mixed-precision GEMM introduces different numerical requirements, matrix-instruction mappings, memory layouts, and correctness protocols. We therefore leave Tensor Core GEMM as a separate follow-up project rather than incorporating a different precision mode into the current SGEMM comparison.

The current kernel achieves 18.962 TFLOP/s, or 74.8% of cuBLAS SGEMM performance. Further optimization should be driven by profiling evidence, resource constraints, and measurable benefits rather than peak-throughput targets alone.

## Repository layout

```
gemm.cu      all seven kernels, CPU reference, verification, timing harness
Makefile     build, resource report, bank-conflict profiling
results/     raw benchmark output per stage
```

Every run prints the whole ladder, one row per stage, so the progression is
visible without checking anything out. Stages are also tagged —
`git checkout stage1` … `stage4c` — with one gap: **4b and 4c landed in a
single commit** (`cd03087`), so no commit holds 4b on its own and there is no
`stage4b` tag. `results/stage4b.txt` has its isolated numbers instead.
