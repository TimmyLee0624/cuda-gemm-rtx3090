NVCC      ?= nvcc
ARCH      ?= sm_86
TARGET    ?= gemm
SRC       := gemm.cu

# -O3 on host, -lineinfo so ncu/nsys can map metrics back to source lines.
NVCCFLAGS := -O3 -std=c++17 -arch=$(ARCH) -lineinfo -Xcompiler "-O3 -march=native -fopenmp"
LDFLAGS   := -Xcompiler "-fopenmp"

ifdef CUBLAS
NVCCFLAGS += -DUSE_CUBLAS
LDFLAGS   += -lcublas
endif

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)

# Register / shared-mem usage per kernel. Read this after EVERY stage.
resource: $(SRC)
	$(NVCC) $(NVCCFLAGS) -Xptxas -v -c -o /dev/null $<

# Shared-memory bank-conflict check. NCU_SUDO=1 if the permission fix
# (NVreg_RestrictProfilingToAdminUsers=0) is not in place yet.
NCU := ncu
ifdef NCU_SUDO
NCU := sudo env "PATH=$(PATH)" ncu
endif
KERNEL ?= gemm_regtile_v1
ID     ?= 4          # kernel id in the registry (must match KERNEL)
BANK_METRICS := smsp__inst_executed_op_shared_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active

bank: $(TARGET)
	$(NCU) --metrics $(BANK_METRICS) --kernel-name $(KERNEL) \
	       --launch-count 1 ./$(TARGET) 2048 $(ID)

# Profiles: nsys = timeline overview, ncu = per-kernel counters
nsys: $(TARGET)
	nsys profile --stats=true -o report_nsys ./$(TARGET) 2048

ncu: $(TARGET)
	ncu --set full -o report_ncu ./$(TARGET) 2048

clean:
	rm -f $(TARGET) report_nsys.* report_ncu.*

.PHONY: all resource bank nsys ncu clean

# Always rebuild before running -- prevents the stale-binary trap.
N ?= 1024
run: $(TARGET)
	./$(TARGET) $(N) $(K)
.PHONY: run
