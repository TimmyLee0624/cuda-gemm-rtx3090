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

# Profiles: nsys = timeline overview, ncu = per-kernel counters
nsys: $(TARGET)
	nsys profile --stats=true -o report_nsys ./$(TARGET) 2048

ncu: $(TARGET)
	ncu --set full -o report_ncu ./$(TARGET) 2048

clean:
	rm -f $(TARGET) report_nsys.* report_ncu.*

.PHONY: all resource nsys ncu clean

# Always rebuild before running -- prevents the stale-binary trap.
N ?= 1024
run: $(TARGET)
	./$(TARGET) $(N) $(K)
.PHONY: run
