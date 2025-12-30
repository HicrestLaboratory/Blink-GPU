REQUIRED_VARS := \
  BLINKGPU_SYSTEM \
  BLINKGPU_CONFIGURATION \
  BLINKGPU_SETUPDIR

$(foreach v,$(REQUIRED_VARS),\
  $(info $(v) = '$($(v))')\
  $(if $(strip $($(v))),,\
    $(error Environment variable $(v) is not set or empty)))


REQUIRED_FILE := ${BLINKGPU_SETUPDIR}${BLINKGPU_SYSTEM}_${BLINKGPU_CONFIGURATION}.conf
ifeq ($(wildcard $(REQUIRED_FILE)),)
$(error Required file '$(REQUIRED_FILE)' does not exist)
endif

#source ${BLINKGPU_SETUPDIR}${BLINKGPU_SYSTEM}_${BLINKGPU_CONFIGURATION}.conf

REQUIRED_VARS := \
  BLINKGPU_CUDA_HOME \
  BLINKGPU_NCCL_HOME \
  BLINKGPU_MPI_HOME \
  BLINKGPU_CUDA_MODULE \
  BLINKGPU_NCCL_MODULE \
  BLINKGPU_MPI_MODULE

$(foreach v,$(REQUIRED_VARS),\
  $(info $(v) = '$($(v))')\
  $(if $(strip $($(v))),,\
    $(error Environment variable $(v) is not set or empty)))

# ------------------ Targets ------------------
COMM_PATTERN  = pp a2a ar hlo mpp
COMM_STRATEGY = Baseline CudaAware Nccl # Nvlink Nvlink temporary disable (to manage and also to change name)

BINFOLDER = bin
DIRS = $(BINFOLDER) out sout
CFLAGS = -DSKIPCPUAFFINITY -arch=sm_80

# This is expanding ALL_TARGETS = {COMM_PATTERN} x {COMM_STRATEGY}
#  i.e. ALL_TARGETS = $(BINFOLDER)/pp_Baseline $(BINFOLDER)/pp_CudaAware ... $(BINFOLDER)/pp_Baseline ...
define ALL_TARGETS
$(foreach p,$(1),$(foreach s,$(2),$(BINFOLDER)/$(p)_$(s)))
endef

ALL_TARGETS_LIST := $(call ALL_TARGETS,$(COMM_PATTERN),$(COMM_STRATEGY))
$(info ALL_TARGETS = $(ALL_TARGETS_LIST))

PP_TARGETS_LIST := $(call ALL_TARGETS,pp,$(COMM_STRATEGY))
A2A_TARGETS_LIST := $(call ALL_TARGETS,a2a,$(COMM_STRATEGY))
AR_TARGETS_LIST := $(call ALL_TARGETS,ar,$(COMM_STRATEGY))
$(info PP_TARGETS_LIST = $(PP_TARGETS_LIST))
$(info A2A_TARGETS_LIST = $(A2A_TARGETS_LIST))
$(info AR_TARGETS_LIST = $(AR_TARGETS_LIST))

pp a2a ar:
	@true

all: $(ALL_TARGETS_LIST)
pp:  $(PP_TARGETS_LIST)
a2a: $(A2A_TARGETS_LIST)
ar:  $(AR_TARGETS_LIST)

# ------------------ Libs & Flags ------------------
MPI=-L$(BLINKGPU_MPI_HOME)/lib -I$(BLINKGPU_MPI_HOME)/include -lmpi
MPICUDA=-L$(BLINKGPU_MPICUDA_HOME)/lib -I$(BLINKGPU_MPICUDA_HOME)/include -lmpi
CUDA=-L$(BLINKGPU_CUDA_HOME)/lib64 -L$(BLINKGPU_CUDA_HOME)/compact -I$(BLINKGPU_CUDA_HOME)/include -lcudart -lcuda
NCCL=-L$(BLINKGPU_NCCL_HOME)/lib -I$(BLINKGPU_NCCL_HOME)/include -lnccl

BASELINE_LIBS = $(MPI) $(CUDA)
CUDAAWARE_LIBS = $(MPICUDA) $(CUDA)
NCCL_LIBS = $(MPI) $(CUDA) $(NCCL)
NVLINK_LIBS = $(MPI) $(CUDA)

# ------------------ Modules ------------------
BASELINE_MODULES = $(BLINKGPU_CUDA_MODULE) $(BLINKGPU_MPI_MODULE)
CUDAAWARE_MODULES = $(BLINKGPU_CUDA_MODULE) $(BLINKGPU_MPICUDA_MODULE)
NCCL_MODULES = $(BLINKGPU_CUDA_MODULE) $(BLINKGPU_NCCL_MODULE)

MODULEFOLDER=module_dir/
BASELINE_MODULE_FILE=${MODULEFOLDER}baseline.mod
CUDAAWARE_MODULE_FILE=${MODULEFOLDER}cudaaware.mod
NCCL_MODULE_FILE=${MODULEFOLDER}nccl.mod

$(BASELINE_MODULE_FILE):
	mkdir -p ${MODULEFOLDER}
	@echo "module purge" > $@
	@echo "module load $(BASELINE_MODULES)" >> $@

$(CUDAAWARE_MODULE_FILE):
	mkdir -p ${MODULEFOLDER}
	@echo "module purge" > $@
	@echo "module load $(CUDAAWARE_MODULES)" >> $@

$(NCCL_MODULE_FILE):
	mkdir -p ${MODULEFOLDER}
	@echo "module purge" > $@
	@echo "module load $(NCCL_MODULES)" >> $@

# ------------------  Rules  ------------------
CC=$(BLINKGPU_CUDA_HOME)/bin/nvcc
# COMPILE = $(CC) $(CFLAGS) $(INCL) $(LIBS) $(LIBFLAGS) \
#           -lstdc++ -lm -Wno-deprecated-gpu-targets $(DBGFLAGS)

$(BINFOLDER)/%_Baseline: src/%_Baseline.cu $(BASELINE_MODULE_FILE) | $(DIRS)
	@echo "Building $@"
	mkdir -p ${BINFOLDER}
	source $(BASELINE_MODULE_FILE) && $(CC) $(CFLAGS) -o $@ $< $(BASELINE_LIBS)

$(BINFOLDER)/%_CudaAware: src/%_CudaAware.cu $(CUDAAWARE_MODULE_FILE) | $(DIRS)
	@echo "Building $@"
	mkdir -p ${BINFOLDER}
	source $(CUDAAWARE_MODULE_FILE) && $(CC) $(CFLAGS) -o $@ $< $(CUDAAWARE_LIBS)

$(BINFOLDER)/%_Nccl: src/%_Nccl.cu $(NCCL_MODULE_FILE) | $(DIRS)
	@echo "Building $@"
	mkdir -p ${BINFOLDER}
	source $(NCCL_MODULE_FILE) && $(CC) $(CFLAGS) -o $@ $< $(NCCL_LIBS)

$(BINFOLDER)/refactorTest: src/refactorTest.cu $(NCCL_MODULE_FILE) | $(DIRS)
	$(CC) $(CFLAGS) -o $@ $< $(NCCL_LIBS)

clean:
	rm -rf $(BINFOLDER)/*
