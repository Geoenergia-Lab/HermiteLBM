# --------------------------------------------------------------------------- #
#                                                                             #
#  HermiteLBM: CUDA-based moment representation Lattice Boltzmann Method      #
#  Developed at UDESC - State University of Santa Catarina                    #
#  Website: https://www.udesc.br                                              #
#  Github: https://github.com/Geoenergia-Lab/HermiteLBM                       #
#                                                                             #
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# common.mk - Shared CUDA/HIP/C++ flags                                       #
# --------------------------------------------------------------------------- #

# The backend is selected by HERMITELBM_API in the sourced bashrc
# ("CUDA" or "HIP"). This file must not be edited to switch backends;
# change the switch in bashrc and re-source it.
ifeq ($(HERMITELBM_API),)
  	$(error HERMITELBM_API is not set. Source the project bashrc first.)
endif

ifeq ($(filter $(HERMITELBM_API),CUDA HIP),)
  	$(error HERMITELBM_API must be CUDA or HIP, got '$(HERMITELBM_API)')
endif

-include $(HERMITELBM_INCLUDE_DIR)/hardware.info

# --------------------------------------------------------------------------- #
#  Compiler selection                                                         #
# --------------------------------------------------------------------------- #

ifeq ($(HERMITELBM_API),CUDA)
	NVCXX = nvcc
else ifeq ($(HERMITELBM_API),HIP)
	NVCXX = hipcc
endif

# --------------------------------------------------------------------------- #
#  Common flags (both backends)                                               #
# --------------------------------------------------------------------------- #

NVCXX_STANDARD = -std=c++20

# Host compiler warnings. These are passed through to the underlying host
# compiler; the exact mechanism differs between nvcc and hipcc, so the
# forwarding flag is set per-API below.
NVCXX_HOST_WFLAGS = -O3 -funroll-loops -march=native -mtune=native \
    -Wall -Wextra -Werror -Wattributes -Wbuiltin-macro-redefined \
    -Wcast-align -Wconversion -Wdiv-by-zero -Wdouble-promotion \
    -Wfloat-equal -Wformat-security -Wformat=2 -Wimplicit-fallthrough=5 \
    -Winline -Wint-to-pointer-cast -Wlogical-op -Woverflow -Wpointer-arith \
    -Wshadow -Wsign-conversion -Wstrict-aliasing=3 -Wstringop-overflow=4 \
    -Wwrite-strings

# Preprocessor defines common to both backends.
NVCXX_DEFINES = -DSCALAR_PRECISION=32 -DLABEL_SIZE=32 -DUSE_SMEM_HALO

# --------------------------------------------------------------------------- #
#  CUDA-specific flags                                                        #
# --------------------------------------------------------------------------- #

ifeq ($(HERMITELBM_API),CUDA)

  	NVCXX_OPTFLAGS = -O3 --restrict -extra-device-vectorization --maxrregcount=128
  	NVCXX_MFLAGS   = -m64

  	# nvcc has NVCC-specific warning names it understands.
  	NVCXX_WFLAGS   = -Werror cross-execution-space-call,reorder,default-stream-launch,missing-launch-bounds,ext-lambda-captures-this

  	# Host flags are forwarded to the host compiler via -Xcompiler.
  	NVCXX_COMPILER_WFLAGS = -Xcompiler "$(NVCXX_HOST_WFLAGS)"

  	# -Xptxas -v prints per-kernel resource usage; keep it out of the default
  	# build to avoid noise, but leave a hook for a debug target.
  	NVCXX_DFLAGS   = $(NVCXX_DEFINES) -lpthread
  	NVCXX_DEBUG_INFOFLAGS = -Xptxas -v

  	# Architecture flags. When HERMITELBM_ARCHITECTURE_DETECTION="Manual",
  	# HERMITELBM_ARCHITECTURE_VERSION holds the compute capability digits
  	# (e.g. 89 for sm_89). When "Automatic", the build system is expected to
  	# fill NVCXX_ALL_ARCHFLAGS itself (e.g. via a hardware.info probe).
  	ifeq ($(HERMITELBM_ARCHITECTURE_DETECTION),Manual)
    	NVCXX_ALL_ARCHFLAGS = -gencode arch=compute_$(HERMITELBM_ARCHITECTURE_VERSION),code=sm_$(HERMITELBM_ARCHITECTURE_VERSION)
  	else
    	NVCXX_ALL_ARCHFLAGS =
  	endif

# --------------------------------------------------------------------------- #
#  HIP-specific flags                                                         #
# --------------------------------------------------------------------------- #

	else ifeq ($(HERMITELBM_API),HIP)

  	# HIP has no --restrict (use __restrict__ in code), no
  	# -extra-device-vectorization (Clang does this under -O3), and no --m64
  	# (64-bit is the default on x86-64). Register limit is passed through to
  	# the AMDGPU backend via -mllvm.
  	NVCXX_OPTFLAGS = -O3 -mllvm -amdgpu-max-num-registers=128
  	NVCXX_MFLAGS   =

  	# HIP does not accept nvcc's named warning list. Use plain -Werror and
  	# rely on the host warning set for diagnostics.
  	NVCXX_WFLAGS   =

  	# hipcc already invokes Clang for the host pass, so host flags go in
  	# directly, without -Xcompiler.
  	NVCXX_COMPILER_WFLAGS = $(NVCXX_HOST_WFLAGS)

  	# -Xptxas is CUDA-only. For HIP, resource usage is available via
  	# -mllvm -stats, but that is noisy; leave it as an opt-in debug flag.
  	NVCXX_DFLAGS   = $(NVCXX_DEFINES) -lpthread
  	NVCXX_DEBUG_INFOFLAGS = -mllvm -stats

  	# Architecture flags. When Manual, HERMITELBM_ARCHITECTURE_VERSION holds
  	# the gfx target suffix without the "gfx" prefix (e.g. 1200 for gfx1200).
  	# When Automatic, the build system fills NVCXX_ALL_ARCHFLAGS itself.
  	ifeq ($(HERMITELBM_ARCHITECTURE_DETECTION),Manual)
    	NVCXX_ALL_ARCHFLAGS = --offload-arch=gfx$(HERMITELBM_ARCHITECTURE_VERSION)
  	else
    	NVCXX_ALL_ARCHFLAGS =
  	endif

endif

# --------------------------------------------------------------------------- #
#  Assembled flag set                                                         #
# --------------------------------------------------------------------------- #

NVCXX_FLAGS = $(NVCXX_STANDARD) \
	$(NVCXX_OPTFLAGS) \
	$(NVCXX_MFLAGS) \
	$(NVCXX_ALL_ARCHFLAGS) \
	$(NVCXX_WFLAGS) \
	$(NVCXX_COMPILER_WFLAGS) \
	$(NVCXX_DFLAGS) \
	$(HAS_MULTI_GPU) \
	$(NVCXX_DEBUG_INFOFLAGS)

# --------------------------------------------------------------------------- #