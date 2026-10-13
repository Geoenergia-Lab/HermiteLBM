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

# Preprocessor defines common to both backends.
NVCXX_DEFINES = -DSCALAR_PRECISION=32 -DLABEL_SIZE=32 -DUSE_SMEM_HALO

# Base host compiler warnings that BOTH GCC and Clang understand.
# Everything here is portable across the two host compilers.
NVCXX_HOST_WFLAGS_COMMON = -O3 -funroll-loops -march=native -mtune=native \
    -Wall -Wextra -Werror -Wattributes -Wbuiltin-macro-redefined \
    -Wcast-align -Wconversion -Wdiv-by-zero -Wdouble-promotion \
    -Wfloat-equal -Wformat-security -Wformat=2 \
    -Winline -Wint-to-pointer-cast -Woverflow -Wpointer-arith \
    -Wshadow -Wsign-conversion -Wwrite-strings

# --------------------------------------------------------------------------- #
#  CUDA-specific flags                                                        #
# --------------------------------------------------------------------------- #

ifeq ($(HERMITELBM_API),CUDA)

  NVCXX_OPTFLAGS = -O3 --restrict -extra-device-vectorization --maxrregcount=128
  NVCXX_MFLAGS   = -m64

  NVCXX_WFLAGS   = -Werror cross-execution-space-call,reorder,default-stream-launch,missing-launch-bounds,ext-lambda-captures-this

  # GCC-only warnings that Clang does not understand. These are kept in the
  # CUDA branch because nvcc's host compiler is GCC.
  NVCXX_HOST_WFLAGS_GCC_ONLY = \
      -Wimplicit-fallthrough=5 \
      -Wlogical-op \
      -Wstrict-aliasing=3 \
      -Wstringop-overflow=4

  NVCXX_COMPILER_WFLAGS = -Xcompiler "$(NVCXX_HOST_WFLAGS_COMMON) $(NVCXX_HOST_WFLAGS_GCC_ONLY)"

  NVCXX_DFLAGS   = $(NVCXX_DEFINES) -lpthread
  NVCXX_DEBUG_INFOFLAGS = -Xptxas -v

  ifeq ($(HERMITELBM_ARCHITECTURE_DETECTION),Manual)
    NVCXX_ALL_ARCHFLAGS = -gencode arch=compute_$(HERMITELBM_ARCHITECTURE_VERSION),code=sm_$(HERMITELBM_ARCHITECTURE_VERSION)
  else
    NVCXX_ALL_ARCHFLAGS =
  endif

# --------------------------------------------------------------------------- #
#  HIP-specific flags                                                         #
# --------------------------------------------------------------------------- #

else ifeq ($(HERMITELBM_API),HIP)

  NVCXX_OPTFLAGS = -O3 -mllvm -amdgpu-max-num-registers=128
  NVCXX_MFLAGS   =

  NVCXX_WFLAGS   =

  # Clang accepts -Wimplicit-fallthrough without the =N level argument.
  # The other three GCC-only flags are dropped entirely for HIP.
  NVCXX_HOST_WFLAGS_CLANG = -Wimplicit-fallthrough

  # Uncomment the following line if you want Clang to silently ignore any
  # unrecognized -W flags instead of splitting the list per backend.
  # This is a stopgap, not a long-term solution.
  # NVCXX_HOST_WFLAGS_CLANG += -Wno-unknown-warning-option

  NVCXX_COMPILER_WFLAGS = $(NVCXX_HOST_WFLAGS_COMMON) $(NVCXX_HOST_WFLAGS_CLANG)

  NVCXX_DFLAGS   = $(NVCXX_DEFINES) -lpthread
  NVCXX_DEBUG_INFOFLAGS = -mllvm -stats

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