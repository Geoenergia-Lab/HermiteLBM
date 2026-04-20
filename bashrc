# --------------------------------------------------------------------------- #
#                                                                             #
#  HermiteLBM: CUDA-based moment representation Lattice Boltzmann Method      #
#  Developed at UDESC - State University of Santa Catarina                    #
#  Website: https://www.udesc.br                                              #
#  Github: https://github.com/Geoenergia-Lab/HermiteLBM                       #
#                                                                             #
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
#  USER-DEFINED ENVIRONMENT VARIABLES
# --------------------------------------------------------------------------- #

# CUDA version (major and minor)
export HERMITELBM_CUDA_VERSION_MAJOR="13"
export HERMITELBM_CUDA_VERSION_MINOR="0"

# Architecture detection mode: "Automatic" or "Manual"
export HERMITELBM_ARCHITECTURE_DETECTION="Automatic"

# Manual architecture version (used only if HERMITELBM_ARCHITECTURE_DETECTION="Manual")
export HERMITELBM_ARCHITECTURE_VERSION="89"

# --------------------------------------------------------------------------- #
#  AUTOMATIC SETUP – DO NOT MODIFY BELOW UNLESS YOU KNOW WHAT YOU ARE DOING
# --------------------------------------------------------------------------- #

# Project root directory (where this bashrc file lives)
export HERMITELBM_PROJECT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Build tree structure
export HERMITELBM_BUILD_DIR="$HERMITELBM_PROJECT_DIR/build"
export HERMITELBM_BIN_DIR="$HERMITELBM_BUILD_DIR/bin"
export HERMITELBM_INCLUDE_DIR="$HERMITELBM_BUILD_DIR/include"

# --------------------------------------------------------------------------- #
#  Automatic distro detection
# --------------------------------------------------------------------------- #

if [[ -z "$HERMITELBM_DISTRO" ]]; then
    if [[ -f /etc/os-release ]]; then
        source /etc/os-release
        case "$ID" in
            ubuntu)
                # Remove dots from version (e.g., 24.04 -> 2404)
                VERSION_NODOTS="${VERSION_ID//./}"
                HERMITELBM_DISTRO="ubuntu${VERSION_NODOTS}"
                ;;
            debian)
                # Debian VERSION_ID is just the major number (e.g., "13")
                HERMITELBM_DISTRO="debian${VERSION_ID}"
                ;;
            *)
                echo "Warning: Unsupported distribution '$ID'. Defaulting to ubuntu2404." >&2
                HERMITELBM_DISTRO="ubuntu2404"
                ;;
        esac
    else
        echo "Warning: /etc/os-release not found. Defaulting to ubuntu2404." >&2
        HERMITELBM_DISTRO="ubuntu2404"
    fi
    # echo "Auto-detected distribution: $HERMITELBM_DISTRO"
fi
export HERMITELBM_DISTRO

# --------------------------------------------------------------------------- #
#  CUDA Toolkit
# --------------------------------------------------------------------------- #

# Determine the correct CUDA directory suffix based on distro and minor version
CUDA_DIR_SUFFIX="${HERMITELBM_CUDA_VERSION_MAJOR}.${HERMITELBM_CUDA_VERSION_MINOR}"
if [[ "$HERMITELBM_DISTRO" =~ ^debian ]] && [[ "${HERMITELBM_CUDA_VERSION_MINOR}" == "0" ]]; then
    CUDA_DIR_SUFFIX="${HERMITELBM_CUDA_VERSION_MAJOR}"
fi

export HERMITELBM_CUDA_DIR="/usr/local/cuda-${CUDA_DIR_SUFFIX}"
export PATH="$HERMITELBM_CUDA_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$HERMITELBM_CUDA_DIR/lib64:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$HERMITELBM_CUDA_DIR/lib64:$LIBRARY_PATH"

# Fallback: If nvcc is found but the default path doesn't exist, resolve from nvcc
if command -v nvcc > /dev/null 2>&1; then
    NVCC_PATH=$(command -v nvcc)
    RESOLVED_NVCC_PATH=$(readlink -f "$NVCC_PATH" 2>/dev/null || echo "$NVCC_PATH")
    export HERMITELBM_CUDA_DIR=$(dirname "$(dirname "$RESOLVED_NVCC_PATH")")
else
    echo "Error: nvcc not found. Ensure CUDA is installed and in your PATH." >&2
    return 1
fi

# --------------------------------------------------------------------------- #
#  UCX (Unified Communication X)
# --------------------------------------------------------------------------- #
export HERMITELBM_UCX_DIR="$HERMITELBM_BUILD_DIR/ucx"
export PATH="$HERMITELBM_UCX_DIR/bin:$PATH"
export LIBRARY_PATH="$HERMITELBM_UCX_DIR/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$HERMITELBM_UCX_DIR/lib:$LD_LIBRARY_PATH"

# --------------------------------------------------------------------------- #
#  OpenMPI
# --------------------------------------------------------------------------- #
export HERMITELBM_MPI_DIR="$HERMITELBM_BUILD_DIR/OpenMPI"
export PATH="$HERMITELBM_MPI_DIR/bin:$PATH"
export LIBRARY_PATH="$HERMITELBM_MPI_DIR/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$HERMITELBM_MPI_DIR/lib:$LD_LIBRARY_PATH"
export C_INCLUDE_PATH="$HERMITELBM_MPI_DIR/include:$C_INCLUDE_PATH"
export CPLUS_INCLUDE_PATH="$HERMITELBM_MPI_DIR/include:$CPLUS_INCLUDE_PATH"

# --------------------------------------------------------------------------- #
#  Add project executables to PATH
# --------------------------------------------------------------------------- #
export PATH="$HERMITELBM_BIN_DIR:$PATH"

# --------------------------------------------------------------------------- #
#  Utility Functions
# --------------------------------------------------------------------------- #

# cleanCase: Remove timeStep and postProcess directories if inside a case folder
cleanCase()
{
    if [[ -f programControl ]]; then
        rm -rf timeStep
        rm -rf postProcess
        return 0
    else
        return 1
    fi
}

# profileRoofline: Wrapper for the roofline profiling script
profileRoofline()
{
    local script_path="$HERMITELBM_PROJECT_DIR/roofline.sh"
    if [[ ! -f "$script_path" ]]; then
        echo "ERROR: Profiling script not found at $script_path"
        return 1
    fi
    "$script_path" "$@"
}

# --------------------------------------------------------------------------- #