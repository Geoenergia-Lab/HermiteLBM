#*---------------------------------------------------------------------------*#
#                                                                             #
#  cudaLBM: CUDA-based moment representation Lattice Boltzmann Method         #
#  Developed at UDESC - State University of Santa Catarina                    #
#  Website: https://www.udesc.br                                              #
#  Github: https://github.com/geoenergiaUDESC/cudaLBM                         #
#                                                                             #
#*---------------------------------------------------------------------------*#

# ------------------------------------------------------------------------------
#  USER-DEFINED ENVIRONMENT VARIABLES
# ------------------------------------------------------------------------------

# Architecture detection mode: "Automatic" or "Manual"
export CUDALBM_ARCHITECTURE_DETECTION="Automatic"

# Manual architecture version (used only if CUDALBM_ARCHITECTURE_DETECTION="Manual")
export CUDALBM_ARCHITECTURE_VERSION="89"

# ------------------------------------------------------------------------------
#  AUTOMATIC SETUP – DO NOT MODIFY BELOW UNLESS YOU KNOW WHAT YOU ARE DOING
# ------------------------------------------------------------------------------

# Project root directory (where this bashrc file lives)
export CUDALBM_PROJECT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Build tree structure
export CUDALBM_BUILD_DIR="$CUDALBM_PROJECT_DIR/build"
export CUDALBM_BIN_DIR="$CUDALBM_BUILD_DIR/bin"
export CUDALBM_INCLUDE_DIR="$CUDALBM_BUILD_DIR/include"

# ------------------------------------------------------------------------------
#  CUDA Toolkit
# ------------------------------------------------------------------------------
if command -v nvcc > /dev/null 2>&1; then
    NVCC_PATH=$(command -v nvcc)
    RESOLVED_NVCC_PATH=$(readlink -f "$NVCC_PATH" 2>/dev/null || echo "$NVCC_PATH")
    export CUDALBM_CUDA_DIR=$(dirname "$(dirname "$RESOLVED_NVCC_PATH")")
else
    echo "Error: nvcc not found. Ensure CUDA is installed and in your PATH." >&2
    return 1
fi

export PATH="$CUDALBM_CUDA_DIR/bin:$PATH"
export LIBRARY_PATH="$CUDALBM_CUDA_DIR/lib64:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$CUDALBM_CUDA_DIR/lib64:$LD_LIBRARY_PATH"

# ------------------------------------------------------------------------------
#  UCX (Unified Communication X)
# ------------------------------------------------------------------------------
export CUDALBM_UCX_DIR="$CUDALBM_BUILD_DIR/ucx"
export PATH="$CUDALBM_UCX_DIR/bin:$PATH"
export LIBRARY_PATH="$CUDALBM_UCX_DIR/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$CUDALBM_UCX_DIR/lib:$LD_LIBRARY_PATH"

# ------------------------------------------------------------------------------
#  OpenMPI
# ------------------------------------------------------------------------------
export CUDALBM_MPI_DIR="$CUDALBM_BUILD_DIR/OpenMPI"
export PATH="$CUDALBM_MPI_DIR/bin:$PATH"
export LIBRARY_PATH="$CUDALBM_MPI_DIR/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$CUDALBM_MPI_DIR/lib:$LD_LIBRARY_PATH"
export C_INCLUDE_PATH="$CUDALBM_MPI_DIR/include:$C_INCLUDE_PATH"
export CPLUS_INCLUDE_PATH="$CUDALBM_MPI_DIR/include:$CPLUS_INCLUDE_PATH"

# ------------------------------------------------------------------------------
#  Add project executables to PATH
# ------------------------------------------------------------------------------
export PATH="$CUDALBM_BIN_DIR:$PATH"

# ------------------------------------------------------------------------------
#  Utility Functions
# ------------------------------------------------------------------------------

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
    local script_path="$CUDALBM_PROJECT_DIR/roofline.sh"
    if [[ ! -f "$script_path" ]]; then
        echo "ERROR: Profiling script not found at $script_path"
        return 1
    fi
    "$script_path" "$@"
}

#*---------------------------------------------------------------------------*#