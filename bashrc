# --------------------------------------------------------------------------- #
#                                                                             #
#  HermiteLBM: CUDA-based moment representation Lattice Boltzmann Method      #
#  Developed at UDESC - State University of Santa Catarina                    #
#  Website: https://www.udesc.br                                              #
#  Github: https://github.com/Geoenergia-Lab/HermiteLBM                       #
#                                                                             #
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
#  Utility Functions                                                          #
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

# printHeader: Prints the file header on start
printHeader()
{
    printf "/*---------------------------------------------------------------------------*\\ \n"
    printf "|                                                                             |\n"
    printf "| HermiteLBM: CUDA-based moment representation Lattice Boltzmann Method       |\n"
    printf "| Developed at UDESC - State University of Santa Catarina                     |\n"
    printf "| Website: https://www.udesc.br                                               |\n"
    printf "| Github: https://github.com/Geoenergia-Lab/HermiteLBM                        |\n"
    printf "|                                                                             |\n"
    printf "\\*---------------------------------------------------------------------------*/\n"
    printf "\n"
}

# printEnd: Prints the file footer on exit
printEnd()
{
    printf "\n"
    printf "End\n"
    printf "\n"
}

# notFound: Check if any listed commands are missing from PATH.
# Arguments: list of executable names
# Returns 0 (true) if at least one executable is NOT found, 1 (false) if all are found.
# Prints a generic error message to stderr for each missing command.
notFound()
{
    local ret=1                # assume all found, return failure (1)
    local cmd
    for cmd in "$@"; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            printf "Error: '%s' not found. Ensure environment is set up.\n" "$cmd" >&2
            ret=0              # at least one missing → success (0)
        fi
    done
    return $ret
}

# Check whether to skip header/footer (default: print them)
skip_header_footer()
{
    case "${SKIP_HEADER_AND_FOOTER:-}" in
        true|1|yes|on) return 0 ;;  # truthy → skip
        *)             return 1 ;;  # unset/false/anything else → print
    esac
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
#  USER-DEFINED ENVIRONMENT VARIABLES                                         #
# --------------------------------------------------------------------------- #

printHeader

# CUDA version (major and minor)
export HERMITELBM_CUDA_VERSION_MAJOR="13"
export HERMITELBM_CUDA_VERSION_MINOR="0"

# Architecture detection mode: "Automatic" or "Manual"
export HERMITELBM_ARCHITECTURE_DETECTION="Automatic"

# Manual architecture version (used only if HERMITELBM_ARCHITECTURE_DETECTION="Manual")
export HERMITELBM_ARCHITECTURE_VERSION="89"

# --------------------------------------------------------------------------- #
#  AUTOMATIC SETUP - DO NOT MODIFY BELOW UNLESS YOU KNOW WHAT YOU ARE DOING   #
# --------------------------------------------------------------------------- #

# Project root directory (where this bashrc file lives)
export HERMITELBM_PROJECT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Build tree structure
export HERMITELBM_BUILD_DIR="$HERMITELBM_PROJECT_DIR/build"
export HERMITELBM_BIN_DIR="$HERMITELBM_BUILD_DIR/bin"
export HERMITELBM_INCLUDE_DIR="$HERMITELBM_BUILD_DIR/include"

# --------------------------------------------------------------------------- #
#  Automatic distro detection (now supports Fedora, RHEL, CentOS)             #
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
            fedora)
                HERMITELBM_DISTRO="fedora${VERSION_ID}"
                ;;
            rhel|centos)
                # Use major version only (e.g., "9")
                HERMITELBM_DISTRO="${ID}${VERSION_ID%%.*}"
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
#  CUDA Toolkit                                                               #
# --------------------------------------------------------------------------- #

# Determine suffix for default CUDA path
CUDA_DIR_SUFFIX="${HERMITELBM_CUDA_VERSION_MAJOR}.${HERMITELBM_CUDA_VERSION_MINOR}"
if [[ "$HERMITELBM_DISTRO" =~ ^(debian|fedora) ]] && [[ "${HERMITELBM_CUDA_VERSION_MINOR}" == "0" ]]; then
    CUDA_DIR_SUFFIX="${HERMITELBM_CUDA_VERSION_MAJOR}"
fi

# Default expected CUDA directory
DEFAULT_CUDA_DIR="/usr/local/cuda-${CUDA_DIR_SUFFIX}"

# 1) Try the default CUDA directory first (most common installation)
if [[ -x "$DEFAULT_CUDA_DIR/bin/nvcc" ]]; then
    export HERMITELBM_CUDA_DIR="$DEFAULT_CUDA_DIR"
    export PATH="$HERMITELBM_CUDA_DIR/bin:$PATH"
    export LD_LIBRARY_PATH="$HERMITELBM_CUDA_DIR/lib64:$LD_LIBRARY_PATH"
    export LIBRARY_PATH="$HERMITELBM_CUDA_DIR/lib64:$LIBRARY_PATH"
    HERMITELBM_CUDA_FOUND=1

# 2) If not in default location, fall back to whatever nvcc is in PATH
elif command -v nvcc > /dev/null 2>&1; then
    NVCC_PATH=$(command -v nvcc)
    RESOLVED_NVCC_PATH=$(readlink -f "$NVCC_PATH" 2>/dev/null || echo "$NVCC_PATH")
    HERMITELBM_CUDA_DIR=$(dirname "$(dirname "$RESOLVED_NVCC_PATH")")

    if [[ -d "$HERMITELBM_CUDA_DIR" ]]; then
        export HERMITELBM_CUDA_DIR
        export PATH="$HERMITELBM_CUDA_DIR/bin:$PATH"
        export LD_LIBRARY_PATH="$HERMITELBM_CUDA_DIR/lib64:$LD_LIBRARY_PATH"
        export LIBRARY_PATH="$HERMITELBM_CUDA_DIR/lib64:$LIBRARY_PATH"
        HERMITELBM_CUDA_FOUND=1
    else
        echo "Warning: CUDA directory $HERMITELBM_CUDA_DIR not found. Skipping CUDA paths." >&2
        HERMITELBM_CUDA_FOUND=0
    fi

# 3) No CUDA found at all
else
    HERMITELBM_CUDA_FOUND=0
    echo "Warning: nvcc not found. CUDA environment will not be configured." >&2
fi

# --------------------------------------------------------------------------- #
#  Print detected environment                                                 #
# --------------------------------------------------------------------------- #

source /etc/os-release
echo "HermiteLBM"
echo "{"
if (( HERMITELBM_CUDA_FOUND )); then
    echo "    CUDA version: ${HERMITELBM_CUDA_VERSION_MAJOR}.${HERMITELBM_CUDA_VERSION_MINOR}"
else
    echo "    CUDA version: Not found"
fi
echo "    Distro: ${ID} ${VERSION_ID}"
echo "    Architecture detection: ${HERMITELBM_ARCHITECTURE_DETECTION}"
echo "    Project directory: ${HERMITELBM_PROJECT_DIR}"
echo "};"
echo ""

# --------------------------------------------------------------------------- #
#  UCX (Unified Communication X)                                              #
# --------------------------------------------------------------------------- #

export HERMITELBM_UCX_DIR="$HERMITELBM_BUILD_DIR/ucx"
export PATH="$HERMITELBM_UCX_DIR/bin:$PATH"
export LIBRARY_PATH="$HERMITELBM_UCX_DIR/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$HERMITELBM_UCX_DIR/lib:$LD_LIBRARY_PATH"

# --------------------------------------------------------------------------- #
#  OpenMPI                                                                    #
# --------------------------------------------------------------------------- #

export HERMITELBM_MPI_DIR="$HERMITELBM_BUILD_DIR/OpenMPI"
export PATH="$HERMITELBM_MPI_DIR/bin:$PATH"
export LIBRARY_PATH="$HERMITELBM_MPI_DIR/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$HERMITELBM_MPI_DIR/lib:$LD_LIBRARY_PATH"
export C_INCLUDE_PATH="$HERMITELBM_MPI_DIR/include:$C_INCLUDE_PATH"
export CPLUS_INCLUDE_PATH="$HERMITELBM_MPI_DIR/include:$CPLUS_INCLUDE_PATH"

# --------------------------------------------------------------------------- #
#  Add project executables to PATH                                            #
# --------------------------------------------------------------------------- #

export PATH="$HERMITELBM_BIN_DIR:$PATH"

# --------------------------------------------------------------------------- #