#!/bin/bash
# ------------------------------------------------------------------------------
#  Roofline Profiling Wrapper for CUDA LBM Executables
#
#  Usage: ./roofline.sh <executable> [executable_args...]
#  Example: ./roofline.sh momentBasedD3Q27 -GPU 0,1 -size 256
#
#  Note: If you encounter ERR_NVGPUCTRPERM, enable performance counters via:
#        - Windows: NVIDIA Control Panel → Developer → Allow access to all users
#        - Linux:   echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia.conf
# ------------------------------------------------------------------------------

set -e  # Exit on any error (but we handle graceful exit manually)

# ------------------------------------------------------------------------------
#  Detect if script is sourced (to avoid closing the terminal)
# ------------------------------------------------------------------------------
(return 0 2>/dev/null) && SOURCED=1 || SOURCED=0

graceful_exit() {
    local code=${1:-1}
    if [[ $SOURCED -eq 1 ]]; then
        return "$code" 2>/dev/null || exit "$code"
    else
        exit "$code"
    fi
}

# ------------------------------------------------------------------------------
#  Check required environment variables
# ------------------------------------------------------------------------------
missing_vars=()
for var in CUDALBM_PROJECT_DIR CUDALBM_BUILD_DIR CUDALBM_BIN_DIR CUDALBM_INCLUDE_DIR; do
    if [ -z "${!var}" ]; then
        missing_vars+=("$var")
    fi
done

if [ ${#missing_vars[@]} -gt 0 ]; then
    echo "ERROR: The following environment variables are not set:"
    printf '  - %s\n' "${missing_vars[@]}"
    echo ""
    echo "Please source the project's bashrc file first:"
    echo "  source /path/to/your/project/bashrc"
    graceful_exit 1
fi

# ------------------------------------------------------------------------------
#  Parse command line arguments
# ------------------------------------------------------------------------------
if [ $# -lt 1 ]; then
    echo "ERROR: Please provide the executable name."
    echo "Usage: $0 <executable> [executable_args...]"
    graceful_exit 1
fi

EXE_NAME="$1"
shift  # Remove executable name; remaining arguments are passed to the executable
EXE_ARGS=("$@")

EXE_PATH="$CUDALBM_BIN_DIR/$EXE_NAME"
HARDWARE_INFO="$CUDALBM_INCLUDE_DIR/hardware.info"

if [ ! -f "$EXE_PATH" ]; then
    echo "ERROR: Executable not found: $EXE_PATH"
    graceful_exit 1
fi
if [ ! -f "$HARDWARE_INFO" ]; then
    echo "ERROR: hardware.info not found at $HARDWARE_INFO"
    graceful_exit 1
fi

# ------------------------------------------------------------------------------
#  Extract mandatory -GPU <list> from executable arguments
# ------------------------------------------------------------------------------
GPU_LIST=""
for ((i=0; i<${#EXE_ARGS[@]}; i++)); do
    if [[ "${EXE_ARGS[i]}" == "-GPU" ]]; then
        if [[ $((i+1)) -lt ${#EXE_ARGS[@]} && "${EXE_ARGS[i+1]}" != -* ]]; then
            GPU_LIST="${EXE_ARGS[i+1]}"
            break
        else
            echo "ERROR: -GPU flag provided but no valid GPU list follows."
            graceful_exit 1
        fi
    fi
done

if [ -z "$GPU_LIST" ]; then
    echo "ERROR: The -GPU argument is mandatory for $EXE_NAME."
    echo "Usage example: $0 $EXE_NAME -GPU 0,1 [other options]"
    graceful_exit 1
fi

# ------------------------------------------------------------------------------
#  Parse GPU IDs and fetch names from hardware.info
# ------------------------------------------------------------------------------
IFS=',' read -ra GPU_IDS <<< "$GPU_LIST"

# Number of GPUs used
NUM_GPUS=${#GPU_IDS[@]}

declare -a GPU_NAMES
for id in "${GPU_IDS[@]}"; do
    gpu_line=$(grep -E "^GPU_NAME_$id\s*=" "$HARDWARE_INFO" | head -n1)
    if [ -z "$gpu_line" ]; then
        echo "ERROR: GPU $id not found in $HARDWARE_INFO"
        graceful_exit 1
    fi
    name=$(echo "$gpu_line" | cut -d'=' -f2- | xargs)
    safe_name=$(echo "$name" | sed 's/ /_/g' | sed 's/[^a-zA-Z0-9_.-]//g')
    GPU_NAMES+=("$safe_name")
done

GPU_NAMES_STR=$(IFS=_; echo "${GPU_NAMES[*]}")

# ------------------------------------------------------------------------------
#  Create timestamped output directory
# ------------------------------------------------------------------------------
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_DIR="./profiling_results/${EXE_NAME}_${TIMESTAMP}_${NUM_GPUS}GPU_${GPU_NAMES_STR}"
mkdir -p "$RESULT_DIR"
echo "Results will be saved to: $(realpath "$RESULT_DIR")"

# Copy hardware information for reference
cp "$HARDWARE_INFO" "$RESULT_DIR/hardware.info"

# ------------------------------------------------------------------------------
#  Run Nsight Compute with roofline metrics
# ------------------------------------------------------------------------------
REPORT_BASE="$RESULT_DIR/${EXE_NAME}_profile"

echo "Profiling $EXE_NAME on GPU(s) $GPU_LIST with arguments: ${EXE_ARGS[*]}"
ncu \
    --metrics sm__sp__ops.sum,dram__bytes.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed \
    --section SpeedOfLight \
    --set full \
    --export "$REPORT_BASE" \
    --force-overwrite \
    "$EXE_PATH" "${EXE_ARGS[@]}"

# ------------------------------------------------------------------------------
#  Extract summary metrics into a human-readable file
# ------------------------------------------------------------------------------
echo "Extracting summary metrics..."
ncu -i "$REPORT_BASE.ncu-rep" --csv --page raw > "$RESULT_DIR/metrics_raw.csv" 2>/dev/null || true

{
    echo "Roofline Profiling Summary"
    echo "=========================="
    echo "Date:           $(date)"
    echo "Executable:     $EXE_NAME"
    echo "Command:        $EXE_NAME ${EXE_ARGS[*]}"
    echo "GPU list:       $GPU_LIST"
    echo "GPU count:      $NUM_GPUS"
    echo "GPU names:      ${GPU_NAMES[*]}"
    echo ""
    echo "Metrics (aggregate over all kernel launches):"
    grep -E "sm__sp__ops.sum|dram__bytes.sum|sm__throughput" "$RESULT_DIR/metrics_raw.csv" 2>/dev/null || echo "Metrics not found."
} > "$RESULT_DIR/summary.txt"

echo "Profiling complete. Results in: $RESULT_DIR"