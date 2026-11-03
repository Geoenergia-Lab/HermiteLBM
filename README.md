![HermiteLBM Logo](HermiteLBM.png)

# HermiteLBM

## Overview

HermiteLBM is a high-performance, GPU-accelerated Lattice Boltzmann Method (LBM) solver. Written in C++, it leverages both **NVIDIA CUDA** and **AMD ROCm (HIP)** to deliver massively parallel execution across a wide range of modern GPU architectures. It is designed to simulate 3D fluid flows using isothermal and moment-based approaches.

## Features

- **Cross-Platform GPU Acceleration**: Compatible with both NVIDIA (CUDA) and AMD (ROCm) GPUs, ensuring high performance regardless of your hardware vendor.
- **Velocity Sets**: Supports **D3Q19** and **D3Q27** lattice configurations.
- **Multiple Solvers**: Includes ready-to-run solvers for:
  * `isothermalD3Q19`
  * `isothermalD3Q27`
  * `momentBasedD3Q19`
  * `momentBasedD3Q27`
- **Advanced Boundary Conditions**: Supports regularized boundary conditions (Dirichlet, No-Slip) and specific configurations for jet flows and lid-driven cavities.
- **Function Objects**: On-the-fly calculation of flow properties including:
  * Kinetic Energy
  * Strain Rate Tensor
  * Turbulence Statistics
  * Mean fields (PiMean, UMean, rhoMean)
- **Post-Processing**: Built-in support for multiple output formats including **VTK** (VTU, VTI, VTS), **Tecplot**, and binary formats (LBMBin).

## Repository Structure

- `src/`: Core library containing the LBM implementation (streaming, collision, memory management, numerical schemes).
- `applications/`: Source code for the main executable solvers and post-processing utilities (`fieldCalculate`, `fieldConvert`).
- `cases/`: Ready-to-run example simulations:
  * `benchmark`
  * `jetFlow`
  * `lidDrivenCavity`
- `unitTests/`: Test suites, including NaN detection (`containsNaN`) and memory sanitization (`computeSanitizer`).
- `preCommit/`: Python scripts for pre-commit formatting and code health checks (e.g., undocumented functions, comment mismatches).

## Building from Source

### Prerequisites

- C++ Compiler (GCC/Clang for Linux, MSVC for Windows)
- **For NVIDIA GPUs:** NVIDIA CUDA Toolkit
- **For AMD GPUs:** AMD ROCm Toolkit (HIP)

### Environment Setup (Linux)

Before compiling on Linux, you must configure your environment and specify your compiler toolkit (HIP or CUDA) based on what you have installed. 

An environment setup script (`bashrc`) is provided in the root of the repository. Open the existing `bashrc` file and modify the `HERMITELBM_API` variable to match your target hardware:

```bash
# GPU backend selection. Must be one of: "CUDA", "HIP"
export HERMITELBM_API="CUDA"
```

You can also adjust the CUDA/ROCm versions and architecture detection settings within this file if necessary. Once configured, source the script to apply the environment variables before building:

```bash
source bashrc
```

### Compilation

The project supports both Linux and Windows build environments via Makefiles and batch scripts. The build system will automatically target your configured GPU environment (`CUDA` via `nvcc` or `ROCm` via `hipcc`).

**Linux:**

```
# Compile the entire project
make install
```

**Windows:**

Note: As of yet, HermiteLBM is not compatible with HIPCC on Windows. Windows compilation currently only supports CUDA.

```
:: Compile the entire project
Make install
```

## Running a Case

To run a simulation, navigate to one of the provided cases (e.g., the lid-driven cavity) and execute the desired solver:

```
cd cases/lidDrivenCavity/D3Q27
isothermalD3Q27 -GPU 0
```

*(Note: Modify the execution command based on your specific path and setup).*

## License

\[Include License Information Here - e.g., MIT, GPLv3\]
