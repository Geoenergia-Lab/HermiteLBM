/*---------------------------------------------------------------------------*\
|                                                                             |
| HermiteLBM: CUDA-based moment representation Lattice Boltzmann Method       |
| Developed at UDESC - State University of Santa Catarina                     |
| Website: https://www.udesc.br                                               |
| Github: https://github.com/Geoenergia-Lab/HermiteLBM                        |
|                                                                             |
\*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*\

Copyright (C) 2023 UDESC Geoenergia Lab
Authors: Nathan Duggins (Geoenergia Lab, UDESC)

This implementation is derived from concepts and algorithms developed in:
  MR-LBM: Moment Representation Lattice Boltzmann Method
  Copyright (C) 2021 CERNN
  Developed at Universidade Federal do Paraná (UFPR)
  Original authors: V. M. de Oliveira, M. A. de Souza, R. F. de Souza
  GitHub: https://github.com/CERNN/MR-LBM
  Licensed under GNU General Public License version 2

License
    This file is part of HermiteLBM.

    HermiteLBM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

Description
    Definitions of CUDA kernels to calculate solution quantities. Unfortunately
    we cannot template CUDA kernels and annotate with launch bounds at the same
    time due to the compiler apparently not noticing that the argument
    preceding the launch bounds is a specification of a template parameter.
    Instead we have to do this preprocessor nonsense. We live in a cruel world.

Namespace
    LBM::functionObjects

SourceFiles
    commonKernelDefinitions.cuh

\*---------------------------------------------------------------------------*/

/**
 * @brief CUDA kernel for calculating the time averaged quantity only
 * @param[in] devPtrs Device pointer collection containing density, velocity and moment fields
 * @param[out] resultMeanPtrs Device pointer collection for the time averaged quantity
 * @param[in] invNewCount Reciprocal of (nTimeSteps + 1) for time averaging
 **/
__launch_bounds__(block::maxThreads(), This::MIN_BLOCKS_PER_MP) __global__ static void meanKernel(
    const device::ptrColl_t devPtrs,
    const device::ptrCollection<This::N, scalar_t> resultMeanPtrs,
    const scalar_t invNewCount)
{
    functionObjects::mean<This>(devPtrs, resultMeanPtrs, invNewCount);
}

/**
 * @brief CUDA kernel for calculating the instantaneous and time averaged quantity
 * @param[in] devPtrs Device pointer collection containing density, velocity and moment fields
 * @param[out] resultPtrs Device pointer collection for the instantaneous quantity
 * @param[out] resultMeanPtrs Device pointer collection for the time averaged quantity
 * @param[in] invNewCount Reciprocal of (nTimeSteps + 1) for time averaging
 **/
__launch_bounds__(block::maxThreads(), This::MIN_BLOCKS_PER_MP) __global__ static void instantaneousAndMeanKernel(
    const device::ptrColl_t devPtrs,
    const device::ptrCollection<This::N, scalar_t> resultPtrs,
    const device::ptrCollection<This::N, scalar_t> resultMeanPtrs,
    const scalar_t invNewCount)
{
    functionObjects::instantaneousAndMean<This>(devPtrs, resultPtrs, resultMeanPtrs, invNewCount);
}

/**
 * @brief CUDA kernel for calculating the instantaneous quantity only
 * @param[in] devPtrs Device pointer collection containing density, velocity and moment fields
 * @param[out] resultPtrs Device pointer collection for the instantaneous quantity
 **/
__launch_bounds__(block::maxThreads(), This::MIN_BLOCKS_PER_MP) __global__ static void instantaneousKernel(
    const device::ptrColl_t devPtrs,
    const device::ptrCollection<This::N, scalar_t> resultPtrs)
{
    functionObjects::instantaneous<This>(devPtrs, resultPtrs);
}

/**
 * @brief CUDA kernel for calculating the prime quantity only
 * @param[in] devPtrs Device pointer collection containing density, velocity and moment fields
 * @param[out] resultPtrs Device pointer collection for the instantaneous quantity
 **/
__launch_bounds__(block::maxThreads(), This::MIN_BLOCKS_PER_MP) __global__ static void primeKernel(
    const device::ptrColl_t devPtrs,
    const device::ptrCollection<This::N, scalar_t> resultMeanPtrs,
    const device::ptrCollection<This::N, scalar_t> resultPtrs)
{
    functionObjects::prime<This>(devPtrs, resultMeanPtrs, resultPtrs);
}

/**
 * @brief CUDA kernel for calculating the time average of the square of the prime quantity
 * @param[in] devPtrs Device pointer collection containing density, velocity and moment fields
 * @param[in] resultMeanPtrs Device pointer collection for the time average quantity
 * @param[out] resultPrimeMeanSqPtrs Device pointer collection for the time average of the square of the prime quantity
 * @param[in] invNewCount Reciprocal of (nTimeSteps + 1) for time averaging
 **/
__launch_bounds__(block::maxThreads(), This::MIN_BLOCKS_PER_MP) __global__ static void primeSqMeanKernel(
    const device::ptrColl_t devPtrs,
    const device::ptrCollection<This::N, scalar_t> resultMeanPtrs,
    const device::ptrCollection<This::N, scalar_t> resultPrimeMeanSqPtrs,
    const scalar_t invNewCount)
{
    functionObjects::primeSqMean<This>(devPtrs, resultMeanPtrs, resultPrimeMeanSqPtrs, invNewCount);
}

struct kernel
{
    /**
     * @brief Returns a function pointer to the instantaneous kernel
     **/
    __host__ [[nodiscard]] static inline consteval auto instantaneous() noexcept { return instantaneousKernel; }

    /**
     * @brief Returns a function pointer to the time average kernel
     **/
    __host__ [[nodiscard]] static inline consteval auto mean() noexcept { return meanKernel; }

    /**
     * @brief Returns a function pointer to the instantaneous and time average kernel
     **/
    __host__ [[nodiscard]] static inline consteval auto instantaneousAndMean() noexcept { return instantaneousAndMeanKernel; }

    /**
     * @brief Returns a function pointer to the prime kernel
     **/
    __host__ [[nodiscard]] static inline consteval auto prime() noexcept { return primeKernel; }

    /**
     * @brief Returns a function pointer to the prime squared time average kernel
     **/
    __host__ [[nodiscard]] static inline consteval auto primeSqMean() noexcept { return primeSqMeanKernel; }
};