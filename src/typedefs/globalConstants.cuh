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
    A list of typedefs used throughout the HermiteLBM source code

Namespace
    LBM

SourceFiles
    globalConstants.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_GLOBALCONSTANTS_CUH
#define __MBLBM_GLOBALCONSTANTS_CUH

namespace LBM
{
    /**
     * @brief Number of hydrodynamic moments
     * @tparam T The return type
     **/
    template <typename T = device::label_t>
    deviceHostFunctionReturnType inline consteval T NUMBER_MOMENTS() noexcept
    {
        return 10;
    }

    /**
     * @brief Total size of the shared memory
     * @tparam Q Number of velocity coefficients of the lattice
     * @tparam nVars Number of moment variables
     * @param[in] variableSize Size of the variable
     **/
    deviceHostFunctionReturnType inline consteval host::label_t sharedMemoryBufferSize(const host::label_t variableSize = 1) noexcept
    {
        return block::size<host::label_t>() * NUMBER_MOMENTS<host::label_t>() * variableSize;
    }

    namespace device
    {
        using ptrColl_t = device::ptrCollection<NUMBER_MOMENTS<host::label_t>(), scalar_t>;
    }

    /**
     * @brief Reference density 1.0
     * @tparam T The return type
     **/
    template <typename T = scalar_t>
    deviceHostFunctionReturnType inline consteval T rho0() noexcept
    {
        return static_cast<T>(1);
    }

    namespace device
    {
        /**
         * @brief Characteristic physical variables
         **/
        __device__ __constant__ scalar_t Ma = static_cast<scalar_t>(0);
        __device__ __constant__ scalar_t Re = static_cast<scalar_t>(0);
        __device__ __constant__ scalar_t nu = static_cast<scalar_t>(0);
        __device__ __constant__ scalar_t tau = static_cast<scalar_t>(0);
        __device__ __constant__ scalar_t L_char = static_cast<scalar_t>(0);
        __device__ __constant__ scalar_t omega = static_cast<scalar_t>(0);
        __device__ __constant__ scalar_t sigma = static_cast<scalar_t>(0);

        /**
         * @brief Force term
         **/
        __device__ __constant__ scalar_t F[3] = {static_cast<scalar_t>(0), static_cast<scalar_t>(0), static_cast<scalar_t>(0)};

        /**
         * @brief Boundary condition variables
         **/
        __device__ __constant__ scalar_t U_North[3] = {static_cast<scalar_t>(0), static_cast<scalar_t>(0), static_cast<scalar_t>(0)};
        __device__ __constant__ scalar_t U_South[3] = {static_cast<scalar_t>(0), static_cast<scalar_t>(0), static_cast<scalar_t>(0)};
        __device__ __constant__ scalar_t U_East[3] = {static_cast<scalar_t>(0), static_cast<scalar_t>(0), static_cast<scalar_t>(0)};
        __device__ __constant__ scalar_t U_West[3] = {static_cast<scalar_t>(0), static_cast<scalar_t>(0), static_cast<scalar_t>(0)};
        __device__ __constant__ scalar_t U_Back[3] = {static_cast<scalar_t>(0), static_cast<scalar_t>(0), static_cast<scalar_t>(0)};
        __device__ __constant__ scalar_t U_Front[3] = {static_cast<scalar_t>(0), static_cast<scalar_t>(0), static_cast<scalar_t>(0)};

        /**
         * @brief Mesh constant variables
         **/
        namespace latticeMesh
        {
            __device__ __constant__ device::label_t nPoints[3] = {static_cast<device::label_t>(0), static_cast<device::label_t>(0), static_cast<device::label_t>(0)};
            __device__ __constant__ device::label_t nBlocks[3] = {static_cast<device::label_t>(0), static_cast<device::label_t>(0), static_cast<device::label_t>(0)};
            __device__ __constant__ device::label_t blockOffset[3] = {static_cast<device::label_t>(0), static_cast<device::label_t>(0), static_cast<device::label_t>(0)};
        }
    }
}

#endif