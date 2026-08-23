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
    A class applying boundary conditions to a purely periodic cube

Namespace
    LBM

SourceFiles
    benchmark.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_BENCHMARK_CUH
#define __MBLBM_BENCHMARK_CUH

namespace LBM
{
    /**
     * @class benchmark
     * @brief Applies boundary conditions for lid-driven cavity simulations using moment representation
     *
     * This class implements the boundary condition treatment for the D3Q19 lattice model
     * in lid-driven cavity flow simulations. It handles both static wall boundaries and
     * moving lid boundaries using moment-based boundary conditions derived from the
     * regularized LBM approach.
     **/
    class benchmark
    {
    public:
        /**
         * @brief Default constructor (constexpr)
         **/
        __device__ __host__ [[nodiscard]] inline consteval benchmark() {}

        /**
         * @brief Periodic boundary definitions
         **/
        __device__ __host__ [[nodiscard]] static inline consteval bool periodicX() noexcept { return true; }
        __device__ __host__ [[nodiscard]] static inline consteval bool periodicY() noexcept { return true; }
        __device__ __host__ [[nodiscard]] static inline consteval bool periodicZ() noexcept { return true; }

        /**
         * @brief Switch determining whether or not the boundary condition actually applies a condition
         **/
        __device__ __host__ [[nodiscard]] static inline consteval bool appliesCondition() noexcept { return false; }

    private:
    };
}

#endif