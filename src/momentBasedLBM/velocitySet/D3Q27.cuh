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
    Definition of the D3Q27 velocity set

Namespace
    LBM

SourceFiles
    D3Q27.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_D3Q27_CUH
#define __MBLBM_D3Q27_CUH

namespace LBM
{
    /**
     * @class D3Q27
     * @brief Implements the D3Q27 velocity set for 3D Lattice Boltzmann simulations
     * @extends velocitySet
     *
     * This class provides the specific implementation for the D3Q27 lattice model,
     * which includes 27 discrete velocity directions in 3D space. It contains:
     * - Velocity components (cx, cy, cz) for each direction
     * - Weight coefficients for each direction
     * - Methods for moment calculation and population reconstruction
     * - Equilibrium distribution functions
     **/
    template <const thermalModel_t ThermalModel>
    class D3Q27 : public velocitySet<27, ThermalModel>
    {
    public:
        using Base = velocitySet<27, ThermalModel>;

        /**
         * @brief Determines the amount of shared memory required for a kernel based on the velocity set
         **/
        __device__ __host__ [[nodiscard]] static inline consteval host::label_t smem_alloc_size() noexcept
        {
            return block::sharedMemoryBufferSize<Base::Q(), NUMBER_MOMENTS<host::label_t>()>(sizeof(scalar_t));
        }
    };
}

#endif