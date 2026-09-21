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
    A class applying the Neumann boundary condition

Namespace
    LBM

SourceFiles
    Neumann.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_NEUMANN_CUH
#define __MBLBM_NEUMANN_CUH

namespace LBM
{
    struct Neumann
    {
        /**
         * @brief Apply the Neumann boundary condition to a particular boundary node type
         * @param[in] moments Moment array (rho, U, Pi)
         * @param[in] incomings The incoming density and second-order moments
         * @param[in] sharedBuffer Shared memory buffer
         * @param[in] tid Thread ID within block
         **/
        template <class VelocitySet, const nodeType_t BoundaryCase, class SharedBuffer>
        __device__ [[nodiscard]] static inline void apply(
            momentsArray &moments,
            const thread::array<scalar_t, 7> &incomings,
            const SharedBuffer &sharedBuffer,
            const device::label_t tid) noexcept
        {
            genericDirichlet<VelocitySet>::apply<BoundaryCase>(moments, incomings, sharedBuffer[idxShared<1>(tid)], sharedBuffer[idxShared<2>(tid)], sharedBuffer[idxShared<3>(tid)]);
        }

        template <const device::label_t Moment>
        __device__ __host__ [[nodiscard]] static inline constexpr device::label_t idxShared(const device::label_t tid) noexcept
        {
            return tid * label_constant<NUMBER_MOMENTS() + 1>() + label_constant<Moment>();
        }
    };
}

#endif