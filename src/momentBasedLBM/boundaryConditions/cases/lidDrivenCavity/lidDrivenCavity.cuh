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
    A class applying boundary conditions to the lid driven cavity case

Namespace
    LBM

SourceFiles
    lidDrivenCavity.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_LIDDRIVENCAVITY_CUH
#define __MBLBM_LIDDRIVENCAVITY_CUH

namespace LBM
{
    /**
     * @class lidDrivenCavity
     * @brief Applies boundary conditions for lid-driven cavity simulations using moment representation
     *
     * This class implements the boundary condition treatment for the D3Q19 lattice model
     * in lid-driven cavity flow simulations. It handles both static wall boundaries and
     * moving lid boundaries using moment-based boundary conditions derived from the
     * regularized LBM approach.
     **/
    class lidDrivenCavity : public boundaryConditionType<false, false, false>
    {
        using Base = boundaryConditionType<false, false, false>;

    public:
        /**
         * @brief Switch determining whether or not the boundary condition actually applies a condition
         **/
        __device__ __host__ [[nodiscard]] static inline consteval bool appliesCondition() noexcept { return true; }

        /**
         * @brief Public method to calculate the post-streaming methods and update boundary conditions
         **/
        template <class VelocitySet, class SharedBuffer>
        __device__ static inline constexpr void calculate_moments(
            const thread::array<scalar_t, VelocitySet::Q()> &pop,
            momentsArray &moments,
            [[maybe_unused]] SharedBuffer &sharedBuffer,
            [[maybe_unused]] const thread::coordinate &Tx,
            const device::pointCoordinate &point,
            [[maybe_unused]] const device::label_t tid) noexcept
        {
            const NormalVector boundaryNormal(point);

            VelocitySet::template calculate_moments(moments, pop);

            if (boundaryNormal.isBoundary())
            {
                Base::apply<VelocitySet>(pop, moments, boundaryNormal.nodeType());
            }
        }
    };
}

#endif