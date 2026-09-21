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
    class lidDrivenCavity : public boundaryConditionType<WALL, WALL, WALL>
    {
    public:
        /**
         * @brief Public method to calculate the post-streaming methods and update boundary conditions
         **/
        template <class VelocitySet, class SharedBuffer>
        __device__ static inline constexpr void calculate_moments(
            const thread::array<scalar_t, VelocitySet::Q()> &pop,
            momentsArray &moments,
            [[maybe_unused]] SharedBuffer &sharedBuffer,
            [[maybe_unused]] const thread::coordinate &Tx,
            [[maybe_unused]] const device::pointCoordinate &point,
            const device::label_t tid) noexcept
        {
            const NormalVector boundaryNormal(point);

            VelocitySet::template calculate_moments(moments, pop, boundaryNormal);

            if (boundaryNormal.isBoundary())
            {
                calculate_moments<VelocitySet>(pop, moments, boundaryNormal);
            }
        }

    private:
        /**
         * @brief Applies the chosen set of boundary conditions to the moment variables at boundary nodes
         * @tparam VelocitySet The velocity set (D3Q19 or D3Q27)
         * @param[in] pop Population density array at current lattice node
         * @param[out] moments Moment array (rho, U, Pi)
         * @param[in] boundaryNormal Normal vector information at boundary node
         **/
        template <class VelocitySet>
        __device__ static inline constexpr void calculate_moments(
            const thread::array<scalar_t, VelocitySet::Q()> &pop,
            momentsArray &moments,
            const NormalVector &boundaryNormal) noexcept
        {
            // Get the incoming moments from the array
            const thread::array<scalar_t, 7> incomings(
                moments[m_i<0>()],
                moments[m_i<4>()],
                moments[m_i<5>()],
                moments[m_i<6>()],
                moments[m_i<7>()],
                moments[m_i<8>()],
                moments[m_i<9>()]);

            switch (boundaryNormal.nodeType())
            {
            // Cardinal no-slip boundaries
            case normalVectorBase::WEST():
            {
                noSlip<VelocitySet>::apply<normalVectorBase::WEST()>(moments, incomings);
                return;
            }
            case normalVectorBase::EAST():
            {
                noSlip<VelocitySet>::apply<normalVectorBase::EAST()>(moments, incomings);
                return;
            }
            case normalVectorBase::BACK():
            {
                noSlip<VelocitySet>::apply<normalVectorBase::BACK()>(moments, incomings);
                return;
            }
            case normalVectorBase::FRONT():
            {
                noSlip<VelocitySet>::apply<normalVectorBase::FRONT()>(moments, incomings);
                return;
            }
            case normalVectorBase::SOUTH():
            {
                noSlip<VelocitySet>::apply<normalVectorBase::SOUTH()>(moments, incomings);
                return;
            }

            // All slip boundaries
            case normalVectorBase::NORTH():
            {
                const scalar_t U_x = device::U_North[0];
                constexpr const scalar_t U_y = static_cast<scalar_t>(0);
                constexpr const scalar_t U_z = static_cast<scalar_t>(0);
                genericDirichlet<VelocitySet>::apply<normalVectorBase::NORTH()>(moments, incomings, U_x, U_y, U_z);

                return;
            }
            case normalVectorBase::NORTH_WEST():
            {
                const scalar_t U_x = device::U_North[0] / static_cast<scalar_t>(2);
                constexpr const scalar_t U_y = static_cast<scalar_t>(0);
                constexpr const scalar_t U_z = static_cast<scalar_t>(0);
                genericDirichlet<VelocitySet>::apply<normalVectorBase::NORTH_WEST()>(moments, incomings, U_x, U_y, U_z);
                return;
            }
            case normalVectorBase::NORTH_EAST():
            {
                const scalar_t U_x = device::U_North[0] / static_cast<scalar_t>(2);
                constexpr const scalar_t U_y = static_cast<scalar_t>(0);
                constexpr const scalar_t U_z = static_cast<scalar_t>(0);
                genericDirichlet<VelocitySet>::apply<normalVectorBase::NORTH_EAST()>(moments, incomings, U_x, U_y, U_z);
                return;
            }
            case normalVectorBase::NORTH_BACK():
            {
                const scalar_t U_x = device::U_North[0] / static_cast<scalar_t>(2);
                constexpr const scalar_t U_y = static_cast<scalar_t>(0);
                constexpr const scalar_t U_z = static_cast<scalar_t>(0);
                genericDirichlet<VelocitySet>::apply<normalVectorBase::NORTH_BACK()>(moments, incomings, U_x, U_y, U_z);
                return;
            }
            case normalVectorBase::NORTH_FRONT():
            {
                const scalar_t U_x = device::U_North[0] / static_cast<scalar_t>(2);
                constexpr const scalar_t U_y = static_cast<scalar_t>(0);
                constexpr const scalar_t U_z = static_cast<scalar_t>(0);
                genericDirichlet<VelocitySet>::apply<normalVectorBase::NORTH_FRONT()>(moments, incomings, U_x, U_y, U_z);
                return;
            }
            case normalVectorBase::NORTH_WEST_FRONT():
            {
                const scalar_t U_x = device::U_North[0] / static_cast<scalar_t>(3);
                constexpr const scalar_t U_y = static_cast<scalar_t>(0);
                constexpr const scalar_t U_z = static_cast<scalar_t>(0);
                genericDirichlet<VelocitySet>::apply<normalVectorBase::NORTH_WEST_FRONT()>(moments, incomings, U_x, U_y, U_z);
                return;
            }
            case normalVectorBase::NORTH_EAST_FRONT():
            {
                const scalar_t U_x = device::U_North[0] / static_cast<scalar_t>(3);
                constexpr const scalar_t U_y = static_cast<scalar_t>(0);
                constexpr const scalar_t U_z = static_cast<scalar_t>(0);
                genericDirichlet<VelocitySet>::apply<normalVectorBase::NORTH_EAST_FRONT()>(moments, incomings, U_x, U_y, U_z);
                return;
            }
            case normalVectorBase::NORTH_WEST_BACK():
            {
                const scalar_t U_x = device::U_North[0] / static_cast<scalar_t>(3);
                constexpr const scalar_t U_y = static_cast<scalar_t>(0);
                constexpr const scalar_t U_z = static_cast<scalar_t>(0);
                genericDirichlet<VelocitySet>::apply<normalVectorBase::NORTH_WEST_BACK()>(moments, incomings, U_x, U_y, U_z);
                return;
            }
            case normalVectorBase::NORTH_EAST_BACK():
            {
                const scalar_t U_x = device::U_North[0] / static_cast<scalar_t>(3);
                constexpr const scalar_t U_y = static_cast<scalar_t>(0);
                constexpr const scalar_t U_z = static_cast<scalar_t>(0);
                genericDirichlet<VelocitySet>::apply<normalVectorBase::NORTH_EAST_BACK()>(moments, incomings, U_x, U_y, U_z);
                return;
            }

            // Edge no-slip boundaries
            case normalVectorBase::SOUTH_WEST():
            {
                noSlip<VelocitySet>::apply<normalVectorBase::SOUTH_WEST()>(moments, incomings);
                return;
            }
            case normalVectorBase::SOUTH_EAST():
            {
                noSlip<VelocitySet>::apply<normalVectorBase::SOUTH_EAST()>(moments, incomings);
                return;
            }
            case normalVectorBase::SOUTH_BACK():
            {
                noSlip<VelocitySet>::apply<normalVectorBase::SOUTH_BACK()>(moments, incomings);
                return;
            }
            case normalVectorBase::SOUTH_FRONT():
            {
                noSlip<VelocitySet>::apply<normalVectorBase::SOUTH_FRONT()>(moments, incomings);
                return;
            }
            case normalVectorBase::EAST_BACK():
            {
                noSlip<VelocitySet>::apply<normalVectorBase::EAST_BACK()>(moments, incomings);
                return;
            }
            case normalVectorBase::EAST_FRONT():
            {
                noSlip<VelocitySet>::apply<normalVectorBase::EAST_FRONT()>(moments, incomings);
                return;
            }
            case normalVectorBase::WEST_BACK():
            {
                noSlip<VelocitySet>::apply<normalVectorBase::WEST_BACK()>(moments, incomings);
                return;
            }
            case normalVectorBase::WEST_FRONT():
            {
                noSlip<VelocitySet>::apply<normalVectorBase::WEST_FRONT()>(moments, incomings);
                return;
            }

            // Corner no-slip boundaries
            case normalVectorBase::SOUTH_WEST_FRONT():
            {
                noSlip<VelocitySet>::apply<normalVectorBase::SOUTH_WEST_FRONT()>(moments, incomings);
                return;
            }
            case normalVectorBase::SOUTH_EAST_FRONT():
            {
                noSlip<VelocitySet>::apply<normalVectorBase::SOUTH_EAST_FRONT()>(moments, incomings);
                return;
            }
            case normalVectorBase::SOUTH_WEST_BACK():
            {
                noSlip<VelocitySet>::apply<normalVectorBase::SOUTH_WEST_BACK()>(moments, incomings);
                return;
            }
            case normalVectorBase::SOUTH_EAST_BACK():
            {
                noSlip<VelocitySet>::apply<normalVectorBase::SOUTH_EAST_BACK()>(moments, incomings);
                return;
            }
            }

            return;
        }
    };
}

#endif