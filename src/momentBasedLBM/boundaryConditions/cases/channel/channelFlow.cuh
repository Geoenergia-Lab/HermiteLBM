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
Authors: Nathan Duggins, Vinicius Czarnobay, Breno Gemelgo (Geoenergia Lab, UDESC)

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
    A class applying boundary conditions to the turbulent jet case

Namespace
    LBM

SourceFiles
    channelFlow.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_CHANNELFLOW_CUH
#define __MBLBM_CHANNELFLOW_CUH

namespace LBM
{
    /**
     * @class jetFlow
     *
     * @brief Applies boundary conditions for turbulent jet simulations using moment representation
     **/
    class channelFlow : public boundaryConditionType<false, false, true>
    {
    public:
        using Base = boundaryConditionType<false, false, true>;

        /**
         * @brief Switch determining whether or not the boundary condition actually applies a condition
         **/
        deviceHostFunctionReturnType static inline consteval bool appliesCondition() noexcept { return true; }

        /**
         * @brief Public method to calculate the post-streaming methods and update boundary conditions
         **/
        template <class VelocitySet>
        __device__ static inline constexpr void calculate_moments(
            momentsArray &moments,
            blockSharedBuffer &sharedBuffer,
            const thread::coordinate &Tx,
            const device::pointCoordinate &point,
            const device::label_t tid,
            const NormalVector &boundaryNormal) noexcept
        {
            if (boundaryNormal.isBoundary())
            {
                calculate_moments<VelocitySet>(moments, boundaryNormal, sharedBuffer, Tx, point);
            }
        }

    private:
        /**
         * @brief Calculate moment variables at boundary nodes
         * @tparam VelocitySet The velocity set (D3Q19 or D3Q27)
         * @param[in] pop Population density array at current lattice node
         * @param[out] moments Moment array (rho, U, Pi)
         * @param[in] boundaryNormal Normal vector information at boundary node
         **/
        template <class VelocitySet>
        __device__ static inline constexpr void calculate_moments(
            momentsArray &moments,
            const NormalVector &boundaryNormal,
            const blockSharedBuffer &sharedBuffer,
            const thread::coordinate &Tx,
            const device::pointCoordinate &point) noexcept
        {
            // Get the ID in the shared memory buffer of the interior point
            const device::label_t tidInterior = block::idx(
                Tx.value<axis::X>() + boundaryNormal.isWest<device::label_t>() - boundaryNormal.isEast<device::label_t>(),
                Tx.value<axis::Y>() + boundaryNormal.isSouth<device::label_t>() - boundaryNormal.isNorth<device::label_t>(),
                Tx.value<axis::Z>());

            // Now get density and the velocities from the interior
            const scalar_t rho_interior = sharedBuffer[tidInterior * (NUMBER_MOMENTS()) + m_i<0>()];
            const scalar_t Ux_interior = sharedBuffer[tidInterior * (NUMBER_MOMENTS()) + m_i<1>()];
            const scalar_t Uy_interior = sharedBuffer[tidInterior * (NUMBER_MOMENTS()) + m_i<2>()];
            const scalar_t Uz_interior = sharedBuffer[tidInterior * (NUMBER_MOMENTS()) + m_i<3>()];

            // Branchless boundary condition implementation
            const scalar_t not_x_boundary = static_cast<scalar_t>(!(boundaryNormal.isWest<bool>() || boundaryNormal.isEast<bool>()));
            const scalar_t not_y_boundary = static_cast<scalar_t>(!(boundaryNormal.isSouth<bool>() || boundaryNormal.isNorth<bool>()));

            const scalar_t x_normal = static_cast<scalar_t>(boundaryNormal.isWest<int8_t>() - boundaryNormal.isEast<int8_t>());
            const scalar_t y_normal = static_cast<scalar_t>(boundaryNormal.isSouth<int8_t>() - boundaryNormal.isNorth<int8_t>());

            // The x derivative is positive if West wall. Subtract if East.
            const scalar_t dUxdx = Ux_interior * x_normal * not_y_boundary;
            const scalar_t dUydx = Uy_interior * x_normal * not_y_boundary;
            const scalar_t dUzdx = Uz_interior * x_normal * not_y_boundary;

            const scalar_t dUxdy = Ux_interior * y_normal * not_x_boundary;
            const scalar_t dUydy = Uy_interior * y_normal * not_x_boundary;
            const scalar_t dUzdy = Uz_interior * y_normal * not_x_boundary;

            // Adiabatic wall
            moments[0] = rho_interior;

            // No-slip wall
            moments[1] = static_cast<scalar_t>(0);
            moments[2] = static_cast<scalar_t>(0);
            moments[3] = static_cast<scalar_t>(0);

            // No-slip wall (velocity gradient)
            moments[axis::index<axis::X, axis::X>()] = -static_cast<scalar_t>(2) * device::tau * velocitySetBase::cs2<scalar_t>() * dUxdx;
            moments[axis::index<axis::X, axis::Y>()] = -device::tau * velocitySetBase::cs2<scalar_t>() * (dUxdy + dUydx);
            moments[axis::index<axis::X, axis::Z>()] = -device::tau * velocitySetBase::cs2<scalar_t>() * (dUzdx);
            moments[axis::index<axis::Y, axis::Y>()] = -static_cast<scalar_t>(2) * device::tau * velocitySetBase::cs2<scalar_t>() * dUydy;
            moments[axis::index<axis::Y, axis::Z>()] = -device::tau * velocitySetBase::cs2<scalar_t>() * (dUzdy);
            moments[axis::index<axis::Z, axis::Z>()] = static_cast<scalar_t>(0);

            return;
        }
    };
}

#endif