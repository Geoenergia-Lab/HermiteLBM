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
    public:
        using Base = boundaryConditionType::This;

        /**
         * @brief Switch determining whether or not the boundary condition actually applies a condition
         **/
        deviceHostFunctionReturnType static inline consteval bool appliesCondition() noexcept { return true; }

        /**
         * @brief Public method to apply the boundary conditions
         * @tparam VelocitySet The velocity set (D3Q19 or D3Q27)
         * @param[out] moments Moment array (rho, U, Pi)
         * @param[in] boundaryNormal Normal vector information at boundary node
         * @param[in] sharedBuffer Shared memory array containing moment data
         * @param[in] Tx Three-dimensional thread coordinates
         * @param[in] point The global point coordinate
         **/
        template <class VelocitySet>
        __device__ static inline constexpr void calculate_moments(
            momentsArray &moments,
            const NormalVector &boundaryNormal,
            const blockSharedBuffer &sharedBuffer,
            const thread::coordinate &Tx,
            const device::pointCoordinate &point) noexcept
        {
            calculate_moments_impl<VelocitySet>(moments, boundaryNormal, sharedBuffer, Tx, point);
        }

    private:
        /**
         * @brief Calculate moment variables at boundary nodes
         * @tparam VelocitySet The velocity set (D3Q19 or D3Q27)
         * @param[out] moments Moment array (rho, U, Pi)
         * @param[in] boundaryNormal Normal vector information at boundary node
         * @param[in] sharedBuffer Shared memory array containing moment data
         * @param[in] Tx Three-dimensional thread coordinates
         * @param[in] point The global point coordinate
         **/
        template <class VelocitySet>
        __device__ static inline constexpr void calculate_moments_impl(
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
                Tx.value<axis::Z>() + boundaryNormal.isBack<device::label_t>() - boundaryNormal.isFront<device::label_t>()); // Check this if NaN

            // Now get density and the velocities from the interior
            const scalar_t Ux_interior = sharedBuffer[tidInterior * (NUMBER_MOMENTS()) + m_i<1>()];
            const scalar_t Uy_interior = sharedBuffer[tidInterior * (NUMBER_MOMENTS()) + m_i<2>()];
            const scalar_t Uz_interior = sharedBuffer[tidInterior * (NUMBER_MOMENTS()) + m_i<3>()];

            // Branchless boundary condition implementation
            const scalar_t not_x_boundary = static_cast<scalar_t>(!(boundaryNormal.isWest<bool>() || boundaryNormal.isEast<bool>()));
            const scalar_t not_y_boundary = static_cast<scalar_t>(!(boundaryNormal.isSouth<bool>() || boundaryNormal.isNorth<bool>()));
            const scalar_t not_z_boundary = static_cast<scalar_t>(!(boundaryNormal.isBack<bool>() || boundaryNormal.isFront<bool>()));

            const scalar_t x_normal = static_cast<scalar_t>(boundaryNormal.isWest<int8_t>() - boundaryNormal.isEast<int8_t>());
            const scalar_t y_normal = static_cast<scalar_t>(boundaryNormal.isSouth<int8_t>() - boundaryNormal.isNorth<int8_t>());
            const scalar_t z_normal = static_cast<scalar_t>(boundaryNormal.isBack<int8_t>() - boundaryNormal.isFront<int8_t>());

            // Fixed density wall (low Mach number)
            moments[0] = rho0();

            // Wall velocities
            const scalar_t nBoundaries = boundaryNormal.template countBoundaries<scalar_t>();
            const symmetricTensor boundarySwitches = {boundaryNormal.template isWest<scalar_t>(), boundaryNormal.template isEast<scalar_t>(), boundaryNormal.template isNorth<scalar_t>(), boundaryNormal.template isSouth<scalar_t>(), boundaryNormal.template isBack<scalar_t>(), boundaryNormal.template isFront<scalar_t>()};
            moments[axis::index<axis::X>()] = U<axis::X>(boundarySwitches, nBoundaries);
            moments[axis::index<axis::Y>()] = U<axis::Y>(boundarySwitches, nBoundaries);
            moments[axis::index<axis::Z>()] = U<axis::Z>(boundarySwitches, nBoundaries);

            // The x derivative is positive if West wall. Subtract if East.
            const scalar_t dUxdx = (Ux_interior - moments[axis::index<axis::X>()]) * x_normal * not_y_boundary * not_z_boundary;
            const scalar_t dUydx = (Uy_interior - moments[axis::index<axis::Y>()]) * x_normal * not_y_boundary * not_z_boundary;
            const scalar_t dUzdx = (Uz_interior - moments[axis::index<axis::Z>()]) * x_normal * not_y_boundary * not_z_boundary;

            const scalar_t dUxdy = (Ux_interior - moments[axis::index<axis::X>()]) * y_normal * not_x_boundary * not_z_boundary;
            const scalar_t dUydy = (Uy_interior - moments[axis::index<axis::Y>()]) * y_normal * not_x_boundary * not_z_boundary;
            const scalar_t dUzdy = (Uz_interior - moments[axis::index<axis::Z>()]) * y_normal * not_x_boundary * not_z_boundary;

            const scalar_t dUxdz = (Ux_interior - moments[axis::index<axis::X>()]) * z_normal * not_x_boundary * not_y_boundary;
            const scalar_t dUydz = (Uy_interior - moments[axis::index<axis::Y>()]) * z_normal * not_x_boundary * not_y_boundary;
            const scalar_t dUzdz = (Uz_interior - moments[axis::index<axis::Z>()]) * z_normal * not_x_boundary * not_y_boundary;

            // No-slip wall (velocity gradient)
            moments[axis::index<axis::X, axis::X>()] = (moments[axis::index<axis::X>()] * moments[axis::index<axis::X>()]) - (device::tau * velocitySetBase::cs2<scalar_t, 2>() * dUxdx);
            moments[axis::index<axis::X, axis::Y>()] = (moments[axis::index<axis::X>()] * moments[axis::index<axis::Y>()]) - (device::tau * velocitySetBase::cs2<scalar_t>() * (dUxdy + dUydx));
            moments[axis::index<axis::X, axis::Z>()] = (moments[axis::index<axis::X>()] * moments[axis::index<axis::Z>()]) - (device::tau * velocitySetBase::cs2<scalar_t>() * (dUxdz + dUzdx));
            moments[axis::index<axis::Y, axis::Y>()] = (moments[axis::index<axis::Y>()] * moments[axis::index<axis::Y>()]) - (device::tau * velocitySetBase::cs2<scalar_t, 2>() * dUydy);
            moments[axis::index<axis::Y, axis::Z>()] = (moments[axis::index<axis::Y>()] * moments[axis::index<axis::Z>()]) - (device::tau * velocitySetBase::cs2<scalar_t>() * (dUydz + dUzdy));
            moments[axis::index<axis::Z, axis::Z>()] = (moments[axis::index<axis::Z>()] * moments[axis::index<axis::Z>()]) - (device::tau * velocitySetBase::cs2<scalar_t, 2>() * dUzdz);

            return;
        }

        /**
         * @brief Branchless computation of the velocity component based on the boundary
         * @tparam alpha The axis direction (X, Y or Z)
         * @param[in] boundarySwitches Switches indicating active boundary conditions
         * @param[in] n_boundaries Number of active boundaries
         * @return Velocity component value
         **/
        template <const axis::type alpha>
        __device__ static inline constexpr scalar_t U(const symmetricTensor &boundarySwitches, const scalar_t n_boundaries) noexcept
        {
            axis::assertions::validate<alpha, axis::NOT_NULL>();

            // Calculate the boundary velocity value
            return ((boundarySwitches[0] * device::U_West[alpha]) +
                    (boundarySwitches[1] * device::U_East[alpha]) +
                    (boundarySwitches[2] * device::U_North[alpha]) +
                    (boundarySwitches[3] * device::U_South[alpha]) +
                    (boundarySwitches[4] * device::U_Back[alpha]) +
                    (boundarySwitches[5] * device::U_Front[alpha])) /
                   n_boundaries;
        }
    };
}

#endif