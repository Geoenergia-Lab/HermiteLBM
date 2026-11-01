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
    jetFlow.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_JETFLOW_CUH
#define __MBLBM_JETFLOW_CUH

namespace LBM
{
    /**
     * @class jetFlow
     *
     * @brief Applies boundary conditions for turbulent jet simulations using moment representation
     **/
    class jetFlow : public boundaryConditionType<false, false, false>
    {
    public:
        using Base = boundaryConditionType::This;

        /**
         * @brief Switch determining whether or not the boundary condition actually applies a condition
         **/
        deviceHostFunctionReturnType static inline consteval bool appliesCondition() noexcept
        {
            return true;
        }

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
            moments[0] = rho0();

            if (Tx.value<axis::Z>() == static_cast<device::label_t>(0))
            {
                const scalar_t x = point.value<axis::X, scalar_t>() - center<axis::X>();
                const scalar_t y = point.value<axis::Y, scalar_t>() - center<axis::Y>();
                const scalar_t r = math::sqrt(math::pow<2>(x) + math::pow<2>(y));

                const scalar_t A = (math::erf((device::L_char - static_cast<scalar_t>(2) * r) / (static_cast<scalar_t>(2) * math::sqrt(device::sigma))) + math::erf((device::L_char + static_cast<scalar_t>(2) * r) / (static_cast<scalar_t>(2) * math::sqrt(device::sigma)))) / (static_cast<scalar_t>(2) * math::erf(device::L_char / (static_cast<scalar_t>(2) * math::sqrt(device::sigma))));

                const scalar_t U_z = A * device::U_Back[2];

                moments[1] = static_cast<scalar_t>(0);
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = U_z;
            }
            else
            {
                const device::label_t tidInterior = block::idx(
                    Tx.value<axis::X>() + boundaryNormal.isWest<device::label_t>() - boundaryNormal.isEast<device::label_t>(),
                    Tx.value<axis::Y>() + boundaryNormal.isSouth<device::label_t>() - boundaryNormal.isNorth<device::label_t>(),
                    Tx.value<axis::Z>() + boundaryNormal.isBack<device::label_t>() - boundaryNormal.isFront<device::label_t>());

                // Now get density and the velocities from the interior
                const scalar_t Ux_interior = sharedBuffer[tidInterior * (NUMBER_MOMENTS()) + m_i<1>()];
                const scalar_t Uy_interior = sharedBuffer[tidInterior * (NUMBER_MOMENTS()) + m_i<2>()];
                const scalar_t Uz_interior = sharedBuffer[tidInterior * (NUMBER_MOMENTS()) + m_i<3>()];

                moments[1] = Ux_interior;
                moments[2] = Uy_interior;
                moments[3] = Uz_interior;
            }

            Base::equilibrium_moment(moments);
        }

        template <const axis::type alpha>
        deviceFunctionReturnType static inline constexpr scalar_t center() noexcept
        {
            return static_cast<scalar_t>(0.5) * static_cast<scalar_t>(device::latticeMesh::nPoints[alpha] - static_cast<device::label_t>(1));
        }

        deviceFunctionReturnType static inline constexpr scalar_t dudx(const scalar_t x, const scalar_t y) noexcept
        {
            return -(x * (math::exp(-math::pow<2>(device::L_char - static_cast<scalar_t>(2) * math::sqrt(math::pow<2>(x) + math::pow<2>(y))) / (static_cast<scalar_t>(4) * device::sigma)) - math::exp(-math::pow<2>(device::L_char + static_cast<scalar_t>(2) * math::sqrt(math::pow<2>(x) + math::pow<2>(y))) / (static_cast<scalar_t>(4) * device::sigma)))) / (math::sqrt(device::sigma) * math::constants::sqrtpi<scalar_t>() * math::erf(device::L_char / (static_cast<scalar_t>(2) * math::sqrt(device::sigma))) * math::sqrt(math::pow<2>(x) + math::pow<2>(y)));
        }

        deviceFunctionReturnType static inline constexpr scalar_t dudy(const scalar_t x, const scalar_t y) noexcept
        {
            return -(y * (math::exp(-math::pow<2>(device::L_char - static_cast<scalar_t>(2) * math::sqrt(math::pow<2>(x) + math::pow<2>(y))) / (static_cast<scalar_t>(4) * device::sigma)) - math::exp(-math::pow<2>(device::L_char + static_cast<scalar_t>(2) * math::sqrt(math::pow<2>(x) + math::pow<2>(y))) / (static_cast<scalar_t>(4) * device::sigma)))) / (math::sqrt(device::sigma) * math::constants::sqrtpi<scalar_t>() * math::erf(device::L_char / (static_cast<scalar_t>(2) * math::sqrt(device::sigma))) * math::sqrt(math::pow<2>(x) + math::pow<2>(y)));
        }
    };
}

#endif