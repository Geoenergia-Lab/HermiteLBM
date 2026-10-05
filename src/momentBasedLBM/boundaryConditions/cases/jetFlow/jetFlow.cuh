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

#ifndef __MBLBM_jetFlow_CUH
#define __MBLBM_jetFlow_CUH

namespace LBM
{
    /**
     * @class jetFlow
     *
     * @brief Applies boundary conditions for turbulent jet simulations using moment representation
     **/
    class jetFlow : public boundaryConditionType<true, true, false>
    {
    public:
        /**
         * @brief Switch determining whether or not the boundary condition actually applies a condition
         **/
        __device__ __host__ [[nodiscard]] static inline consteval bool appliesCondition() noexcept { return true; }

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
            // Update the shared buffer with the refreshed moments
            device::constexpr_for<0, NUMBER_MOMENTS()>(
                [&](const auto moment)
                {
                    const device::label_t ID = tid * label_constant<NUMBER_MOMENTS()>() + label_constant<moment>();
                    sharedBuffer[ID] = moments[moment];
                });

            block::sync();

            if (boundaryNormal.isBoundary())
            {
                calculate_moments<VelocitySet>(moments, boundaryNormal, sharedBuffer, Tx, point);
            }
        }

    private:
        template <const bool FixDensity>
        __device__ static inline constexpr void smemNeumann(momentsArray &moments, const device::label_t tidOutlet, const blockSharedBuffer &sharedBuffer) noexcept
        {
            if constexpr (FixDensity)
            {
                moments[0] = rho0();
                device::constexpr_for<1, NUMBER_MOMENTS()>(
                    [&](const auto moment)
                    {
                        moments[m_i<moment>()] = sharedBuffer[tidOutlet * (NUMBER_MOMENTS()) + m_i<moment>()];
                    });
            }
            else
            {
                device::constexpr_for<0, NUMBER_MOMENTS()>(
                    [&](const auto moment)
                    {
                        moments[m_i<moment>()] = sharedBuffer[tidOutlet * (NUMBER_MOMENTS()) + m_i<moment>()];
                    });
            }
        }

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
#include "jetBoundaryCondition.cuh"
        }

        __device__ [[nodiscard]] static inline scalar_t center_x() noexcept
        {
            return static_cast<scalar_t>(0.5) * static_cast<scalar_t>(device::n<axis::X>() - 1);
        }

        __device__ [[nodiscard]] static inline scalar_t center_y() noexcept
        {
            return static_cast<scalar_t>(0.5) * static_cast<scalar_t>(device::n<axis::Y>() - 1);
        }

        static constexpr const scalar_t sigma = static_cast<scalar_t>(8);

        static constexpr const scalar_t pi = static_cast<scalar_t>(std::numbers::pi);

        __device__ [[nodiscard]] static inline scalar_t dudx(const scalar_t x, const scalar_t y) noexcept
        {
            return -(x * (std::exp(-pow<2>(device::L_char - static_cast<scalar_t>(2) * std::sqrt(pow<2>(x) + pow<2>(y))) / (static_cast<scalar_t>(4) * sigma)) - std::exp(-pow<2>(device::L_char + static_cast<scalar_t>(2) * std::sqrt(pow<2>(x) + pow<2>(y))) / (static_cast<scalar_t>(4) * sigma)))) / (std::sqrt(sigma) * std::sqrt(pi) * std::erf(device::L_char / (static_cast<scalar_t>(2) * std::sqrt(sigma))) * std::sqrt(pow<2>(x) + pow<2>(y)));
        }

        __device__ [[nodiscard]] static inline scalar_t dudy(const scalar_t x, const scalar_t y) noexcept
        {
            return -(y * (std::exp(-pow<2>(device::L_char - static_cast<scalar_t>(2) * std::sqrt(pow<2>(x) + pow<2>(y))) / (static_cast<scalar_t>(4) * sigma)) - std::exp(-pow<2>(device::L_char + static_cast<scalar_t>(2) * std::sqrt(pow<2>(x) + pow<2>(y))) / (static_cast<scalar_t>(4) * sigma)))) / (std::sqrt(sigma) * std::sqrt(pi) * std::erf(device::L_char / (static_cast<scalar_t>(2) * std::sqrt(sigma))) * std::sqrt(pow<2>(x) + pow<2>(y)));
        }

        __device__ [[nodiscard]] static inline scalar_t radius() noexcept
        {
            return static_cast<scalar_t>(0.5) * static_cast<scalar_t>(device::L_char);
        }

        __device__ [[nodiscard]] static inline scalar_t r2() noexcept
        {
            return radius() * radius();
        }
    };
}

#endif