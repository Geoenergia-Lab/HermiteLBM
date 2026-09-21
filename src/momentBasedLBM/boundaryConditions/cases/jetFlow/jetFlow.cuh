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
     *
     * This class implements the boundary condition treatment for the D3Q19 lattice
     * model in turbulent jet flow simulations. It handles static wall, inflow, and
     * outflow boundaries using moment-based boundary conditions derived from the
     * regularized LBM approach.
     **/
    class jetFlow : public boundaryConditionType<PERIODIC, PERIODIC, WALL>
    {
    public:
        /**
         * @brief Public method to calculate the post-streaming methods and update boundary conditions
         **/
        template <class VelocitySet, class SharedBuffer>
        __device__ static inline constexpr void calculate_moments(
            const thread::array<scalar_t, VelocitySet::Q()> &pop,
            momentsArray &moments,
            SharedBuffer &sharedBuffer,
            const thread::coordinate &Tx,
            const device::pointCoordinate &point,
            const device::label_t tid) noexcept
        {
            const NormalVector boundaryNormal(point);

            VelocitySet::template calculate_moments(moments, pop, boundaryNormal);

            // Update the shared buffer with the refreshed moments
            device::constexpr_for<1, 4>(
                [&](const auto moment)
                {
                    sharedBuffer[idxShared<moment>(tid)] = moments[moment];
                });

            block::sync();

            if (boundaryNormal.isBoundary())
            {
                calculate_moments<VelocitySet>(pop, moments, boundaryNormal, sharedBuffer, Tx, point);
            }
        }

    private:
        template <const device::label_t Moment>
        __device__ __host__ [[nodiscard]] static inline constexpr device::label_t idxShared(const device::label_t tid) noexcept
        {
            return tid * label_constant<NUMBER_MOMENTS() + 1>() + label_constant<Moment>();
        }

        /**
         * @brief Calculate moment variables at boundary nodes
         * @tparam VelocitySet The velocity set (D3Q19 or D3Q27)
         * @param[in] pop Population density array at current lattice node
         * @param[out] moments Moment array (rho, U, Pi)
         * @param[in] boundaryNormal Normal vector information at boundary node
         *
         * This method implements the moment-based boundary condition treatment
         * for the D3Q19 lattice model. Currently, it handles both the inflow
         * (jet) boundary located at the BACK face of the domain and the outflow
         * boundary located at the FRONT face.
         *
         * This method implements the moment-based boundary condition treatment for
         * the D3Q19 lattice model. It handles various boundary types including:
         * - Static wall boundaries (all velocity components zero)
         * - Moving lid boundaries (prescribed tangential velocity)
         * - Corner and edge cases with specialized treatment
         *
         * The method uses the regularized LBM approach to reconstruct boundary
         * moments from available population information, ensuring mass conservation
         * and appropriate stress conditions at boundaries.
         **/
        template <class VelocitySet, class SharedBuffer>
        __device__ static inline constexpr void calculate_moments(
            const thread::array<scalar_t, VelocitySet::Q()> &pop,
            momentsArray &moments,
            const NormalVector &boundaryNormal,
            const SharedBuffer &sharedBuffer,
            const thread::coordinate &Tx,
            const device::pointCoordinate &point) noexcept
        {
            const thread::array<scalar_t, 7> incomings(
                moments[m_i<0>()],
                moments[m_i<4>()],
                moments[m_i<5>()],
                moments[m_i<6>()],
                moments[m_i<7>()],
                moments[m_i<8>()],
                moments[m_i<9>()]);

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