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
    Definition of second order collision

Namespace
    LBM

SourceFiles
    BGK.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_COLLISION_BGK_CUH
#define __MBLBM_COLLISION_BGK_CUH

namespace LBM
{
    /**
     * @class BGK
     * @brief Implements the BGK collision operator for LBM simulations
     *
     * This class provides a specialized collision operator that handles
     * second-order moment updates in the Lattice Boltzmann Method. It assumes
     * zero force terms and updates both diagonal and off-diagonal moments
     * using relaxation parameters and velocity components.
     *
     * The collision operation follows the standard BGK approximation with
     * specialized treatment for second-order moments in the moment space.
     **/
    template <class Force>
    class BGK
    {
    public:
        /**
         * @brief Perform second-order collision and forcing operation on moments
         * @param[out] moments Moment array (rho, U, Pi)
         **/
        __device__ static inline void collide(momentsArray &moments) noexcept
        {
            if constexpr (Force::appliesForce())
            {
                const scalar_t u_x = moments[1];
                const scalar_t u_y = moments[2];
                const scalar_t u_z = moments[3];

                const scalar_t A = (static_cast<scalar_t>(3) * (device::omega - static_cast<scalar_t>(2))) / (static_cast<scalar_t>(2) * moments[0]);

                moments[1] = u_x + (A * Force::template apply<axis::X>());
                moments[2] = u_y + (A * Force::template apply<axis::Y>());
                moments[3] = u_z + (A * Force::template apply<axis::Z>());

                moments[4] = (static_cast<scalar_t>(0.5) * device::omega * math::pow<2>(u_x)) - ((device::omega - static_cast<scalar_t>(1)) * moments[4]) - (A * Force::template apply<axis::X>() * u_x);
                moments[5] = (device::omega * u_x * u_y) - ((device::omega - static_cast<scalar_t>(1)) * moments[5]) - (A * ((Force::template apply<axis::X>() * u_y) + (Force::template apply<axis::Y>() * u_x)));
                moments[6] = (device::omega * u_x * u_z) - ((device::omega - static_cast<scalar_t>(1)) * moments[6]) - (A * ((Force::template apply<axis::X>() * u_z) + (Force::template apply<axis::Z>() * u_x)));
                moments[7] = (static_cast<scalar_t>(0.5) * device::omega * math::pow<2>(u_y)) - ((device::omega - static_cast<scalar_t>(1)) * moments[7]) - (A * Force::template apply<axis::Y>() * u_y);
                moments[8] = (device::omega * u_y * u_z) - ((device::omega - static_cast<scalar_t>(1)) * moments[8]) - (A * ((Force::template apply<axis::Y>() * u_z) + (Force::template apply<axis::Z>() * u_y)));
                moments[9] = (static_cast<scalar_t>(0.5) * device::omega * math::pow<2>(u_z)) - ((device::omega - static_cast<scalar_t>(1)) * moments[9]) - (A * Force::template apply<axis::Z>() * u_z);
            }
            else
            {
                apply<axis::X, axis::X>(moments);
                apply<axis::X, axis::Y>(moments);
                apply<axis::X, axis::Z>(moments);
                apply<axis::Y, axis::Y>(moments);
                apply<axis::Y, axis::Z>(moments);
                apply<axis::Z, axis::Z>(moments);
            }

            // if constexpr (Force::appliesForce())
            // {
            //     apply<axis::X>(moments);
            //     apply<axis::Y>(moments);
            //     apply<axis::Z>(moments);
            // }

            // apply<axis::X, axis::X>(moments);
            // apply<axis::X, axis::Y>(moments);
            // apply<axis::X, axis::Z>(moments);
            // apply<axis::Y, axis::Y>(moments);
            // apply<axis::Y, axis::Z>(moments);
            // apply<axis::Z, axis::Z>(moments);
        }

        __host__ static inline void print() noexcept
        {
            IO::printBlock(
                std::cout, "collision",
                "{", "};",
                "type", "BGK",
                "forcing", Force::name());
            std::cout << std::endl;
        }

    private:
        /**
         * @brief Helper for moment-based collision and optional forces
         * @param[in] moments Moment array (rho, U, Pi)
         * @return Updated moment after collision
         **/
        template <const axis::type alpha, const axis::type beta>
        deviceFunctionReturnType static inline constexpr void apply(momentsArray &moments) noexcept
        {
            if constexpr (Force::appliesForce())
            {
                moments[axis::index<alpha, beta>()] = collide<alpha, beta>(moments) + Force::template apply<alpha, beta>(moments);
            }
            else
            {
                moments[axis::index<alpha, beta>()] = collide<alpha, beta>(moments);
            }
        }

        template <const axis::type alpha>
        __device__ static inline constexpr void apply(momentsArray &moments) noexcept
        {
            moments[axis::index<alpha, axis::NO_DIRECTION>()] =
                moments[axis::index<alpha, axis::NO_DIRECTION>()] +
                ((static_cast<scalar_t>(2) * velocitySetBase::scale_i<scalar_t>() / moments[0]) * Force::template apply<alpha>());
        }

        /**
         * @brief Collision helper for pure moment-based collision
         * @param[in] moments Moment array (rho, U, Pi)
         * @return Updated moment after collision
         **/
        template <const axis::type alpha, const axis::type beta>
        deviceFunctionReturnType static inline constexpr scalar_t collide(const momentsArray &moments) noexcept
        {
            axis::assertions::validate<alpha, axis::NOT_NULL>();
            axis::assertions::validate<beta, axis::NOT_NULL>();
            if constexpr (alpha == beta)
            {
                return ((static_cast<scalar_t>(0.5) * math::pow<2>(moments[axis::index<alpha>()])) - moments[axis::index<alpha, alpha>()]) * device::omega + moments[axis::index<alpha, alpha>()];
            }
            else
            {
                return (moments[axis::index<alpha>()] * moments[axis::index<beta>()] - moments[axis::index<alpha, beta>()]) * device::omega + moments[axis::index<alpha, beta>()];
            }
        }
    };
}

#endif