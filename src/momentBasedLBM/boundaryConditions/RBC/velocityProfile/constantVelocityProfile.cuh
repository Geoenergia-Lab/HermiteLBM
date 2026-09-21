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
    Struct to apply a constant velocity profile to the velocity field

Namespace
    LBM

SourceFiles
    constantVelocityProfile.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_CONSTANTVELOCITYPROFILE_CUH
#define __MBLBM_CONSTANTVELOCITYPROFILE_CUH

namespace LBM
{
    /**
     * @brief Apply a constant velocity profile to the velocity field
     **/
    struct constantVelocityProfile
    {
        /**
         * @brief Placeholder WIP
         **/
        template <const nodeType_t NodeType, const axis::type beta>
        __device__ [[nodiscard]] static inline constexpr scalar_t apply() noexcept
        {
            return 0;
        }

        /**
         * @brief Get the boundary condition value
         **/
        template <const axis::type alpha, const int coeff, const axis::type beta>
        __device__ [[nodiscard]] static inline constexpr scalar_t value() noexcept
        {
            if constexpr (alpha == axis::X)
            {
                if constexpr (coeff == -1)
                {
                    return device::U_West[static_cast<device::label_t>(beta)];
                }

                if constexpr (coeff == +1)
                {
                    return device::U_East[static_cast<device::label_t>(beta)];
                }
            }

            if constexpr (alpha == axis::Y)
            {
                if constexpr (coeff == -1)
                {
                    return device::U_South[static_cast<device::label_t>(beta)];
                }

                if constexpr (coeff == +1)
                {
                    return device::U_North[static_cast<device::label_t>(beta)];
                }
            }

            if constexpr (alpha == axis::Z)
            {
                if constexpr (coeff == -1)
                {
                    return device::U_Back[static_cast<device::label_t>(beta)];
                }

                if constexpr (coeff == +1)
                {
                    return device::U_Front[static_cast<device::label_t>(beta)];
                }
            }
        }
    };
}

#endif