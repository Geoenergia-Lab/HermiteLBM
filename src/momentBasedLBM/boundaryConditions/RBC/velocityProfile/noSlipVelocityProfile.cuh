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
    Struct to apply a no-slip velocity profile to the velocity field

Namespace
    LBM

SourceFiles
    noSlipVelocityProfile.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_NOSLIPVELOCITYPROFILE_CUH
#define __MBLBM_NOSLIPVELOCITYPROFILE_CUH

namespace LBM
{
    /**
     * @brief Apply the no-slip boundary condition to the velocity field
     **/
    struct noSlipVelocityProfile
    {
        /**
         * @brief Get the boundary condition value
         **/
        __device__ [[nodiscard]] static inline consteval scalar_t value() noexcept
        {
            return static_cast<scalar_t>(0);
        }

        /**
         * @brief Apply the boundary condition to the velocity field
         * @param[out] moments Moment array (rho, U, Pi)
         **/
        __device__ [[nodiscard]] static inline constexpr void apply(momentsArray &moments) noexcept
        {
            moments[q_i<1>()] = value();
            moments[q_i<2>()] = value();
            moments[q_i<3>()] = value();
        }
    };
}

#endif