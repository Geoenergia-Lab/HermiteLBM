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
    velocityBC.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_VELOCITYBC_CUH
#define __MBLBM_VELOCITYBC_CUH

namespace LBM
{
    struct constantVelocityBC
    {
        template <const nodeType_t BoundaryCase>
        __device__ __host__ [[nodiscard]] static inline constexpr void apply(momentsArray &moments) noexcept
        {
            if constexpr (BoundaryCase == normalVectorBase::SOUTH_WEST_BACK())
            {
                moments[q_i<1>()] = (device::U_West[0] + device::U_South[0] + device::U_Back[0]) / static_cast<scalar_t>(3);
                moments[q_i<2>()] = (device::U_West[1] + device::U_South[1] + device::U_Back[1]) / static_cast<scalar_t>(3);
                moments[q_i<3>()] = (device::U_West[2] + device::U_South[2] + device::U_Back[2]) / static_cast<scalar_t>(3);
            }
            if constexpr (BoundaryCase == normalVectorBase::SOUTH_WEST())
            {
                moments[q_i<1>()] = (device::U_West[0] + device::U_South[0]) / static_cast<scalar_t>(2);
                moments[q_i<2>()] = (device::U_West[1] + device::U_South[1]) / static_cast<scalar_t>(2);
                moments[q_i<3>()] = (device::U_West[2] + device::U_South[2]) / static_cast<scalar_t>(2);
            }
            if constexpr (BoundaryCase == normalVectorBase::SOUTH_WEST_FRONT())
            {
                moments[q_i<1>()] = (device::U_West[0] + device::U_South[0] + device::U_Front[0]) / static_cast<scalar_t>(3);
                moments[q_i<2>()] = (device::U_West[1] + device::U_South[1] + device::U_Front[1]) / static_cast<scalar_t>(3);
                moments[q_i<3>()] = (device::U_West[2] + device::U_South[2] + device::U_Front[2]) / static_cast<scalar_t>(3);
            }
            if constexpr (BoundaryCase == normalVectorBase::WEST_BACK())
            {
                moments[q_i<1>()] = (device::U_West[0] + device::U_Back[0]) / static_cast<scalar_t>(2);
                moments[q_i<2>()] = (device::U_West[1] + device::U_Back[1]) / static_cast<scalar_t>(2);
                moments[q_i<3>()] = (device::U_West[2] + device::U_Back[2]) / static_cast<scalar_t>(2);
            }
            if constexpr (BoundaryCase == normalVectorBase::WEST())
            {
                moments[q_i<1>()] = device::U_West[0];
                moments[q_i<2>()] = device::U_West[1];
                moments[q_i<3>()] = device::U_West[2];
            }
            if constexpr (BoundaryCase == normalVectorBase::WEST_FRONT())
            {
                moments[q_i<1>()] = (device::U_West[0] + device::U_Front[0]) / static_cast<scalar_t>(2);
                moments[q_i<2>()] = (device::U_West[1] + device::U_Front[1]) / static_cast<scalar_t>(2);
                moments[q_i<3>()] = (device::U_West[2] + device::U_Front[2]) / static_cast<scalar_t>(2);
            }
            if constexpr (BoundaryCase == normalVectorBase::NORTH_WEST_BACK())
            {
                moments[q_i<1>()] = (device::U_West[0] + device::U_North[0] + device::U_Back[0]) / static_cast<scalar_t>(3);
                moments[q_i<2>()] = (device::U_West[1] + device::U_North[1] + device::U_Back[1]) / static_cast<scalar_t>(3);
                moments[q_i<3>()] = (device::U_West[2] + device::U_North[2] + device::U_Back[2]) / static_cast<scalar_t>(3);
            }
            if constexpr (BoundaryCase == normalVectorBase::NORTH_WEST())
            {
                moments[q_i<1>()] = (device::U_West[0] + device::U_North[0]) / static_cast<scalar_t>(2);
                moments[q_i<2>()] = (device::U_West[1] + device::U_North[1]) / static_cast<scalar_t>(2);
                moments[q_i<3>()] = (device::U_West[2] + device::U_North[2]) / static_cast<scalar_t>(2);
            }
            if constexpr (BoundaryCase == normalVectorBase::NORTH_WEST_FRONT())
            {
                moments[q_i<1>()] = (device::U_West[0] + device::U_North[0] + device::U_Front[0]) / static_cast<scalar_t>(3);
                moments[q_i<2>()] = (device::U_West[1] + device::U_North[1] + device::U_Front[1]) / static_cast<scalar_t>(3);
                moments[q_i<3>()] = (device::U_West[2] + device::U_North[2] + device::U_Front[2]) / static_cast<scalar_t>(3);
            }
            if constexpr (BoundaryCase == normalVectorBase::SOUTH_BACK())
            {
                moments[q_i<1>()] = (device::U_South[0] + device::U_Back[0]) / static_cast<scalar_t>(2);
                moments[q_i<2>()] = (device::U_South[1] + device::U_Back[1]) / static_cast<scalar_t>(2);
                moments[q_i<3>()] = (device::U_South[2] + device::U_Back[2]) / static_cast<scalar_t>(2);
            }
            if constexpr (BoundaryCase == normalVectorBase::SOUTH())
            {
                moments[q_i<1>()] = device::U_South[0];
                moments[q_i<2>()] = device::U_South[1];
                moments[q_i<3>()] = device::U_South[2];
            }
            if constexpr (BoundaryCase == normalVectorBase::SOUTH_FRONT())
            {
                moments[q_i<1>()] = (device::U_South[0] + device::U_Front[0]) / static_cast<scalar_t>(2);
                moments[q_i<2>()] = (device::U_South[1] + device::U_Front[1]) / static_cast<scalar_t>(2);
                moments[q_i<3>()] = (device::U_South[2] + device::U_Front[2]) / static_cast<scalar_t>(2);
            }
            if constexpr (BoundaryCase == normalVectorBase::BACK())
            {
                moments[q_i<1>()] = device::U_Back[0];
                moments[q_i<2>()] = device::U_Back[1];
                moments[q_i<3>()] = device::U_Back[2];
            }
            if constexpr (BoundaryCase == normalVectorBase::FRONT())
            {
                moments[q_i<1>()] = device::U_Front[0];
                moments[q_i<2>()] = device::U_Front[1];
                moments[q_i<3>()] = device::U_Front[2];
            }
            if constexpr (BoundaryCase == normalVectorBase::NORTH_BACK())
            {
                moments[q_i<1>()] = (device::U_North[0] + device::U_Back[0]) / static_cast<scalar_t>(2);
                moments[q_i<2>()] = (device::U_North[1] + device::U_Back[1]) / static_cast<scalar_t>(2);
                moments[q_i<3>()] = (device::U_North[2] + device::U_Back[2]) / static_cast<scalar_t>(2);
            }
            if constexpr (BoundaryCase == normalVectorBase::NORTH())
            {
                moments[q_i<1>()] = device::U_North[0];
                moments[q_i<2>()] = device::U_North[1];
                moments[q_i<3>()] = device::U_North[2];
            }
            if constexpr (BoundaryCase == normalVectorBase::NORTH_FRONT())
            {
                moments[q_i<1>()] = (device::U_North[0] + device::U_Front[0]) / static_cast<scalar_t>(2);
                moments[q_i<2>()] = (device::U_North[1] + device::U_Front[1]) / static_cast<scalar_t>(2);
                moments[q_i<3>()] = (device::U_North[2] + device::U_Front[2]) / static_cast<scalar_t>(2);
            }
            if constexpr (BoundaryCase == normalVectorBase::SOUTH_EAST_BACK())
            {
                moments[q_i<1>()] = (device::U_East[0] + device::U_South[0] + device::U_Back[0]) / static_cast<scalar_t>(3);
                moments[q_i<2>()] = (device::U_East[1] + device::U_South[1] + device::U_Back[1]) / static_cast<scalar_t>(3);
                moments[q_i<3>()] = (device::U_East[2] + device::U_South[2] + device::U_Back[2]) / static_cast<scalar_t>(3);
            }
            if constexpr (BoundaryCase == normalVectorBase::SOUTH_EAST())
            {
                moments[q_i<1>()] = (device::U_East[0] + device::U_South[0]) / static_cast<scalar_t>(2);
                moments[q_i<2>()] = (device::U_East[1] + device::U_South[1]) / static_cast<scalar_t>(2);
                moments[q_i<3>()] = (device::U_East[2] + device::U_South[2]) / static_cast<scalar_t>(2);
            }
            if constexpr (BoundaryCase == normalVectorBase::SOUTH_EAST_FRONT())
            {
                moments[q_i<1>()] = (device::U_East[0] + device::U_South[0] + device::U_Front[0]) / static_cast<scalar_t>(3);
                moments[q_i<2>()] = (device::U_East[1] + device::U_South[1] + device::U_Front[1]) / static_cast<scalar_t>(3);
                moments[q_i<3>()] = (device::U_East[2] + device::U_South[2] + device::U_Front[2]) / static_cast<scalar_t>(3);
            }
            if constexpr (BoundaryCase == normalVectorBase::EAST_BACK())
            {
                moments[q_i<1>()] = (device::U_East[0] + device::U_Back[0]) / static_cast<scalar_t>(2);
                moments[q_i<2>()] = (device::U_East[1] + device::U_Back[1]) / static_cast<scalar_t>(2);
                moments[q_i<3>()] = (device::U_East[2] + device::U_Back[2]) / static_cast<scalar_t>(2);
            }
            if constexpr (BoundaryCase == normalVectorBase::EAST())
            {
                moments[q_i<1>()] = device::U_East[0];
                moments[q_i<2>()] = device::U_East[1];
                moments[q_i<3>()] = device::U_East[2];
            }
            if constexpr (BoundaryCase == normalVectorBase::EAST_FRONT())
            {
                moments[q_i<1>()] = (device::U_East[0] + device::U_Front[0]) / static_cast<scalar_t>(2);
                moments[q_i<2>()] = (device::U_East[1] + device::U_Front[1]) / static_cast<scalar_t>(2);
                moments[q_i<3>()] = (device::U_East[2] + device::U_Front[2]) / static_cast<scalar_t>(2);
            }
            if constexpr (BoundaryCase == normalVectorBase::NORTH_EAST_BACK())
            {
                moments[q_i<1>()] = (device::U_East[0] + device::U_North[0] + device::U_Back[0]) / static_cast<scalar_t>(3);
                moments[q_i<2>()] = (device::U_East[1] + device::U_North[1] + device::U_Back[1]) / static_cast<scalar_t>(3);
                moments[q_i<3>()] = (device::U_East[2] + device::U_North[2] + device::U_Back[2]) / static_cast<scalar_t>(3);
            }
            if constexpr (BoundaryCase == normalVectorBase::NORTH_EAST())
            {
                moments[q_i<1>()] = (device::U_East[0] + device::U_North[0]) / static_cast<scalar_t>(2);
                moments[q_i<2>()] = (device::U_East[1] + device::U_North[1]) / static_cast<scalar_t>(2);
                moments[q_i<3>()] = (device::U_East[2] + device::U_North[2]) / static_cast<scalar_t>(2);
            }
            if constexpr (BoundaryCase == normalVectorBase::NORTH_EAST_FRONT())
            {
                moments[q_i<1>()] = (device::U_East[0] + device::U_North[0] + device::U_Front[0]) / static_cast<scalar_t>(3);
                moments[q_i<2>()] = (device::U_East[1] + device::U_North[1] + device::U_Front[1]) / static_cast<scalar_t>(3);
                moments[q_i<3>()] = (device::U_East[2] + device::U_North[2] + device::U_Front[2]) / static_cast<scalar_t>(3);
            }
        }
    };

    struct noSlipVelocityBC
    {
        template <const nodeType_t BoundaryCase>
        __device__ __host__ [[nodiscard]] static inline constexpr void apply(momentsArray &moments) noexcept
        {
            moments[q_i<1>()] = static_cast<scalar_t>(0);
            moments[q_i<2>()] = static_cast<scalar_t>(0);
            moments[q_i<3>()] = static_cast<scalar_t>(0);
        }
    };
}

#endif