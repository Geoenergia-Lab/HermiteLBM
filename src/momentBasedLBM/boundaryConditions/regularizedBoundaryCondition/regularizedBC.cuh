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
    regularizedBC.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_REGULARIZEDBC_CUH
#define __MBLBM_REGULARIZEDBC_CUH

namespace LBM
{
    struct regularizedBC
    {
        using ValueType = thread::array<scalar_t, 7>;

        template <class VelocitySet, const nodeType_t BoundaryCase>
        __device__ __host__ [[nodiscard]] static inline constexpr const ValueType incomingMoments(const thread::array<scalar_t, VelocitySet::Q()> &pop) noexcept
        {
            if constexpr (VelocitySet::Q() == 27)
            {
                if constexpr (BoundaryCase == normalVectorBase::SOUTH_WEST_BACK())
                {
                    const scalar_t rho_I = pop[0] + pop[2] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[20];
                    return ValueType(
                        rho_I,
                        (pop[2] + pop[8] + pop[10] + pop[20]) / (rho_I)-VelocitySet::B(),
                        (pop[8] + pop[20]) / (rho_I),
                        (pop[10] + pop[20]) / (rho_I),
                        (pop[4] + pop[8] + pop[12] + pop[20]) / (rho_I)-VelocitySet::B(),
                        (pop[12] + pop[20]) / (rho_I),
                        (pop[6] + pop[10] + pop[12] + pop[20]) / (rho_I)-VelocitySet::B());
                }

                if constexpr (BoundaryCase == normalVectorBase::SOUTH_WEST())
                {
                    const scalar_t rho_I = pop[0] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[12] + pop[16] + pop[18] + pop[20] + pop[22];
                    return ValueType(
                        rho_I,
                        (pop[2] + pop[8] + pop[10] + pop[16] + pop[20] + pop[22]) / (rho_I)-VelocitySet::B(),
                        (pop[8] + pop[20] + pop[22]) / (rho_I),
                        (pop[10] - pop[16] + pop[20] - pop[22]) / (rho_I),
                        (pop[4] + pop[8] + pop[12] + pop[18] + pop[20] + pop[22]) / (rho_I)-VelocitySet::B(),
                        (pop[12] - pop[18] + pop[20] - pop[22]) / (rho_I),
                        VelocitySet::A() - (pop[0] + pop[2] + pop[4] + pop[8]) / (rho_I));
                }

                if constexpr (BoundaryCase == normalVectorBase::SOUTH_WEST_FRONT())
                {
                    const scalar_t rho_I = pop[0] + pop[2] + pop[4] + pop[5] + pop[8] + pop[16] + pop[18] + pop[22];
                    return ValueType(
                        rho_I,
                        (pop[2] + pop[8] + pop[16] + pop[22]) / (rho_I)-VelocitySet::B(),
                        (pop[8] + pop[22]) / (rho_I),
                        -(pop[16] + pop[22]) / (rho_I),
                        (pop[4] + pop[8] + pop[18] + pop[22]) / (rho_I)-VelocitySet::B(),
                        -(pop[18] + pop[22]) / (rho_I),
                        (pop[5] + pop[16] + pop[18] + pop[22]) / (rho_I)-VelocitySet::B());
                }

                if constexpr (BoundaryCase == normalVectorBase::WEST_BACK())
                {
                    const scalar_t rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[14] + pop[17] + pop[20] + pop[24];
                    return ValueType(
                        rho_I,
                        (pop[2] + pop[8] + pop[10] + pop[14] + pop[20] + pop[24]) / (rho_I)-VelocitySet::B(),
                        (pop[8] - pop[14] + pop[20] - pop[24]) / (rho_I),
                        (pop[10] + pop[20] + pop[24]) / (rho_I),
                        VelocitySet::A() - (pop[0] + pop[2] + pop[6] + pop[10]) / (rho_I),
                        (pop[12] - pop[17] + pop[20] - pop[24]) / (rho_I),
                        (pop[6] + pop[10] + pop[12] + pop[17] + pop[20] + pop[24]) / (rho_I)-VelocitySet::B());
                }

                if constexpr (BoundaryCase == normalVectorBase::WEST())
                {
                    const scalar_t rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[11] + pop[12] + pop[14] + pop[16] + pop[17] + pop[18] + pop[20] + pop[22] + pop[24] + pop[25];
                    return ValueType(
                        rho_I,
                        (pop[2] + pop[8] + pop[10] + pop[14] + pop[16] + pop[20] + pop[22] + pop[24] + pop[25]) / (rho_I)-VelocitySet::B(),
                        (pop[8] - pop[14] + pop[20] + pop[22] - pop[24] - pop[25]) / (rho_I),
                        (pop[10] - pop[16] + pop[20] - pop[22] + pop[24] - pop[25]) / (rho_I),
                        VelocitySet::A() - (pop[0] + pop[2] + pop[5] + pop[6] + pop[10] + pop[16]) / (rho_I),
                        (pop[11] + pop[12] - pop[17] - pop[18] + pop[20] - pop[22] - pop[24] + pop[25]) / (rho_I),
                        VelocitySet::A() - (pop[0] + pop[2] + pop[3] + pop[4] + pop[8] + pop[14]) / (rho_I));
                }

                if constexpr (BoundaryCase == normalVectorBase::WEST_FRONT())
                {
                    const scalar_t rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[8] + pop[11] + pop[14] + pop[16] + pop[18] + pop[22] + pop[25];
                    return ValueType(
                        rho_I,
                        (pop[2] + pop[8] + pop[14] + pop[16] + pop[22] + pop[25]) / (rho_I)-VelocitySet::B(),
                        (pop[8] - pop[14] + pop[22] - pop[25]) / (rho_I),
                        -(pop[16] + pop[22] + pop[25]) / (rho_I),
                        VelocitySet::A() - (pop[0] + pop[2] + pop[5] + pop[16]) / (rho_I),
                        (pop[11] - pop[18] - pop[22] + pop[25]) / (rho_I),
                        (pop[5] + pop[11] + pop[16] + pop[18] + pop[22] + pop[25]) / (rho_I)-VelocitySet::B());
                }

                if constexpr (BoundaryCase == normalVectorBase::NORTH_WEST_BACK())
                {
                    const scalar_t rho_I = pop[0] + pop[2] + pop[3] + pop[6] + pop[10] + pop[14] + pop[17] + pop[24];
                    return ValueType(
                        rho_I,
                        (pop[2] + pop[10] + pop[14] + pop[24]) / (rho_I)-VelocitySet::B(),
                        -(pop[14] + pop[24]) / (rho_I),
                        (pop[10] + pop[24]) / (rho_I),
                        (pop[3] + pop[14] + pop[17] + pop[24]) / (rho_I)-VelocitySet::B(),
                        -(pop[17] + pop[24]) / (rho_I),
                        (pop[6] + pop[10] + pop[17] + pop[24]) / (rho_I)-VelocitySet::B());
                }

                if constexpr (BoundaryCase == normalVectorBase::NORTH_WEST())
                {
                    const scalar_t rho_I = pop[0] + pop[2] + pop[3] + pop[5] + pop[6] + pop[10] + pop[11] + pop[14] + pop[16] + pop[17] + pop[24] + pop[25];
                    return ValueType(
                        rho_I,
                        (pop[2] + pop[10] + pop[14] + pop[16] + pop[24] + pop[25]) / (rho_I)-VelocitySet::B(),
                        -(pop[14] + pop[24] + pop[25]) / (rho_I),
                        (pop[10] - pop[16] + pop[24] - pop[25]) / (rho_I),
                        (pop[3] + pop[11] + pop[14] + pop[17] + pop[24] + pop[25]) / (rho_I)-VelocitySet::B(),
                        (pop[11] - pop[17] - pop[24] + pop[25]) / (rho_I),
                        VelocitySet::A() - (pop[0] + pop[2] + pop[3] + pop[14]) / (rho_I));
                }

                if constexpr (BoundaryCase == normalVectorBase::NORTH_WEST_FRONT())
                {
                    const scalar_t rho_I = pop[0] + pop[2] + pop[3] + pop[5] + pop[11] + pop[14] + pop[16] + pop[25];
                    return ValueType(
                        rho_I,
                        (pop[2] + pop[14] + pop[16] + pop[25]) / (rho_I)-VelocitySet::B(),
                        -(pop[14] + pop[25]) / (rho_I),
                        -(pop[16] + pop[25]) / (rho_I),
                        (pop[3] + pop[11] + pop[14] + pop[25]) / (rho_I)-VelocitySet::B(),
                        (pop[11] + pop[25]) / (rho_I),
                        (pop[5] + pop[11] + pop[16] + pop[25]) / (rho_I)-VelocitySet::B());
                }

                if constexpr (BoundaryCase == normalVectorBase::SOUTH_BACK())
                {
                    const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[13] + pop[15] + pop[20] + pop[26];
                    return ValueType(
                        rho_I,
                        VelocitySet::A() - (pop[0] + pop[4] + pop[6] + pop[12]) / (rho_I),
                        (pop[8] - pop[13] + pop[20] - pop[26]) / (rho_I),
                        (pop[10] - pop[15] + pop[20] - pop[26]) / (rho_I),
                        (pop[4] + pop[8] + pop[12] + pop[13] + pop[20] + pop[26]) / (rho_I)-VelocitySet::B(),
                        (pop[12] + pop[20] + pop[26]) / (rho_I),
                        (pop[6] + pop[10] + pop[12] + pop[15] + pop[20] + pop[26]) / (rho_I)-VelocitySet::B());
                }

                if constexpr (BoundaryCase == normalVectorBase::SOUTH())
                {
                    const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[15] + pop[16] + pop[18] + pop[20] + pop[22] + pop[23] + pop[26];
                    return ValueType(
                        rho_I,
                        VelocitySet::A() - (pop[0] + pop[4] + pop[5] + pop[6] + pop[12] + pop[18]) / (rho_I),
                        (pop[8] - pop[13] + pop[20] + pop[22] - pop[23] - pop[26]) / (rho_I),
                        (pop[9] + pop[10] - pop[15] - pop[16] + pop[20] - pop[22] + pop[23] - pop[26]) / (rho_I),
                        (pop[4] + pop[8] + pop[12] + pop[13] + pop[18] + pop[20] + pop[22] + pop[23] + pop[26]) / (rho_I)-VelocitySet::B(),
                        (pop[12] - pop[18] + pop[20] - pop[22] - pop[23] + pop[26]) / (rho_I),
                        VelocitySet::A() - (pop[0] + pop[1] + pop[2] + pop[4] + pop[8] + pop[13]) / (rho_I));
                }

                if constexpr (BoundaryCase == normalVectorBase::SOUTH_FRONT())
                {
                    const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[8] + pop[9] + pop[13] + pop[16] + pop[18] + pop[22] + pop[23];
                    return ValueType(
                        rho_I,
                        VelocitySet::A() - (pop[0] + pop[4] + pop[5] + pop[18]) / (rho_I),
                        (pop[8] - pop[13] + pop[22] - pop[23]) / (rho_I),
                        (pop[9] - pop[16] - pop[22] + pop[23]) / (rho_I),
                        (pop[4] + pop[8] + pop[13] + pop[18] + pop[22] + pop[23]) / (rho_I)-VelocitySet::B(),
                        -(pop[18] + pop[22] + pop[23]) / (rho_I),
                        (pop[5] + pop[9] + pop[16] + pop[18] + pop[22] + pop[23]) / (rho_I)-VelocitySet::B());
                }

                if constexpr (BoundaryCase == normalVectorBase::BACK())
                {
                    const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[6] + pop[7] + pop[8] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17] + pop[20] + pop[21] + pop[24] + pop[26];
                    return ValueType(
                        rho_I,
                        VelocitySet::A() - (pop[0] + pop[3] + pop[4] + pop[6] + pop[12] + pop[17]) / (rho_I),
                        (pop[7] + pop[8] - pop[13] - pop[14] + pop[20] + pop[21] - pop[24] - pop[26]) / (rho_I),
                        (pop[10] - pop[15] + pop[20] - pop[21] + pop[24] - pop[26]) / (rho_I),
                        VelocitySet::A() - (pop[0] + pop[1] + pop[2] + pop[6] + pop[10] + pop[15]) / (rho_I),
                        (pop[12] - pop[17] + pop[20] - pop[21] - pop[24] + pop[26]) / (rho_I),
                        (pop[6] + pop[10] + pop[12] + pop[15] + pop[17] + pop[20] + pop[21] + pop[24] + pop[26]) / (rho_I)-VelocitySet::B());
                }

                if constexpr (BoundaryCase == normalVectorBase::BACK())
                {
                    const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26];
                    return ValueType(
                        rho_I,
                        VelocitySet::A() - (pop[0] + pop[3] + pop[4] + pop[5] + pop[6] + pop[11] + pop[12] + pop[17] + pop[18]) / (rho_I),
                        (pop[7] + pop[8] - pop[13] - pop[14] + pop[19] + pop[20] + pop[21] + pop[22] - pop[23] - pop[24] - pop[25] - pop[26]) / (rho_I),
                        (pop[9] + pop[10] - pop[15] - pop[16] + pop[19] + pop[20] - pop[21] - pop[22] + pop[23] + pop[24] - pop[25] - pop[26]) / (rho_I),
                        VelocitySet::A() - (pop[0] + pop[1] + pop[2] + pop[5] + pop[6] + pop[9] + pop[10] + pop[15] + pop[16]) / (rho_I),
                        (pop[11] + pop[12] - pop[17] - pop[18] + pop[19] + pop[20] - pop[21] - pop[22] - pop[23] - pop[24] + pop[25] + pop[26]) / (rho_I),
                        VelocitySet::A() - (pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[7] + pop[8] + pop[13] + pop[14]) / (rho_I));
                }

                if constexpr (BoundaryCase == normalVectorBase::FRONT())
                {
                    const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[7] + pop[8] + pop[9] + pop[11] + pop[13] + pop[14] + pop[16] + pop[18] + pop[19] + pop[22] + pop[23] + pop[25];
                    return ValueType(
                        rho_I,
                        VelocitySet::A() - (pop[0] + pop[3] + pop[4] + pop[5] + pop[11] + pop[18]) / (rho_I),
                        (pop[7] + pop[8] - pop[13] - pop[14] + pop[19] + pop[22] - pop[23] - pop[25]) / (rho_I),
                        (pop[9] - pop[16] + pop[19] - pop[22] + pop[23] - pop[25]) / (rho_I),
                        VelocitySet::A() - (pop[0] + pop[1] + pop[2] + pop[5] + pop[9] + pop[16]) / (rho_I),
                        (pop[11] - pop[18] + pop[19] - pop[22] - pop[23] + pop[25]) / (rho_I),
                        (pop[5] + pop[9] + pop[11] + pop[16] + pop[18] + pop[19] + pop[22] + pop[23] + pop[25]) / (rho_I)-VelocitySet::B());
                }

                if constexpr (BoundaryCase == normalVectorBase::NORTH_BACK())
                {
                    const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[6] + pop[7] + pop[10] + pop[14] + pop[15] + pop[17] + pop[21] + pop[24];
                    return ValueType(
                        rho_I,
                        VelocitySet::A() - (pop[0] + pop[3] + pop[6] + pop[17]) / (rho_I),
                        (pop[7] - pop[14] + pop[21] - pop[24]) / (rho_I),
                        (pop[10] - pop[15] - pop[21] + pop[24]) / (rho_I),
                        (pop[3] + pop[7] + pop[14] + pop[17] + pop[21] + pop[24]) / (rho_I)-VelocitySet::B(),
                        -(pop[17] + pop[21] + pop[24]) / (rho_I),
                        (pop[6] + pop[10] + pop[15] + pop[17] + pop[21] + pop[24]) / (rho_I)-VelocitySet::B());
                }

                if constexpr (BoundaryCase == normalVectorBase::NORTH())
                {
                    const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[14] + pop[15] + pop[16] + pop[17] + pop[19] + pop[21] + pop[24] + pop[25];
                    return ValueType(
                        rho_I,
                        VelocitySet::A() - (pop[0] + pop[3] + pop[5] + pop[6] + pop[11] + pop[17]) / (rho_I),
                        (pop[7] - pop[14] + pop[19] + pop[21] - pop[24] - pop[25]) / (rho_I),
                        (pop[9] + pop[10] - pop[15] - pop[16] + pop[19] - pop[21] + pop[24] - pop[25]) / (rho_I),
                        (pop[3] + pop[7] + pop[11] + pop[14] + pop[17] + pop[19] + pop[21] + pop[24] + pop[25]) / (rho_I)-VelocitySet::B(),
                        (pop[11] - pop[17] + pop[19] - pop[21] - pop[24] + pop[25]) / (rho_I),
                        VelocitySet::A() - (pop[0] + pop[1] + pop[2] + pop[3] + pop[7] + pop[14]) / (rho_I));
                }

                if constexpr (BoundaryCase == normalVectorBase::NORTH_FRONT())
                {
                    const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[7] + pop[9] + pop[11] + pop[14] + pop[16] + pop[19] + pop[25];
                    return ValueType(
                        rho_I,
                        VelocitySet::A() - (pop[0] + pop[3] + pop[5] + pop[11]) / (rho_I),
                        (pop[7] - pop[14] + pop[19] - pop[25]) / (rho_I),
                        (pop[9] - pop[16] + pop[19] - pop[25]) / (rho_I),
                        (pop[3] + pop[7] + pop[11] + pop[14] + pop[19] + pop[25]) / (rho_I)-VelocitySet::B(),
                        (pop[11] + pop[19] + pop[25]) / (rho_I),
                        (pop[5] + pop[9] + pop[11] + pop[16] + pop[19] + pop[25]) / (rho_I)-VelocitySet::B());
                }

                if constexpr (BoundaryCase == normalVectorBase::SOUTH_EAST_BACK())
                {
                    const scalar_t rho_I = pop[0] + pop[1] + pop[4] + pop[6] + pop[12] + pop[13] + pop[15] + pop[26];
                    return ValueType(
                        rho_I,
                        (pop[1] + pop[13] + pop[15] + pop[26]) / (rho_I)-VelocitySet::B(),
                        -(pop[13] + pop[26]) / (rho_I),
                        -(pop[15] + pop[26]) / (rho_I),
                        (pop[4] + pop[12] + pop[13] + pop[26]) / (rho_I)-VelocitySet::B(),
                        (pop[12] + pop[26]) / (rho_I),
                        (pop[6] + pop[12] + pop[15] + pop[26]) / (rho_I)-VelocitySet::B());
                }

                if constexpr (BoundaryCase == normalVectorBase::SOUTH_EAST())
                {
                    const scalar_t rho_I = pop[0] + pop[1] + pop[4] + pop[5] + pop[6] + pop[9] + pop[12] + pop[13] + pop[15] + pop[18] + pop[23] + pop[26];
                    return ValueType(
                        rho_I,
                        (pop[1] + pop[9] + pop[13] + pop[15] + pop[23] + pop[26]) / (rho_I)-VelocitySet::B(),
                        -(pop[13] + pop[23] + pop[26]) / (rho_I),
                        (pop[9] - pop[15] + pop[23] - pop[26]) / (rho_I),
                        (pop[4] + pop[12] + pop[13] + pop[18] + pop[23] + pop[26]) / (rho_I)-VelocitySet::B(),
                        (pop[12] - pop[18] - pop[23] + pop[26]) / (rho_I),
                        VelocitySet::A() - (pop[0] + pop[1] + pop[4] + pop[13]) / (rho_I));
                }

                if constexpr (BoundaryCase == normalVectorBase::SOUTH_EAST_FRONT())
                {
                    const scalar_t rho_I = pop[0] + pop[1] + pop[4] + pop[5] + pop[9] + pop[13] + pop[18] + pop[23];
                    return ValueType(
                        rho_I,
                        (pop[1] + pop[9] + pop[13] + pop[23]) / (rho_I)-VelocitySet::B(),
                        -(pop[13] + pop[23]) / (rho_I),
                        (pop[9] + pop[23]) / (rho_I),
                        (pop[4] + pop[13] + pop[18] + pop[23]) / (rho_I)-VelocitySet::B(),
                        -(pop[18] + pop[23]) / (rho_I),
                        (pop[5] + pop[9] + pop[18] + pop[23]) / (rho_I)-VelocitySet::B());
                }

                if constexpr (BoundaryCase == normalVectorBase::EAST_BACK())
                {
                    const scalar_t rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[6] + pop[7] + pop[12] + pop[13] + pop[15] + pop[17] + pop[21] + pop[26];
                    return ValueType(
                        rho_I,
                        (pop[1] + pop[7] + pop[13] + pop[15] + pop[21] + pop[26]) / (rho_I)-VelocitySet::B(),
                        (pop[7] - pop[13] + pop[21] - pop[26]) / (rho_I),
                        -(pop[15] + pop[21] + pop[26]) / (rho_I),
                        VelocitySet::A() - (pop[0] + pop[1] + pop[6] + pop[15]) / (rho_I),
                        (pop[12] - pop[17] - pop[21] + pop[26]) / (rho_I),
                        (pop[6] + pop[12] + pop[15] + pop[17] + pop[21] + pop[26]) / (rho_I)-VelocitySet::B());
                }

                if constexpr (BoundaryCase == normalVectorBase::EAST())
                {
                    const scalar_t rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[12] + pop[13] + pop[15] + pop[17] + pop[18] + pop[19] + pop[21] + pop[23] + pop[26];
                    return ValueType(
                        rho_I,
                        (pop[1] + pop[7] + pop[9] + pop[13] + pop[15] + pop[19] + pop[21] + pop[23] + pop[26]) / (rho_I)-VelocitySet::B(),
                        (pop[7] - pop[13] + pop[19] + pop[21] - pop[23] - pop[26]) / (rho_I),
                        (pop[9] - pop[15] + pop[19] - pop[21] + pop[23] - pop[26]) / (rho_I),
                        VelocitySet::A() - (pop[0] + pop[1] + pop[5] + pop[6] + pop[9] + pop[15]) / (rho_I),
                        (pop[11] + pop[12] - pop[17] - pop[18] + pop[19] - pop[21] - pop[23] + pop[26]) / (rho_I),
                        VelocitySet::A() - (pop[0] + pop[1] + pop[3] + pop[4] + pop[7] + pop[13]) / (rho_I));
                }

                if constexpr (BoundaryCase == normalVectorBase::EAST_FRONT())
                {
                    const scalar_t rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[7] + pop[9] + pop[11] + pop[13] + pop[18] + pop[19] + pop[23];
                    return ValueType(
                        rho_I,
                        (pop[1] + pop[7] + pop[9] + pop[13] + pop[19] + pop[23]) / (rho_I)-VelocitySet::B(),
                        (pop[7] - pop[13] + pop[19] - pop[23]) / (rho_I),
                        (pop[9] + pop[19] + pop[23]) / (rho_I),
                        VelocitySet::A() - (pop[0] + pop[1] + pop[5] + pop[9]) / (rho_I),
                        (pop[11] - pop[18] + pop[19] - pop[23]) / (rho_I),
                        (pop[5] + pop[9] + pop[11] + pop[18] + pop[19] + pop[23]) / (rho_I)-VelocitySet::B());
                }

                if constexpr (BoundaryCase == normalVectorBase::NORTH_EAST_BACK())
                {
                    const scalar_t rho_I = pop[0] + pop[1] + pop[3] + pop[6] + pop[7] + pop[15] + pop[17] + pop[21];
                    return ValueType(
                        rho_I,
                        (pop[1] + pop[7] + pop[15] + pop[21]) / (rho_I)-VelocitySet::B(),
                        (pop[7] + pop[21]) / (rho_I),
                        -(pop[15] + pop[21]) / (rho_I),
                        (pop[3] + pop[7] + pop[17] + pop[21]) / (rho_I)-VelocitySet::B(),
                        -(pop[17] + pop[21]) / (rho_I),
                        (pop[6] + pop[15] + pop[17] + pop[21]) / (rho_I)-VelocitySet::B());
                }

                if constexpr (BoundaryCase == normalVectorBase::NORTH_EAST())
                {
                    const scalar_t rho_I = pop[0] + pop[1] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[15] + pop[17] + pop[19] + pop[21];
                    return ValueType(
                        rho_I,
                        (pop[1] + pop[7] + pop[9] + pop[15] + pop[19] + pop[21]) / (rho_I)-VelocitySet::B(),
                        (pop[7] + pop[19] + pop[21]) / (rho_I),
                        (pop[9] - pop[15] + pop[19] - pop[21]) / (rho_I),
                        (pop[3] + pop[7] + pop[11] + pop[17] + pop[19] + pop[21]) / (rho_I)-VelocitySet::B(),
                        (pop[11] - pop[17] + pop[19] - pop[21]) / (rho_I),
                        VelocitySet::A() - (pop[0] + pop[1] + pop[3] + pop[7]) / (rho_I));
                }

                if constexpr (BoundaryCase == normalVectorBase::NORTH_EAST_FRONT())
                {
                    const scalar_t rho_I = pop[0] + pop[1] + pop[3] + pop[5] + pop[7] + pop[9] + pop[11] + pop[19];
                    return ValueType(
                        rho_I,
                        (pop[1] + pop[7] + pop[9] + pop[19]) / (rho_I)-VelocitySet::B(),
                        (pop[7] + pop[19]) / (rho_I),
                        (pop[9] + pop[19]) / (rho_I),
                        (pop[3] + pop[7] + pop[11] + pop[19]) / (rho_I)-VelocitySet::B(),
                        (pop[11] + pop[19]) / (rho_I),
                        (pop[5] + pop[9] + pop[11] + pop[19]) / (rho_I)-VelocitySet::B());
                }
            }

            if constexpr (VelocitySet::Q() == 19)
            {
                if constexpr (BoundaryCase == normalVectorBase::SOUTH_WEST_BACK())
                {
                    const scalar_t rho_I = pop[0] + pop[2] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12];
                    return ValueType(
                        rho_I,
                        (pop[2] + pop[8] + pop[10]) / (rho_I)-VelocitySet::B(),
                        pop[8] / (rho_I),
                        pop[10] / (rho_I),
                        (pop[4] + pop[8] + pop[12]) / (rho_I)-VelocitySet::B(),
                        pop[12] / (rho_I),
                        (pop[6] + pop[10] + pop[12]) / (rho_I)-VelocitySet::B());
                }

                if constexpr (BoundaryCase == normalVectorBase::SOUTH_WEST())
                {
                    const scalar_t rho_I = pop[0] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[12] + pop[16] + pop[18];
                    return ValueType(
                        rho_I,
                        (pop[2] + pop[8] + pop[10] + pop[16]) / (rho_I)-VelocitySet::B(),
                        pop[8] / (rho_I),
                        (pop[10] - pop[16]) / (rho_I),
                        (pop[4] + pop[8] + pop[12] + pop[18]) / (rho_I)-VelocitySet::B(),
                        (pop[12] - pop[18]) / (rho_I),
                        VelocitySet::A() - (pop[0] + pop[2] + pop[4] + pop[8]) / (rho_I));
                }

                if constexpr (BoundaryCase == normalVectorBase::SOUTH_WEST_FRONT())
                {
                    const scalar_t rho_I = pop[0] + pop[2] + pop[4] + pop[5] + pop[8] + pop[16] + pop[18];
                    return ValueType(
                        rho_I,
                        (pop[2] + pop[8] + pop[16]) / (rho_I)-VelocitySet::B(),
                        pop[8] / (rho_I),
                        -pop[16] / (rho_I),
                        (pop[4] + pop[8] + pop[18]) / (rho_I)-VelocitySet::B(),
                        -pop[18] / (rho_I),
                        (pop[5] + pop[16] + pop[18]) / (rho_I)-VelocitySet::B());
                }

                if constexpr (BoundaryCase == normalVectorBase::WEST_BACK())
                {
                    const scalar_t rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[14] + pop[17];
                    return ValueType(
                        rho_I,
                        (pop[2] + pop[8] + pop[10] + pop[14]) / (rho_I)-VelocitySet::B(),
                        (pop[8] - pop[14]) / (rho_I),
                        pop[10] / (rho_I),
                        VelocitySet::A() - (pop[0] + pop[2] + pop[6] + pop[10]) / (rho_I),
                        (pop[12] - pop[17]) / (rho_I),
                        (pop[6] + pop[10] + pop[12] + pop[17]) / (rho_I)-VelocitySet::B());
                }

                if constexpr (BoundaryCase == normalVectorBase::WEST())
                {
                    const scalar_t rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[11] + pop[12] + pop[14] + pop[16] + pop[17] + pop[18];
                    return ValueType(
                        rho_I,
                        (pop[2] + pop[8] + pop[10] + pop[14] + pop[16]) / (rho_I)-VelocitySet::B(),
                        (pop[8] - pop[14]) / (rho_I),
                        (pop[10] - pop[16]) / (rho_I),
                        VelocitySet::A() - (pop[0] + pop[2] + pop[5] + pop[6] + pop[10] + pop[16]) / (rho_I),
                        (pop[11] + pop[12] - pop[17] - pop[18]) / (rho_I),
                        VelocitySet::A() - (pop[0] + pop[2] + pop[3] + pop[4] + pop[8] + pop[14]) / (rho_I));
                }

                if constexpr (BoundaryCase == normalVectorBase::WEST_FRONT())
                {
                    const scalar_t rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[8] + pop[11] + pop[14] + pop[16] + pop[18];
                    return ValueType(
                        rho_I,
                        (pop[2] + pop[8] + pop[14] + pop[16]) / (rho_I)-VelocitySet::B(),
                        (pop[8] - pop[14]) / (rho_I),
                        -pop[16] / (rho_I),
                        VelocitySet::A() - (pop[0] + pop[2] + pop[5] + pop[16]) / (rho_I),
                        (pop[11] - pop[18]) / (rho_I),
                        (pop[5] + pop[11] + pop[16] + pop[18]) / (rho_I)-VelocitySet::B());
                }

                if constexpr (BoundaryCase == normalVectorBase::NORTH_WEST_BACK())
                {
                    const scalar_t rho_I = pop[0] + pop[2] + pop[3] + pop[6] + pop[10] + pop[14] + pop[17];
                    return ValueType(
                        rho_I,
                        (pop[2] + pop[10] + pop[14]) / (rho_I)-VelocitySet::B(),
                        -pop[14] / (rho_I),
                        pop[10] / (rho_I),
                        (pop[3] + pop[14] + pop[17]) / (rho_I)-VelocitySet::B(),
                        -pop[17] / (rho_I),
                        (pop[6] + pop[10] + pop[17]) / (rho_I)-VelocitySet::B());
                }

                if constexpr (BoundaryCase == normalVectorBase::NORTH_WEST())
                {
                    const scalar_t rho_I = pop[0] + pop[2] + pop[3] + pop[5] + pop[6] + pop[10] + pop[11] + pop[14] + pop[16] + pop[17];
                    return ValueType(
                        rho_I,
                        (pop[2] + pop[10] + pop[14] + pop[16]) / (rho_I)-VelocitySet::B(),
                        -pop[14] / (rho_I),
                        (pop[10] - pop[16]) / (rho_I),
                        (pop[3] + pop[11] + pop[14] + pop[17]) / (rho_I)-VelocitySet::B(),
                        (pop[11] - pop[17]) / (rho_I),
                        VelocitySet::A() - (pop[0] + pop[2] + pop[3] + pop[14]) / (rho_I));
                }

                if constexpr (BoundaryCase == normalVectorBase::NORTH_WEST_FRONT())
                {
                    const scalar_t rho_I = pop[0] + pop[2] + pop[3] + pop[5] + pop[11] + pop[14] + pop[16];
                    return ValueType(
                        rho_I,
                        (pop[2] + pop[14] + pop[16]) / (rho_I)-VelocitySet::B(),
                        -pop[14] / (rho_I),
                        -pop[16] / (rho_I),
                        (pop[3] + pop[11] + pop[14]) / (rho_I)-VelocitySet::B(),
                        pop[11] / (rho_I),
                        (pop[5] + pop[11] + pop[16]) / (rho_I)-VelocitySet::B());
                }

                if constexpr (BoundaryCase == normalVectorBase::SOUTH_BACK())
                {
                    const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[13] + pop[15];
                    return ValueType(
                        rho_I,
                        VelocitySet::A() - (pop[0] + pop[4] + pop[6] + pop[12]) / (rho_I),
                        (pop[8] - pop[13]) / (rho_I),
                        (pop[10] - pop[15]) / (rho_I),
                        (pop[4] + pop[8] + pop[12] + pop[13]) / (rho_I)-VelocitySet::B(),
                        pop[12] / (rho_I),
                        (pop[6] + pop[10] + pop[12] + pop[15]) / (rho_I)-VelocitySet::B());
                }

                if constexpr (BoundaryCase == normalVectorBase::SOUTH())
                {
                    const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[15] + pop[16] + pop[18];
                    return ValueType(
                        rho_I,
                        VelocitySet::A() - (pop[0] + pop[4] + pop[5] + pop[6] + pop[12] + pop[18]) / (rho_I),
                        (pop[8] - pop[13]) / (rho_I),
                        (pop[9] + pop[10] - pop[15] - pop[16]) / (rho_I),
                        (pop[4] + pop[8] + pop[12] + pop[13] + pop[18]) / (rho_I)-VelocitySet::B(),
                        (pop[12] - pop[18]) / (rho_I),
                        VelocitySet::A() - (pop[0] + pop[1] + pop[2] + pop[4] + pop[8] + pop[13]) / (rho_I));
                }

                if constexpr (BoundaryCase == normalVectorBase::SOUTH_FRONT())
                {
                    const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[8] + pop[9] + pop[13] + pop[16] + pop[18];
                    return ValueType(
                        rho_I,
                        VelocitySet::A() - (pop[0] + pop[4] + pop[5] + pop[18]) / (rho_I),
                        (pop[8] - pop[13]) / (rho_I),
                        (pop[9] - pop[16]) / (rho_I),
                        (pop[4] + pop[8] + pop[13] + pop[18]) / (rho_I)-VelocitySet::B(),
                        -pop[18] / (rho_I),
                        (pop[5] + pop[9] + pop[16] + pop[18]) / (rho_I)-VelocitySet::B());
                }

                if constexpr (BoundaryCase == normalVectorBase::BACK())
                {
                    const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[6] + pop[7] + pop[8] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17];
                    return ValueType(
                        rho_I,
                        VelocitySet::A() - (pop[0] + pop[3] + pop[4] + pop[6] + pop[12] + pop[17]) / (rho_I),
                        (pop[7] + pop[8] - pop[13] - pop[14]) / (rho_I),
                        (pop[10] - pop[15]) / (rho_I),
                        VelocitySet::A() - (pop[0] + pop[1] + pop[2] + pop[6] + pop[10] + pop[15]) / (rho_I),
                        (pop[12] - pop[17]) / (rho_I),
                        (pop[6] + pop[10] + pop[12] + pop[15] + pop[17]) / (rho_I)-VelocitySet::B());
                }

                if constexpr (BoundaryCase == normalVectorBase::BACK())
                {
                    const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
                    return ValueType(
                        rho_I,
                        VelocitySet::A() - (pop[0] + pop[3] + pop[4] + pop[5] + pop[6] + pop[11] + pop[12] + pop[17] + pop[18]) / (rho_I),
                        (pop[7] + pop[8] - pop[13] - pop[14]) / (rho_I),
                        (pop[9] + pop[10] - pop[15] - pop[16]) / (rho_I),
                        VelocitySet::A() - (pop[0] + pop[1] + pop[2] + pop[5] + pop[6] + pop[9] + pop[10] + pop[15] + pop[16]) / (rho_I),
                        (pop[11] + pop[12] - pop[17] - pop[18]) / (rho_I),
                        VelocitySet::A() - (pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[7] + pop[8] + pop[13] + pop[14]) / (rho_I));
                }

                if constexpr (BoundaryCase == normalVectorBase::FRONT())
                {
                    const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[7] + pop[8] + pop[9] + pop[11] + pop[13] + pop[14] + pop[16] + pop[18];
                    return ValueType(
                        rho_I,
                        VelocitySet::A() - (pop[0] + pop[3] + pop[4] + pop[5] + pop[11] + pop[18]) / (rho_I),
                        (pop[7] + pop[8] - pop[13] - pop[14]) / (rho_I),
                        (pop[9] - pop[16]) / (rho_I),
                        VelocitySet::A() - (pop[0] + pop[1] + pop[2] + pop[5] + pop[9] + pop[16]) / (rho_I),
                        (pop[11] - pop[18]) / (rho_I),
                        (pop[5] + pop[9] + pop[11] + pop[16] + pop[18]) / (rho_I)-VelocitySet::B());
                }

                if constexpr (BoundaryCase == normalVectorBase::NORTH_BACK())
                {
                    const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[6] + pop[7] + pop[10] + pop[14] + pop[15] + pop[17];
                    return ValueType(
                        rho_I,
                        VelocitySet::A() - (pop[0] + pop[3] + pop[6] + pop[17]) / (rho_I),
                        (pop[7] - pop[14]) / (rho_I),
                        (pop[10] - pop[15]) / (rho_I),
                        (pop[3] + pop[7] + pop[14] + pop[17]) / (rho_I)-VelocitySet::B(),
                        -pop[17] / (rho_I),
                        (pop[6] + pop[10] + pop[15] + pop[17]) / (rho_I)-VelocitySet::B());
                }

                if constexpr (BoundaryCase == normalVectorBase::NORTH())
                {
                    const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[14] + pop[15] + pop[16] + pop[17];
                    return ValueType(
                        rho_I,
                        VelocitySet::A() - (pop[0] + pop[3] + pop[5] + pop[6] + pop[11] + pop[17]) / (rho_I),
                        (pop[7] - pop[14]) / (rho_I),
                        (pop[9] + pop[10] - pop[15] - pop[16]) / (rho_I),
                        (pop[3] + pop[7] + pop[11] + pop[14] + pop[17]) / (rho_I)-VelocitySet::B(),
                        (pop[11] - pop[17]) / (rho_I),
                        VelocitySet::A() - (pop[0] + pop[1] + pop[2] + pop[3] + pop[7] + pop[14]) / (rho_I));
                }

                if constexpr (BoundaryCase == normalVectorBase::NORTH_FRONT())
                {
                    const scalar_t rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[7] + pop[9] + pop[11] + pop[14] + pop[16];
                    return ValueType(
                        rho_I,
                        VelocitySet::A() - (pop[0] + pop[3] + pop[5] + pop[11]) / (rho_I),
                        (pop[7] - pop[14]) / (rho_I),
                        (pop[9] - pop[16]) / (rho_I),
                        (pop[3] + pop[7] + pop[11] + pop[14]) / (rho_I)-VelocitySet::B(),
                        pop[11] / (rho_I),
                        (pop[5] + pop[9] + pop[11] + pop[16]) / (rho_I)-VelocitySet::B());
                }

                if constexpr (BoundaryCase == normalVectorBase::SOUTH_EAST_BACK())
                {
                    const scalar_t rho_I = pop[0] + pop[1] + pop[4] + pop[6] + pop[12] + pop[13] + pop[15];
                    return ValueType(
                        rho_I,
                        (pop[1] + pop[13] + pop[15]) / (rho_I)-VelocitySet::B(),
                        -pop[13] / (rho_I),
                        -pop[15] / (rho_I),
                        (pop[4] + pop[12] + pop[13]) / (rho_I)-VelocitySet::B(),
                        pop[12] / (rho_I),
                        (pop[6] + pop[12] + pop[15]) / (rho_I)-VelocitySet::B());
                }

                if constexpr (BoundaryCase == normalVectorBase::SOUTH_EAST())
                {
                    const scalar_t rho_I = pop[0] + pop[1] + pop[4] + pop[5] + pop[6] + pop[9] + pop[12] + pop[13] + pop[15] + pop[18];
                    return ValueType(
                        rho_I,
                        (pop[1] + pop[9] + pop[13] + pop[15]) / (rho_I)-VelocitySet::B(),
                        -pop[13] / (rho_I),
                        (pop[9] - pop[15]) / (rho_I),
                        (pop[4] + pop[12] + pop[13] + pop[18]) / (rho_I)-VelocitySet::B(),
                        (pop[12] - pop[18]) / (rho_I),
                        VelocitySet::A() - (pop[0] + pop[1] + pop[4] + pop[13]) / (rho_I));
                }

                if constexpr (BoundaryCase == normalVectorBase::SOUTH_EAST_FRONT())
                {
                    const scalar_t rho_I = pop[0] + pop[1] + pop[4] + pop[5] + pop[9] + pop[13] + pop[18];
                    return ValueType(
                        rho_I,
                        (pop[1] + pop[9] + pop[13]) / (rho_I)-VelocitySet::B(),
                        -pop[13] / (rho_I),
                        pop[9] / (rho_I),
                        (pop[4] + pop[13] + pop[18]) / (rho_I)-VelocitySet::B(),
                        -pop[18] / (rho_I),
                        (pop[5] + pop[9] + pop[18]) / (rho_I)-VelocitySet::B());
                }

                if constexpr (BoundaryCase == normalVectorBase::EAST_BACK())
                {
                    const scalar_t rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[6] + pop[7] + pop[12] + pop[13] + pop[15] + pop[17];
                    return ValueType(
                        rho_I,
                        (pop[1] + pop[7] + pop[13] + pop[15]) / (rho_I)-VelocitySet::B(),
                        (pop[7] - pop[13]) / (rho_I),
                        -pop[15] / (rho_I),
                        VelocitySet::A() - (pop[0] + pop[1] + pop[6] + pop[15]) / (rho_I),
                        (pop[12] - pop[17]) / (rho_I),
                        (pop[6] + pop[12] + pop[15] + pop[17]) / (rho_I)-VelocitySet::B());
                }

                if constexpr (BoundaryCase == normalVectorBase::EAST())
                {
                    const scalar_t rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[12] + pop[13] + pop[15] + pop[17] + pop[18];
                    return ValueType(
                        rho_I,
                        (pop[1] + pop[7] + pop[9] + pop[13] + pop[15]) / (rho_I)-VelocitySet::B(),
                        (pop[7] - pop[13]) / (rho_I),
                        (pop[9] - pop[15]) / (rho_I),
                        VelocitySet::A() - (pop[0] + pop[1] + pop[5] + pop[6] + pop[9] + pop[15]) / (rho_I),
                        (pop[11] + pop[12] - pop[17] - pop[18]) / (rho_I),
                        VelocitySet::A() - (pop[0] + pop[1] + pop[3] + pop[4] + pop[7] + pop[13]) / (rho_I));
                }

                if constexpr (BoundaryCase == normalVectorBase::EAST_FRONT())
                {
                    const scalar_t rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[7] + pop[9] + pop[11] + pop[13] + pop[18];
                    return ValueType(
                        rho_I,
                        (pop[1] + pop[7] + pop[9] + pop[13]) / (rho_I)-VelocitySet::B(),
                        (pop[7] - pop[13]) / (rho_I),
                        pop[9] / (rho_I),
                        VelocitySet::A() - (pop[0] + pop[1] + pop[5] + pop[9]) / (rho_I),
                        (pop[11] - pop[18]) / (rho_I),
                        (pop[5] + pop[9] + pop[11] + pop[18]) / (rho_I)-VelocitySet::B());
                }

                if constexpr (BoundaryCase == normalVectorBase::NORTH_EAST_BACK())
                {
                    const scalar_t rho_I = pop[0] + pop[1] + pop[3] + pop[6] + pop[7] + pop[15] + pop[17];
                    return ValueType(
                        rho_I,
                        (pop[1] + pop[7] + pop[15]) / (rho_I)-VelocitySet::B(),
                        pop[7] / (rho_I),
                        -pop[15] / (rho_I),
                        (pop[3] + pop[7] + pop[17]) / (rho_I)-VelocitySet::B(),
                        -pop[17] / (rho_I),
                        (pop[6] + pop[15] + pop[17]) / (rho_I)-VelocitySet::B());
                }

                if constexpr (BoundaryCase == normalVectorBase::NORTH_EAST())
                {
                    const scalar_t rho_I = pop[0] + pop[1] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[15] + pop[17];
                    return ValueType(
                        rho_I,
                        (pop[1] + pop[7] + pop[9] + pop[15]) / (rho_I)-VelocitySet::B(),
                        pop[7] / (rho_I),
                        (pop[9] - pop[15]) / (rho_I),
                        (pop[3] + pop[7] + pop[11] + pop[17]) / (rho_I)-VelocitySet::B(),
                        (pop[11] - pop[17]) / (rho_I),
                        VelocitySet::A() - (pop[0] + pop[1] + pop[3] + pop[7]) / (rho_I));
                }

                if constexpr (BoundaryCase == normalVectorBase::NORTH_EAST_FRONT())
                {
                    const scalar_t rho_I = pop[0] + pop[1] + pop[3] + pop[5] + pop[7] + pop[9] + pop[11];
                    return ValueType(
                        rho_I,
                        (pop[1] + pop[7] + pop[9]) / (rho_I)-VelocitySet::B(),
                        pop[7] / (rho_I),
                        pop[9] / (rho_I),
                        (pop[3] + pop[7] + pop[11]) / (rho_I)-VelocitySet::B(),
                        pop[11] / (rho_I),
                        (pop[5] + pop[9] + pop[11]) / (rho_I)-VelocitySet::B());
                }
            }
        }
    };
}

#include "DirichletBC.cuh"
#include "noSlipBC.cuh"

#endif