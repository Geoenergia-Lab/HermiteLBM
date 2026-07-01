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
    Definition of the D3Q27 velocity set

Namespace
    LBM

SourceFiles
    D3Q27.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_D3Q27_CUH
#define __MBLBM_D3Q27_CUH

#include "velocitySet.cuh"

namespace LBM
{
    /**
     * @class D3Q27
     * @brief Implements the D3Q27 velocity set for 3D Lattice Boltzmann simulations
     * @extends velocitySet
     *
     * This class provides the specific implementation for the D3Q27 lattice model,
     * which includes 27 discrete velocity directions in 3D space. It contains:
     * - Velocity components (cx, cy, cz) for each direction
     * - Weight coefficients for each direction
     * - Methods for moment calculation and population reconstruction
     * - Equilibrium distribution functions
     **/
    template <const thermalModel_t ThermalModel>
    class D3Q27 : public velocitySet<27>, public thermalModelBase<ThermalModel>
    {
    public:
        using Base = velocitySet<27>;

        /**
         * @brief Reconstruct population distribution from moments (in-place)
         * @param[out] pop Population array to be filled
         * @param[in] moments Moment array (rho, U, Pi)
         **/
        template <const bool CalculateRest = true>
        __device__ __host__ static inline void reconstruct(thread::array<scalar_t, Base::Q()> &pop, const thread::array<scalar_t, NUMBER_MOMENTS<host::label_t>()> &moments) noexcept
        {
            if constexpr (ThermalModel)
            {
                const thread::array<scalar_t, 3> diagonalTerm = velocitySet::diagonal_term(moments);

                const scalar_t rhow_1 = moments[m_i<0>()] * w_1<scalar_t>();
                const scalar_t rhow_2 = moments[m_i<0>()] * w_2<scalar_t>();
                const scalar_t rhow_3 = moments[m_i<0>()] * w_3<scalar_t>();
                const scalar_t pics2 = static_cast<scalar_t>(1) - cs2<scalar_t>() * (diagonalTerm[q_i<0>()] + diagonalTerm[q_i<1>()] + diagonalTerm[q_i<2>()]);

                if constexpr (CalculateRest)
                {
                    const scalar_t rhow_0 = moments[m_i<0>()] * w_0<scalar_t>();
                    pop[0] = rhow_0 * (pics2);
                }

                pop[1] = rhow_1 * (pics2 + moments[q_i<1>()] + diagonalTerm[q_i<0>()]);
                pop[2] = rhow_1 * (pics2 - moments[q_i<1>()] + diagonalTerm[q_i<0>()]);
                pop[3] = rhow_1 * (pics2 + moments[q_i<2>()] + diagonalTerm[q_i<1>()]);
                pop[4] = rhow_1 * (pics2 - moments[q_i<2>()] + diagonalTerm[q_i<1>()]);
                pop[5] = rhow_1 * (pics2 + moments[q_i<3>()] + diagonalTerm[q_i<2>()]);
                pop[6] = rhow_1 * (pics2 - moments[q_i<3>()] + diagonalTerm[q_i<2>()]);

                pop[7] = rhow_2 * (pics2 + moments[q_i<1>()] + moments[q_i<2>()] + diagonalTerm[q_i<0>()] + diagonalTerm[q_i<1>()] + moments[q_i<5>()]);
                pop[8] = rhow_2 * (pics2 - moments[q_i<1>()] - moments[q_i<2>()] + diagonalTerm[q_i<0>()] + diagonalTerm[q_i<1>()] + moments[q_i<5>()]);
                pop[9] = rhow_2 * (pics2 + moments[q_i<1>()] + moments[q_i<3>()] + diagonalTerm[q_i<0>()] + diagonalTerm[q_i<2>()] + moments[q_i<6>()]);
                pop[10] = rhow_2 * (pics2 - moments[q_i<1>()] - moments[q_i<3>()] + diagonalTerm[q_i<0>()] + diagonalTerm[q_i<2>()] + moments[q_i<6>()]);
                pop[11] = rhow_2 * (pics2 + moments[q_i<2>()] + moments[q_i<3>()] + diagonalTerm[q_i<1>()] + diagonalTerm[q_i<2>()] + moments[q_i<8>()]);
                pop[12] = rhow_2 * (pics2 - moments[q_i<2>()] - moments[q_i<3>()] + diagonalTerm[q_i<1>()] + diagonalTerm[q_i<2>()] + moments[q_i<8>()]);
                pop[13] = rhow_2 * (pics2 + moments[q_i<1>()] - moments[q_i<2>()] + diagonalTerm[q_i<0>()] + diagonalTerm[q_i<1>()] - moments[q_i<5>()]);
                pop[14] = rhow_2 * (pics2 - moments[q_i<1>()] + moments[q_i<2>()] + diagonalTerm[q_i<0>()] + diagonalTerm[q_i<1>()] - moments[q_i<5>()]);
                pop[15] = rhow_2 * (pics2 + moments[q_i<1>()] - moments[q_i<3>()] + diagonalTerm[q_i<0>()] + diagonalTerm[q_i<2>()] - moments[q_i<6>()]);
                pop[16] = rhow_2 * (pics2 - moments[q_i<1>()] + moments[q_i<3>()] + diagonalTerm[q_i<0>()] + diagonalTerm[q_i<2>()] - moments[q_i<6>()]);
                pop[17] = rhow_2 * (pics2 + moments[q_i<2>()] - moments[q_i<3>()] + diagonalTerm[q_i<1>()] + diagonalTerm[q_i<2>()] - moments[q_i<8>()]);
                pop[18] = rhow_2 * (pics2 - moments[q_i<2>()] + moments[q_i<3>()] + diagonalTerm[q_i<1>()] + diagonalTerm[q_i<2>()] - moments[q_i<8>()]);

                pop[19] = rhow_3 * (pics2 + moments[q_i<1>()] + moments[q_i<2>()] + moments[q_i<3>()] + diagonalTerm[q_i<0>()] + diagonalTerm[q_i<1>()] + diagonalTerm[q_i<2>()] + (moments[q_i<5>()] + moments[q_i<6>()] + moments[q_i<8>()]));
                pop[20] = rhow_3 * (pics2 - moments[q_i<1>()] - moments[q_i<2>()] - moments[q_i<3>()] + diagonalTerm[q_i<0>()] + diagonalTerm[q_i<1>()] + diagonalTerm[q_i<2>()] + (moments[q_i<5>()] + moments[q_i<6>()] + moments[q_i<8>()]));
                pop[21] = rhow_3 * (pics2 + moments[q_i<1>()] + moments[q_i<2>()] - moments[q_i<3>()] + diagonalTerm[q_i<0>()] + diagonalTerm[q_i<1>()] + diagonalTerm[q_i<2>()] + (moments[q_i<5>()] - moments[q_i<6>()] - moments[q_i<8>()]));
                pop[22] = rhow_3 * (pics2 - moments[q_i<1>()] - moments[q_i<2>()] + moments[q_i<3>()] + diagonalTerm[q_i<0>()] + diagonalTerm[q_i<1>()] + diagonalTerm[q_i<2>()] + (moments[q_i<5>()] - moments[q_i<6>()] - moments[q_i<8>()]));
                pop[23] = rhow_3 * (pics2 + moments[q_i<1>()] - moments[q_i<2>()] + moments[q_i<3>()] + diagonalTerm[q_i<0>()] + diagonalTerm[q_i<1>()] + diagonalTerm[q_i<2>()] - (moments[q_i<5>()] - moments[q_i<6>()] + moments[q_i<8>()]));
                pop[24] = rhow_3 * (pics2 - moments[q_i<1>()] + moments[q_i<2>()] - moments[q_i<3>()] + diagonalTerm[q_i<0>()] + diagonalTerm[q_i<1>()] + diagonalTerm[q_i<2>()] - (moments[q_i<5>()] - moments[q_i<6>()] + moments[q_i<8>()]));
                pop[25] = rhow_3 * (pics2 - moments[q_i<1>()] + moments[q_i<2>()] + moments[q_i<3>()] + diagonalTerm[q_i<0>()] + diagonalTerm[q_i<1>()] + diagonalTerm[q_i<2>()] - (moments[q_i<5>()] + moments[q_i<6>()] - moments[q_i<8>()]));
                pop[26] = rhow_3 * (pics2 + moments[q_i<1>()] - moments[q_i<2>()] - moments[q_i<3>()] + diagonalTerm[q_i<0>()] + diagonalTerm[q_i<1>()] + diagonalTerm[q_i<2>()] - (moments[q_i<5>()] + moments[q_i<6>()] - moments[q_i<8>()]));
            }
            else
            {
                const scalar_t pics2 = static_cast<scalar_t>(1.0) - cs2<scalar_t>() * (moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<9>()]);

                if constexpr (CalculateRest)
                {
                    const scalar_t rhow_0 = moments[m_i<0>()] * w_0<scalar_t>();
                    pop[q_i<0>()] = rhow_0 * pics2;
                }

                const scalar_t rhow_1 = moments[m_i<0>()] * w_1<scalar_t>();
                const scalar_t rhow_2 = moments[m_i<0>()] * w_2<scalar_t>();
                const scalar_t rhow_3 = moments[m_i<0>()] * w_3<scalar_t>();

                pop[q_i<1>()] = rhow_1 * (pics2 + moments[m_i<1>()] + moments[m_i<4>()]);
                pop[q_i<2>()] = rhow_1 * (pics2 - moments[m_i<1>()] + moments[m_i<4>()]);
                pop[q_i<3>()] = rhow_1 * (pics2 + moments[m_i<2>()] + moments[m_i<7>()]);
                pop[q_i<4>()] = rhow_1 * (pics2 - moments[m_i<2>()] + moments[m_i<7>()]);
                pop[q_i<5>()] = rhow_1 * (pics2 + moments[m_i<3>()] + moments[m_i<9>()]);
                pop[q_i<6>()] = rhow_1 * (pics2 - moments[m_i<3>()] + moments[m_i<9>()]);

                pop[q_i<7>()] = rhow_2 * (pics2 + moments[m_i<1>()] + moments[m_i<2>()] + moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<5>()]);
                pop[q_i<8>()] = rhow_2 * (pics2 - moments[m_i<1>()] - moments[m_i<2>()] + moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<5>()]);
                pop[q_i<9>()] = rhow_2 * (pics2 + moments[m_i<1>()] + moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<9>()] + moments[m_i<6>()]);
                pop[q_i<10>()] = rhow_2 * (pics2 - moments[m_i<1>()] - moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<9>()] + moments[m_i<6>()]);
                pop[q_i<11>()] = rhow_2 * (pics2 + moments[m_i<2>()] + moments[m_i<3>()] + moments[m_i<7>()] + moments[m_i<9>()] + moments[m_i<8>()]);
                pop[q_i<12>()] = rhow_2 * (pics2 - moments[m_i<2>()] - moments[m_i<3>()] + moments[m_i<7>()] + moments[m_i<9>()] + moments[m_i<8>()]);
                pop[q_i<13>()] = rhow_2 * (pics2 + moments[m_i<1>()] - moments[m_i<2>()] + moments[m_i<4>()] + moments[m_i<7>()] - moments[m_i<5>()]);
                pop[q_i<14>()] = rhow_2 * (pics2 - moments[m_i<1>()] + moments[m_i<2>()] + moments[m_i<4>()] + moments[m_i<7>()] - moments[m_i<5>()]);
                pop[q_i<15>()] = rhow_2 * (pics2 + moments[m_i<1>()] - moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<9>()] - moments[m_i<6>()]);
                pop[q_i<16>()] = rhow_2 * (pics2 - moments[m_i<1>()] + moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<9>()] - moments[m_i<6>()]);
                pop[q_i<17>()] = rhow_2 * (pics2 + moments[m_i<2>()] - moments[m_i<3>()] + moments[m_i<7>()] + moments[m_i<9>()] - moments[m_i<8>()]);
                pop[q_i<18>()] = rhow_2 * (pics2 - moments[m_i<2>()] + moments[m_i<3>()] + moments[m_i<7>()] + moments[m_i<9>()] - moments[m_i<8>()]);

                pop[q_i<19>()] = rhow_3 * (pics2 + moments[m_i<1>()] + moments[m_i<2>()] + moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<9>()] + (moments[m_i<5>()] + moments[m_i<6>()] + moments[m_i<8>()]));
                pop[q_i<20>()] = rhow_3 * (pics2 - moments[m_i<1>()] - moments[m_i<2>()] - moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<9>()] + (moments[m_i<5>()] + moments[m_i<6>()] + moments[m_i<8>()]));
                pop[q_i<21>()] = rhow_3 * (pics2 + moments[m_i<1>()] + moments[m_i<2>()] - moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<9>()] + (moments[m_i<5>()] - moments[m_i<6>()] - moments[m_i<8>()]));
                pop[q_i<22>()] = rhow_3 * (pics2 - moments[m_i<1>()] - moments[m_i<2>()] + moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<9>()] + (moments[m_i<5>()] - moments[m_i<6>()] - moments[m_i<8>()]));
                pop[q_i<23>()] = rhow_3 * (pics2 + moments[m_i<1>()] - moments[m_i<2>()] + moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<9>()] - (moments[m_i<5>()] - moments[m_i<6>()] + moments[m_i<8>()]));
                pop[q_i<24>()] = rhow_3 * (pics2 - moments[m_i<1>()] + moments[m_i<2>()] - moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<9>()] - (moments[m_i<5>()] - moments[m_i<6>()] + moments[m_i<8>()]));
                pop[q_i<25>()] = rhow_3 * (pics2 - moments[m_i<1>()] + moments[m_i<2>()] + moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<9>()] - (moments[m_i<5>()] + moments[m_i<6>()] - moments[m_i<8>()]));
                pop[q_i<26>()] = rhow_3 * (pics2 + moments[m_i<1>()] - moments[m_i<2>()] - moments[m_i<3>()] + moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<9>()] - (moments[m_i<5>()] + moments[m_i<6>()] - moments[m_i<8>()]));
            }
        }

        /**
         * @brief Reconstruct population distribution from moments (return)
         * @param[in] moments Moment array (rho, U, Pi)
         * @return Population array with 27 components
         **/
        __device__ __host__ [[nodiscard]] static inline thread::array<scalar_t, Base::Q()> reconstruct(
            const thread::array<scalar_t, NUMBER_MOMENTS<host::label_t>()> &moments) noexcept
        {
            thread::array<scalar_t, Base::Q()> pop;

            reconstruct(pop, moments);

            return pop;
        }

    private:
    };
}

#endif