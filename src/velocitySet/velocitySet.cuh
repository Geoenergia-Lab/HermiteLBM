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
    Top-level header file for the definition of the velocity set class template

Namespace
    LBM

SourceFiles
    velocitySet.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_VELOCITYSET_CUH
#define __MBLBM_VELOCITYSET_CUH

#include "../LBMIncludes.cuh"
#include "../typedefs/typedefs.cuh"
#include "../globalFunctions.cuh"
#include "../array/threadArray.cuh"
#include "lattice.cuh"
#include "velocitySetBase.cuh"
#include "thermalModelBase.cuh"

namespace LBM
{
    /**
     * @class velocitySet
     * @brief Base class for LBM velocity sets providing common constants and scaling operations
     *
     * This class serves as a base for specific velocity set implementations (e.g., D3Q19, D3Q27)
     * and provides common constants, scaling factors, and utility functions used across
     * different velocity set configurations in the Lattice Boltzmann Method.
     **/
    template <const host::label_t Q_, const thermalModel_t ThermalModel>
    class velocitySet : public velocitySetBase, public lattice<Q_>, public thermalModelBase<ThermalModel>
    {
    public:
        using Lattice = lattice<Q_>;
        using ThermoModel = thermalModelBase<ThermalModel>;

        /**
         * @brief Default constructor (consteval)
         **/
        __device__ __host__ [[nodiscard]] inline consteval velocitySet() noexcept {}

        /**
         * @brief Calculates the diagonal correction term for the isothermal velocity set
         * @param[in] moments Moment array (rho, U, Pi)
         **/
        __device__ __host__ [[nodiscard]] static inline constexpr const thread::array<scalar_t, 3> diagonal_term(const thread::array<scalar_t, NUMBER_MOMENTS()> &moments) noexcept
        {
            const scalar_t Delta_m = (moments[q_i<1>()] * moments[q_i<1>()] + moments[q_i<2>()] * moments[q_i<2>()] + moments[q_i<3>()] * moments[q_i<3>()] - moments[q_i<4>()] - moments[q_i<7>()] - moments[q_i<9>()]) / static_cast<scalar_t>(3);

            return {moments[q_i<4>()] + Delta_m, moments[q_i<7>()] + Delta_m, moments[q_i<9>()] + Delta_m};
        }

        /**
         * @brief Calculate a specific moment of the distribution function
         * @tparam alpha The first axis direction (X, Y, or Z)
         * @tparam beta The second axis direction (X, Y, or Z)
         * @param[in] pop The distribution function array
         * @return The calculated moment value
         **/
        template <const axis::type alpha, const axis::type beta>
        __device__ __host__ [[nodiscard]] static inline constexpr scalar_t calculate_moment(const thread::array<scalar_t, Lattice::Q()> &pop) noexcept
        {
            constexpr const thread::array<int, Lattice::Q()> c_AB = c_AlphaBeta<alpha, beta>();
            constexpr const host::label_t N = c_AB.number_non_zero();
            constexpr const thread::array<int, N> C = c_AB.template non_zero_values<N>();
            constexpr const thread::array<host::label_t, N> indices = c_AB.template non_zero_indices<N>();

            return [&]<const host::label_t... Is>(std::index_sequence<Is...>)
            {
                return (process_momentum_element<C[Is]>(pop[indices[Is]]) + ...);
            }(std::make_index_sequence<N>{});
        }

        /**
         * @brief Calculate a specific moment of the distribution function
         * @tparam alpha The first axis direction (X, Y, or Z)
         * @tparam beta The second axis direction (X, Y, or Z)
         * @tparam BoundaryNormal The boundary normal vector type
         * @param[in] pop The distribution function array
         * @param[in] boundaryNormal Normal vector information at boundary node
         * @return The calculated moment value
         **/
        template <const axis::type alpha, const axis::type beta, class BoundaryNormal>
        __device__ __host__ [[nodiscard]] static inline constexpr scalar_t calculate_moment(const thread::array<scalar_t, Lattice::Q()> &pop, const BoundaryNormal &boundaryNormal) noexcept
        {
            constexpr const thread::array<int, Lattice::Q()> c_AB = c_AlphaBeta<alpha, beta>();
            constexpr const host::label_t N = c_AB.number_non_zero();
            constexpr const thread::array<int, N> C = c_AB.template non_zero_values<N>();
            constexpr const thread::array<host::label_t, N> indices = c_AB.template non_zero_indices<N>();

            return [&]<const host::label_t... Is>(std::index_sequence<Is...>)
            {
                return (process_momentum_element<C[Is], indices[Is]>(pop[indices[Is]], boundaryNormal) + ...);
            }(std::make_index_sequence<N>{});
        }

        /**
         * @brief Calculate all moments of the distribution function
         * @param[in] pop The distribution function array
         * @param[out] moments The calculated moments array
         **/
        __device__ __host__ static inline void calculate_moments(const thread::array<scalar_t, Lattice::Q()> &pop, thread::array<scalar_t, NUMBER_MOMENTS()> &moments) noexcept
        {
            // Density
            moments[m_i<0>()] = calculate_moment<axis::NO_DIRECTION, axis::NO_DIRECTION>(pop);
            const scalar_t inv_rho = static_cast<scalar_t>(1) / moments[m_i<0>()];

            // Velocity
            moments[m_i<1>()] = calculate_moment<axis::X, axis::NO_DIRECTION>(pop) * inv_rho;
            moments[m_i<2>()] = calculate_moment<axis::Y, axis::NO_DIRECTION>(pop) * inv_rho;
            moments[m_i<3>()] = calculate_moment<axis::Z, axis::NO_DIRECTION>(pop) * inv_rho;

            // Second order moments
            moments[m_i<4>()] = (calculate_moment<axis::X, axis::X>(pop) * inv_rho) - cs2<scalar_t>();
            moments[m_i<5>()] = calculate_moment<axis::X, axis::Y>(pop) * inv_rho;
            moments[m_i<6>()] = calculate_moment<axis::X, axis::Z>(pop) * inv_rho;
            moments[m_i<7>()] = (calculate_moment<axis::Y, axis::Y>(pop) * inv_rho) - cs2<scalar_t>();
            moments[m_i<8>()] = calculate_moment<axis::Y, axis::Z>(pop) * inv_rho;
            moments[m_i<9>()] = (calculate_moment<axis::Z, axis::Z>(pop) * inv_rho) - cs2<scalar_t>();
        }

        /**
         * @brief Calculate all moments of the distribution function
         * @tparam BoundaryNormal The boundary normal vector type
         * @param[in] pop The distribution function array
         * @param[out] moments The calculated moments array
         * @param[in] boundaryNormal Normal vector information at boundary node
         **/
        template <class BoundaryNormal>
        __device__ __host__ static inline void calculate_moments(const thread::array<scalar_t, Lattice::Q()> &pop, thread::array<scalar_t, NUMBER_MOMENTS()> &moments, const BoundaryNormal &boundaryNormal) noexcept
        {
            // Density
            moments[m_i<0>()] = calculate_moment<axis::NO_DIRECTION, axis::NO_DIRECTION>(pop, boundaryNormal);
            const scalar_t inv_rho = static_cast<scalar_t>(1) / moments[m_i<0>()];

            // Velocity
            moments[m_i<1>()] = calculate_moment<axis::X, axis::NO_DIRECTION>(pop, boundaryNormal) * inv_rho;
            moments[m_i<2>()] = calculate_moment<axis::Y, axis::NO_DIRECTION>(pop, boundaryNormal) * inv_rho;
            moments[m_i<3>()] = calculate_moment<axis::Z, axis::NO_DIRECTION>(pop, boundaryNormal) * inv_rho;

            // Second order moments
            moments[m_i<4>()] = (calculate_moment<axis::X, axis::X>(pop, boundaryNormal) * inv_rho) - cs2<scalar_t>();
            moments[m_i<5>()] = calculate_moment<axis::X, axis::Y>(pop, boundaryNormal) * inv_rho;
            moments[m_i<6>()] = calculate_moment<axis::X, axis::Z>(pop, boundaryNormal) * inv_rho;
            moments[m_i<7>()] = (calculate_moment<axis::Y, axis::Y>(pop, boundaryNormal) * inv_rho) - cs2<scalar_t>();
            moments[m_i<8>()] = calculate_moment<axis::Y, axis::Z>(pop, boundaryNormal) * inv_rho;
            moments[m_i<9>()] = (calculate_moment<axis::Z, axis::Z>(pop, boundaryNormal) * inv_rho) - cs2<scalar_t>();
        }

        /**
         * @brief Returns the indices of the distribution functions on a specific face
         * @tparam alpha The axis direction (X, Y or Z)
         * @tparam coeff The coefficient indicating the direction along the axis (must be -1 or 1)
         * @return Indices of the distribution on a specific face
         **/
        template <const axis::type alpha, const int coeff>
        __device__ __host__ [[nodiscard]] static inline consteval thread::array<host::label_t, Lattice::QF()> indices_on_face() noexcept
        {
            axis::assertions::validate<alpha, axis::NOT_NULL>();

            velocityCoefficient::assertions::validate<coeff, velocityCoefficient::NOT_NULL>();

            constexpr const thread::array<int, Lattice::Q()> vals = Lattice::template c<int, alpha>();

            thread::array<host::label_t, Lattice::QF()> indices;

            host::label_t j = 0;

            for (host::label_t i = 0; i < Lattice::Q(); i++)
            {
                if (vals[i] == coeff)
                {
                    indices[j] = i;
                    j++;
                }
            }

            return indices;
        }

        /**
         * @brief Print velocity set information to terminal
         **/
        __host__ static void print() noexcept
        {
            std::cout << "D3Q" << Lattice::Q() << " {w, cx, cy, cz}:" << std::endl;
            std::cout << "{" << std::endl;
            printAll();
            std::cout << "};" << std::endl;
            std::cout << std::endl;
        }

        /**
         * @brief Calculate the regularized distribution function from the moments
         * @tparam CalculateRest Whether to calculate the rest population (f_0) or not
         * @param[out] pop The distribution function array
         * @param[in] moments The calculated moments array
         **/
        template <const bool CalculateRest = true>
        __device__ __host__ static inline void reconstruct(thread::array<scalar_t, Lattice::Q()> &pop, const thread::array<scalar_t, NUMBER_MOMENTS<host::label_t>()> &moments) noexcept
        {
            if constexpr (ThermoModel::thermalModel() == thermalModel_t::Isothermal)
            {
                const thread::array<scalar_t, 3> diagonalTerm = velocitySet::diagonal_term(moments);
                const scalar_t pics2 = static_cast<scalar_t>(1) - cs2<scalar_t>() * (diagonalTerm[q_i<0>()] + diagonalTerm[q_i<1>()] + diagonalTerm[q_i<2>()]);

                if constexpr (CalculateRest)
                {
                    const scalar_t rhow_0 = moments[m_i<0>()] * Lattice::template w_0<scalar_t>();
                    pop[0] = rhow_0 * (pics2);
                }

                const scalar_t rhow_1 = moments[m_i<0>()] * Lattice::template w_1<scalar_t>();
                pop[1] = rhow_1 * (pics2 + moments[q_i<1>()] + diagonalTerm[q_i<0>()]);
                pop[2] = rhow_1 * (pics2 - moments[q_i<1>()] + diagonalTerm[q_i<0>()]);
                pop[3] = rhow_1 * (pics2 + moments[q_i<2>()] + diagonalTerm[q_i<1>()]);
                pop[4] = rhow_1 * (pics2 - moments[q_i<2>()] + diagonalTerm[q_i<1>()]);
                pop[5] = rhow_1 * (pics2 + moments[q_i<3>()] + diagonalTerm[q_i<2>()]);
                pop[6] = rhow_1 * (pics2 - moments[q_i<3>()] + diagonalTerm[q_i<2>()]);

                const scalar_t rhow_2 = moments[m_i<0>()] * Lattice::template w_2<scalar_t>();
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

                if constexpr (Lattice::Q() == 27)
                {
                    const scalar_t rhow_3 = moments[m_i<0>()] * Lattice::template w_3<scalar_t>();
                    pop[19] = rhow_3 * (pics2 + moments[q_i<1>()] + moments[q_i<2>()] + moments[q_i<3>()] + diagonalTerm[q_i<0>()] + diagonalTerm[q_i<1>()] + diagonalTerm[q_i<2>()] + (moments[q_i<5>()] + moments[q_i<6>()] + moments[q_i<8>()]));
                    pop[20] = rhow_3 * (pics2 - moments[q_i<1>()] - moments[q_i<2>()] - moments[q_i<3>()] + diagonalTerm[q_i<0>()] + diagonalTerm[q_i<1>()] + diagonalTerm[q_i<2>()] + (moments[q_i<5>()] + moments[q_i<6>()] + moments[q_i<8>()]));
                    pop[21] = rhow_3 * (pics2 + moments[q_i<1>()] + moments[q_i<2>()] - moments[q_i<3>()] + diagonalTerm[q_i<0>()] + diagonalTerm[q_i<1>()] + diagonalTerm[q_i<2>()] + (moments[q_i<5>()] - moments[q_i<6>()] - moments[q_i<8>()]));
                    pop[22] = rhow_3 * (pics2 - moments[q_i<1>()] - moments[q_i<2>()] + moments[q_i<3>()] + diagonalTerm[q_i<0>()] + diagonalTerm[q_i<1>()] + diagonalTerm[q_i<2>()] + (moments[q_i<5>()] - moments[q_i<6>()] - moments[q_i<8>()]));
                    pop[23] = rhow_3 * (pics2 + moments[q_i<1>()] - moments[q_i<2>()] + moments[q_i<3>()] + diagonalTerm[q_i<0>()] + diagonalTerm[q_i<1>()] + diagonalTerm[q_i<2>()] - (moments[q_i<5>()] - moments[q_i<6>()] + moments[q_i<8>()]));
                    pop[24] = rhow_3 * (pics2 - moments[q_i<1>()] + moments[q_i<2>()] - moments[q_i<3>()] + diagonalTerm[q_i<0>()] + diagonalTerm[q_i<1>()] + diagonalTerm[q_i<2>()] - (moments[q_i<5>()] - moments[q_i<6>()] + moments[q_i<8>()]));
                    pop[25] = rhow_3 * (pics2 - moments[q_i<1>()] + moments[q_i<2>()] + moments[q_i<3>()] + diagonalTerm[q_i<0>()] + diagonalTerm[q_i<1>()] + diagonalTerm[q_i<2>()] - (moments[q_i<5>()] + moments[q_i<6>()] - moments[q_i<8>()]));
                    pop[26] = rhow_3 * (pics2 + moments[q_i<1>()] - moments[q_i<2>()] - moments[q_i<3>()] + diagonalTerm[q_i<0>()] + diagonalTerm[q_i<1>()] + diagonalTerm[q_i<2>()] - (moments[q_i<5>()] + moments[q_i<6>()] - moments[q_i<8>()]));
                }
            }
            else
            {
                const scalar_t pics2 = static_cast<scalar_t>(1.0) - cs2<scalar_t>() * (moments[m_i<4>()] + moments[m_i<7>()] + moments[m_i<9>()]);

                if constexpr (CalculateRest)
                {
                    const scalar_t rhow_0 = moments[m_i<0>()] * Lattice::template w_0<scalar_t>();
                    pop[q_i<0>()] = rhow_0 * pics2;
                }

                const scalar_t rhow_1 = moments[m_i<0>()] * Lattice::template w_1<scalar_t>();
                pop[q_i<1>()] = rhow_1 * (pics2 + moments[m_i<1>()] + moments[m_i<4>()]);
                pop[q_i<2>()] = rhow_1 * (pics2 - moments[m_i<1>()] + moments[m_i<4>()]);
                pop[q_i<3>()] = rhow_1 * (pics2 + moments[m_i<2>()] + moments[m_i<7>()]);
                pop[q_i<4>()] = rhow_1 * (pics2 - moments[m_i<2>()] + moments[m_i<7>()]);
                pop[q_i<5>()] = rhow_1 * (pics2 + moments[m_i<3>()] + moments[m_i<9>()]);
                pop[q_i<6>()] = rhow_1 * (pics2 - moments[m_i<3>()] + moments[m_i<9>()]);

                const scalar_t rhow_2 = moments[m_i<0>()] * Lattice::template w_2<scalar_t>();
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

                if constexpr (Lattice::Q() == 27)
                {
                    const scalar_t rhow_3 = moments[m_i<0>()] * Lattice::template w_3<scalar_t>();
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
        }

        /**
         * @brief Reconstruct population distribution from moments (return)
         * @param[in] moments Moment array (rho, U, Pi)
         * @return Population array with Q components
         **/
        __device__ __host__ [[nodiscard]] static inline thread::array<scalar_t, Lattice::Q()> reconstruct(
            const thread::array<scalar_t, NUMBER_MOMENTS<host::label_t>()> &moments) noexcept
        {
            thread::array<scalar_t, Lattice::Q()> pop;

            reconstruct(pop, moments);

            return pop;
        }

    private:
        /**
         * @brief Determines if a discrete velocity direction is incoming relative to a boundary normal
         * @tparam T Return type (typically numeric type)
         * @tparam BoundaryNormal Type of boundary normal object with directional methods
         * @tparam q_ Compile-time velocity direction index
         * @param[in] q Compile-time constant representing velocity direction
         * @param[in] boundaryNormal Boundary normal information with directional methods
         * @return T 1 if velocity is incoming (pointing into domain), 0 if outgoing
         *
         * @details Checks if velocity components oppose boundary normal direction:
         * - For East boundary (normal.x > 0): checks negative x-velocity component
         * - For West boundary (normal.x < 0): checks positive x-velocity component
         * - For North boundary (normal.y > 0): checks negative y-velocity component
         * - For South boundary (normal.y < 0): checks positive y-velocity component
         * - For Front boundary (normal.z > 0): checks negative z-velocity component
         * - For Back boundary (normal.z < 0): checks positive z-velocity component
         * Returns 1 only if no incoming component is detected on any axis
         **/
        template <typename T, class BoundaryNormal, const device::label_t q_>
        __device__ __host__ [[nodiscard]] static inline constexpr T is_incoming(const q_i<q_> q, const BoundaryNormal &boundaryNormal) noexcept
        {
            // boundaryNormal.x > 0  => EAST boundary
            // boundaryNormal.x < 0  => WEST boundary
            const bool cond_x = (boundaryNormal.isEast() & is_negative<axis::X>(q)) | (boundaryNormal.isWest() & is_positive<axis::X>(q));

            // boundaryNormal.y > 0  => NORTH boundary
            // boundaryNormal.y < 0  => SOUTH boundary
            const bool cond_y = (boundaryNormal.isNorth() & is_negative<axis::Y>(q)) | (boundaryNormal.isSouth() & is_positive<axis::Y>(q));

            // boundaryNormal.z > 0  => FRONT boundary
            // boundaryNormal.z < 0  => BACK boundary
            const bool cond_z = (boundaryNormal.isFront() & is_negative<axis::Z>(q)) | (boundaryNormal.isBack() & is_positive<axis::Z>(q));

            return static_cast<T>(!(cond_x | cond_y | cond_z));
        }

        /**
         * @brief Returns the product of the c values for two directions
         * @tparam alpha The axis direction (X, Y, Z or NULL)
         **/
        template <const axis::type alpha, const axis::type beta>
        __device__ __host__ [[nodiscard]] static inline consteval const thread::array<int, Lattice::Q()> c_AlphaBeta() noexcept
        {
            axis::assertions::validate<alpha, axis::CAN_BE_NULL>();
            axis::assertions::validate<beta, axis::CAN_BE_NULL>();

            return Lattice::template c<int, alpha>() * Lattice::template c<int, beta>();
        }

        /**
         * @brief Adds or subtracts a particular population based on the sign of the coefficient
         * @tparam coeff The coefficient indicating the direction along the axis (must be -1 or 1)
         * @param[in] pop_value A particular population
         * @return Plus or minus pop_value depending on the value of coeff
         **/
        template <const int coeff>
        __device__ __host__ [[nodiscard]] static inline constexpr scalar_t process_momentum_element(const scalar_t pop_value) noexcept
        {
            velocityCoefficient::assertions::validate<coeff, velocityCoefficient::NOT_NULL>();

            if constexpr (coeff == 1)
            {
                return pop_value;
            }

            if constexpr (coeff == -1)
            {
                return -pop_value;
            }
        }

        /**
         * @brief Adds or subtracts a particular population based on the sign of the coefficient
         * @tparam coeff The coefficient indicating the direction along the axis (must be -1 or 1)
         * @tparam BoundaryNormal The boundary normal vector type
         * @param[in] pop_value A particular population
         * @param[in] boundaryNormal
         * @return Plus or minus pop_value depending on the value of coeff
         **/
        template <const int coeff, const device::label_t I, class BoundaryNormal>
        __device__ __host__ [[nodiscard]] static inline constexpr scalar_t process_momentum_element(const scalar_t pop_value, const BoundaryNormal &boundaryNormal) noexcept
        {
            velocityCoefficient::assertions::validate<coeff, velocityCoefficient::NOT_NULL>();

            if constexpr (coeff == 1)
            {
                return is_incoming<scalar_t>(q_i<I>(), boundaryNormal) * pop_value;
            }

            if constexpr (coeff == -1)
            {
                return -is_incoming<scalar_t>(q_i<I>(), boundaryNormal) * pop_value;
            }
        }

        /**
         * @brief Determines whether or not a particular lattice coefficient is negative
         * @tparam alpha The axis direction (X, Y or Z)
         * @param[in] q The lattice index
         * @return True if the lattice coefficient is negative, false otherwise
         **/
        template <const axis::type alpha, const device::label_t q_>
        __device__ __host__ [[nodiscard]] static inline consteval bool is_negative(const q_i<q_> q) noexcept
        {
            axis::assertions::validate<alpha, axis::NOT_NULL>();

            return (Lattice::template c<int, alpha>()[q] < 0);
        }

        /**
         * @brief Determines whether or not a particular lattice coefficient is positive
         * @tparam alpha The axis direction (X, Y, or Z)
         * @param[in] The lattice index
         * @return True if the lattice coefficient is positive, false otherwise
         **/
        template <const axis::type alpha, const device::label_t q_>
        __device__ __host__ [[nodiscard]] static inline consteval bool is_positive(const q_i<q_> q) noexcept
        {
            axis::assertions::validate<alpha, axis::NOT_NULL>();

            return (Lattice::template c<int, alpha>()[q] > 0);
        }

    protected:
        /**
         * @brief Returns the string corresponding to a lattice velocity coefficient
         * @tparam coeff The coefficient indicating the direction along the axis (must be -1 or 1)
         **/
        template <const int coeff>
        __host__ [[nodiscard]] static inline consteval const char *c_str() noexcept
        {
            if constexpr (coeff == 0)
            {
                return "0";
            }

            if constexpr (coeff == -1)
            {
                return "-1";
            }

            if constexpr (coeff == 1)
            {
                return "+1";
            }
        }

        /**
         * @brief Implementation of the print loop
         * @note This function effectively unrolls the loop at compile-time and checks for its bounds
         **/
        __host__ static void printAll() noexcept
        {
            host::constexpr_for<0, Lattice::Q()>(
                [&](const auto i)
                {
                    constexpr auto idx = q_i<decltype(i)::value>();
                    std::cout
                        << "    {w, cx, cy, cz}[" << idx << "] = {"
                        << Lattice::template w_q<double>()[idx] << ", "
                        << c_str<Lattice::template c<int, axis::X>()[idx]>() << ", "
                        << c_str<Lattice::template c<int, axis::Y>()[idx]>() << ", "
                        << c_str<Lattice::template c<int, axis::Z>()[idx]>() << "};" << std::endl;
                });
        }
    };
}

#include "D3Q19.cuh"
#include "D3Q27.cuh"

#endif