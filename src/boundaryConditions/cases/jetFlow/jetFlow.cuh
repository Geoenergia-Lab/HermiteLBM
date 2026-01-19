/*---------------------------------------------------------------------------*\
|                                                                             |
| cudaLBM: CUDA-based moment representation Lattice Boltzmann Method          |
| Developed at UDESC - State University of Santa Catarina                     |
| Website: https://www.udesc.br                                               |
| Github: https://github.com/geoenergiaUDESC/cudaLBM                          |
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
    This file is part of cudaLBM.

    cudaLBM is free software: you can redistribute it and/or modify it
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

// Fallback print tracker for all 26 boundary cases
__device__ int printedFallback[26] = {false};

namespace LBM
{

    __device__ __host__ [[nodiscard]] inline consteval scalar_t L_char() noexcept
    {
        return static_cast<scalar_t>(10);
    }

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
    class jetFlow
    {
    public:
        /**
         * @brief Default constructor (constexpr)
         **/
        __device__ __host__ [[nodiscard]] inline consteval jetFlow(){};

        /**
         * @brief Calculate moment variables at boundary nodes
         * @tparam VelocitySet Velocity set configuration defining lattice structure
         * @param[in] pop Population density array at current lattice node
         * @param[out] moments Moment variables array to be populated
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
        template <class VelocitySet>
        __device__ static inline constexpr void calculate_moments(
            const thread::array<scalar_t, VelocitySet::Q()> &pop,
            thread::array<scalar_t, NUMBER_MOMENTS()> &moments,
            const normalVector &boundaryNormal,
            const scalar_t *const ptrRestrict shared_buffer) noexcept
        {
            static_assert((VelocitySet::Q() == 19) || (VelocitySet::Q() == 27), "Error: jetFlow::calculate_moments only supports D3Q19 and D3Q27.");

            const scalar_t rho_I = velocitySet::calculate_moment<VelocitySet, NO_DIRECTION, NO_DIRECTION>(pop, boundaryNormal);
            const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;

            bool already_handled = false;

            switch (boundaryNormal.nodeType())
            {
            // Round inflow + no-slip
            case normalVector::BACK():
            {
                const label_t x = threadIdx.x + block::nx() * blockIdx.x;
                const label_t y = threadIdx.y + block::ny() * blockIdx.y;

                const scalar_t is_jet = static_cast<scalar_t>((static_cast<scalar_t>(x) - center_x()) * (static_cast<scalar_t>(x) - center_x()) + (static_cast<scalar_t>(y) - center_y()) * (static_cast<scalar_t>(y) - center_y()) < r2());

                const scalar_t mxz_I = BACK_mxz_I(pop, inv_rho_I);
                const scalar_t myz_I = BACK_myz_I(pop, inv_rho_I);

                const scalar_t rho = rho0<scalar_t>();
                const scalar_t mxz = static_cast<scalar_t>(2) * mxz_I * rho_I / rho;
                const scalar_t myz = static_cast<scalar_t>(2) * myz_I * rho_I / rho;

                moments[m_i<0>()] = rho;                                            // rho
                moments[m_i<1>()] = static_cast<scalar_t>(0);                       // ux
                moments[m_i<2>()] = static_cast<scalar_t>(0);                       // uy
                moments[m_i<3>()] = is_jet * device::u_inf;                         // uz
                moments[m_i<4>()] = static_cast<scalar_t>(0);                       // mxx
                moments[m_i<5>()] = static_cast<scalar_t>(0);                       // mxy
                moments[m_i<6>()] = mxz;                                            // mxz
                moments[m_i<7>()] = static_cast<scalar_t>(0);                       // myy
                moments[m_i<8>()] = myz;                                            // myz
                moments[m_i<9>()] = is_jet * (rho * device::u_inf * device::u_inf); // mzz

                already_handled = true;
                return;
            }

// Periodic
#include "include/periodic.cuh"

// Outflow (zero-gradient) at front face
#include "include/IRBCN.cuh"

            // Call static boundaries for uncovered cases
            default:
            {
                if (!already_handled)
                {
                    switch (boundaryNormal.nodeType())
                    {
#include "include/fallback.cuh"
                    }
                }

                break;
            }
            }
        }

        template <class VelocitySet, const label_t N>
        __device__ static inline constexpr void calculate_moments(
            const thread::array<scalar_t, VelocitySet::Q()> &pop,
            thread::array<scalar_t, NUMBER_MOMENTS()> &moments,
            const normalVector &boundaryNormal,
            const thread::array<scalar_t, N> &shared_buffer) noexcept
        {
            static_assert((VelocitySet::Q() == 19) || (VelocitySet::Q() == 27), "Error: jetFlow::calculate_moments only supports D3Q19 and D3Q27.");

            // const scalar_t rho_I = velocitySet::rho_I<VelocitySet>(pop, boundaryNormal);
            const scalar_t rho_I = velocitySet::calculate_moment<VelocitySet, NO_DIRECTION, NO_DIRECTION>(pop, boundaryNormal);
            const scalar_t inv_rho_I = static_cast<scalar_t>(1) / rho_I;

            bool already_handled = false;

            switch (boundaryNormal.nodeType())
            {
            // Round inflow + no-slip
            case normalVector::BACK():
            {
                const label_t x = threadIdx.x + block::nx() * blockIdx.x;
                const label_t y = threadIdx.y + block::ny() * blockIdx.y;

                const scalar_t is_jet = static_cast<scalar_t>((static_cast<scalar_t>(x) - center_x()) * (static_cast<scalar_t>(x) - center_x()) + (static_cast<scalar_t>(y) - center_y()) * (static_cast<scalar_t>(y) - center_y()) < r2());

                const scalar_t mxz_I = BACK_mxz_I(pop, inv_rho_I);
                const scalar_t myz_I = BACK_myz_I(pop, inv_rho_I);

                const scalar_t rho = rho0<scalar_t>();
                const scalar_t mxz = static_cast<scalar_t>(2) * mxz_I * rho_I / rho;
                const scalar_t myz = static_cast<scalar_t>(2) * myz_I * rho_I / rho;

                moments[m_i<0>()] = rho;                                            // rho
                moments[m_i<1>()] = static_cast<scalar_t>(0);                       // ux
                moments[m_i<2>()] = static_cast<scalar_t>(0);                       // uy
                moments[m_i<3>()] = is_jet * device::u_inf;                         // uz
                moments[m_i<4>()] = static_cast<scalar_t>(0);                       // mxx
                moments[m_i<5>()] = static_cast<scalar_t>(0);                       // mxy
                moments[m_i<6>()] = mxz;                                            // mxz
                moments[m_i<7>()] = static_cast<scalar_t>(0);                       // myy
                moments[m_i<8>()] = myz;                                            // myz
                moments[m_i<9>()] = is_jet * (rho * device::u_inf * device::u_inf); // mzz

                already_handled = true;
                return;
            }

// Periodic
#include "include/periodic.cuh"

// Outflow (zero-gradient) at front face
#include "include/IRBCN.cuh"

            // Call static boundaries for uncovered cases
            default:
            {
                if (!already_handled)
                {
                    switch (boundaryNormal.nodeType())
                    {
#include "include/fallback.cuh"
                    }
                }

                break;
            }
            }
        }

    private:
        __device__ [[nodiscard]] static inline scalar_t center_x() noexcept
        {
            return static_cast<scalar_t>(0.5) * static_cast<scalar_t>(device::nx - 1);
        }

        __device__ [[nodiscard]] static inline scalar_t center_y() noexcept
        {
            return static_cast<scalar_t>(0.5) * static_cast<scalar_t>(device::ny - 1);
        }

        __device__ [[nodiscard]] static inline scalar_t radius() noexcept
        {
            return static_cast<scalar_t>(0.5) * static_cast<scalar_t>(L_char());
        }

        __device__ [[nodiscard]] static inline scalar_t r2() noexcept
        {
            return radius() * radius();
        }

        template <typename T = uint8_t>
        __device__ static inline void printOnce(
            const T caseId,
            const char *name) noexcept
        {
            if (atomicExch(&printedFallback[caseId], true) == false)
            {
                printf("[fallback] %s applied\n", name);
            }
        }

        __device__ [[nodiscard]] static inline constexpr int boundaryTarget(
            const int nThreads,
            const int offset) noexcept
        {
            return (offset > 0) ? 0 : (offset < 0 ? nThreads - 1 : (nThreads >> 1));
        }

        __device__ [[nodiscard]] static inline bool isBoundaryThread(const int3 offset) noexcept
        {
            const int tx = boundaryTarget(block::nx(), offset.x);
            const int ty = boundaryTarget(block::ny(), offset.y);
            const int tz = boundaryTarget(block::nz(), offset.z);

            return (static_cast<int>(threadIdx.x) == tx) && (static_cast<int>(threadIdx.y) == ty) && (static_cast<int>(threadIdx.z) == tz);
        }

        template <typename T = const char *>
        __device__ static inline void printThreadMapping(
            const T label,
            const int3 offset) noexcept
        {
            if (!isBoundaryThread(offset))
                return;

            const int gx = threadIdx.x + block::nx() * blockIdx.x;
            const int gy = threadIdx.y + block::ny() * blockIdx.y;
            const int gz = threadIdx.z + block::nz() * blockIdx.z;

            const int gx_int = gx + offset.x;
            const int gy_int = gy + offset.y;
            const int gz_int = gz + offset.z;

            const int bx = (offset.x == 0);
            const int by = (offset.y == 0);
            const int bz = (offset.z == 0);
            const int key = (bx) | (by << 1) | (bz << 2);

            switch (key)
            {
            case 0:
                printf("[%s] global=(%d,%d,%d) -> interior=(%d,%d,%d)\n",
                       label, gx, gy, gz, gx_int, gy_int, gz_int);
                break;
            case 1:
                printf("[%s] global=(%c,%d,%d) -> interior=(%c,%d,%d)\n",
                       label, 'x', gy, gz, 'x', gy_int, gz_int);
                break;
            case 2:
                printf("[%s] global=(%d,%c,%d) -> interior=(%d,%c,%d)\n",
                       label, gx, 'y', gz, gx_int, 'y', gz_int);
                break;
            case 3:
                printf("[%s] global=(%c,%c,%d) -> interior=(%c,%c,%d)\n",
                       label, 'x', 'y', gz, 'x', 'y', gz_int);
                break;
            case 4:
                printf("[%s] global=(%d,%d,%c) -> interior=(%d,%d,%c)\n",
                       label, gx, gy, 'z', gx_int, gy_int, 'z');
                break;
            case 5:
                printf("[%s] global=(%c,%d,%c) -> interior=(%c,%d,%c)\n",
                       label, 'x', gy, 'z', 'x', gy_int, 'z');
                break;
            case 6:
                printf("[%s] global=(%d,%c,%c) -> interior=(%d,%c,%c)\n",
                       label, gx, 'y', 'z', gx_int, 'y', 'z');
                break;
            case 7:
                printf("[%s] global=(%c,%c,%c) -> interior=(%c,%c,%c)\n",
                       label, 'x', 'y', 'z', 'x', 'y', 'z');
                break;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t SOUTH_WEST_mxy_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return pop[q_i<8>()] * inv_rho_I;
            }
            else
            {
                return (pop[q_i<8>()] + pop[q_i<20>()] + pop[q_i<22>()]) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t SOUTH_EAST_mxy_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return -pop[q_i<13>()] * inv_rho_I;
            }
            else
            {
                return -(pop[q_i<13>()] + pop[q_i<23>()] + pop[q_i<26>()]) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t WEST_BACK_mxz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return (pop[q_i<10>()]) * inv_rho_I;
            }
            else
            {
                return (pop[q_i<10>()] + pop[q_i<20>()] + pop[q_i<24>()]) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t WEST_FRONT_mxz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return -(pop[q_i<16>()]) * inv_rho_I;
            }
            else
            {
                return -(pop[q_i<16>()] + pop[q_i<22>()] + pop[q_i<25>()]) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t EAST_BACK_mxz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return -(pop[q_i<15>()]) * inv_rho_I;
            }
            else
            {
                return -(pop[q_i<15>()] + pop[q_i<21>()] + pop[q_i<26>()]) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t EAST_FRONT_mxz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return (pop[q_i<9>()]) * inv_rho_I;
            }
            else
            {
                return (pop[q_i<9>()] + pop[q_i<19>()] + pop[q_i<23>()]) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t SOUTH_BACK_myz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return (pop[q_i<12>()]) * inv_rho_I;
            }
            else
            {
                return (pop[q_i<12>()] + pop[q_i<20>()] + pop[q_i<26>()]) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t SOUTH_FRONT_myz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return -(pop[q_i<18>()]) * inv_rho_I;
            }
            else
            {
                return -(pop[q_i<18>()] + pop[q_i<22>()] + pop[q_i<23>()]) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t WEST_mxy_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop[q_i<8>()]) - (pop[q_i<14>()])) * inv_rho_I;
            }
            else
            {
                return ((pop[q_i<8>()] + pop[q_i<20>()] + pop[q_i<22>()]) - (pop[q_i<14>()] + pop[q_i<24>()] + pop[q_i<25>()])) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t WEST_mxz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop[q_i<10>()]) - (pop[q_i<16>()])) * inv_rho_I;
            }
            else
            {
                return ((pop[q_i<10>()] + pop[q_i<20>()] + pop[q_i<24>()]) - (pop[q_i<16>()] + pop[q_i<22>()] + pop[q_i<25>()])) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t EAST_mxy_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop[q_i<7>()]) - (pop[q_i<13>()])) * inv_rho_I;
            }
            else
            {
                return ((pop[q_i<7>()] + pop[q_i<19>()] + pop[q_i<21>()]) - (pop[q_i<13>()] + pop[q_i<23>()] + pop[q_i<26>()])) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t EAST_mxz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop[q_i<9>()]) - (pop[q_i<15>()])) * inv_rho_I;
            }
            else
            {
                return ((pop[q_i<9>()] + pop[q_i<19>()] + pop[q_i<23>()]) - (pop[q_i<15>()] + pop[q_i<21>()] + pop[q_i<26>()])) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t SOUTH_mxy_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop[q_i<8>()]) - (pop[q_i<13>()])) * inv_rho_I;
            }
            else
            {
                return ((pop[q_i<8>()] + pop[q_i<20>()] + pop[q_i<22>()]) - (pop[q_i<13>()] + pop[q_i<23>()] + pop[q_i<26>()])) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t SOUTH_myz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop[q_i<12>()]) - (pop[q_i<18>()])) * inv_rho_I;
            }
            else
            {
                return ((pop[q_i<12>()] + pop[q_i<20>()] + pop[q_i<26>()]) - (pop[q_i<18>()] + pop[q_i<22>()] + pop[q_i<23>()])) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t BACK_mxz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop[q_i<10>()]) - (pop[q_i<15>()])) * inv_rho_I;
            }
            else
            {
                return ((pop[q_i<10>()] + pop[q_i<20>()] + pop[q_i<24>()]) - (pop[q_i<15>()] + pop[q_i<21>()] + pop[q_i<26>()])) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t BACK_myz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop[q_i<12>()]) - (pop[q_i<17>()])) * inv_rho_I;
            }
            else
            {
                return ((pop[q_i<12>()] + pop[q_i<20>()] + pop[q_i<26>()]) - (pop[q_i<17>()] + pop[q_i<21>()] + pop[q_i<24>()])) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t FRONT_mxz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop[q_i<9>()]) - (pop[q_i<16>()])) * inv_rho_I;
            }
            else
            {
                return ((pop[q_i<9>()] + pop[q_i<19>()] + pop[q_i<23>()]) - (pop[q_i<16>()] + pop[q_i<22>()] + pop[q_i<25>()])) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t FRONT_myz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop[q_i<11>()]) - (pop[q_i<18>()])) * inv_rho_I;
            }
            else
            {
                return ((pop[q_i<11>()] + pop[q_i<19>()] + pop[q_i<25>()]) - (pop[q_i<18>()] + pop[q_i<22>()] + pop[q_i<23>()])) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t NORTH_mxy_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop[q_i<7>()]) - (pop[q_i<14>()])) * inv_rho_I;
            }
            else
            {
                return ((pop[q_i<7>()] + pop[q_i<19>()] + pop[q_i<21>()]) - (pop[q_i<14>()] + pop[q_i<24>()] + pop[q_i<25>()])) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t NORTH_myz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return ((pop[q_i<11>()]) - (pop[q_i<17>()])) * inv_rho_I;
            }
            else
            {
                return ((pop[q_i<11>()] + pop[q_i<19>()] + pop[q_i<25>()]) - (pop[q_i<17>()] + pop[q_i<21>()] + pop[q_i<24>()])) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t NORTH_BACK_myz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return -(pop[q_i<17>()]) * inv_rho_I;
            }
            else
            {
                return -(pop[q_i<17>()] + pop[q_i<21>()] + pop[q_i<24>()]) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t NORTH_FRONT_myz_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return (pop[q_i<11>()]) * inv_rho_I;
            }
            else
            {
                return (pop[q_i<11>()] + pop[q_i<19>()] + pop[q_i<25>()]) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t NORTH_EAST_mxy_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return (pop[q_i<7>()]) * inv_rho_I;
            }
            else
            {
                return (pop[q_i<7>()] + pop[q_i<19>()] + pop[q_i<21>()]) * inv_rho_I;
            }
        }

        template <const label_t Q>
        __device__ [[nodiscard]] static inline constexpr scalar_t NORTH_WEST_mxy_I(
            const thread::array<scalar_t, Q> &pop,
            const scalar_t inv_rho_I) noexcept
        {
            if constexpr (Q == 19)
            {
                return -(pop[q_i<14>()]) * inv_rho_I;
            }
            else
            {
                return -(pop[q_i<14>()] + pop[q_i<24>()] + pop[q_i<25>()]) * inv_rho_I;
            }
        }
    };
}

#endif