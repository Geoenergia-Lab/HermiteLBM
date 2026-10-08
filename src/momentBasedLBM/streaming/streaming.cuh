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
    Class handling the streaming step

Namespace
    LBM

SourceFiles
    streaming.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_STREAMING_CUH
#define __MBLBM_STREAMING_CUH

namespace LBM
{
    /**
     * @class streaming
     * @brief Handles the streaming step in Lattice Boltzmann Method simulations
     *
     * This class manages the streaming (propagation) step of the LBM algorithm,
     * where particle distributions move to neighboring lattice sites. It provides
     * efficient shared memory operations for storing and retrieving population
     * data with optimized periodic boundary handling.
     **/
    template <class VelocitySet>
    class streaming
    {
    public:
        /**
         * @brief Reads global moments into the shared memory array
         * @param[in] devPtrs Device pointer collection containing density, velocity and moment fields
         * @param[out] sharedBuffer Shared memory array for population storage
         * @param[in] tid Thread ID within block
         * @param[in] idx Global index of the lattice node
         **/
        __device__ static inline void save(
            const device::ptrColl_t &devPtrs,
            blockSharedBuffer &sharedBuffer,
            const device::label_t tid,
            const device::label_t idx) noexcept
        {
            sharedBuffer[q_i<0 * block::size()>() + tid] = devPtrs.ptr<axis::index<axis::NO_DIRECTION>()>()[idx] + rho0();

            device::constexpr_for<1, NUMBER_MOMENTS<device::label_t>()>(
                [&](const auto i)
                {
                    sharedBuffer[q_i<i * block::size()>() + tid] = devPtrs.ptr<i>()[idx];
                });

            block::sync();
        }

        /**
         * @brief Pulls population density from shared memory with periodic boundaries
         * @tparam N Size of shared memory array
         * @param[out] pop Population density array to be populated
         * @param[in] sharedBuffer Shared memory array containing moment data
         **/
        __device__ static inline void pull(
            thread::array<scalar_t, VelocitySet::template Q()> &pop,
            const blockSharedBuffer &sharedBuffer,
            const thread::coordinate &Tx) noexcept
        {
            device::constexpr_for<0, VelocitySet::template Q()>(
                [&](const auto i)
                {
                    const device::label_t idxIncoming = block::idx(
                        periodic_index<-VelocitySet::template c<int, axis::X>(q_i<i>()), block::template nx<device::label_t>()>(Tx.value<axis::X>()),
                        periodic_index<-VelocitySet::template c<int, axis::Y>(q_i<i>()), block::template ny<device::label_t>()>(Tx.value<axis::Y>()),
                        periodic_index<-VelocitySet::template c<int, axis::Z>(q_i<i>()), block::template nz<device::label_t>()>(Tx.value<axis::Z>()));

                    const momentsArray incomingMoments{
                        sharedBuffer[q_i<0 * block::size()>() + idxIncoming],
                        sharedBuffer[q_i<1 * block::size()>() + idxIncoming],
                        sharedBuffer[q_i<2 * block::size()>() + idxIncoming],
                        sharedBuffer[q_i<3 * block::size()>() + idxIncoming],
                        sharedBuffer[q_i<4 * block::size()>() + idxIncoming],
                        sharedBuffer[q_i<5 * block::size()>() + idxIncoming],
                        sharedBuffer[q_i<6 * block::size()>() + idxIncoming],
                        sharedBuffer[q_i<7 * block::size()>() + idxIncoming],
                        sharedBuffer[q_i<8 * block::size()>() + idxIncoming],
                        sharedBuffer[q_i<9 * block::size()>() + idxIncoming]};

                    VelocitySet::template reconstruct<i>(pop, incomingMoments);
                });
        }

    private:
        /**
         * @brief Computes periodic boundary index with optimization for power-of-two dimensions
         * @tparam coeff Direction shift (-1 for backward, +1 for forward)
         * @tparam Dim Dimension size (periodic length)
         * @param[in] idx Current index position
         * @return Shifted index with periodic wrapping
         *
         * This function uses bitwise AND optimization when Dim is power-of-two
         * for improved performance, falling back to modulo arithmetic otherwise.
         **/
        template <const int coeff, const device::label_t Dim>
        [[nodiscard]] __device__ static inline device::label_t periodic_index(const device::label_t idx) noexcept
        {
            velocityCoefficient::assertions::validate<coeff, velocityCoefficient::CAN_BE_NULL>();

            if constexpr (Dim > 0 && (Dim & (Dim - 1)) == 0)
            {
                // Power-of-two: use bitwise AND
                if constexpr (coeff == -1)
                {
                    return (idx - 1) & (Dim - 1);
                }

                if constexpr (coeff == 1)
                {
                    return (idx + 1) & (Dim - 1);
                }

                if constexpr (coeff == 0)
                {
                    return idx & (Dim - 1);
                }
            }
            else
            {
                // General case: adjust by adding Dim to ensure nonnegative modulo
                if constexpr (coeff == -1)
                {
                    return (idx - 1 + Dim) % Dim;
                }

                if constexpr (coeff == 1)
                {
                    return (idx + 1) % Dim;
                }

                if constexpr (coeff == 0)
                {
                    return idx % Dim;
                }
            }
        }
    };

}

#endif