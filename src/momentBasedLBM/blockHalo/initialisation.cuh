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
    This file contains the implementation of the kernel for the initialisation
    of the block halo in the moment representation Lattice Boltzmann method.
    The kernel reconstructs the population distribution functions from the
    moments and saves them to the block halo buffers.

Namespace
    LBM::device

SourceFiles
    initialisation.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_INITIALISATION_CUH
#define __MBLBM_INITIALISATION_CUH

#include "../ptrCollection.cuh"

namespace LBM
{
    template <class VelocitySet, class BoundaryConditions>
    struct blockHaloInitialisationKernel
    {
        /**
         * @brief Saves the reconstructed halo populations into both halo buffers.
         *
         * @param[in] moments Moment array (rho, U, Pi)
         * @param[in] readBuffer Halo storage used for reads during the streaming step.
         * @param[in] writeBuffer Halo storage used for writes after the streaming step.
         * @param[in] Tx Thread coordinates within the current block.
         * @param[in] Bx Block coordinates in the global mesh.
         * @param[in] point Global point coordinate associated with the current halo element.
         *
         * @details Reconstructs the distribution function from the moments and writes it
         * to both the read and write halo buffers so the neighbouring data is available
         * to the next streaming step.
         **/
        template <const bool periodic_x, const bool periodic_y, const bool periodic_z>
        __device__ static inline void saveHalo(
            const momentsArray &moments,
            const device::ptrCollection<6, scalar_t> &readBuffer,
            const device::ptrCollection<6, scalar_t> &writeBuffer,
            const thread::coordinate &Tx,
            const block::coordinate &Bx,
            const device::pointCoordinate &point) noexcept
        {
            device::halo<VelocitySet, boundaryConditionType<periodic_x, periodic_y, periodic_z>>::save(moments, readBuffer, Tx, Bx, point);
            device::halo<VelocitySet, boundaryConditionType<periodic_x, periodic_y, periodic_z>>::save(moments, writeBuffer, Tx, Bx, point);
        }

        __device__ __forceinline__ static constexpr uint32_t hash_u32(uint32_t x) noexcept
        {
            x ^= x >> 16;
            x *= 0x7feb352dU;
            x ^= x >> 15;
            x *= 0x846ca68bU;
            x ^= x >> 16;
            return x;
        }

        // Uniform in [-1, 1)
        __device__ __forceinline__ static constexpr scalar_t uniform_signed(
            const uint32_t seed, const uint32_t idx, const uint32_t comp) noexcept
        {
            const uint32_t h = hash_u32(seed ^ (idx * 0x9E3779B9U) ^ (comp * 0x85EBCA6BU));
            return static_cast<scalar_t>(h >> 8) * (2.0f / 16777216.0f) - static_cast<scalar_t>(1);
        }

        /**
         * @brief Initialises the block halo for a single lattice block.
         *
         * @param[in] devPtrs Device pointer collection containing the moment arrays.
         * @param[in] haloBuffer Pointer collection holding the halo buffers for all six faces.
         *
         * @details Loads the local moments for the current lattice point, synchronises the
         * block, and writes the reconstructed population data to the halo storage.
         **/
        __device__ static inline void haloInitialisation(
            const device::ptrCollection<NUMBER_MOMENTS<host::label_t>(), scalar_t> &devPtrs,
            const device::ptrCollection<12, scalar_t> &haloBuffer,
            const bool firstTimeStep) noexcept
        {
            const device::ptrCollection<6, scalar_t> readBuffer(
                haloBuffer.ptr<0>(), haloBuffer.ptr<1>(), haloBuffer.ptr<2>(),
                haloBuffer.ptr<3>(), haloBuffer.ptr<4>(), haloBuffer.ptr<5>());

            const device::ptrCollection<6, scalar_t> writeBuffer(
                haloBuffer.ptr<6>(), haloBuffer.ptr<7>(), haloBuffer.ptr<8>(),
                haloBuffer.ptr<9>(), haloBuffer.ptr<10>(), haloBuffer.ptr<11>());

            const thread::coordinate Tx;

            const block::coordinate Bx;

            const device::pointCoordinate point(Tx, Bx);

            // Index into global arrays
            const device::label_t idx = device::idx(Tx, Bx);

            // Into block arrays
            const device::label_t tid = block::idx(Tx);

            // Always a multiple of 32, so no need to check this(I think)
            if (device::out_of_bounds(point))
            {
                return;
            }

            // constexpr const scalar_t u0 = static_cast<scalar_t>(0.1 / math::constexpr_sqrt<double, static_cast<double>(3)>());

            // const scalar_t fx = cuda::std::sin(point.value<axis::X>()) * u0;
            // const scalar_t fy = cuda::std::sin(point.value<axis::Y>()) * u0;
            // const scalar_t fz = cuda::std::sin(point.value<axis::Z>()) * u0;

            // Coalesced read from global memory
            momentsArray moments{
                rho0(),
                static_cast<scalar_t>(0),
                static_cast<scalar_t>(0),
                static_cast<scalar_t>(0),
                static_cast<scalar_t>(0),
                static_cast<scalar_t>(0),
                static_cast<scalar_t>(0),
                static_cast<scalar_t>(0),
                static_cast<scalar_t>(0),
                static_cast<scalar_t>(0)};

            // momentsArray moments{
            //     static_cast<scalar_t>(0),
            //     static_cast<scalar_t>(0),
            //     static_cast<scalar_t>(0),
            //     static_cast<scalar_t>(0),
            //     static_cast<scalar_t>(0),
            //     static_cast<scalar_t>(0),
            //     static_cast<scalar_t>(0),
            //     static_cast<scalar_t>(0),
            //     static_cast<scalar_t>(0),
            //     static_cast<scalar_t>(0)};

            // if (firstTimeStep)
            // {
            //     constexpr uint32_t SEED = 0xDEADBEEFu;
            //     constexpr scalar_t A = static_cast<scalar_t>(0.1);

            //     const uint32_t i = static_cast<uint32_t>(idx);

            //     moments[1] = (A * velocitySetBase::cs2<scalar_t>()) * uniform_signed(SEED, i, 0);                                            // ux
            //     moments[2] = (A * velocitySetBase::cs2<scalar_t>()) * uniform_signed(SEED, i, 1);                                            // uy
            //     moments[3] = (A * velocitySetBase::cs2<scalar_t>()) + ((A * velocitySetBase::cs2<scalar_t>()) * uniform_signed(SEED, i, 2)); // uz
            //     // Second moments at equilibrium: m_ab = rho * u_a * u_b
            //     moments[4] = moments[1] * moments[1]; // mxx
            //     moments[5] = moments[1] * moments[2]; // mxy
            //     moments[6] = moments[1] * moments[3]; // mxz
            //     moments[7] = moments[2] * moments[2]; // myy
            //     moments[8] = moments[2] * moments[3]; // myz
            //     moments[9] = moments[3] * moments[3]; // mzz
            // }

            if (!firstTimeStep)
            {
                device::constexpr_for<0, NUMBER_MOMENTS()>(
                    [&](const auto moment)
                    {
                        kernel::readFromPtr<moment>(devPtrs, moments, idx);
                    });

                // Add some random noise to the velocity field
                if constexpr (false)
                {
                    constexpr uint32_t SEED = 0xDEADBEEFu;
                    constexpr scalar_t A = static_cast<scalar_t>(0.1);

                    const uint32_t i = static_cast<uint32_t>(idx);

                    const scalar_t ux = (A * velocitySetBase::cs2<scalar_t>()) * uniform_signed(SEED, i, 0);
                    const scalar_t uy = (A * velocitySetBase::cs2<scalar_t>()) * uniform_signed(SEED, i, 1);
                    const scalar_t uz = (A * velocitySetBase::cs2<scalar_t>()) * uniform_signed(SEED, i, 2);

                    moments[1] = moments[1] + ux;
                    moments[2] = moments[2] + uy;
                    moments[3] = moments[3] + uz;
                }
            }
            else
            {
                __shared__ blockSharedBuffer sharedBuffer;

                // Update the post-streaming moments according to the interior and/or boundary conditions
                if constexpr (BoundaryConditions::appliesCondition())
                {
                    using NormalVector = normalVector<var3<bool>(BoundaryConditions::template periodic<axis::X>(), BoundaryConditions::template periodic<axis::Y>(), BoundaryConditions::template periodic<axis::Z>())>;
                    const NormalVector boundaryNormal(point);
                    BoundaryConditions::Base::update_smem(moments, sharedBuffer, tid);
                    BoundaryConditions::template calculate_moments<VelocitySet>(moments, sharedBuffer, Tx, point, tid, boundaryNormal);
                }
            }

            block::sync();

            // velocitySetBase::scale(moments);

            // Temp weird fix
            // moments[6] = -moments[6];
            // moments[8] = -moments[8];

            // Save the halo
            saveHalo<BoundaryConditions::template periodic<axis::X>(), BoundaryConditions::template periodic<axis::Y>(), BoundaryConditions::template periodic<axis::Z>()>(moments, readBuffer, writeBuffer, Tx, Bx, point);

            device::constexpr_for<0, NUMBER_MOMENTS()>(
                [&](const auto moment)
                {
                    kernel::saveToPtr<moment>(devPtrs, moments, idx);
                });
        }
    };

    namespace momentBasedLBM
    {
        /**
         * @brief Launches the initialisation kernel.
         * @param[in] devPtrs Device pointer collection containing the moment arrays.
         * @param[in] haloBuffer Pointer collection holding the halo buffers for the block.
         * @param[in] firstTimeStep True if initialising the first time step, false otherwise
         **/
        template <class VelocitySet>
        __launch_bounds__(block::maxThreads(), 1) __global__ static void initialise(
            const device::ptrCollection<NUMBER_MOMENTS<host::label_t>(), scalar_t> devPtrs,
            const device::ptrCollection<12, scalar_t> haloBuffer,
            const bool firstTimeStep)
        {
            blockHaloInitialisationKernel<VelocitySet, BoundaryConditionCase>::haloInitialisation(devPtrs, haloBuffer, firstTimeStep);
        }

        /**
         * @brief Helper struct to return a pointer to the initialisation kernel
         * @tparam VelocitySet The velocity set
         **/
        template <class VelocitySet>
        struct initialisation
        {
            hostFunctionReturnType inline static consteval auto kernel() noexcept
            {
                return initialise<VelocitySet>;
            }
        };
    }
}

#endif