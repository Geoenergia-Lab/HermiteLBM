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
    Definition of the main GPU kernel

Namespace
    LBM::host, LBM::device

SourceFiles
    kernel.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_MOMENTBASEDLBM_KERNEL_CUH
#define __MBLBM_MOMENTBASEDLBM_KERNEL_CUH

#include "deviceCommunicator.cuh"
#include "launchConfig.cuh"
#include "ptrCollection.cuh"

namespace LBM
{
    namespace detail
    {
        using Streaming = streaming<VelocitySet>;

        /**
         * @brief Implements solution of the lattice Boltzmann method using the moment representation and a chosen velocity set
         * @tparam BoundaryConditions The boundary conditions of the solver
         * @tparam VelocitySet The velocity set to use for streaming
         * @tparam Collision The collision model
         * @tparam BlockHalo The class handling inter-block streaming
         * @param[in] devPtrs Collection of 10 pointers to device arrays on the GPU
         * @param[in] readBuffer Collection of read-only pointers to the block halo faces used during streaming
         * @param[in] writeBuffer Collection of mutable pointers to the block halo faces used after streaming
         * @param[in] sharedBuffer Inline or externally stored shared memory buffer
         **/
        template <class BoundaryConditions, class VelocitySet, class Collision, class BlockHalo, class SharedBuffer>
        __device__ inline void momentBasedLBM(
            const device::ptrCollection<NUMBER_MOMENTS<host::label_t>(), scalar_t> &devPtrs,
            const device::ptrCollection<6, const scalar_t> &readBuffer,
            const device::ptrCollection<6, scalar_t> &writeBuffer,
            SharedBuffer &sharedBuffer,
            const device::label_t bzOffset)
        {
            static_assert(std::is_same_v<BlockHalo, device::halo<VelocitySet, BoundaryConditions::periodicX(), BoundaryConditions::periodicY(), BoundaryConditions::periodicZ()>>);

            const thread::coordinate Tx;

            const block::coordinate Bx(blockIdx.x, blockIdx.y, blockIdx.z + bzOffset);

            const device::pointCoordinate point(Tx, Bx);

            // Index into global arrays
            const device::label_t idx = device::idx(Tx, Bx);

            // Into block arrays
            const device::label_t tid = block::idx(Tx);

            // Always a multiple of 32, so no need to check this(I think)
            if constexpr (out_of_bounds_check())
            {
                if (device::out_of_bounds(point))
                {
                    return;
                }
            }

            // Prefetch devPtrs into L2
            device::constexpr_for<0, NUMBER_MOMENTS()>(
                [&](const auto moment)
                {
                    cache::prefetch<cache::Level::L2, cache::Policy::evict_last>(&(devPtrs.ptr<moment>()[idx]));
                });

            // Coalesced read from global memory
            momentsArray moments;
            device::constexpr_for<0, NUMBER_MOMENTS()>(
                [&](const auto moment)
                {
                    const device::label_t ID = tid * m_i<NUMBER_MOMENTS() + 1>() + m_i<moment>();
                    sharedBuffer[ID] = devPtrs.ptr<moment>()[idx];
                    if constexpr (moment == index::rho)
                    {
                        moments[moment] = sharedBuffer[ID] + rho0();
                    }
                    else
                    {
                        moments[moment] = sharedBuffer[ID];
                    }
                });

            block::sync();

            // Reconstruct the population from the moments
            thread::array<scalar_t, VelocitySet::Q()> pop = VelocitySet::reconstruct(moments);

            // Save/pull from shared memory
            {
                // Save populations in shared memory
                Streaming::save(pop, sharedBuffer, tid);

                block::sync();

                // Pull from shared memory
                Streaming::pull(pop, sharedBuffer, Tx);

                // Pull pop from global memory in cover nodes
                BlockHalo::pull(pop, readBuffer, Tx, Bx, point);

                block::sync();
            }

            // Update the post-streaming moments according to the interior and/or boundary conditions
            if constexpr (BoundaryConditions::appliesCondition())
            {
                BoundaryConditions::template calculate_moments<VelocitySet>(pop, moments, sharedBuffer, Tx, point, tid);
            }
            else
            {
                VelocitySet::template calculate_moments(pop, moments);
            }

            // Scale the moments correctly
            velocitySetBase::scale(moments);

            // Collide
            Collision::collide(moments);

            // Coalesced write to global memory
            device::constexpr_for<0, NUMBER_MOMENTS()>(
                [&](const auto moment)
                {
                    if constexpr (moment == index::rho)
                    {
                        devPtrs.ptr<moment>()[idx] = moments[moment] - rho0();
                    }
                    else
                    {
                        devPtrs.ptr<moment>()[idx] = moments[moment];
                    }
                });

            // Save the populations to the block halo
            if constexpr (use_cooperative_halo())
            {
                VelocitySet::reconstruct<false>(pop, moments);
                BlockHalo::transpose_to_shared(pop, writeBuffer, sharedBuffer, Tx, Bx, point);
                BlockHalo::save_from_shared(sharedBuffer, writeBuffer, Tx, Bx);
            }
            else
            {
                BlockHalo::save(pop, moments, writeBuffer, Tx, Bx, point);
            }
        }
    }

    namespace kernel
    {
        /**
         * @brief Implements solution of the lattice Boltzmann method using the moment representation and the D3Q19 velocity set
         * @param[in] devPtrs Collection of 10 pointers to device arrays on the GPU
         * @param[in] readBuffer Collection of read-only pointers to the block halo faces used during streaming
         * @param[in] writeBuffer Collection of mutable pointers to the block halo faces used after streaming
         **/
        __launch_bounds__(block::maxThreads(), MIN_BLOCKS_PER_MP<VelocitySet>()) __global__ void momentBasedLBM(
            const device::ptrCollection<NUMBER_MOMENTS<host::label_t>(), scalar_t> devPtrs,
            const device::ptrCollection<6, const scalar_t> readBuffer,
            const device::ptrCollection<6, scalar_t> writeBuffer,
            const device::label_t bzOffset)
        {
            if constexpr (VelocitySet::smem_alloc_size() == 0)
            {
                __shared__ thread::array<scalar_t, block::sharedMemoryBufferSize<VelocitySet::Q(), NUMBER_MOMENTS<host::label_t>()>()> sharedBuffer;

                detail::momentBasedLBM<BoundaryConditions, VelocitySet, Collision, BlockHalo>(devPtrs, readBuffer, writeBuffer, sharedBuffer, bzOffset);
            }
            else
            {
                extern __shared__ scalar_t sharedBuffer[];

                detail::momentBasedLBM<BoundaryConditions, VelocitySet, Collision, BlockHalo>(devPtrs, readBuffer, writeBuffer, sharedBuffer, bzOffset);
            }
        }

        template <const host::label_t N>
        __host__ void launchHelper(
            const host::latticeMesh &mesh,
            const programControl &programCtrl,
            const ptrCollection &devPtrs,
            const haloBuffer<VelocitySet> &haloPtrs,
            const host::label_t timeStep,
            const std::array<host::label_t, N> &idxStreams,
            const std::array<device::label_t, N> &bzOffsets) noexcept
        {
            // Pre-sync and launch the kernels
            for (host::label_t deviceIdx = 0; deviceIdx < programCtrl.deviceList().size(); deviceIdx++)
            {
                // Set the active device
                errorHandler::checkInline(cudaSetDevice(programCtrl.deviceList()[deviceIdx]));

                // Sync the streams to ensure previous operations are complete before launching new kernels
                for (const host::label_t idxStream : idxStreams)
                {
                    programCtrl.streams().synchronize(device::idxStream(deviceIdx, idxStream));
                }

                // Launch the kernels for the specified streams and block offsets
                for (host::label_t idxStream = 0; idxStream < idxStreams.size(); idxStream++)
                {
                    kernel::launch<momentBasedLBM, VelocitySet::smem_alloc_size()>(
                        mesh.gridBlock()[device::idxStream(deviceIdx, idxStreams[idxStream])],
                        programCtrl.streams()[GPU::internalStreamID(deviceIdx)],
                        devPtrs[deviceIdx],
                        haloPtrs.readBuffer(deviceIdx, timeStep),
                        haloPtrs.writeBuffer(deviceIdx, timeStep),
                        bzOffsets[idxStream]);
                }
            }

            // Sync the streams
            for (host::label_t deviceIdx = 0; deviceIdx < programCtrl.deviceList().size(); deviceIdx++)
            {
                for (const host::label_t idxStream : idxStreams)
                {
                    programCtrl.streams().synchronize(device::idxStream(deviceIdx, idxStream));
                }
            }
        }

        /**
         * @brief Launches the lattice Boltzmann kernel for all devices and streams, ensuring proper synchronization and device selection
         * @param[in] mesh Lattice mesh object containing information about the grid and block dimensions
         * @param[in] programCtrl Program control object containing information about the devices and streams
         * @param[in] devPtrs Collection of pointers to device arrays on the GPU, used to pass the data to the kernel
         * @param[in] haloPtrs Collection of pointers to the block halo faces used during streaming
         * @param[in] timeStep Current time step of the simulation, used to determine which halo buffers to use for reading and writing
         **/
        __host__ inline void launchInternal(
            const host::latticeMesh &mesh,
            const programControl &programCtrl,
            const ptrCollection &devPtrs,
            const haloBuffer<VelocitySet> &haloPtrs,
            const host::label_t timeStep) noexcept
        {
            constexpr const std::array<host::label_t, 1> idxStreams = {static_cast<device::label_t>(1)};
            constexpr const std::array<device::label_t, 1> bzOffsets = {static_cast<device::label_t>(1)};
            launchHelper(mesh, programCtrl, devPtrs, haloPtrs, timeStep, idxStreams, bzOffsets);
        }

        /**
         * @brief Launches the lattice Boltzmann kernel for all devices and streams, ensuring proper synchronization and device selection
         * @param[in] mesh Lattice mesh object containing information about the grid and block dimensions
         * @param[in] programCtrl Program control object containing information about the devices and streams
         * @param[in] devPtrs Collection of pointers to device arrays on the GPU, used to pass the data to the kernel
         * @param[in] haloPtrs Collection of pointers to the block halo faces used during streaming
         * @param[in] devComm Device communicator object used to handle inter-device communication of halo buffers
         * @param[in] timeStep Current time step of the simulation, used to determine which halo buffers to use for reading and writing
         **/
        __host__ inline void launchBoundary(
            const host::latticeMesh &mesh,
            const programControl &programCtrl,
            const ptrCollection &devPtrs,
            const haloBuffer<VelocitySet> &haloPtrs,
            const deviceCommunicator<VelocitySet> &devComm,
            const host::label_t timeStep) noexcept
        {
            constexpr const std::array<host::label_t, 2> idxStreams = {static_cast<device::label_t>(0), static_cast<device::label_t>(2)};
            const std::array<device::label_t, 2> bzOffsets = {static_cast<device::label_t>(0), static_cast<device::label_t>(mesh.blocksPerDevice<axis::Z>() - static_cast<host::label_t>(1))};
            launchHelper(mesh, programCtrl, devPtrs, haloPtrs, timeStep, idxStreams, bzOffsets);
            devComm.exchange(timeStep);
        }
    }
}

#endif