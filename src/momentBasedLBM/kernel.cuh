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

namespace LBM
{
    namespace detail
    {
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
            SharedBuffer &sharedBuffer)
        {
            static_assert(std::is_same_v<BlockHalo, device::halo<VelocitySet, BoundaryConditions::periodicX(), BoundaryConditions::periodicY(), BoundaryConditions::periodicZ()>>);

            const thread::coordinate Tx;

            const block::coordinate Bx;

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
            thread::array<scalar_t, NUMBER_MOMENTS()> moments;
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
                streaming::save<VelocitySet>(pop, sharedBuffer, tid);

                block::sync();

                // Pull from shared memory
                streaming::pull<VelocitySet>(pop, sharedBuffer, Tx);

                // Pull pop from global memory in cover nodes
                BlockHalo::pull(pop, readBuffer, Tx, Bx, point);

                block::sync();
            }

            if constexpr (std::is_same_v<BoundaryConditions, lidDrivenCavity>)
            {
                // Calculate the moments either at the boundary or interior
                {
                    const normalVector boundaryNormal(point);

                    velocitySet::calculate_moments<VelocitySet>(pop, moments, boundaryNormal);

                    if (boundaryNormal.isBoundary())
                    {
                        BoundaryConditions::template calculate_moments<VelocitySet>(pop, moments, boundaryNormal, sharedBuffer, Tx, point);
                    }
                }
            }

            if constexpr (std::is_same_v<BoundaryConditions, jetFlow>)
            {
                // Compute post-stream moments
                velocitySet::calculate_moments<VelocitySet>(pop, moments);
                {
                    // Update the shared buffer with the refreshed moments
                    device::constexpr_for<0, NUMBER_MOMENTS()>(
                        [&](const auto moment)
                        {
                            const device::label_t ID = tid * label_constant<NUMBER_MOMENTS() + 1>() + label_constant<moment>();
                            sharedBuffer[ID] = moments[moment];
                        });
                }

                block::sync();

                // Calculate the moments at the boundary
                {
                    const normalVector boundaryNormal(point);

                    if (boundaryNormal.isBoundary())
                    {
                        BoundaryConditions::template calculate_moments<VelocitySet>(pop, moments, boundaryNormal, sharedBuffer, Tx, point);
                    }
                }
            }

            // Scale the moments correctly
            velocitySet::scale(moments);

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
            if constexpr (use_cooperative_halo() && ((std::is_same_v<VelocitySet, D3Q19<Thermal>>) || (std::is_same_v<VelocitySet, D3Q19<Isothermal>>)))
            {
                BlockHalo::transpose_to_shared(pop, sharedBuffer, Tx, point);
                BlockHalo::save_from_shared(sharedBuffer, writeBuffer);
            }
            else
            {
                BlockHalo::save(pop, moments, writeBuffer, Tx, Bx, point);
            }
        }
    }

    namespace kernel
    {
        class ptrCollection
        {
        public:
            /**
             * @brief Alias for the collection of pointers to device arrays on the GPU, used to pass the data to the kernel
             **/
            using CollectionType = device::ptrCollection<NUMBER_MOMENTS<host::label_t>(), scalar_t>;
            using Type = std::vector<CollectionType>;

            /**
             * @brief Constructor for the collection of pointers to device arrays on the GPU, used to pass the data to the kernel
             * @param[in] rho Device scalar field for density
             * @param[in] U Device vector field for velocity
             * @param[in] Pi Device symmetric tensor field for the second-order moments
             * @param[in] programCtrl Program control object containing information about the devices and streams
             **/
            __host__ [[nodiscard]] ptrCollection(
                const device::scalarField<VelocitySet, time::instantaneous> &rho,
                const device::vectorField<VelocitySet, time::instantaneous> &U,
                const device::symmetricTensorField<VelocitySet, time::instantaneous> &Pi,
                const programControl &programCtrl) noexcept
                : devPtrs_(initialisePtrs(rho, U, Pi, programCtrl)) {}

            /**
             * @brief Access operator for the collection of pointers to device arrays on the GPU, used to pass the data to the kernel
             * @param[in] index Index of the device/stream to access
             * @return Collection of pointers to device arrays for the specified device/stream
             **/
            __host__ [[nodiscard]] inline constexpr const CollectionType &operator[](const host::label_t index) const noexcept
            {
                return devPtrs_[index];
            }

        private:
            /**
             * @brief Collection of pointers to device arrays on the GPU, used to pass the data to the kernel
             **/
            const Type devPtrs_;

            /**
             * @brief Initializes the collection of pointers to device arrays on the GPU, used to pass the data to the kernel
             * @param[in] rho Device scalar field for density
             * @param[in] U Device vector field for velocity
             * @param[in] Pi Device symmetric tensor field for the second-order moments
             * @param[in] programCtrl Program control object containing information about the devices and streams
             * @return Collection of pointers to device arrays for all devices/streams
             **/
            __host__ [[nodiscard]] static const Type initialisePtrs(
                const device::scalarField<VelocitySet, time::instantaneous> &rho,
                const device::vectorField<VelocitySet, time::instantaneous> &U,
                const device::symmetricTensorField<VelocitySet, time::instantaneous> &Pi,
                const programControl &programCtrl)
            {
                Type ptrs;

                for (host::label_t stream = 0; stream < programCtrl.deviceList().size(); stream++)
                {
                    errorHandler::checkInline(cudaSetDevice(programCtrl.deviceList()[stream]));
                    errorHandler::checkInline(cudaDeviceSynchronize());
                    programCtrl.streams().synchronize(stream);

                    ptrs.emplace_back(
                        device::ptrCollection<NUMBER_MOMENTS<host::label_t>(), scalar_t>(
                            rho.self().mutPtr(stream),
                            U.x().mutPtr(stream),
                            U.y().mutPtr(stream),
                            U.z().mutPtr(stream),
                            Pi.xx().mutPtr(stream),
                            Pi.xy().mutPtr(stream),
                            Pi.xz().mutPtr(stream),
                            Pi.yy().mutPtr(stream),
                            Pi.yz().mutPtr(stream),
                            Pi.zz().mutPtr(stream)));
                }

                return ptrs;
            }
        };

        /**
         * @brief Implements solution of the lattice Boltzmann method using the moment representation and the D3Q19 velocity set
         * @param[in] devPtrs Collection of 10 pointers to device arrays on the GPU
         * @param[in] readBuffer Collection of read-only pointers to the block halo faces used during streaming
         * @param[in] writeBuffer Collection of mutable pointers to the block halo faces used after streaming
         **/
        __launch_bounds__(block::maxThreads(), MIN_BLOCKS_PER_MP<VelocitySet>())
            __global__ void momentBasedLBM(
                const device::ptrCollection<NUMBER_MOMENTS<host::label_t>(), scalar_t> devPtrs,
                const device::ptrCollection<6, const scalar_t> readBuffer,
                const device::ptrCollection<6, scalar_t> writeBuffer)
        {
            if constexpr ((std::is_same_v<VelocitySet, D3Q19<Thermal>>) || (std::is_same_v<VelocitySet, D3Q19<Isothermal>>))
            {
                __shared__ thread::array<scalar_t, block::sharedMemoryBufferSize<VelocitySet, NUMBER_MOMENTS<host::label_t>()>()> shared_buffer;

                detail::momentBasedLBM<BoundaryConditions, VelocitySet, Collision, BlockHalo>(devPtrs, readBuffer, writeBuffer, shared_buffer);
            }
            else
            {
                extern __shared__ scalar_t shared_buffer[];

                detail::momentBasedLBM<BoundaryConditions, VelocitySet, Collision, BlockHalo>(devPtrs, readBuffer, writeBuffer, shared_buffer);
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
        __host__ inline void launch(
            const host::latticeMesh &mesh,
            const programControl &programCtrl,
            const ptrCollection &devPtrs,
            const haloBuffer<VelocitySet> &haloPtrs,
            const host::label_t timeStep) noexcept
        {
            for (host::label_t stream = 0; stream < programCtrl.deviceList().size(); stream++)
            {
                errorHandler::checkInline(cudaSetDevice(programCtrl.deviceList()[stream]));
                programCtrl.streams().synchronize(stream);

                kernel::momentBasedLBM<<<mesh.gridBlock(), mesh.threadBlock(), smem_alloc_size<VelocitySet>(), programCtrl.streams()[stream]>>>(
                    devPtrs[stream],
                    haloPtrs.readBuffer(stream, timeStep),
                    haloPtrs.writeBuffer(stream, timeStep));
            }

            programCtrl.allsync();
        }
    }
}

#endif