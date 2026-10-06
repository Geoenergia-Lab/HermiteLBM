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
        __device__ static inline void saveHalo(
            const momentsArray &moments,
            const device::ptrCollection<6, scalar_t> &readBuffer,
            const device::ptrCollection<6, scalar_t> &writeBuffer,
            const thread::coordinate &Tx,
            const block::coordinate &Bx,
            const device::pointCoordinate &point) noexcept
        {
            device::halo<VelocitySet, boundaryConditionType<true, true, true>>::save(moments, readBuffer, Tx, Bx, point);
            device::halo<VelocitySet, boundaryConditionType<true, true, true>>::save(moments, writeBuffer, Tx, Bx, point);
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

            if (!firstTimeStep)
            {
                device::constexpr_for<0, NUMBER_MOMENTS()>(
                    [&](const auto moment)
                    {
                        kernel::readFromPtr<moment>(devPtrs, moments, idx);
                    });
            }

            block::sync();

            // Reconstruct the population from the moments
            thread::array<scalar_t, VelocitySet::Q()> pop;
            VelocitySet::reconstruct(pop, moments);

            __shared__ blockSharedBuffer sharedBuffer;

            // Update the post-streaming moments according to the interior and/or boundary conditions
            if constexpr (BoundaryConditions::appliesCondition())
            {
                using NormalVector = normalVector<var3<bool>(BoundaryConditions::template periodic<axis::X>(), BoundaryConditions::template periodic<axis::Y>(), BoundaryConditions::template periodic<axis::Z>())>;
                const NormalVector boundaryNormal(point);
                VelocitySet::template calculate_moments(moments, pop, boundaryNormal);
                BoundaryConditions::template calculate_moments<VelocitySet>(moments, sharedBuffer, Tx, point, tid, boundaryNormal);
            }
            else
            {
                VelocitySet::template calculate_moments(moments, pop);
            }

            velocitySetBase::scale(moments);

            // Save the halo
            saveHalo(moments, readBuffer, writeBuffer, Tx, Bx, point);

            device::constexpr_for<0, NUMBER_MOMENTS()>(
                [&](const auto moment)
                {
                    kernel::saveToPtr<moment>(devPtrs, moments, idx);
                });
        }
    };

    namespace kernel
    {
        /**
         * @brief Launches the D3Q19 thermal halo initialisation kernel.
         *
         * @param[in] devPtrs Device pointer collection containing the moment arrays.
         * @param[in] haloBuffer Pointer collection holding the halo buffers for the block.
         **/
        __launch_bounds__(block::maxThreads(), 1) __global__ void momentBasedLBMInitialisationD3Q19Thermal(
            const device::ptrCollection<NUMBER_MOMENTS<host::label_t>(), scalar_t> devPtrs,
            const device::ptrCollection<12, scalar_t> haloBuffer,
            const bool firstTimeStep)
        {
            blockHaloInitialisationKernel<D3Q19<Thermal>, BoundaryConditionCase>::haloInitialisation(devPtrs, haloBuffer, firstTimeStep);
        }

        /**
         * @brief Launches the D3Q19 isothermal halo initialisation kernel.
         *
         * @param[in] devPtrs Device pointer collection containing the moment arrays.
         * @param[in] haloBuffer Pointer collection holding the halo buffers for the block.
         **/
        __launch_bounds__(block::maxThreads(), 1) __global__ void momentBasedLBMInitialisationD3Q19Isothermal(
            const device::ptrCollection<NUMBER_MOMENTS<host::label_t>(), scalar_t> devPtrs,
            const device::ptrCollection<12, scalar_t> haloBuffer,
            const bool firstTimeStep)
        {
            blockHaloInitialisationKernel<D3Q19<Isothermal>, BoundaryConditionCase>::haloInitialisation(devPtrs, haloBuffer, firstTimeStep);
        }

        /**
         * @brief Launches the D3Q27 thermal halo initialisation kernel.
         *
         * @param[in] devPtrs Device pointer collection containing the moment arrays.
         * @param[in] haloBuffer Pointer collection holding the halo buffers for the block.
         **/
        __launch_bounds__(block::maxThreads(), 1) __global__ void momentBasedLBMInitialisationD3Q27Thermal(
            const device::ptrCollection<NUMBER_MOMENTS<host::label_t>(), scalar_t> devPtrs,
            const device::ptrCollection<12, scalar_t> haloBuffer,
            const bool firstTimeStep)
        {
            blockHaloInitialisationKernel<D3Q27<Thermal>, BoundaryConditionCase>::haloInitialisation(devPtrs, haloBuffer, firstTimeStep);
        }

        /**
         * @brief Launches the D3Q27 isothermal halo initialisation kernel.
         *
         * @param[in] devPtrs Device pointer collection containing the moment arrays.
         * @param[in] haloBuffer Pointer collection holding the halo buffers for the block.
         **/
        __launch_bounds__(block::maxThreads(), 1) __global__ void momentBasedLBMInitialisationD3Q27Isothermal(
            const device::ptrCollection<NUMBER_MOMENTS<host::label_t>(), scalar_t> devPtrs,
            const device::ptrCollection<12, scalar_t> haloBuffer,
            const bool firstTimeStep)
        {
            blockHaloInitialisationKernel<D3Q27<Isothermal>, BoundaryConditionCase>::haloInitialisation(devPtrs, haloBuffer, firstTimeStep);
        }

        /**
         * @brief Returns the correct halo-initialisation kernel for the requested velocity set.
         *
         * @tparam VelocitySet Velocity set type used by the current simulation.
         * @return Function pointer to the corresponding kernel launch entry point.
         *
         * @details Selects the appropriate D3Q19/D3Q27 thermal or isothermal initialisation
         * kernel based on the compile-time velocity set type.
         **/
        template <class VelocitySet>
        __host__ inline consteval auto momentBasedLBMInitialisation() noexcept
        {
            if constexpr (std::is_same_v<VelocitySet, D3Q19<Thermal>>)
            {
                return momentBasedLBMInitialisationD3Q19Thermal;
            }

            if constexpr (std::is_same_v<VelocitySet, D3Q19<Isothermal>>)
            {
                return momentBasedLBMInitialisationD3Q19Isothermal;
            }

            if constexpr (std::is_same_v<VelocitySet, D3Q27<Thermal>>)
            {
                return momentBasedLBMInitialisationD3Q27Thermal;
            }

            if constexpr (std::is_same_v<VelocitySet, D3Q27<Isothermal>>)
            {
                return momentBasedLBMInitialisationD3Q27Isothermal;
            }
        }
    }
}

#endif