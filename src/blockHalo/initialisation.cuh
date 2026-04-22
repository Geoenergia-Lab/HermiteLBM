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

namespace LBM
{
    namespace detail
    {
        /**
         * @brief Saves the halo for the given velocity set and moments
         * @param[in] moments The moments of the distribution function at the current point
         **/
        template <typename VelocitySet>
        __device__ inline void saveHalo(
            const thread::array<scalar_t, NUMBER_MOMENTS()> &moments,
            const device::ptrCollection<6, scalar_t> &readBuffer,
            const device::ptrCollection<6, scalar_t> &writeBuffer,
            const thread::coordinate &Tx,
            const block::coordinate &Bx,
            const device::pointCoordinate &point)
        {
            thread::array<scalar_t, VelocitySet::Q()> pop = VelocitySet::reconstruct(moments);
            device::halo<VelocitySet, true, true, true>::save(pop, moments, readBuffer, Tx, Bx, point);
            device::halo<VelocitySet, true, true, true>::save(pop, moments, writeBuffer, Tx, Bx, point);
        }

        /**
         * @brief Implements solution of the lattice Boltzmann method using the moment representation and the D3Q19 velocity set
         * @param[in] devPtrs Collection of 10 pointers to device arrays on the GPU
         * @param[in] readBuffer Collection of 6 pointers to device arrays on the GPU used for the block halo
         * @param[in] writeBuffer Collection of 6 pointers to device arrays on the GPU used for the block halo
         * @param[in] Q The number of discrete velocities in the velocity set
         * @param[in] thermalModel The thermal model used in the simulation (thermal or isothermal)
         **/
        __device__ inline void haloInitialisation(
            const device::ptrCollection<NUMBER_MOMENTS<host::label_t>(), const scalar_t> &devPtrs,
            const device::ptrCollection<6, scalar_t> &readBuffer,
            const device::ptrCollection<6, scalar_t> &writeBuffer,
            const host::label_t Q,
            const thermalModel_t thermalModel)
        {
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
            thread::array<scalar_t, NUMBER_MOMENTS()> moments;
            device::constexpr_for<0, NUMBER_MOMENTS()>(
                [&](const auto moment)
                {
                    if constexpr (moment == index::rho)
                    {
                        moments[moment] = devPtrs.ptr<moment>()[idx] + rho0();
                    }
                    else
                    {
                        moments[moment] = devPtrs.ptr<moment>()[idx];
                    }
                });

            block::sync();

            // Save the halo
            if (Q == constants::D3Q19::Q())
            {
                if (thermalModel == thermalModel_t::Thermal)
                {
                    saveHalo<D3Q19<Thermal>>(moments, readBuffer, writeBuffer, Tx, Bx, point);
                }

                if (thermalModel == thermalModel_t::Isothermal)
                {
                    saveHalo<D3Q19<Isothermal>>(moments, readBuffer, writeBuffer, Tx, Bx, point);
                }
            }

            if (Q == constants::D3Q27::Q())
            {
                if (thermalModel == thermalModel_t::Thermal)
                {
                    saveHalo<D3Q27<Thermal>>(moments, readBuffer, writeBuffer, Tx, Bx, point);
                }

                if (thermalModel == thermalModel_t::Isothermal)
                {
                    saveHalo<D3Q27<Isothermal>>(moments, readBuffer, writeBuffer, Tx, Bx, point);
                }
            }
        }
    }

    namespace kernel
    {
        /**
         * @brief Implements solution of the lattice Boltzmann method using the moment representation and the D3Q19 velocity set
         * @param[in] devPtrs Collection of 10 pointers to device arrays on the GPU
         * @param[in] haloBuffer Collection of 12 pointers to device arrays on the GPU used for the block halo
         * @param[in] Q The number of discrete velocities in the velocity set
         * @param[in] thermalModel The thermal model used in the simulation (thermal or isothermal)
         **/
        __launch_bounds__(block::maxThreads(), 1)
            __global__ void momentBasedLBMInitialisation(
                const device::ptrCollection<NUMBER_MOMENTS<host::label_t>(), const scalar_t> devPtrs,
                const device::ptrCollection<12, scalar_t> haloBuffer,
                const host::label_t Q,
                const thermalModel_t thermalModel)
        {
            const device::ptrCollection<6, scalar_t> readBuffer(
                haloBuffer.ptr<0>(), haloBuffer.ptr<1>(), haloBuffer.ptr<2>(),
                haloBuffer.ptr<3>(), haloBuffer.ptr<4>(), haloBuffer.ptr<5>());

            const device::ptrCollection<6, scalar_t> writeBuffer(
                haloBuffer.ptr<6>(), haloBuffer.ptr<7>(), haloBuffer.ptr<8>(),
                haloBuffer.ptr<9>(), haloBuffer.ptr<10>(), haloBuffer.ptr<11>());

            detail::haloInitialisation(devPtrs, readBuffer, writeBuffer, Q, thermalModel);
        }
    }
}

#endif