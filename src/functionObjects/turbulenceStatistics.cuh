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
Authors: Gustavo Choiare (Geoenergia Lab, UDESC)

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
    File containing kernels and class definitions for the turbulence statistics

Namespace
    LBM::functionObjects

SourceFiles
    turbulenceStatistics.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_TURBULENCESTATISTICS_CUH
#define __MBLBM_TURBULENCESTATISTICS_CUH

#include "../momentBasedLBM/ptrCollection.cuh"

namespace LBM
{
    __device__ __host__ [[nodiscard]] inline constexpr const scalar calculate_P(const symmetricTensor &R, const symmetricTensor &S) noexcept
    {
        const symmetricTensor RdotS = R * S;
        return {RdotS[0] + RdotS[1] + RdotS[1] + RdotS[2] + RdotS[2] + RdotS[3] + RdotS[4] + RdotS[4] + RdotS[5]};
    }

    __launch_bounds__(block::maxThreads(), 1) __global__ void turbulenceStatisticsCalculate(
        const device::ptrCollection<NUMBER_MOMENTS<host::label_t>(), scalar_t> devPtrs, // Pointers to the moments
        const device::ptrCollection<6, scalar_t> RPtrs,                                 // Reynolds stress tensor
        const device::ptrCollection<1, scalar_t> PPtrs,                                 // Production term
        const device::ptrCollection<1, scalar_t> epsilonPtrs,                           // Dissipation term
        const device::ptrCollection<1, scalar_t> kPtrs,                                 // Turbulent kinetic energy
        const device::ptrCollection<3, scalar_t> UMeanPtrs,                             // Mean velocity field
        const device::ptrCollection<6, scalar_t> SMeanPtrs,                             // Mean strain rate tensor
        const scalar_t invNewCount)
    {
        // Get the index
        const device::label_t idx = device::idx(thread::coordinate(), block::coordinate());

        // Read the data from memory
        const vector U = functionObjects::read_from_moments<1, 2, 3>(devPtrs, idx);
        const vector UMean = functionObjects::read<3>(UMeanPtrs, idx);
        const symmetricTensor S = functionObjects::S::calculate(devPtrs, idx);
        const symmetricTensor SMean = functionObjects::read<6>(SMeanPtrs, idx);

        // Read the turbulence fields from memory
        const symmetricTensor R = functionObjects::read<6>(RPtrs, idx);    // Reynolds stress tensor
        const scalar P = functionObjects::read<1>(PPtrs, idx);             // Production term
        const scalar epsilon = functionObjects::read<1>(epsilonPtrs, idx); // Dissipation term
        const scalar k = functionObjects::read<1>(kPtrs, idx);             // Turbulent kinetic energy

        // Calculate the perturbation quantities
        const vector UPrime = U - UMean;          // Fluctuation of velocity
        const symmetricTensor SPrime = S - SMean; // Fluctuation of strain rate tensor

        // Update the Reynolds stress tensor and production term using time averaging
        const symmetricTensor RNew = {UPrime[0] * UPrime[0], UPrime[0] * UPrime[1], UPrime[0] * UPrime[2], UPrime[1] * UPrime[1], UPrime[1] * UPrime[2], UPrime[2] * UPrime[2]};
        const symmetricTensor RMean = functionObjects::time_average(R, RNew, invNewCount);
        const scalar PMean = calculate_P(RMean, SMean);

        // Update the dissipation term using time averaging
        const scalar epsilonNew = {static_cast<scalar_t>(2) * device::nu * ((SPrime[0] * SPrime[0]) + (SPrime[1] * SPrime[1]) + (SPrime[2] * SPrime[2]) + static_cast<scalar_t>(2) * ((SPrime[3] * SPrime[3]) + (SPrime[4] * SPrime[4]) + (SPrime[5] * SPrime[5])))};
        const scalar epsilonMean = functionObjects::time_average(epsilon, epsilonNew, invNewCount);

        // Update the turbulent kinetic energy using time averaging
        const scalar kNew = {static_cast<scalar_t>(0.5) * ((UPrime[0] * UPrime[0]) + (UPrime[1] * UPrime[1]) + (UPrime[2] * UPrime[2]))};
        const scalar kMean = functionObjects::time_average(k, kNew, invNewCount);

        // Save the updated turbulence fields to memory
        functionObjects::save(RMean, RPtrs, idx);
        functionObjects::save(PMean, PPtrs, idx);
        functionObjects::save(epsilonMean, epsilonPtrs, idx);
        functionObjects::save(kMean, kPtrs, idx);
    }

    template <class VelocitySet>
    class turbulenceStatistics
    {
    public:
        __host__ [[nodiscard]] turbulenceStatistics(
            const kernel::ptrCollection &devPtrs,
            const host::latticeMesh &mesh,
            const programControl &programCtrl,
            host::array<host::PINNED, scalar_t, VelocitySet> &hostWriteBuffer)
            : devPtrs_(devPtrs),
              calculate_(functionObjects::initialiserSwitch("turbulenceStatistics")),
              programCtrl_(programCtrl),
              mesh_(mesh),
              UMean_("UMean", mesh, programCtrl, calculate_),
              SMean_("SMean", mesh, programCtrl, calculate_),
              R_("R", mesh, zeros<scalar_t, 6>(), programCtrl, calculate_),
              P_("P", mesh, zeros<scalar_t, 1>(), programCtrl, calculate_),
              epsilon_("epsilon", mesh, zeros<scalar_t, 1>(), programCtrl, calculate_),
              k_("k", mesh, zeros<scalar_t, 1>(), programCtrl, calculate_),
              hostWriteBuffer_(hostWriteBuffer)
        {
        }

        ~turbulenceStatistics() {}

        __host__ inline void calculate()
        {
            if (calculate_)
            {
                const scalar_t invNewCount = static_cast<scalar_t>(1) / static_cast<scalar_t>(R_.meanCountRef() + 1);

                for (host::label_t deviceIdx = 0; deviceIdx < programCtrl_.deviceList().size(); deviceIdx++)
                {
                    kernel::launch<turbulenceStatisticsCalculate>(
                        mesh_,
                        programCtrl_.streams()[GPU::internalStreamID(deviceIdx)],
                        devPtrs_[deviceIdx],
                        R_.ptr(deviceIdx),
                        P_.ptr(deviceIdx),
                        epsilon_.ptr(deviceIdx),
                        k_.ptr(deviceIdx),
                        UMean_.ptr(deviceIdx),
                        SMean_.ptr(deviceIdx),
                        invNewCount);
                }

                R_.meanCountRef()++;
                P_.meanCountRef()++;
                epsilon_.meanCountRef()++;
                k_.meanCountRef()++;
            }
        }

        __host__ inline void save(const host::label_t timeStep)
        {
            if (calculate_)
            {
                R_.template save<postProcess::LBMBin>(hostWriteBuffer_, timeStep);
                P_.template save<postProcess::LBMBin>(hostWriteBuffer_, timeStep);
                epsilon_.template save<postProcess::LBMBin>(hostWriteBuffer_, timeStep);
                k_.template save<postProcess::LBMBin>(hostWriteBuffer_, timeStep);
            }
        }

    private:
        const kernel::ptrCollection &devPtrs_;
        const bool calculate_;

        const programControl &programCtrl_;
        const host::latticeMesh &mesh_;

        const device::vectorField<VelocitySet, time::timeAverage> UMean_;
        const device::symmetricTensorField<VelocitySet, time::timeAverage> SMean_;

        device::symmetricTensorField<VelocitySet, time::timeAverage> R_;
        device::scalarField<VelocitySet, time::timeAverage> P_;
        device::scalarField<VelocitySet, time::timeAverage> epsilon_;
        device::scalarField<VelocitySet, time::timeAverage> k_;

        host::array<host::PINNED, scalar_t, VelocitySet> &hostWriteBuffer_;

        host::label_t meanCount_ = 0;
    };
}

#endif