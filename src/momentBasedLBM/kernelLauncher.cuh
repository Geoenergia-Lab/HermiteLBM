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

#ifndef __MBLBM_MOMENTBASEDLBM_KERNELLAUNCHER_CUH
#define __MBLBM_MOMENTBASEDLBM_KERNELLAUNCHER_CUH

namespace LBM
{
    __host__ inline void launch_multi_GPU(
        const host::latticeMesh &mesh,
        const programControl &programCtrl,
        const kernel::ptrCollection &devPtrs,
        const haloBuffer<VelocitySet> &haloPtrs,
        const deviceCommunicator<VelocitySet> &devComm) noexcept
    {
        std::thread boundaryThread(
            std::addressof(kernel::launchBoundary),
            std::cref(mesh),
            std::cref(programCtrl),
            std::cref(devPtrs),
            std::cref(haloPtrs),
            std::cref(devComm),
            programCtrl.timeStep());
        std::thread internalThread(
            std::addressof(kernel::launchInternal),
            std::cref(mesh),
            std::cref(programCtrl),
            std::cref(devPtrs),
            std::cref(haloPtrs),
            programCtrl.timeStep());

        // Synchronize computation and communication
        boundaryThread.join();
        internalThread.join();
    }

    __host__ inline void launch_single_GPU(
        const host::latticeMesh &mesh,
        const programControl &programCtrl,
        const kernel::ptrCollection &devPtrs,
        const haloBuffer<VelocitySet> &haloPtrs) noexcept
    {
        kernel::launch<kernel::momentBasedLBM, VelocitySet::smem_alloc_size()>(
            mesh,
            programCtrl.streams()[GPU::internalStreamID(0)],
            devPtrs[0],
            haloPtrs.readBuffer(0, programCtrl.timeStep()),
            haloPtrs.writeBuffer(0, programCtrl.timeStep()),
            static_cast<device::label_t>(0));
    }

    class MultiGPULauncher
    {
    public:
        MultiGPULauncher(
            const host::latticeMesh &mesh,
            const programControl &programCtrl,
            const device::scalarField<VelocitySet, time::instantaneous> &rho,
            const device::vectorField<VelocitySet, time::instantaneous> &U,
            const device::symmetricTensorField<VelocitySet, time::instantaneous> &Pi)
            : mesh_(mesh),
              programCtrl_(programCtrl),
              devPtrs_(rho, U, Pi, programCtrl),
              haloPtrs_(rho, U, Pi, mesh, programCtrl),
              devComm_(mesh, programCtrl, haloPtrs_)
        {
        }

        inline void launch() const noexcept
        {
            launch_multi_GPU(mesh_, programCtrl_, devPtrs_, haloPtrs_, devComm_);
        }

        __host__ [[nodiscard]] inline constexpr const kernel::ptrCollection &devPtrs() const noexcept { return devPtrs_; }

    private:
        const host::latticeMesh &mesh_;
        const programControl &programCtrl_;

        // Construct devPtrs, haloPtrs and devComm
        const kernel::ptrCollection devPtrs_;
        const haloBuffer<VelocitySet> haloPtrs_;
        const deviceCommunicator<VelocitySet> devComm_;
    };

    class SingleGPULauncher
    {
    public:
        SingleGPULauncher(
            const host::latticeMesh &mesh,
            const programControl &programCtrl,
            const device::scalarField<VelocitySet, time::instantaneous> &rho,
            const device::vectorField<VelocitySet, time::instantaneous> &U,
            const device::symmetricTensorField<VelocitySet, time::instantaneous> &Pi)
            : mesh_(mesh),
              programCtrl_(programCtrl),
              devPtrs_(rho, U, Pi, programCtrl),
              haloPtrs_(rho, U, Pi, mesh, programCtrl)
        {
        }

        inline void launch() const noexcept
        {
            launch_single_GPU(mesh_, programCtrl_, devPtrs_, haloPtrs_);
        }

        __host__ [[nodiscard]] inline constexpr const kernel::ptrCollection &devPtrs() const noexcept { return devPtrs_; }

    private:
        const host::latticeMesh &mesh_;
        const programControl &programCtrl_;

        // Construct devPtrs and haloPtrs
        const kernel::ptrCollection devPtrs_;
        const haloBuffer<VelocitySet> haloPtrs_;
    };

    // -----------------------------------------------------------------------------
    // Wrapper class using std::variant to hold the appropriate concrete launcher
    // -----------------------------------------------------------------------------
    class KernelLauncher
    {
    public:
        KernelLauncher(
            const host::latticeMesh &mesh,
            const programControl &programCtrl,
            const device::scalarField<VelocitySet, time::instantaneous> &rho,
            const device::vectorField<VelocitySet, time::instantaneous> &U,
            const device::symmetricTensorField<VelocitySet, time::instantaneous> &Pi)
            : variant_(makeVariant(mesh, programCtrl, rho, U, Pi))
        {
        }

        inline void launch() const noexcept
        {
            std::visit([](const auto &launcher)
                       { launcher.launch(); }, variant_);
        }

        __host__ [[nodiscard]] inline const kernel::ptrCollection &devPtrs() const noexcept
        {
            return std::visit(
                [](const auto &launcher) -> const kernel::ptrCollection &
                {
                    return launcher.devPtrs();
                },
                variant_);
        }

    private:
        static const std::variant<MultiGPULauncher, SingleGPULauncher> makeVariant(
            const host::latticeMesh &mesh,
            const programControl &programCtrl,
            const device::scalarField<VelocitySet, time::instantaneous> &rho,
            const device::vectorField<VelocitySet, time::instantaneous> &U,
            const device::symmetricTensorField<VelocitySet, time::instantaneous> &Pi) noexcept
        {
            if constexpr (system::hasMultiGPU())
            {
                if (programCtrl.deviceList().size() > 1)
                {
                    return std::variant<MultiGPULauncher, SingleGPULauncher>(std::in_place_type<MultiGPULauncher>, mesh, programCtrl, rho, U, Pi);
                }
            }
            return std::variant<MultiGPULauncher, SingleGPULauncher>(std::in_place_type<SingleGPULauncher>, mesh, programCtrl, rho, U, Pi);
        }

        const std::variant<MultiGPULauncher, SingleGPULauncher> variant_;
    };
}

#endif