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
    Common main function definition for all LBM solvers

Namespace
    LBM

SourceFiles
    main.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_MAIN_CUH
#define __MBLBM_MAIN_CUH

using namespace LBM;

void sigint_handler([[maybe_unused]] const int code)
{
    std::cout << "Abort signal received" << std::endl;
    program_status.store(BAD, std::memory_order_relaxed);
}

int main(const int argc, const char *const argv[])
{
    struct sigaction sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.sa_handler = sigint_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    if (sigaction(SIGINT, &sa, nullptr) != 0)
    {
        return 1;
    }

    programControl programCtrl(argc, argv);

    const host::latticeMesh mesh(programCtrl);

    if (!((mesh.nDevices<axis::X>() * mesh.nDevices<axis::Y>() * mesh.nDevices<axis::Z>()) == programCtrl.deviceList().size()))
    {
        errorHandler::handle(error::INCORRECT_NUMBER_OF_GPUS);
        return 0;
    }

    if ((mesh.nDevices<axis::X>() > 1) || (mesh.nDevices<axis::Y>() > 1))
    {
        errorHandler::handle(error::INVALID_DEVICE_DECOMPOSITION);
        return 0;
    }

    VelocitySet::print();

    // Allocate the arrays on the device
    const device::scalarField<VelocitySet, time::instantaneous> rho("rho", mesh, programCtrl);
    const device::vectorField<VelocitySet, time::instantaneous> U("U", mesh, programCtrl);
    const device::symmetricTensorField<VelocitySet, time::instantaneous> Pi("Pi", mesh, programCtrl);

    programCtrl.configure<VelocitySet::smem_alloc_size()>(kernel::momentBasedLBM);

    const KernelLauncher momentBasedLBM(mesh, programCtrl, rho, U, Pi);

    objectRegistry<VelocitySet> runTimeObjects(mesh, programCtrl, momentBasedLBM.devPtrs());
    turbulenceStatistics<VelocitySet> turbulenceStats(mesh, programCtrl, momentBasedLBM.devPtrs());

    programCtrl.allsync();

    if (program_status.load() == GOOD)
    {
        runTimeIO<VelocitySet> IO(mesh, programCtrl, rho, U, Pi, runTimeObjects, turbulenceStats);

        for (programCtrl.timeStep() = programCtrl.latestTime(); programCtrl.end(); programCtrl.timeStep()++)
        {
            // Do the run-time IO
            if (programCtrl.print(programCtrl.timeStep()))
            {
                std::cout << "Time: " << programCtrl.timeStep() << std::endl;
            }

            // Checkpoint
            if constexpr (boundaryConditions::save())
            {
                if (programCtrl.save(programCtrl.timeStep()))
                {
                    IO.save<postProcess::LBMBin>();
                }
            }

            // Main kernel launch
            momentBasedLBM.launch();

            // Evaluate the run-time function objects
            if constexpr (boundaryConditions::save())
            {
                runTimeObjects.calculate();
                turbulenceStats.calculate();
            }
        }

        programCtrl.allsync();
    }

    return 0;
}

#endif