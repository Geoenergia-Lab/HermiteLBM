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

#include "../../../src/momentBasedLBM/momentBasedLBM.cuh"

namespace LBM
{
    /**
     * @brief Make some templated types concrete
     **/
    using ScalarField = device::scalarField<VelocitySet, time::instantaneous>;
    using VectorField = device::vectorField<VelocitySet, time::instantaneous>;
    using SymmericTensorField = device::symmetricTensorField<VelocitySet, time::instantaneous>;
    using ObjectRegistry = objectRegistry<VelocitySet>;
    using TurbulenceStatistics = turbulenceStatistics<VelocitySet>;
    using RunTimeIO = runTimeIO<VelocitySet>;
}

using namespace LBM;

int main(const int argc, const char *const argv[])
{
    const signalHandler sigint_manager;

    programControl programCtrl(argc, argv);

    const host::latticeMesh mesh(programCtrl);

    if (!((mesh.nDevices<axis::X>() * mesh.nDevices<axis::Y>() * mesh.nDevices<axis::Z>()) == programCtrl.deviceList().size()))
    {
        errorHandler::handle(runTime::error::INCORRECT_NUMBER_OF_GPUS);
        return 0;
    }

    if ((mesh.nDevices<axis::X>() > 1) || (mesh.nDevices<axis::Y>() > 1))
    {
        errorHandler::handle(runTime::error::INVALID_DEVICE_DECOMPOSITION);
        return 0;
    }

    VelocitySet::print();

    // Allocate the arrays on the device
    const ScalarField rho("rho", mesh, programCtrl);
    const VectorField U("U", mesh, programCtrl);
    const SymmericTensorField Pi("Pi", mesh, programCtrl);

    programCtrl.configure<VelocitySet::smem_alloc_size()>(kernel::momentBasedLBM);

    const KernelLauncher momentBasedLBM(mesh, programCtrl, rho, U, Pi);

    ObjectRegistry runTimeObjects(mesh, programCtrl, momentBasedLBM.devPtrs());
    TurbulenceStatistics turbulenceStats(mesh, programCtrl, momentBasedLBM.devPtrs());

    programCtrl.allsync();

    if (runTime::program_status.load() == runTime::GOOD)
    {
        RunTimeIO IO(mesh, programCtrl, rho, U, Pi, runTimeObjects, turbulenceStats);

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