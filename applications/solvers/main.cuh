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

int main(const int argc, const char *const argv[])
{
    const programControl programCtrl(argc, argv);

    const host::latticeMesh mesh(programCtrl);

    if (!((mesh.nDevices<axis::X>() * mesh.nDevices<axis::Y>() * mesh.nDevices<axis::Z>()) == programCtrl.deviceList().size()))
    {
        errorHandler::check<throws::NO_THROW>(-1, "Number of GPUs must match the number of devices in the mesh decomposition");
        return 0;
    }

    if ((mesh.nDevices<axis::X>() > 1) || (mesh.nDevices<axis::Y>() > 1))
    {
        errorHandler::check<throws::NO_THROW>(-1, "HermiteLBM currently only supports decomposition in the z axis");
        return 0;
    }

    VelocitySet::print();

    // Allocate the arrays on the device
    device::scalarField<VelocitySet, time::instantaneous> rho("rho", mesh, programCtrl);
    device::vectorField<VelocitySet, time::instantaneous> U("U", mesh, programCtrl);
    device::symmetricTensorField<VelocitySet, time::instantaneous> Pi("Pi", mesh, programCtrl);

    haloBuffer<VelocitySet> haloPtrs(rho, U, Pi, mesh, programCtrl);

    host::array<host::PINNED, scalar_t, VelocitySet, time::instantaneous> hostWriteBuffer(mesh.size() * 6, mesh);

    programCtrl.configure<smem_alloc_size<VelocitySet>()>(kernel::momentBasedLBM);

    objectRegistry<VelocitySet> runTimeObjects(hostWriteBuffer, mesh, rho, U, Pi, programCtrl.streams(), programCtrl);

    const runTimeIO IO(mesh, programCtrl);

    const kernel::ptrCollection devPtrs(rho, U, Pi, programCtrl);

    const deviceCommunicator devComm(mesh, programCtrl, haloPtrs);

    for (host::label_t timeStep = programCtrl.latestTime(); timeStep < programCtrl.nt(); timeStep++)
    {
        // Do the run-time IO
        if (programCtrl.print(timeStep))
        {
            std::cout << "Time: " << timeStep << std::endl;
        }

        // Checkpoint
        if (programCtrl.save(timeStep))
        {
            rho.save<postProcess::LBMBin>(hostWriteBuffer, timeStep);

            U.save<postProcess::LBMBin>(hostWriteBuffer, timeStep);

            Pi.save<postProcess::LBMBin>(hostWriteBuffer, timeStep);

            runTimeObjects.save(timeStep);
        }

        // Main kernel
        kernel::launch(mesh, programCtrl, devPtrs, haloPtrs, timeStep);

        runTimeObjects.calculate();

        // Sync all devices and streams
        programCtrl.allsync();

        // Exchange memory between devices
        devComm.exchange(timeStep);
    }

    return 0;
}

#endif