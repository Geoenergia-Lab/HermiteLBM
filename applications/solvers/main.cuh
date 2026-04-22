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

__host__ inline void allsync(const programControl &programCtrl)
{
    for (host::label_t stream = 0; stream < programCtrl.deviceList().size(); stream++)
    {
        errorHandler::checkInline(cudaSetDevice(programCtrl.deviceList()[stream]));
        errorHandler::checkInline(cudaDeviceSynchronize());
        programCtrl.streams().synchronize(stream);
    }
}

__host__ inline void launch(
    const host::latticeMesh &mesh,
    const programControl &programCtrl,
    device::scalarField<VelocitySet, time::instantaneous> &rho,
    device::vectorField<VelocitySet, time::instantaneous> &U,
    device::symmetricTensorField<VelocitySet, time::instantaneous> &Pi,
    const haloBuffer<VelocitySet> &haloPtrs,
    const host::label_t timeStep) noexcept
{
    for (host::label_t stream = 0; stream < programCtrl.deviceList().size(); stream++)
    {
        errorHandler::checkInline(cudaSetDevice(programCtrl.deviceList()[stream]));
        errorHandler::checkInline(cudaDeviceSynchronize());
        programCtrl.streams().synchronize(stream);

        const device::ptrCollection<NUMBER_MOMENTS<host::label_t>(), scalar_t> devPtrs(
            rho.self().ptr(stream),
            U.x().ptr(stream),
            U.y().ptr(stream),
            U.z().ptr(stream),
            Pi.xx().ptr(stream),
            Pi.xy().ptr(stream),
            Pi.xz().ptr(stream),
            Pi.yy().ptr(stream),
            Pi.yz().ptr(stream),
            Pi.zz().ptr(stream));

        kernel::momentBasedLBM<<<mesh.gridBlock(), mesh.threadBlock(), smem_alloc_size<VelocitySet>(), programCtrl.streams()[stream]>>>(
            devPtrs,
            haloPtrs.readBuffer(stream, timeStep),
            haloPtrs.writeBuffer(stream, timeStep));
    }

    allsync(programCtrl);
}

int main(const int argc, const char *const argv[])
{
    const programControl programCtrl(argc, argv);

    if (programCtrl.deviceList().size() > 1)
    {
        errorHandler::check<throws::NO_THROW>(-1, "Multi GPU not yet implemented in all solvers - please use multiGPUD3Q27");
        return 0;
    }

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

    // Begin the main time loop
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
        launch(mesh, programCtrl, rho, U, Pi, haloPtrs, timeStep);

        errorHandler::check(cudaDeviceSynchronize());
    }

    return 0;
}

#endif