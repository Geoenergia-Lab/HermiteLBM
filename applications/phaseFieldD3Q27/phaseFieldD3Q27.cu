/*---------------------------------------------------------------------------*\
|                                                                             |
| cudaLBM: CUDA-based moment representation Lattice Boltzmann Method          |
| Developed at UDESC - State University of Santa Catarina                     |
| Website: https://www.udesc.br                                               |
| Github: https://github.com/geoenergiaUDESC/cudaLBM                          |
|                                                                             |
\*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*\

Copyright (C) 2023 UDESC Geoenergia Lab
Authors: Nathan Duggins, Breno Gemelgo (Geoenergia Lab, UDESC)

This implementation is derived from concepts and algorithms developed in:
  MR-LBM: Moment Representation Lattice Boltzmann Method
  Copyright (C) 2021 CERNN
  Developed at Universidade Federal do Paraná (UFPR)
  Original authors: V. M. de Oliveira, M. A. de Souza, R. F. de Souza
  GitHub: https://github.com/CERNN/MR-LBM
  Licensed under GNU General Public License version 2

License
    This file is part of cudaLBM.

    cudaLBM is free software: you can redistribute it and/or modify it
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
    Implementation of the multiphase moment representation with the D3Q27
    velocity set for hydrodynamics and D3Q7 for phase field evolution

Namespace
    LBM

SourceFiles
    phaseFieldD3Q27.cu

\*---------------------------------------------------------------------------*/

#include "phaseFieldD3Q27.cuh"

using namespace LBM;

__host__ [[nodiscard]] inline consteval device::label_t NStreams() noexcept { return 1; }

int main(const int argc, const char *const argv[])
{
    const programControl programCtrl(argc, argv);

    // Set cuda device
    errorHandler::check(cudaDeviceSynchronize());
    errorHandler::check(cudaSetDevice(programCtrl.deviceList()[0]));
    errorHandler::check(cudaDeviceSynchronize());

    const host::latticeMesh mesh(programCtrl);

    VelocitySet::print();

    // Allocate the arrays on the device
    device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> rho("rho", mesh, programCtrl);
    device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> u("u", mesh, programCtrl);
    device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> v("v", mesh, programCtrl);
    device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> w("w", mesh, programCtrl);
    device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> mxx("m_xx", mesh, programCtrl);
    device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> mxy("m_xy", mesh, programCtrl);
    device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> mxz("m_xz", mesh, programCtrl);
    device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> myy("m_yy", mesh, programCtrl);
    device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> myz("m_yz", mesh, programCtrl);
    device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> mzz("m_zz", mesh, programCtrl);

    // Phase field arrays
    device::array<field::FULL_FIELD, scalar_t, PhaseVelocitySet, time::instantaneous> phi("phi", mesh, programCtrl);

    // Setup Streams
    const streamHandler streamsLBM(programCtrl);

    // Allocate a buffer of pinned memory on the host for writing
    host::array<host::PINNED, scalar_t, VelocitySet, time::instantaneous> hostWriteBuffer(mesh.size() * NUMBER_MOMENTS<true>(), mesh);

    objectRegistry<VelocitySet> runTimeObjects(hostWriteBuffer, mesh, rho, u, v, w, mxx, mxy, mxz, myy, myz, mzz, streamsLBM, programCtrl);

    device::haloSingle<VelocitySet, BoundaryConditions::periodicX(), BoundaryConditions::periodicY(), BoundaryConditions::periodicZ()> fBlockHalo(mesh, programCtrl);      // Hydrodynamic population halo
    device::haloSingle<PhaseVelocitySet, BoundaryConditions::periodicX(), BoundaryConditions::periodicY(), BoundaryConditions::periodicZ()> gBlockHalo(mesh, programCtrl); // Phase population halo
    device::halo<PhaseVelocitySet, BoundaryConditions::periodicX(), BoundaryConditions::periodicY(), BoundaryConditions::periodicZ()> phiBlockHalo(mesh, programCtrl);     // Scalar phi halo
#if defined(FORCE_MULTI_GPU_SCALAR_HALO_TEST)
    const bool enableScalarHalo = true;
#else
    const bool enableScalarHalo = (mesh.nDevices<axis::X>() * mesh.nDevices<axis::Y>() * mesh.nDevices<axis::Z>()) > static_cast<host::label_t>(1);
#endif

#ifdef DEBUG_PHI_EXCHANGE
    const auto debugPhiFaceCopy =
        [&](const char *const stage,
            const char *const faceTag,
            const host::label_t destinationDevice,
            scalar_t *const destinationStart,
            const host::label_t sourceDevice,
            const scalar_t *const sourceStart,
            const host::label_t nValues)
    {
        if (nValues == static_cast<host::label_t>(0))
        {
            return;
        }

        const host::label_t sampleIDs[6] = {
            static_cast<host::label_t>(0),
            static_cast<host::label_t>(1),
            nValues / static_cast<host::label_t>(4),
            nValues / static_cast<host::label_t>(2),
            (nValues > static_cast<host::label_t>(2)) ? (nValues - static_cast<host::label_t>(2)) : static_cast<host::label_t>(0),
            nValues - static_cast<host::label_t>(1)};

        host::label_t mismatchedSamples = static_cast<host::label_t>(0);
        host::label_t nanSamples = static_cast<host::label_t>(0);
        scalar_t maxAbsDiff = static_cast<scalar_t>(0);

        for (const host::label_t rawID : sampleIDs)
        {
            const host::label_t sampleID = (rawID < nValues) ? rawID : (nValues - static_cast<host::label_t>(1));

            scalar_t sourceValue = static_cast<scalar_t>(0);
            scalar_t destinationValue = static_cast<scalar_t>(0);

            errorHandler::checkInline(cudaSetDevice(programCtrl.deviceList()[sourceDevice]));
            errorHandler::checkInline(cudaMemcpy(&sourceValue, sourceStart + sampleID, sizeof(scalar_t), cudaMemcpyDeviceToHost));

            errorHandler::checkInline(cudaSetDevice(programCtrl.deviceList()[destinationDevice]));
            errorHandler::checkInline(cudaMemcpy(&destinationValue, destinationStart + sampleID, sizeof(scalar_t), cudaMemcpyDeviceToHost));

            const bool finiteSource = std::isfinite(sourceValue);
            const bool finiteDestination = std::isfinite(destinationValue);

            if (!(finiteSource && finiteDestination))
            {
                nanSamples++;
                continue;
            }

            const scalar_t absDiff = (sourceValue > destinationValue) ? (sourceValue - destinationValue) : (destinationValue - sourceValue);
            maxAbsDiff = (absDiff > maxAbsDiff) ? absDiff : maxAbsDiff;

            if (absDiff > static_cast<scalar_t>(1e-6))
            {
                mismatchedSamples++;
            }
        }

        if ((mismatchedSamples > static_cast<host::label_t>(0)) || (nanSamples > static_cast<host::label_t>(0)))
        {
            std::cout << "[DEBUG_PHI_EXCHANGE] stage=" << stage
                      << " face=" << faceTag
                      << " srcDevice=" << sourceDevice
                      << " dstDevice=" << destinationDevice
                      << " mismatchSamples=" << mismatchedSamples
                      << " nanSamples=" << nanSamples
                      << " maxAbsDiff=" << maxAbsDiff
                      << std::endl;
        }
    };
#endif

    if (enableScalarHalo)
    {
        programCtrl.configure<smem_alloc_size<VelocitySet>()>(phaseFieldStreamScalarHalo);
    }
    else
    {
        programCtrl.configure<smem_alloc_size<VelocitySet>()>(phaseFieldStreamLocal);
    }

    const runTimeIO IO(mesh, programCtrl);

    if (enableScalarHalo)
    {
        for (device::label_t VirtualDeviceIndex = 0; VirtualDeviceIndex < mesh.nDevices().size(); VirtualDeviceIndex++)
        {
            errorHandler::checkInline(cudaSetDevice(programCtrl.deviceList()[VirtualDeviceIndex]));

            const device::ptrCollection<NUMBER_MOMENTS<true>(), scalar_t> devPtrs(
                rho.ptr(VirtualDeviceIndex),
                u.ptr(VirtualDeviceIndex),
                v.ptr(VirtualDeviceIndex),
                w.ptr(VirtualDeviceIndex),
                mxx.ptr(VirtualDeviceIndex),
                mxy.ptr(VirtualDeviceIndex),
                mxz.ptr(VirtualDeviceIndex),
                myy.ptr(VirtualDeviceIndex),
                myz.ptr(VirtualDeviceIndex),
                mzz.ptr(VirtualDeviceIndex),
                phi.ptr(VirtualDeviceIndex));

            phaseFieldPrimePhiHalo<<<mesh.gridBlock(), mesh.threadBlock(), 0, streamsLBM.streams()[VirtualDeviceIndex]>>>(
                devPtrs,
                phiBlockHalo.writeBuffer(VirtualDeviceIndex));

            errorHandler::checkLast();
        }

        for (device::label_t VirtualDeviceIndex = 0; VirtualDeviceIndex < mesh.nDevices().size(); VirtualDeviceIndex++)
        {
            errorHandler::checkInline(cudaSetDevice(programCtrl.deviceList()[VirtualDeviceIndex]));
            errorHandler::checkInline(cudaDeviceSynchronize());
            streamsLBM.synchronize(VirtualDeviceIndex);
        }

        if (mesh.nDevices<axis::Z>() > static_cast<host::label_t>(1))
        {
            if ((mesh.nDevices<axis::X>() != static_cast<host::label_t>(1)) ||
                (mesh.nDevices<axis::Y>() != static_cast<host::label_t>(1)))
            {
                throw std::runtime_error("phaseField exchange currently mirrors testExecutable and supports Z-only device decomposition.");
            }

            const host::label_t nxb = mesh.nBlocks<axis::X>();
            const host::label_t nyb = mesh.nBlocks<axis::Y>();

            constexpr const host::threadLabel threadStart(
                static_cast<host::label_t>(0),
                static_cast<host::label_t>(0),
                static_cast<host::label_t>(0));

            const host::label_t Size = static_cast<host::label_t>(sizeof(scalar_t)) * static_cast<host::label_t>(2) * block::nx<host::label_t>() * block::ny<host::label_t>() * mesh.blocksPerDevice<axis::X>() * mesh.blocksPerDevice<axis::Y>();

            const host::blockLabel destinationBackBlock(0, 0, 0);
            const host::label_t destinationBackID = host::idxPop<axis::Z, static_cast<host::label_t>(2)>(0, threadStart, destinationBackBlock, nxb, nyb);
            const host::blockLabel sourceBackBlock(0, 0, 0);
            const host::label_t sourceBackID = host::idxPop<axis::Z, static_cast<host::label_t>(2)>(0, threadStart, sourceBackBlock, nxb, nyb);

            const host::blockLabel destinationFrontBlock(0, 0, mesh.blocksPerDevice<axis::Z>() - 1);
            const host::label_t destinationFrontID = host::idxPop<axis::Z, static_cast<host::label_t>(2)>(0, threadStart, destinationFrontBlock, nxb, nyb);
            const host::blockLabel sourceFrontBlock(0, 0, mesh.blocksPerDevice<axis::Z>() - 1);
            const host::label_t sourceFrontID = host::idxPop<axis::Z, static_cast<host::label_t>(2)>(0, threadStart, sourceFrontBlock, nxb, nyb);

            for (host::label_t westDevice = 0; westDevice + 1 < mesh.nDevices<axis::Z>(); westDevice++)
            {
                const host::label_t eastDevice = westDevice + 1;
#ifdef DEBUG_PHI_EXCHANGE
                const host::label_t nValues = Size / static_cast<host::label_t>(sizeof(scalar_t));
#endif

                errorHandler::check(cudaMemcpyPeer(
                    &(phiBlockHalo.writeBuffer(westDevice).template ptr<static_cast<host::label_t>(4)>()[destinationBackID]),
                    programCtrl.deviceList()[westDevice],
                    &(phiBlockHalo.writeBuffer(eastDevice).template ptr<static_cast<host::label_t>(4)>()[sourceBackID]),
                    programCtrl.deviceList()[eastDevice],
                    Size));
#ifdef DEBUG_PHI_EXCHANGE
                debugPhiFaceCopy(
                    "prime",
                    "back(4)",
                    westDevice,
                    &(phiBlockHalo.writeBuffer(westDevice).template ptr<static_cast<host::label_t>(4)>()[destinationBackID]),
                    eastDevice,
                    &(phiBlockHalo.writeBuffer(eastDevice).template ptr<static_cast<host::label_t>(4)>()[sourceBackID]),
                    nValues);
#endif

                errorHandler::check(cudaMemcpyPeer(
                    &(phiBlockHalo.writeBuffer(eastDevice).template ptr<static_cast<host::label_t>(5)>()[destinationFrontID]),
                    programCtrl.deviceList()[eastDevice],
                    &(phiBlockHalo.writeBuffer(westDevice).template ptr<static_cast<host::label_t>(5)>()[sourceFrontID]),
                    programCtrl.deviceList()[westDevice],
                    Size));
#ifdef DEBUG_PHI_EXCHANGE
                debugPhiFaceCopy(
                    "prime",
                    "front(5)",
                    eastDevice,
                    &(phiBlockHalo.writeBuffer(eastDevice).template ptr<static_cast<host::label_t>(5)>()[destinationFrontID]),
                    westDevice,
                    &(phiBlockHalo.writeBuffer(westDevice).template ptr<static_cast<host::label_t>(5)>()[sourceFrontID]),
                    nValues);
#endif
            }
        }

        for (device::label_t VirtualDeviceIndex = 0; VirtualDeviceIndex < mesh.nDevices().size(); VirtualDeviceIndex++)
        {
            errorHandler::checkInline(cudaSetDevice(programCtrl.deviceList()[VirtualDeviceIndex]));
            errorHandler::checkInline(cudaDeviceSynchronize());
            streamsLBM.synchronize(VirtualDeviceIndex);
        }

        for (device::label_t VirtualDeviceIndex = 0; VirtualDeviceIndex < mesh.nDevices().size(); VirtualDeviceIndex++)
        {
            phiBlockHalo.swapNoSync(VirtualDeviceIndex);
        }
    }

    for (host::label_t timeStep = programCtrl.latestTime(); timeStep < programCtrl.nt(); timeStep++)
    {
        // Checkpoint
        if (programCtrl.save(timeStep))
        {
            // Do this in a loop
            for (host::label_t VirtualDeviceIndex = 0; VirtualDeviceIndex < mesh.nDevices().size(); VirtualDeviceIndex++)
            {
                hostWriteBuffer.copy_from_device(
                    device::ptrCollection<11, scalar_t>{
                        rho.ptr(VirtualDeviceIndex),
                        u.ptr(VirtualDeviceIndex),
                        v.ptr(VirtualDeviceIndex),
                        w.ptr(VirtualDeviceIndex),
                        mxx.ptr(VirtualDeviceIndex),
                        mxy.ptr(VirtualDeviceIndex),
                        mxz.ptr(VirtualDeviceIndex),
                        myy.ptr(VirtualDeviceIndex),
                        myz.ptr(VirtualDeviceIndex),
                        mzz.ptr(VirtualDeviceIndex),
                        phi.ptr(VirtualDeviceIndex)},
                    mesh,
                    VirtualDeviceIndex);
            }

            fileIO::writeFile<time::instantaneous>(
                programCtrl.caseName() + "_" + std::to_string(timeStep) + ".LBMBin",
                mesh,
                functionObjects::solutionVariableNames(true),
                hostWriteBuffer.data(),
                timeStep,
                rho.meanCount());

            runTimeObjects.save(timeStep);
        }

        for (device::label_t VirtualDeviceIndex = 0; VirtualDeviceIndex < mesh.nDevices().size(); VirtualDeviceIndex++)
        {
            errorHandler::checkInline(cudaSetDevice(programCtrl.deviceList()[VirtualDeviceIndex]));
            errorHandler::checkInline(cudaDeviceSynchronize());
            streamsLBM.synchronize(VirtualDeviceIndex);
        }

        // Stream kernel (per GPU)
        for (device::label_t VirtualDeviceIndex = 0; VirtualDeviceIndex < mesh.nDevices().size(); VirtualDeviceIndex++)
        {
            errorHandler::checkInline(cudaSetDevice(programCtrl.deviceList()[VirtualDeviceIndex]));
            streamsLBM.synchronize(VirtualDeviceIndex);

            const device::ptrCollection<NUMBER_MOMENTS<true>(), scalar_t> devPtrs{
                rho.ptr(VirtualDeviceIndex),
                u.ptr(VirtualDeviceIndex),
                v.ptr(VirtualDeviceIndex),
                w.ptr(VirtualDeviceIndex),
                mxx.ptr(VirtualDeviceIndex),
                mxy.ptr(VirtualDeviceIndex),
                mxz.ptr(VirtualDeviceIndex),
                myy.ptr(VirtualDeviceIndex),
                myz.ptr(VirtualDeviceIndex),
                mzz.ptr(VirtualDeviceIndex),
                phi.ptr(VirtualDeviceIndex)};

            if (enableScalarHalo)
            {
                phaseFieldStreamScalarHalo<<<mesh.gridBlock(), mesh.threadBlock(), smem_alloc_size<VelocitySet>(), streamsLBM.streams()[VirtualDeviceIndex]>>>(
                    devPtrs,
                    fBlockHalo.readBuffer(VirtualDeviceIndex),
                    gBlockHalo.readBuffer(VirtualDeviceIndex),
                    phiBlockHalo.readBuffer(VirtualDeviceIndex),
                    phiBlockHalo.writeBuffer(VirtualDeviceIndex));
            }
            else
            {
                phaseFieldStreamLocal<<<mesh.gridBlock(), mesh.threadBlock(), smem_alloc_size<VelocitySet>(), streamsLBM.streams()[VirtualDeviceIndex]>>>(
                    devPtrs,
                    fBlockHalo.readBuffer(VirtualDeviceIndex),
                    gBlockHalo.readBuffer(VirtualDeviceIndex),
                    phiBlockHalo.readBuffer(VirtualDeviceIndex),
                    phiBlockHalo.writeBuffer(VirtualDeviceIndex));
            }

            errorHandler::checkLast();
        }

        for (device::label_t VirtualDeviceIndex = 0; VirtualDeviceIndex < mesh.nDevices().size(); VirtualDeviceIndex++)
        {
            errorHandler::checkInline(cudaSetDevice(programCtrl.deviceList()[VirtualDeviceIndex]));
            errorHandler::checkInline(cudaDeviceSynchronize());
            streamsLBM.synchronize(VirtualDeviceIndex);
        }

        // Extra phase-specific exchange required between stream and collide.
        if (enableScalarHalo)
        {
            if (mesh.nDevices<axis::Z>() > static_cast<host::label_t>(1))
            {
                if ((mesh.nDevices<axis::X>() != static_cast<host::label_t>(1)) ||
                    (mesh.nDevices<axis::Y>() != static_cast<host::label_t>(1)))
                {
                    throw std::runtime_error("phaseField exchange currently mirrors testExecutable and supports Z-only device decomposition.");
                }

                const host::label_t nxb = mesh.nBlocks<axis::X>();
                const host::label_t nyb = mesh.nBlocks<axis::Y>();

                constexpr const host::threadLabel threadStart(
                    static_cast<host::label_t>(0),
                    static_cast<host::label_t>(0),
                    static_cast<host::label_t>(0));

                const host::label_t Size = static_cast<host::label_t>(sizeof(scalar_t)) * static_cast<host::label_t>(2) * block::nx<host::label_t>() * block::ny<host::label_t>() * mesh.blocksPerDevice<axis::X>() * mesh.blocksPerDevice<axis::Y>();

                const host::blockLabel destinationBackBlock(0, 0, 0);
                const host::label_t destinationBackID = host::idxPop<axis::Z, static_cast<host::label_t>(2)>(0, threadStart, destinationBackBlock, nxb, nyb);
                const host::blockLabel sourceBackBlock(0, 0, 0);
                const host::label_t sourceBackID = host::idxPop<axis::Z, static_cast<host::label_t>(2)>(0, threadStart, sourceBackBlock, nxb, nyb);

                const host::blockLabel destinationFrontBlock(0, 0, mesh.blocksPerDevice<axis::Z>() - 1);
                const host::label_t destinationFrontID = host::idxPop<axis::Z, static_cast<host::label_t>(2)>(0, threadStart, destinationFrontBlock, nxb, nyb);
                const host::blockLabel sourceFrontBlock(0, 0, mesh.blocksPerDevice<axis::Z>() - 1);
                const host::label_t sourceFrontID = host::idxPop<axis::Z, static_cast<host::label_t>(2)>(0, threadStart, sourceFrontBlock, nxb, nyb);

                for (host::label_t westDevice = 0; westDevice + 1 < mesh.nDevices<axis::Z>(); westDevice++)
                {
                    const host::label_t eastDevice = westDevice + 1;
#ifdef DEBUG_PHI_EXCHANGE
                    const host::label_t nValues = Size / static_cast<host::label_t>(sizeof(scalar_t));
#endif

                    errorHandler::check(cudaMemcpyPeer(
                        &(phiBlockHalo.writeBuffer(westDevice).template ptr<static_cast<host::label_t>(4)>()[destinationBackID]),
                        programCtrl.deviceList()[westDevice],
                        &(phiBlockHalo.writeBuffer(eastDevice).template ptr<static_cast<host::label_t>(4)>()[sourceBackID]),
                        programCtrl.deviceList()[eastDevice],
                        Size));
#ifdef DEBUG_PHI_EXCHANGE
                    debugPhiFaceCopy(
                        "stream_to_collide",
                        "back(4)",
                        westDevice,
                        &(phiBlockHalo.writeBuffer(westDevice).template ptr<static_cast<host::label_t>(4)>()[destinationBackID]),
                        eastDevice,
                        &(phiBlockHalo.writeBuffer(eastDevice).template ptr<static_cast<host::label_t>(4)>()[sourceBackID]),
                        nValues);
#endif

                    errorHandler::check(cudaMemcpyPeer(
                        &(phiBlockHalo.writeBuffer(eastDevice).template ptr<static_cast<host::label_t>(5)>()[destinationFrontID]),
                        programCtrl.deviceList()[eastDevice],
                        &(phiBlockHalo.writeBuffer(westDevice).template ptr<static_cast<host::label_t>(5)>()[sourceFrontID]),
                        programCtrl.deviceList()[westDevice],
                        Size));
#ifdef DEBUG_PHI_EXCHANGE
                    debugPhiFaceCopy(
                        "stream_to_collide",
                        "front(5)",
                        eastDevice,
                        &(phiBlockHalo.writeBuffer(eastDevice).template ptr<static_cast<host::label_t>(5)>()[destinationFrontID]),
                        westDevice,
                        &(phiBlockHalo.writeBuffer(westDevice).template ptr<static_cast<host::label_t>(5)>()[sourceFrontID]),
                        nValues);
#endif
                }
            }
        }

        // Collide kernel (per GPU)
        for (device::label_t VirtualDeviceIndex = 0; VirtualDeviceIndex < mesh.nDevices().size(); VirtualDeviceIndex++)
        {
            errorHandler::checkInline(cudaSetDevice(programCtrl.deviceList()[VirtualDeviceIndex]));
            streamsLBM.synchronize(VirtualDeviceIndex);

            const device::ptrCollection<NUMBER_MOMENTS<true>(), scalar_t> devPtrs{
                rho.ptr(VirtualDeviceIndex),
                u.ptr(VirtualDeviceIndex),
                v.ptr(VirtualDeviceIndex),
                w.ptr(VirtualDeviceIndex),
                mxx.ptr(VirtualDeviceIndex),
                mxy.ptr(VirtualDeviceIndex),
                mxz.ptr(VirtualDeviceIndex),
                myy.ptr(VirtualDeviceIndex),
                myz.ptr(VirtualDeviceIndex),
                mzz.ptr(VirtualDeviceIndex),
                phi.ptr(VirtualDeviceIndex)};

            if (enableScalarHalo)
            {
                phaseFieldCollideScalarHalo<<<mesh.gridBlock(), mesh.threadBlock(), 0, streamsLBM.streams()[VirtualDeviceIndex]>>>(
                    devPtrs,
                    fBlockHalo.writeBuffer(VirtualDeviceIndex),
                    gBlockHalo.writeBuffer(VirtualDeviceIndex),
                    phiBlockHalo.writeBufferConst(VirtualDeviceIndex));
            }
            else
            {
                phaseFieldCollideLocal<<<mesh.gridBlock(), mesh.threadBlock(), 0, streamsLBM.streams()[VirtualDeviceIndex]>>>(
                    devPtrs,
                    fBlockHalo.writeBuffer(VirtualDeviceIndex),
                    gBlockHalo.writeBuffer(VirtualDeviceIndex),
                    phiBlockHalo.writeBufferConst(VirtualDeviceIndex));
            }

            errorHandler::checkLast();
        }

        // Sync all devices and streams
        for (device::label_t VirtualDeviceIndex = 0; VirtualDeviceIndex < mesh.nDevices().size(); VirtualDeviceIndex++)
        {
            errorHandler::checkInline(cudaSetDevice(programCtrl.deviceList()[VirtualDeviceIndex]));
            errorHandler::checkInline(cudaDeviceSynchronize());
            streamsLBM.synchronize(VirtualDeviceIndex);
        }

        if (enableScalarHalo)
        {
            // Mirrors testExecutable exchange placement for population halos.
            if (mesh.nDevices<axis::Z>() > static_cast<host::label_t>(1))
            {
                if ((mesh.nDevices<axis::X>() != static_cast<host::label_t>(1)) ||
                    (mesh.nDevices<axis::Y>() != static_cast<host::label_t>(1)))
                {
                    throw std::runtime_error("phaseField exchange currently mirrors testExecutable and supports Z-only device decomposition.");
                }

                const host::label_t nxb = mesh.nBlocks<axis::X>();
                const host::label_t nyb = mesh.nBlocks<axis::Y>();

                constexpr const host::threadLabel threadStart(
                    static_cast<host::label_t>(0),
                    static_cast<host::label_t>(0),
                    static_cast<host::label_t>(0));

                {
                    const host::label_t Size = static_cast<host::label_t>(sizeof(scalar_t)) * VelocitySet::QF<host::label_t>() * block::nx<host::label_t>() * block::ny<host::label_t>() * mesh.blocksPerDevice<axis::X>() * mesh.blocksPerDevice<axis::Y>();

                    const host::blockLabel destinationBackBlock(0, 0, 0);
                    const host::label_t destinationBackID = host::idxPop<axis::Z, VelocitySet::QF<host::label_t>()>(0, threadStart, destinationBackBlock, nxb, nyb);
                    const host::blockLabel sourceBackBlock(0, 0, 0);
                    const host::label_t sourceBackID = host::idxPop<axis::Z, VelocitySet::QF<host::label_t>()>(0, threadStart, sourceBackBlock, nxb, nyb);

                    const host::blockLabel destinationFrontBlock(0, 0, mesh.blocksPerDevice<axis::Z>() - 1);
                    const host::label_t destinationFrontID = host::idxPop<axis::Z, VelocitySet::QF<host::label_t>()>(0, threadStart, destinationFrontBlock, nxb, nyb);
                    const host::blockLabel sourceFrontBlock(0, 0, mesh.blocksPerDevice<axis::Z>() - 1);
                    const host::label_t sourceFrontID = host::idxPop<axis::Z, VelocitySet::QF<host::label_t>()>(0, threadStart, sourceFrontBlock, nxb, nyb);

                    for (host::label_t westDevice = 0; westDevice + 1 < mesh.nDevices<axis::Z>(); westDevice++)
                    {
                        const host::label_t eastDevice = westDevice + 1;

                        errorHandler::check(cudaMemcpyPeer(
                            &(fBlockHalo.writeBuffer(westDevice).template ptr<static_cast<host::label_t>(4)>()[destinationBackID]),
                            programCtrl.deviceList()[westDevice],
                            &(fBlockHalo.writeBuffer(eastDevice).template ptr<static_cast<host::label_t>(4)>()[sourceBackID]),
                            programCtrl.deviceList()[eastDevice],
                            Size));

                        errorHandler::check(cudaMemcpyPeer(
                            &(fBlockHalo.writeBuffer(eastDevice).template ptr<static_cast<host::label_t>(5)>()[destinationFrontID]),
                            programCtrl.deviceList()[eastDevice],
                            &(fBlockHalo.writeBuffer(westDevice).template ptr<static_cast<host::label_t>(5)>()[sourceFrontID]),
                            programCtrl.deviceList()[westDevice],
                            Size));
                    }
                }

                {
                    const host::label_t Size = static_cast<host::label_t>(sizeof(scalar_t)) * static_cast<host::label_t>(2) * block::nx<host::label_t>() * block::ny<host::label_t>() * mesh.blocksPerDevice<axis::X>() * mesh.blocksPerDevice<axis::Y>();

                    const host::blockLabel destinationBackBlock(0, 0, 0);
                    const host::label_t destinationBackID = host::idxPop<axis::Z, static_cast<host::label_t>(2)>(0, threadStart, destinationBackBlock, nxb, nyb);
                    const host::blockLabel sourceBackBlock(0, 0, 0);
                    const host::label_t sourceBackID = host::idxPop<axis::Z, static_cast<host::label_t>(2)>(0, threadStart, sourceBackBlock, nxb, nyb);

                    const host::blockLabel destinationFrontBlock(0, 0, mesh.blocksPerDevice<axis::Z>() - 1);
                    const host::label_t destinationFrontID = host::idxPop<axis::Z, static_cast<host::label_t>(2)>(0, threadStart, destinationFrontBlock, nxb, nyb);
                    const host::blockLabel sourceFrontBlock(0, 0, mesh.blocksPerDevice<axis::Z>() - 1);
                    const host::label_t sourceFrontID = host::idxPop<axis::Z, static_cast<host::label_t>(2)>(0, threadStart, sourceFrontBlock, nxb, nyb);

                    for (host::label_t westDevice = 0; westDevice + 1 < mesh.nDevices<axis::Z>(); westDevice++)
                    {
                        const host::label_t eastDevice = westDevice + 1;

                        errorHandler::check(cudaMemcpyPeer(
                            &(gBlockHalo.writeBuffer(westDevice).template ptr<static_cast<host::label_t>(4)>()[destinationBackID]),
                            programCtrl.deviceList()[westDevice],
                            &(gBlockHalo.writeBuffer(eastDevice).template ptr<static_cast<host::label_t>(4)>()[sourceBackID]),
                            programCtrl.deviceList()[eastDevice],
                            Size));

                        errorHandler::check(cudaMemcpyPeer(
                            &(gBlockHalo.writeBuffer(eastDevice).template ptr<static_cast<host::label_t>(5)>()[destinationFrontID]),
                            programCtrl.deviceList()[eastDevice],
                            &(gBlockHalo.writeBuffer(westDevice).template ptr<static_cast<host::label_t>(5)>()[sourceFrontID]),
                            programCtrl.deviceList()[westDevice],
                            Size));
                    }
                }
            }
        }

        // Sync all devices and streams
        for (device::label_t VirtualDeviceIndex = 0; VirtualDeviceIndex < mesh.nDevices().size(); VirtualDeviceIndex++)
        {
            errorHandler::checkInline(cudaSetDevice(programCtrl.deviceList()[VirtualDeviceIndex]));
            errorHandler::checkInline(cudaDeviceSynchronize());
            streamsLBM.synchronize(VirtualDeviceIndex);
        }

        // Calculate S kernel
        runTimeObjects.calculate(timeStep);

        // Promote collide-written scalar halos for the next stream/collide step
        if (enableScalarHalo)
        {
            for (device::label_t VirtualDeviceIndex = 0; VirtualDeviceIndex < mesh.nDevices().size(); VirtualDeviceIndex++)
            {
                errorHandler::checkInline(cudaDeviceSynchronize());
                errorHandler::checkInline(cudaSetDevice(programCtrl.deviceList()[VirtualDeviceIndex]));
                errorHandler::checkInline(cudaDeviceSynchronize());
                phiBlockHalo.swapNoSync(VirtualDeviceIndex);
                errorHandler::checkInline(cudaDeviceSynchronize());
            }
        }

        // Do the run-time IO
        if (programCtrl.print(timeStep))
        {
            std::cout << "Time: " << timeStep << std::endl;
        }
    }

    return 0;
}
