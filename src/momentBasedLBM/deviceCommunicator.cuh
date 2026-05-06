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
    LBM

SourceFiles
    deviceCommunicator.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_DEVICECOMMUNICATOR_CUH
#define __MBLBM_DEVICECOMMUNICATOR_CUH

namespace LBM
{
    template <class VelocitySet>
    class deviceCommunicator
    {
        using exchangeFunction = std::function<void(const host::label_t)>;

    public:
        __host__ [[nodiscard]] deviceCommunicator(
            const host::latticeMesh &mesh,
            const programControl &programCtrl,
            const haloBuffer<VelocitySet> &haloPtrs)
            : mesh_(mesh),
              programCtrl_(programCtrl),
              haloPtrs_(haloPtrs),
              commList_(assembleCommList(programCtrl)) {}

        __host__ ~deviceCommunicator() {}

        /**
         * @brief Disable copying
         **/
        __host__ [[nodiscard]] deviceCommunicator(const deviceCommunicator &) = delete;
        __host__ [[nodiscard]] deviceCommunicator &operator=(const deviceCommunicator &) = delete;

        __host__ void exchange(const host::label_t timeStep) const
        {
            for (const exchangeFunction &commFunction : commList_)
            {
                commFunction(timeStep);
            }
        }

    private:
        const host::latticeMesh &mesh_;
        const programControl &programCtrl_;
        const haloBuffer<VelocitySet> &haloPtrs_;

        const std::vector<exchangeFunction> commList_;

        __host__ [[nodiscard]] const std::vector<exchangeFunction> assembleCommList(const programControl &programCtrl) const
        {
            std::vector<exchangeFunction> commList;

            if (programCtrl.deviceList().size() > 1)
            {
                commList.push_back(
                    [this](const host::label_t timeStep)
                    {
                        this->exchangeImpl(timeStep);
                    });
            }

            return commList;
        }

        __host__ void exchangeImpl(const host::label_t timeStep) const
        {
            const host::label_t nxb = mesh_.nBlocks<axis::X>();
            const host::label_t nyb = mesh_.nBlocks<axis::Y>();

            constexpr const host::threadLabel threadStart(static_cast<device::label_t>(0), static_cast<device::label_t>(0), static_cast<device::label_t>(0));

            const host::label_t Size = static_cast<host::label_t>(sizeof(scalar_t)) * VelocitySet::template QF<host::label_t>() * block::nx<host::label_t>() * block::ny<host::label_t>() * mesh_.blocksPerDevice<axis::X>() * mesh_.blocksPerDevice<axis::Y>();

            constexpr const host::label_t WestDevice = 0;
            constexpr const host::label_t EastDevice = 1;

            constexpr const host::label_t WestPtr_x0 = 4;
            constexpr const host::label_t EastPtr_x1 = 5;

            // Right to Left exchange
            constexpr const host::blockLabel WestDeviceDestinationBlock(0, 0, 0);
            const host::label_t WestDestinationID = host::idxPop<axis::Z, VelocitySet::QF()>(0, threadStart, WestDeviceDestinationBlock, nxb, nyb);
            constexpr const host::blockLabel EastDeviceSourceBlock(0, 0, 0);
            const host::label_t EastSourceID = host::idxPop<axis::Z, VelocitySet::QF()>(0, threadStart, EastDeviceSourceBlock, nxb, nyb);

            // Left to Right exchange
            const host::blockLabel EastDeviceDestinationBlock(0, 0, mesh_.blocksPerDevice<axis::Z>() - 1);
            const host::label_t EastDestinationID = host::idxPop<axis::Z, VelocitySet::QF()>(0, threadStart, EastDeviceDestinationBlock, nxb, nyb);
            const host::blockLabel WestDeviceSourceBlock(0, 0, mesh_.blocksPerDevice<axis::Z>() - 1);
            const host::label_t WestSourceID = host::idxPop<axis::Z, VelocitySet::QF()>(0, threadStart, WestDeviceSourceBlock, nxb, nyb);

            errorHandler::check(cudaMemcpyPeerAsync(
                &(haloPtrs_.writeBuffer(WestDevice, timeStep).template ptr<WestPtr_x0>()[WestDestinationID]),
                programCtrl_.deviceList()[WestDevice],
                &(haloPtrs_.writeBuffer(EastDevice, timeStep).template ptr<WestPtr_x0>()[EastSourceID]),
                programCtrl_.deviceList()[EastDevice],
                Size,
                programCtrl_.streams()[WestDevice]));

            errorHandler::check(cudaMemcpyPeerAsync(
                &(haloPtrs_.writeBuffer(EastDevice, timeStep).template ptr<EastPtr_x1>()[EastDestinationID]),
                programCtrl_.deviceList()[EastDevice],
                &(haloPtrs_.writeBuffer(WestDevice, timeStep).template ptr<EastPtr_x1>()[WestSourceID]),
                programCtrl_.deviceList()[WestDevice],
                Size,
                programCtrl_.streams()[EastDevice]));

            // Sync devices and streams - the cudaDeviceSynchronize() may not be 100% necessary, not sure yet
            errorHandler::checkInline(cudaSetDevice(programCtrl_.deviceList()[WestDevice]));
            errorHandler::checkInline(cudaDeviceSynchronize());
            errorHandler::checkInline(cudaSetDevice(programCtrl_.deviceList()[EastDevice]));
            errorHandler::checkInline(cudaDeviceSynchronize());
            programCtrl_.streams().synchronize(WestDevice);
            programCtrl_.streams().synchronize(EastDevice);
        }
    };
}

#endif