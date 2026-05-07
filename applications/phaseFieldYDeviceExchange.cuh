/*---------------------------------------------------------------------------*\
|                                                                             |
| cudaLBM: CUDA-based moment representation Lattice Boltzmann Method          |
| Developed at UDESC - State University of Santa Catarina                     |
| Website: https://www.udesc.br                                               |
| Github: https://github.com/geoenergiaUDESC/cudaLBM                          |
|                                                                             |
\*---------------------------------------------------------------------------*/

#ifndef __CUDALBM_PHASE_FIELD_Y_DEVICE_EXCHANGE_CUH
#define __CUDALBM_PHASE_FIELD_Y_DEVICE_EXCHANGE_CUH

namespace LBM::phaseFieldApplication
{
    __host__ [[nodiscard]] static bool hasMultipleDevices(const host::latticeMesh &mesh) noexcept
    {
        return (mesh.nDevices<axis::X>() * mesh.nDevices<axis::Y>() * mesh.nDevices<axis::Z>()) > static_cast<host::label_t>(1);
    }

    __host__ static void validateYDeviceDecomposition(const host::latticeMesh &mesh)
    {
        if (!hasMultipleDevices(mesh))
        {
            return;
        }

        if ((mesh.nDevices<axis::X>() != static_cast<host::label_t>(1)) ||
            (mesh.nDevices<axis::Y>() <= static_cast<host::label_t>(1)) ||
            (mesh.nDevices<axis::Z>() != static_cast<host::label_t>(1)))
        {
            throw std::runtime_error("phaseField exchange supports Y-only device decomposition (1 x N x 1).");
        }
    }

    __host__ static void copyStridedYFacePeer(
        scalar_t *const dst,
        const host::label_t dstVirtualDevice,
        scalar_t *const src,
        const host::label_t srcVirtualDevice,
        const std::size_t faceRowBytes,
        const std::size_t facePitchBytes,
        const std::size_t zRows,
        const programControl &programCtrl)
    {
        cudaMemcpy3DPeerParms copyParams{};
        copyParams.dstPtr = make_cudaPitchedPtr(dst, facePitchBytes, faceRowBytes, zRows);
        copyParams.dstDevice = programCtrl.deviceList()[dstVirtualDevice];
        copyParams.srcPtr = make_cudaPitchedPtr(src, facePitchBytes, faceRowBytes, zRows);
        copyParams.srcDevice = programCtrl.deviceList()[srcVirtualDevice];
        copyParams.extent = make_cudaExtent(faceRowBytes, zRows, static_cast<std::size_t>(1));

        errorHandler::check(cudaMemcpy3DPeer(&copyParams));
    }

    template <host::label_t QF, class Halo>
    __host__ void exchangeAdjacentYDeviceHalos(
        Halo &halo,
        const host::latticeMesh &mesh,
        const programControl &programCtrl)
    {
        validateYDeviceDecomposition(mesh);

        if (mesh.nDevices<axis::Y>() <= static_cast<host::label_t>(1))
        {
            return;
        }

        constexpr host::label_t yMinusPtr = static_cast<host::label_t>(axis::pointerIndex_t::South);
        constexpr host::label_t yPlusPtr = static_cast<host::label_t>(axis::pointerIndex_t::North);

        const host::label_t nxb = mesh.blocksPerDevice<axis::X>();
        const host::label_t nyb = mesh.blocksPerDevice<axis::Y>();
        const host::label_t nzb = mesh.blocksPerDevice<axis::Z>();

        constexpr const host::threadLabel threadStart(
            static_cast<host::label_t>(0),
            static_cast<host::label_t>(0),
            static_cast<host::label_t>(0));

        // A local Y face is contiguous across X blocks for one Z-block row; Z rows are separated by the local Y-block stride.
        const std::size_t faceRowBytes =
            sizeof(scalar_t) *
            static_cast<std::size_t>(QF) *
            static_cast<std::size_t>(block::nx<host::label_t>()) *
            static_cast<std::size_t>(block::nz<host::label_t>()) *
            static_cast<std::size_t>(nxb);
        const std::size_t facePitchBytes = faceRowBytes * static_cast<std::size_t>(nyb);
        const std::size_t zRows = static_cast<std::size_t>(nzb);

        const host::blockLabel destinationSouthBlock(0, 0, 0);
        const host::label_t destinationSouthID = host::idxPop<axis::Y, QF>(0, threadStart, destinationSouthBlock, nxb, nyb);
        const host::blockLabel sourceSouthBlock(0, 0, 0);
        const host::label_t sourceSouthID = host::idxPop<axis::Y, QF>(0, threadStart, sourceSouthBlock, nxb, nyb);

        const host::blockLabel destinationNorthBlock(0, nyb - static_cast<host::label_t>(1), 0);
        const host::label_t destinationNorthID = host::idxPop<axis::Y, QF>(0, threadStart, destinationNorthBlock, nxb, nyb);
        const host::blockLabel sourceNorthBlock(0, nyb - static_cast<host::label_t>(1), 0);
        const host::label_t sourceNorthID = host::idxPop<axis::Y, QF>(0, threadStart, sourceNorthBlock, nxb, nyb);

        for (host::label_t lowerYDevice = 0; lowerYDevice + static_cast<host::label_t>(1) < mesh.nDevices<axis::Y>(); lowerYDevice++)
        {
            const host::label_t upperYDevice = lowerYDevice + static_cast<host::label_t>(1);

            copyStridedYFacePeer(
                &(halo.writeBuffer(lowerYDevice).template ptr<yMinusPtr>()[destinationSouthID]),
                lowerYDevice,
                &(halo.writeBuffer(upperYDevice).template ptr<yMinusPtr>()[sourceSouthID]),
                upperYDevice,
                faceRowBytes,
                facePitchBytes,
                zRows,
                programCtrl);

            copyStridedYFacePeer(
                &(halo.writeBuffer(upperYDevice).template ptr<yPlusPtr>()[destinationNorthID]),
                upperYDevice,
                &(halo.writeBuffer(lowerYDevice).template ptr<yPlusPtr>()[sourceNorthID]),
                lowerYDevice,
                faceRowBytes,
                facePitchBytes,
                zRows,
                programCtrl);
        }
    }
}

#endif
