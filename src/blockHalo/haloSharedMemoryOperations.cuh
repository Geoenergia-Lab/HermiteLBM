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
    A class handling the device halo. This class is used to exchange the
    microscopic velocity components at the edge of a CUDA block

Namespace
    LBM::device

SourceFiles
    haloSharedMemoryOperations.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_HALOSHAREDMEMORYOPERATIONS_CUH
#define __MBLBM_HALOSHAREDMEMORYOPERATIONS_CUH

/**
 * @brief Computes linear index for a thread within a block
 * @param[in] tx Thread x-coordinate within block
 * @param[in] ty Thread y-coordinate within block
 * @param[in] tz Thread z-coordinate within block
 * @return Linearized index in shared memory
 *
 * Memory layout: [tz][ty][tx] (tz slowest varying, tx fastest)
 **/
__device__ __host__ [[nodiscard]] static inline constexpr device::label_t idx_block(const device::label_t tx, const device::label_t ty, const device::label_t tz) noexcept
{
    return tx + block::nx() * (ty + block::ny() * tz);
}

/**
 * @overload Passes a pre-constructed thread coordinate
 * @param[in] Tx Three-dimensional thread coordinates
 **/
__device__ __host__ [[nodiscard]] static inline constexpr device::label_t idx_block(const thread::coordinate &Tx) noexcept
{
    return idx_block(Tx.value<axis::X>(), Tx.value<axis::Y>(), Tx.value<axis::Z>());
}

/**
 * @brief Computes the warp number of a particular thread within a block
 * @param[in] tx Thread x-coordinate within block
 * @param[in] ty Thread y-coordinate within block
 * @param[in] tz Thread z-coordinate within block
 * @return The unique ID of the warp corresponding to a particular thread
 *
 * Memory layout: [tz][ty][tx] (tz slowest varying, tx fastest)
 **/
__device__ __host__ [[nodiscard]] static inline constexpr device::label_t warpID(const device::label_t tx, const device::label_t ty, const device::label_t tz) noexcept
{
    return idx_block(tx, ty, tz) / block::warp_size();
}

/**
 * @overload Passes a pre-constructed thread coordinate
 * @param[in] Tx Three-dimensional thread coordinates
 **/
__device__ __host__ [[nodiscard]] static inline constexpr device::label_t warpID(const thread::coordinate &Tx) noexcept
{
    return warpID(Tx.value<axis::X>(), Tx.value<axis::Y>(), Tx.value<axis::Z>());
}

/**
 * @brief Computes the linear index of a thread within a warp
 * @param[in] tx Thread x-coordinate within block
 * @param[in] ty Thread y-coordinate within block
 * @param[in] tz Thread z-coordinate within block
 * @return The unique ID of a thread within a warp, in the range [0, warp_size]
 *
 * Memory layout: [tz][ty][tx] (tz slowest varying, tx fastest)
 **/
__device__ __host__ [[nodiscard]] static inline constexpr device::label_t idxWarp(const device::label_t tx, const device::label_t ty, const device::label_t tz) noexcept
{
    return idx_block(tx, ty, tz) % block::warp_size();
}

/**
 * @overload Passes a pre-constructed thread coordinate
 * @param[in] Tx Three-dimensional thread coordinates
 **/
__device__ __host__ [[nodiscard]] static inline constexpr device::label_t idxWarp(const thread::coordinate &Tx) noexcept
{
    return idxWarp(Tx.value<axis::X>(), Tx.value<axis::Y>(), Tx.value<axis::Z>());
}

/**
 * @brief Total area of lattice units on a block face
 * @tparam alpha The axis direction (X, Y or Z)
 * @tparam T The return type
 * @return Linearized face index>
 **/
template <const axis::type alpha, typename T = device::label_t>
__device__ __host__ [[nodiscard]] static inline consteval T faceArea() noexcept
{
    return block::n<axis::orthogonal<alpha, 0>(), T>() * block::n<axis::orthogonal<alpha, 1>(), T>();
}

/**
 * @brief Index for points located on a block face
 * @tparam alpha The axis direction (X, Y or Z)
 * @return Linearized face index>
 **/
template <const axis::type alpha>
__device__ __host__ [[nodiscard]] static inline constexpr device::label_t idxFace(const thread::coordinate &Tx) noexcept
{
    return Tx.value<axis::orthogonal<alpha, 0>()>() + (Tx.value<axis::orthogonal<alpha, 1>()>() * block::n<axis::orthogonal<alpha, 0>()>());
}

/**
 * @brief Transposes an individual face into the shared memory
 * @tparam alpha The axis direction (X, Y or Z)
 * @tparam coeff The coefficient indicating the direction along the axis (must be -1 or 1)
 * @tparam idxoffset The constant offset into the shared memory for the particular block configuration
 * @param[in] Tx Three-dimensional thread coordinates
 * @param[in] pop Array to store loaded population values
 * @param[in] sharedBuffer Inline or externally stored shared memory buffer
 **/
template <const axis::type alpha, const int coeff, const device::label_t idxOffset, class SharedBuffer>
__device__ static inline constexpr void transpose(const thread::coordinate &Tx, const thread::array<scalar_t, VelocitySet::Q()> &pop, SharedBuffer &sharedBuffer) noexcept
{
    axis::assertions::validate<alpha, axis::NOT_NULL>();

    velocityCoefficient::assertions::validate<coeff, velocityCoefficient::NOT_NULL>();

    const device::label_t base_idx = idxFace<alpha>(Tx);
    device::constexpr_for<0, VelocitySet::QF()>(
        [&](const auto i)
        {
            sharedBuffer[idxOffset + base_idx + (static_cast<device::label_t>(i) * faceArea<alpha>())] = pop[q_i<streaming_index<alpha, coeff>(i)>()];
        });
}

/**
 * @brief Helper function for smemOffset
 **/
template <const int FaceIdx>
__device__ __host__ [[nodiscard]] static inline consteval device::label_t sumFaceAreasBefore() noexcept
{
    return []<host::label_t... I>(std::index_sequence<I...>)
    {
        return (device::label_t{0} + ... + ((I < FaceIdx) ? faceArea<static_cast<axis::type>(I / 2)>() : device::label_t{0}));
    }(std::make_index_sequence<6>{});
}

/**
 * @brief Calculates the offset into shared memory for a particular halo transpose operation
 * @tparam alpha The axis direction (X, Y or Z)
 * @tparam coeff The coefficient indicating the direction along the axis (must be -1 or 1)
 **/
template <const axis::type alpha, const int coeff>
__device__ __host__ [[nodiscard]] static inline consteval device::label_t smemOffset() noexcept
{
    axis::assertions::validate<alpha, axis::NOT_NULL>();

    velocityCoefficient::assertions::validate<coeff, velocityCoefficient::NOT_NULL>();

    return VelocitySet::QF() * sumFaceAreasBefore<static_cast<int>(alpha) * 2 + (coeff == -1 ? 0 : 1)>();
}

/**
 * @brief Transposes population data in halo regions via the shared memory
 * @tparam alpha The axis direction (X, Y or Z)
 * @param[in] pop Array containing population values to save
 * @param[out] sharedBuffer Inline or externally stored shared memory buffer
 * @param[in] point The global point coordinate
 * @param[in] Tx Three-dimensional thread coordinates
 **/
template <const axis::type alpha, class SharedBuffer>
__device__ static inline constexpr void transpose_direction(
    const thread::array<scalar_t, VelocitySet::Q()> &pop,
    SharedBuffer &sharedBuffer,
    const device::pointCoordinate &point,
    const thread::coordinate &Tx) noexcept
{
    axis::assertions::validate<alpha, axis::NOT_NULL>();

    if (boundaryCheck<alpha, -1, is_periodic<alpha>()>(point.value<alpha>(), Tx))
    {
        transpose<alpha, -1, smemOffset<alpha, -1>()>(Tx, pop, sharedBuffer);
    }
    else if (boundaryCheck<alpha, +1, is_periodic<alpha>()>(point.value<alpha>(), Tx))
    {
        transpose<alpha, +1, smemOffset<alpha, +1>()>(Tx, pop, sharedBuffer);
    }
}

/**
 * @brief Transposes the block halo into the shared memory for X and Y axes, saves the Z halo
 * @param[in] pop Array containing the populations for the particular thread
 * @param[out] writeBuffer Collection of pointers to the halo faces
 * @param[out] sharedBuffer Inline or externally stored shared memory buffer
 * @param[in] Tx Three-dimensional thread coordinates
 * @param[in] Bx Three-dimensional block coordinates
 * @param[in] point The global point coordinate
 **/
template <class SharedBuffer>
__device__ static inline constexpr void transpose_to_shared(
    const thread::array<scalar_t, VelocitySet::Q()> &pop,
    const device::ptrCollection<6, scalar_t> &writeBuffer,
    SharedBuffer &sharedBuffer,
    const thread::coordinate &Tx,
    const block::coordinate &Bx,
    const device::pointCoordinate &point) noexcept
{
    // X axis halo transposition
    transpose_direction<axis::X>(pop, sharedBuffer, point, Tx);

    // Y axis halo transposition
    transpose_direction<axis::Y>(pop, sharedBuffer, point, Tx);

    block::sync();

    // Z halos: these halos coalesce naturally, so no transposition is needed
    save_direction<axis::Z>(pop, writeBuffer, Tx, Bx, point);
}

/**
 * @brief Saves population data to halo regions for neighboring blocks
 * @param[in] sharedBuffer Shared array containing the packed population halos
 * @param[out] writeBuffer Collection of pointers to the halo faces
 * @param[in] Tx Three-dimensional thread coordinates
 * @param[in] Bx Three-dimensional block coordinates
 * @note This device function saves population values to halo regions for neighboring blocks to read.
 **/
template <class SharedBuffer>
__device__ static inline constexpr void save_from_shared(
    const SharedBuffer &sharedBuffer,
    const device::ptrCollection<6, scalar_t> &writeBuffer,
    const thread::coordinate &Tx,
    const block::coordinate &Bx) noexcept
{
    const device::label_t warpId = warpID(Tx);
    const device::label_t offset = block::warp_size() * (warpId % static_cast<device::label_t>(2));
    const device::label_t idx_in_warp = idxWarp(Tx);

    // Equivalent of threadIdx.alpha, threadIdx.beta
    const dim2<axis::X> yz(idx_in_warp + offset);
    const dim2<axis::Y> xz(idx_in_warp + offset);

    const device::label_t ID = idx_block(Tx);

    constexpr device::label_t padded_stride = block::size() + static_cast<device::label_t>(0);

    if constexpr ((std::is_same_v<VelocitySet, D3Q19<Thermal>>) || (std::is_same_v<VelocitySet, D3Q19<Isothermal>>))
    {
        const thread::array<scalar_t, 3> val(
            sharedBuffer[ID + (0 * padded_stride)],
            sharedBuffer[ID + (1 * padded_stride)],
            sharedBuffer[ID + (2 * padded_stride)]);

        block::sync();

        switch (warpId / 2)
        {
        case 0:
        {
            writeBuffer.ptr<0>()[idxPop<axis::X, 0, VelocitySet::QF()>(yz.i(), yz.j(), Bx)] = val[0];
            writeBuffer.ptr<1>()[idxPop<axis::X, 3, VelocitySet::QF()>(yz.i(), yz.j(), Bx)] = val[1];
            writeBuffer.ptr<3>()[idxPop<axis::Y, 1, VelocitySet::QF()>(xz.i(), xz.j(), Bx)] = val[2];

            return;
        }
        case 1:
        {
            writeBuffer.ptr<0>()[idxPop<axis::X, 1, VelocitySet::QF()>(yz.i(), yz.j(), Bx)] = val[0];
            writeBuffer.ptr<1>()[idxPop<axis::X, 4, VelocitySet::QF()>(yz.i(), yz.j(), Bx)] = val[1];
            writeBuffer.ptr<3>()[idxPop<axis::Y, 2, VelocitySet::QF()>(xz.i(), xz.j(), Bx)] = val[2];

            return;
        }
        case 2:
        {
            writeBuffer.ptr<0>()[idxPop<axis::X, 2, VelocitySet::QF()>(yz.i(), yz.j(), Bx)] = val[0];
            writeBuffer.ptr<2>()[idxPop<axis::Y, 0, VelocitySet::QF()>(xz.i(), xz.j(), Bx)] = val[1];
            writeBuffer.ptr<3>()[idxPop<axis::Y, 3, VelocitySet::QF()>(xz.i(), xz.j(), Bx)] = val[2];

            return;
        }
        case 3:
        {
            writeBuffer.ptr<0>()[idxPop<axis::X, 3, VelocitySet::QF()>(yz.i(), yz.j(), Bx)] = val[0];
            writeBuffer.ptr<2>()[idxPop<axis::Y, 1, VelocitySet::QF()>(xz.i(), xz.j(), Bx)] = val[1];
            writeBuffer.ptr<3>()[idxPop<axis::Y, 4, VelocitySet::QF()>(xz.i(), xz.j(), Bx)] = val[2];

            return;
        }
        case 4:
        {
            writeBuffer.ptr<0>()[idxPop<axis::X, 4, VelocitySet::QF()>(yz.i(), yz.j(), Bx)] = val[0];
            writeBuffer.ptr<2>()[idxPop<axis::Y, 2, VelocitySet::QF()>(xz.i(), xz.j(), Bx)] = val[1];

            return;
        }
        case 5:
        {
            writeBuffer.ptr<1>()[idxPop<axis::X, 0, VelocitySet::QF()>(yz.i(), yz.j(), Bx)] = val[0];
            writeBuffer.ptr<2>()[idxPop<axis::Y, 3, VelocitySet::QF()>(xz.i(), xz.j(), Bx)] = val[1];

            return;
        }
        case 6:
        {
            writeBuffer.ptr<1>()[idxPop<axis::X, 1, VelocitySet::QF()>(yz.i(), yz.j(), Bx)] = val[0];
            writeBuffer.ptr<2>()[idxPop<axis::Y, 4, VelocitySet::QF()>(xz.i(), xz.j(), Bx)] = val[1];

            return;
        }
        case 7:
        {
            writeBuffer.ptr<1>()[idxPop<axis::X, 2, VelocitySet::QF()>(yz.i(), yz.j(), Bx)] = val[0];
            writeBuffer.ptr<3>()[idxPop<axis::Y, 0, VelocitySet::QF()>(xz.i(), xz.j(), Bx)] = val[1];

            return;
        }
        }
    }

    if constexpr ((std::is_same_v<VelocitySet, D3Q27<Thermal>>) || (std::is_same_v<VelocitySet, D3Q27<Isothermal>>))
    {
        const thread::array<scalar_t, 5> val(
            sharedBuffer[ID + (0 * padded_stride)],
            sharedBuffer[ID + (1 * padded_stride)],
            sharedBuffer[ID + (2 * padded_stride)],
            sharedBuffer[ID + (3 * padded_stride)],
            sharedBuffer[ID + (4 * padded_stride)]);

        switch (warpId / 2)
        {
        case 0:
        {
            writeBuffer.ptr<0>()[idxPop<axis::X, 0, VelocitySet::QF()>(yz.i(), yz.j(), Bx)] = val[0];
            writeBuffer.ptr<0>()[idxPop<axis::X, 8, VelocitySet::QF()>(yz.i(), yz.j(), Bx)] = val[1];
            writeBuffer.ptr<1>()[idxPop<axis::X, 7, VelocitySet::QF()>(yz.i(), yz.j(), Bx)] = val[2];

            writeBuffer.ptr<2>()[idxPop<axis::Y, 6, VelocitySet::QF()>(xz.i(), xz.j(), Bx)] = val[3];
            writeBuffer.ptr<3>()[idxPop<axis::Y, 5, VelocitySet::QF()>(xz.i(), xz.j(), Bx)] = val[4];

            return;
        }
        case 1:
        {
            writeBuffer.ptr<0>()[idxPop<axis::X, 1, VelocitySet::QF()>(yz.i(), yz.j(), Bx)] = val[0];
            writeBuffer.ptr<1>()[idxPop<axis::X, 0, VelocitySet::QF()>(yz.i(), yz.j(), Bx)] = val[1];
            writeBuffer.ptr<1>()[idxPop<axis::X, 8, VelocitySet::QF()>(yz.i(), yz.j(), Bx)] = val[2];

            writeBuffer.ptr<2>()[idxPop<axis::Y, 7, VelocitySet::QF()>(xz.i(), xz.j(), Bx)] = val[3];
            writeBuffer.ptr<3>()[idxPop<axis::Y, 6, VelocitySet::QF()>(xz.i(), xz.j(), Bx)] = val[4];

            return;
        }
        case 2:
        {
            writeBuffer.ptr<0>()[idxPop<axis::X, 2, VelocitySet::QF()>(yz.i(), yz.j(), Bx)] = val[0];
            writeBuffer.ptr<1>()[idxPop<axis::X, 1, VelocitySet::QF()>(yz.i(), yz.j(), Bx)] = val[1];

            writeBuffer.ptr<2>()[idxPop<axis::Y, 0, VelocitySet::QF()>(xz.i(), xz.j(), Bx)] = val[2];
            writeBuffer.ptr<2>()[idxPop<axis::Y, 8, VelocitySet::QF()>(xz.i(), xz.j(), Bx)] = val[3];
            writeBuffer.ptr<3>()[idxPop<axis::Y, 7, VelocitySet::QF()>(xz.i(), xz.j(), Bx)] = val[4];

            return;
        }
        case 3:
        {
            writeBuffer.ptr<0>()[idxPop<axis::X, 3, VelocitySet::QF()>(yz.i(), yz.j(), Bx)] = val[0];
            writeBuffer.ptr<1>()[idxPop<axis::X, 2, VelocitySet::QF()>(yz.i(), yz.j(), Bx)] = val[1];

            writeBuffer.ptr<2>()[idxPop<axis::Y, 1, VelocitySet::QF()>(xz.i(), xz.j(), Bx)] = val[2];
            writeBuffer.ptr<3>()[idxPop<axis::Y, 0, VelocitySet::QF()>(xz.i(), xz.j(), Bx)] = val[3];
            writeBuffer.ptr<3>()[idxPop<axis::Y, 8, VelocitySet::QF()>(xz.i(), xz.j(), Bx)] = val[4];

            return;
        }
        case 4:
        {
            writeBuffer.ptr<0>()[idxPop<axis::X, 4, VelocitySet::QF()>(yz.i(), yz.j(), Bx)] = val[0];
            writeBuffer.ptr<1>()[idxPop<axis::X, 3, VelocitySet::QF()>(yz.i(), yz.j(), Bx)] = val[1];

            writeBuffer.ptr<2>()[idxPop<axis::Y, 2, VelocitySet::QF()>(xz.i(), xz.j(), Bx)] = val[2];
            writeBuffer.ptr<3>()[idxPop<axis::Y, 1, VelocitySet::QF()>(xz.i(), xz.j(), Bx)] = val[3];

            return;
        }
        case 5:
        {
            writeBuffer.ptr<0>()[idxPop<axis::X, 5, VelocitySet::QF()>(yz.i(), yz.j(), Bx)] = val[0];
            writeBuffer.ptr<1>()[idxPop<axis::X, 4, VelocitySet::QF()>(yz.i(), yz.j(), Bx)] = val[1];

            writeBuffer.ptr<2>()[idxPop<axis::Y, 3, VelocitySet::QF()>(xz.i(), xz.j(), Bx)] = val[2];
            writeBuffer.ptr<3>()[idxPop<axis::Y, 2, VelocitySet::QF()>(xz.i(), xz.j(), Bx)] = val[3];

            return;
        }
        case 6:
        {
            writeBuffer.ptr<0>()[idxPop<axis::X, 6, VelocitySet::QF()>(yz.i(), yz.j(), Bx)] = val[0];
            writeBuffer.ptr<1>()[idxPop<axis::X, 5, VelocitySet::QF()>(yz.i(), yz.j(), Bx)] = val[1];

            writeBuffer.ptr<2>()[idxPop<axis::Y, 4, VelocitySet::QF()>(xz.i(), xz.j(), Bx)] = val[2];
            writeBuffer.ptr<3>()[idxPop<axis::Y, 3, VelocitySet::QF()>(xz.i(), xz.j(), Bx)] = val[3];

            return;
        }
        case 7:
        {
            writeBuffer.ptr<0>()[idxPop<axis::X, 7, VelocitySet::QF()>(yz.i(), yz.j(), Bx)] = val[0];
            writeBuffer.ptr<1>()[idxPop<axis::X, 6, VelocitySet::QF()>(yz.i(), yz.j(), Bx)] = val[1];

            writeBuffer.ptr<2>()[idxPop<axis::Y, 5, VelocitySet::QF()>(xz.i(), xz.j(), Bx)] = val[2];
            writeBuffer.ptr<3>()[idxPop<axis::Y, 4, VelocitySet::QF()>(xz.i(), xz.j(), Bx)] = val[3];

            return;
        }
        }
    }
}

#endif