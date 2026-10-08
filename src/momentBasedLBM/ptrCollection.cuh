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
    ptrCollection.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_MOMENTBASEDLBM_PTRCOLLECTION_CUH
#define __MBLBM_MOMENTBASEDLBM_PTRCOLLECTION_CUH

namespace LBM
{
    namespace kernel
    {
        class ptrCollection
        {
        public:
            /**
             * @brief Alias for the collection of pointers to device arrays on the GPU, used to pass the data to the kernel
             **/
            using CollectionType = device::ptrColl_t;
            using Type = std::vector<CollectionType>;

            /**
             * @brief Constructor for the collection of pointers to device arrays on the GPU, used to pass the data to the kernel
             * @param[in] rho Device scalar field containing the density values on the GPU
             * @param[in] U Device vector field containing the velocity values on the GPU
             * @param[in] Pi Device symmetric tensor field containing the stress tensor values on the GPU
             * @param[in] programCtrl Program control object containing information about the devices and streams
             **/
            template <class VelocitySet>
            [[nodiscard]] __host__ ptrCollection(
                const device::scalarField<VelocitySet, time::instantaneous, solutionField> &rho,
                const device::vectorField<VelocitySet, time::instantaneous, solutionField> &U,
                const device::symmetricTensorField<VelocitySet, time::instantaneous, solutionField> &Pi,
                const programControl &programCtrl) noexcept
                : devPtrs_(initialisePtrs(rho, U, Pi, programCtrl)) {}

            /**
             * @brief Access operator for the collection of pointers to device arrays on the GPU, used to pass the data to the kernel
             * @param[in] index Index of the device/stream to access
             * @return Collection of pointers to device arrays for the specified device/stream
             **/
            [[nodiscard]] __host__ inline constexpr const CollectionType &operator[](const host::label_t index) const noexcept
            {
                return devPtrs_[index];
            }

        private:
            /**
             * @brief Collection of pointers to device arrays on the GPU, used to pass the data to the kernel
             **/
            const Type devPtrs_;

            /**
             * @brief Initializes the collection of pointers to device arrays on the GPU, used to pass the data to the kernel
             * @param[in] rho Device scalar field containing the density values on the GPU
             * @param[in] U Device vector field containing the velocity values on the GPU
             * @param[in] Pi Device symmetric tensor field containing the stress tensor values on the GPU
             * @param[in] programCtrl Program control object containing information about the devices and streams
             * @return Collection of pointers to device arrays for all devices/streams
             **/
            template <class VelocitySet>
            [[nodiscard]] __host__ static const Type initialisePtrs(
                const device::scalarField<VelocitySet, time::instantaneous, solutionField> &rho,
                const device::vectorField<VelocitySet, time::instantaneous, solutionField> &U,
                const device::symmetricTensorField<VelocitySet, time::instantaneous, solutionField> &Pi,
                const programControl &programCtrl) noexcept
            {
                Type ptrs;

                programCtrl.allsync();

                for (host::label_t stream = 0; stream < programCtrl.deviceList().size(); stream++)
                {
                    errorHandler::handleInline(device::API::setDevice(programCtrl.deviceList()[stream]));
                    errorHandler::handleInline(device::API::deviceSynchronize());

                    ptrs.emplace_back(
                        device::ptrColl_t(
                            rho.self().mutPtr(stream),
                            U.x().mutPtr(stream),
                            U.y().mutPtr(stream),
                            U.z().mutPtr(stream),
                            Pi.xx().mutPtr(stream),
                            Pi.xy().mutPtr(stream),
                            Pi.xz().mutPtr(stream),
                            Pi.yy().mutPtr(stream),
                            Pi.yz().mutPtr(stream),
                            Pi.zz().mutPtr(stream)));
                }

                return ptrs;
            }
        };

        /**
         * @brief Saves a momentsArray object to its original pointers
         * @param[out] devPtrs The pointers to save to
         * @param[in] moments Moment array (rho, U, Pi)
         * @param[in] idx The index into the global array
         **/
        template <const host::label_t i>
        __device__ static inline constexpr void saveToPtr(
            const device::ptrColl_t &devPtrs,
            const momentsArray &moments,
            const device::label_t idx) noexcept
        {
            if constexpr (i == axis::index<axis::NO_DIRECTION>())
            {
                devPtrs.ptr<i>()[idx] = moments[i] - rho0();
            }
            else
            {
                devPtrs.ptr<i>()[idx] = moments[i];
            }
        }

        /**
         * @brief Reads a momentsArray object from its original pointers
         * @param[in] devPtrs The pointers to read from
         * @param[out] moments Moment array (rho, U, Pi)
         * @param[in] idx The index into the global array
         **/
        template <const host::label_t i>
        __device__ static inline constexpr void readFromPtr(
            const device::ptrColl_t &devPtrs,
            momentsArray &moments,
            const device::label_t idx) noexcept
        {
            if constexpr (i == axis::index<axis::NO_DIRECTION>())
            {
                moments[i] = devPtrs.ptr<i>()[idx] + rho0();
            }
            else
            {
                moments[i] = devPtrs.ptr<i>()[idx];
            }
        }
    }
}

#endif