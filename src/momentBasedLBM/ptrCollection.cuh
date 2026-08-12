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
            using CollectionType = device::ptrCollection<NUMBER_MOMENTS<host::label_t>(), scalar_t>;
            using Type = std::vector<CollectionType>;

            /**
             * @brief Constructor for the collection of pointers to device arrays on the GPU, used to pass the data to the kernel
             * @param[in] rho Device scalar field for density
             * @param[in] U Device vector field for velocity
             * @param[in] Pi Device symmetric tensor field for the second-order moments
             * @param[in] programCtrl Program control object containing information about the devices and streams
             **/
            template <class VelocitySet>
            __host__ [[nodiscard]] ptrCollection(
                const device::scalarField<VelocitySet, time::instantaneous> &rho,
                const device::vectorField<VelocitySet, time::instantaneous> &U,
                const device::symmetricTensorField<VelocitySet, time::instantaneous> &Pi,
                const programControl &programCtrl) noexcept
                : devPtrs_(initialisePtrs(rho, U, Pi, programCtrl)) {}

            /**
             * @brief Access operator for the collection of pointers to device arrays on the GPU, used to pass the data to the kernel
             * @param[in] index Index of the device/stream to access
             * @return Collection of pointers to device arrays for the specified device/stream
             **/
            __host__ [[nodiscard]] inline constexpr const CollectionType &operator[](const host::label_t index) const noexcept
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
             * @param[in] rho Device scalar field for density
             * @param[in] U Device vector field for velocity
             * @param[in] Pi Device symmetric tensor field for the second-order moments
             * @param[in] programCtrl Program control object containing information about the devices and streams
             * @return Collection of pointers to device arrays for all devices/streams
             **/
            template <class VelocitySet>
            __host__ [[nodiscard]] static const Type initialisePtrs(
                const device::scalarField<VelocitySet, time::instantaneous> &rho,
                const device::vectorField<VelocitySet, time::instantaneous> &U,
                const device::symmetricTensorField<VelocitySet, time::instantaneous> &Pi,
                const programControl &programCtrl)
            {
                Type ptrs;

                programCtrl.allsync();

                for (host::label_t stream = 0; stream < programCtrl.deviceList().size(); stream++)
                {
                    errorHandler::checkInline(cudaSetDevice(programCtrl.deviceList()[stream]));
                    errorHandler::checkInline(cudaDeviceSynchronize());

                    ptrs.emplace_back(
                        device::ptrCollection<NUMBER_MOMENTS<host::label_t>(), scalar_t>(
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

    }
}

#endif