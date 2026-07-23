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
    Base class for LBM function objects, containing common data members.

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_FUNCTIONOBJECTBASE_CUH
#define __MBLBM_FUNCTIONOBJECTBASE_CUH

namespace LBM
{
    namespace functionObjects
    {
        __host__ [[nodiscard]] inline consteval bool functionObjectReady() noexcept { return true; }

        /**
         * @brief Base class for LBM function objects, providing common data members.
         * @tparam VelocitySet The velocity set (D3Q19 or D3Q27)
         * @tparam N The number of spatial components of the function object
         **/
        template <class VelocitySet, const host::label_t N>
        class FunctionObjectBase
        {
        protected:
            /**
             * @brief Name of the field and its time-averaged counterpart
             **/
            const name_t name_;
            const name_t nameMean_;
            const name_t namePrime_;
            const name_t namePrimeSqMean_;

            /**
             * @brief Name of the field's components and their time-averaged counterpart
             **/
            const words_t componentNames_;
            const words_t componentNamesMean_;
            const words_t componentNamesPrime_;
            const words_t componentNamesPrimeSqMean_;

            /**
             * @brief Switches to determine whether or not the field is to be calculated
             **/
            const bool calculate_;
            const bool calculateMean_;
            const bool calculatePrime_;
            const bool calculatePrimeSqMean_;

            /**
             * @brief Reference to the write buffer
             **/
            host::array<host::PINNED, scalar_t, VelocitySet> &hostWriteBuffer_;

            /**
             * @brief Reference to lattice mesh
             **/
            const host::latticeMesh &mesh_;

            /**
             * @brief Device pointer collection
             **/
            const device::scalarField<VelocitySet, time::instantaneous> &rho_;
            const device::vectorField<VelocitySet, time::instantaneous> &U_;
            const device::symmetricTensorField<VelocitySet, time::instantaneous> &Pi_;

            /**
             * @brief Stream handler for CUDA operations
             **/
            const programControl &programCtrl_;

            /**
             * @brief Configures the kernels to allocate no dynamic shared memory and prefer L1 cache
             * @param[in] programCtrl The program control object
             **/
            template <class Kernel>
            __host__ static inline constexpr void configure(const programControl &programCtrl) noexcept
            {
                programCtrl.configure<0, false>(Kernel::instantaneous());
                programCtrl.configure<0, false>(Kernel::instantaneousAndMean());
                programCtrl.configure<0, false>(Kernel::mean());
                programCtrl.configure<0, false>(Kernel::prime());
                programCtrl.configure<0, false>(Kernel::primeSqMean());
            }

            /**
             * @brief Return the pointers that correspond to a particular device partition
             * @param[in] idx The device index
             * @return Pointers to the 10 solution variables allocated on device idx
             **/
            __host__ [[nodiscard]] inline constexpr const device::ptrCollection<NUMBER_MOMENTS<host::label_t>(), const scalar_t> devPtrs(const host::label_t idx) const noexcept
            {
                return {rho_.self().constPtr(idx),
                        U_.x().constPtr(idx), U_.y().constPtr(idx), U_.z().constPtr(idx),
                        Pi_.xx().constPtr(idx), Pi_.xy().constPtr(idx), Pi_.xz().constPtr(idx),
                        Pi_.yy().constPtr(idx), Pi_.yz().constPtr(idx), Pi_.zz().constPtr(idx)};
            }

            /**
             * @brief Calculate a time-averaged quantity
             * @param[in] func The kernel to execute
             * @param[out] object The function object to calculate
             * @param[out] meanCount Counter of time averaging steps
             **/
            template <class FunctionObject, class F>
            __host__ inline void mean(
                F *func,
                FunctionObject &object,
                host::label_t &meanCount)
            {
                static_assert(functionObjectReady(), "Need to fix the launch configuration and stream handling in functionObjectBase");

                const scalar_t invNewCount = static_cast<scalar_t>(1) / static_cast<scalar_t>(meanCount + 1);

                const dim3 nBlocks(static_cast<uint32_t>(mesh_.blocksPerDevice<axis::X>()), static_cast<uint32_t>(mesh_.blocksPerDevice<axis::Y>()), static_cast<uint32_t>(mesh_.blocksPerDevice<axis::Z>()));
                for (host::label_t deviceIdx = 0; deviceIdx < programCtrl_.deviceList().size(); deviceIdx++)
                {
                    func<<<nBlocks, host::latticeMesh::threadBlock(), 0, programCtrl_.streams()[(deviceIdx * 3) + 1]>>>(
                        devPtrs(deviceIdx),
                        object.meanPtrs(deviceIdx),
                        invNewCount);
                }

                meanCount++;
            }

            /**
             * @brief Calculate an instantaneous quantity
             * @param[in] func The kernel to execute
             * @param[out] object The function object to calculate
             **/
            template <class FunctionObject, class F>
            __host__ inline void instantaneous(
                F *func,
                FunctionObject &object)
            {
                static_assert(functionObjectReady(), "Need to fix the launch configuration and stream handling in functionObjectBase");

                const dim3 nBlocks(static_cast<uint32_t>(mesh_.blocksPerDevice<axis::X>()), static_cast<uint32_t>(mesh_.blocksPerDevice<axis::Y>()), static_cast<uint32_t>(mesh_.blocksPerDevice<axis::Z>()));
                for (host::label_t deviceIdx = 0; deviceIdx < programCtrl_.deviceList().size(); deviceIdx++)
                {
                    func<<<nBlocks, host::latticeMesh::threadBlock(), 0, programCtrl_.streams()[(deviceIdx * 3) + 1]>>>(
                        devPtrs(deviceIdx),
                        object.meanPtrs(deviceIdx));
                }
            }

            /**
             * @brief Calculate both an instantaneous and a time-averaged quantity
             * @param[in] func The kernel to execute
             * @param[out] object The function object to calculate
             * @param[out] meanCount Counter of time averaging steps
             **/
            template <class FunctionObject, class F>
            __host__ inline void instantaneousAndMean(
                F *func,
                FunctionObject &object,
                host::label_t &meanCount)
            {
                static_assert(functionObjectReady(), "Need to fix the launch configuration and stream handling in functionObjectBase");

                const scalar_t invNewCount = static_cast<scalar_t>(1) / static_cast<scalar_t>(meanCount + 1);

                const dim3 nBlocks(static_cast<uint32_t>(mesh_.blocksPerDevice<axis::X>()), static_cast<uint32_t>(mesh_.blocksPerDevice<axis::Y>()), static_cast<uint32_t>(mesh_.blocksPerDevice<axis::Z>()));
                for (host::label_t deviceIdx = 0; deviceIdx < programCtrl_.deviceList().size(); deviceIdx++)
                {
                    func<<<nBlocks, host::latticeMesh::threadBlock(), 0, programCtrl_.streams()[(deviceIdx * 3) + 1]>>>(
                        devPtrs(deviceIdx),
                        object.instantaneousPtrs(deviceIdx),
                        object.meanPtrs(deviceIdx),
                        invNewCount);
                }

                meanCount++;
            }

            /**
             * @brief Calculate the perturbation quantity
             * @param[in] func The kernel to execute
             * @param[out] object The function object to calculate
             * @param[out] meanCount Counter of time averaging steps
             **/
            template <class FunctionObject, class F>
            __host__ inline void prime(
                F *func,
                FunctionObject &object)
            {
                static_assert(functionObjectReady(), "Need to fix the launch configuration and stream handling in functionObjectBase");

                const dim3 nBlocks(static_cast<uint32_t>(mesh_.blocksPerDevice<axis::X>()), static_cast<uint32_t>(mesh_.blocksPerDevice<axis::Y>()), static_cast<uint32_t>(mesh_.blocksPerDevice<axis::Z>()));
                for (host::label_t deviceIdx = 0; deviceIdx < programCtrl_.deviceList().size(); deviceIdx++)
                {
                    func<<<nBlocks, host::latticeMesh::threadBlock(), 0, programCtrl_.streams()[(deviceIdx * 3) + 1]>>>(
                        devPtrs(deviceIdx),
                        object.meanPtrs(deviceIdx),
                        object.primePtrs(deviceIdx));
                }
            }

            /**
             * @brief Calculate the time average of the square of the perturbation quantity
             * @param[in] func The kernel to execute
             * @param[out] object The function object to calculate
             * @param[out] meanCount Counter of time averaging steps
             **/
            template <class FunctionObject, class F>
            __host__ inline void primeSqMean(
                F *func,
                FunctionObject &object,
                host::label_t &meanCount)
            {
                const scalar_t invNewCount = static_cast<scalar_t>(1) / static_cast<scalar_t>(meanCount + 1);

                const dim3 nBlocks(static_cast<uint32_t>(mesh_.blocksPerDevice<axis::X>()), static_cast<uint32_t>(mesh_.blocksPerDevice<axis::Y>()), static_cast<uint32_t>(mesh_.blocksPerDevice<axis::Z>()));
                for (host::label_t deviceIdx = 0; deviceIdx < programCtrl_.deviceList().size(); deviceIdx++)
                {
                    func<<<nBlocks, host::latticeMesh::threadBlock(), 0, programCtrl_.streams()[(deviceIdx * 3) + 1]>>>(
                        devPtrs(deviceIdx),
                        object.meanPtrs(deviceIdx),
                        object.primeSqMeanPtrs(deviceIdx),
                        invNewCount);
                }
            }

        public:
            /**
             * @brief Constructs a function object base with common input data.
             * @param[in] hostWriteBuffer Host buffer for writing output data.
             * @param[in] mesh Lattice mesh.
             * @param[in] rho Device scalar field containing the density values on the GPU
             * @param[in] U Device vector field containing the velocity values on the GPU
             * @param[in] Pi Device symmetric tensor field containing the stress tensor values on the GPU
             * @param[in] programCtrl The program control object
             **/
            __host__ [[nodiscard]] FunctionObjectBase(
                const name_t &name,
                host::array<host::PINNED, scalar_t, VelocitySet> &hostWriteBuffer,
                const host::latticeMesh &mesh,
                const device::scalarField<VelocitySet, time::instantaneous> &rho,
                const device::vectorField<VelocitySet, time::instantaneous> &U,
                const device::symmetricTensorField<VelocitySet, time::instantaneous> &Pi,
                const programControl &programCtrl) noexcept
                : name_(name),
                  nameMean_(name + "Mean"),
                  namePrime_(name + "Prime"),
                  namePrimeSqMean_(name + "PrimeSqMean"),
                  componentNames_(fieldType<N>::template makeComponentNames<words_t>(name_)),
                  componentNamesMean_(fieldType<N>::template makeComponentNames<words_t>(nameMean_)),
                  componentNamesPrime_(fieldType<N>::template makeComponentNames<words_t>(namePrime_)),
                  componentNamesPrimeSqMean_(fieldType<N>::template makeComponentNames<words_t>(namePrimeSqMean_)),
                  calculate_(initialiserSwitch(name_)),
                  calculateMean_(initialiserSwitch(nameMean_)),
                  calculatePrime_(initialiserSwitch(namePrime_)),
                  calculatePrimeSqMean_(initialiserSwitch(namePrimeSqMean_)),
                  hostWriteBuffer_(hostWriteBuffer),
                  mesh_(mesh),
                  rho_(rho),
                  U_(U),
                  Pi_(Pi),
                  programCtrl_(programCtrl) {}

            /**
             * @brief Check if calculation of the instantaneous quantity is enabled
             **/
            __host__ [[nodiscard]] inline constexpr bool doInstantaneous() const noexcept
            {
                return calculate_;
            }

            /**
             * @brief Check if calculation of the time average is enabled
             **/
            __host__ [[nodiscard]] inline constexpr bool doMean() const noexcept
            {
                return calculateMean_;
            }

            /**
             * @brief Check if calculation of the perturbation is enabled
             **/
            __host__ [[nodiscard]] inline constexpr bool doPrime() const noexcept
            {
                return calculatePrime_;
            }

            /**
             * @brief Check if calculation of the time average of the square of the perturbation is enabled
             **/
            __host__ [[nodiscard]] inline constexpr bool doPrimeSqMean() const noexcept
            {
                return calculatePrimeSqMean_;
            }
        };
    }
}

#endif // __MBLBM_FUNCTIONOBJECTBASE_CUH