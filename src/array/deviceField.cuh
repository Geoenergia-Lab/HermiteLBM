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
    Implementation of scalar, vector and tensor fields on the device

Namespace
    LBM

SourceFiles
    deviceField.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_DEVICEFIELD_CUH
#define __MBLBM_DEVICEFIELD_CUH

namespace LBM
{
    namespace device
    {
        template <class VelocitySet, const time::type TimeType, const host::label_t N>
        class fieldBase
        {
        protected:
            /**
             * @brief Type alias for the components of a field
             **/
            using ComponentType = device::array<field::FULL_FIELD, scalar_t, VelocitySet, TimeType>;

            /**
             * @brief Name of the field
             **/
            const name_t name_;

            /**
             * @brief Count of time steps averaged (for time-averaged fields)
             **/
            host::label_t meanCount_;

            /**
             * @brief Array of field components
             **/
            std::array<ComponentType, N> components_;

        public:
            /**
             * @brief Constructor for fieldBase
             * @param[in] name Name of the field
             * @param[in] mesh Reference to the lattice mesh
             * @param[in] values Initial values for each component (only used if allocate=true)
             * @param[in] programCtrl Reference to the program control object
             * @param[in] allocate Whether to allocate memory and initialize values
             **/
            __host__ [[nodiscard]] fieldBase(
                const name_t &name,
                const host::latticeMesh &mesh,
                const std::array<scalar_t, N> values,
                const programControl &programCtrl,
                const bool allocate = true)
                : name_(name),
                  meanCount_(initialiseMeanCount(name, programCtrl)),
                  components_(makeComponents(std::make_index_sequence<N>{}, name, mesh, values, programCtrl, allocate)) {}

            /**
             * @brief Constructor for fieldBase
             * @param[in] name Name of the field
             * @param[in] mesh Reference to the lattice mesh
             * @param[in] programCtrl Reference to the program control object
             * @param[in] allocate Whether to allocate memory and initialize values
             **/
            __host__ [[nodiscard]] fieldBase(
                const name_t &name,
                const host::latticeMesh &mesh,
                const programControl &programCtrl,
                const bool allocate = true)
                : name_(name),
                  meanCount_(initialiseMeanCount(name, programCtrl)),
                  components_(makeComponents(std::make_index_sequence<N>{}, name, mesh, programCtrl, allocate)) {}

            /**
             * @brief Get the field name.
             * @return Const reference to the name string.
             **/
            __host__ [[nodiscard]] inline constexpr const name_t &name() const noexcept
            {
                return name_;
            }

            /**
             * @brief Get the current averaging count (for time‑averaged fields).
             * @return Number of time steps averaged so far.
             **/
            __host__ [[nodiscard]] inline constexpr host::label_t meanCount() const noexcept
            {
                return meanCount_;
            }

            /**
             * @brief Get a reference to the averaging count (for modification).
             * @return Reference to meanCount_.
             **/
            __host__ [[nodiscard]] inline constexpr host::label_t &meanCountRef() noexcept
            {
                return meanCount_;
            }

            /**
             * @brief Destructor for fieldBase
             */
            ~fieldBase() {}

            /**
             * @brief Save the field data to disk using the provided Writer class.
             * @tparam Writer Class that implements a static write() method for output.
             * @param[in] hostWriteBuffer Host buffer used for copying data from the device before writing.
             * @param[in] timeStep Current time step index (used for output file naming).
             **/
            template <class Writer>
            __host__ void save(
                host::array<host::PINNED, scalar_t, VelocitySet, time::instantaneous> &hostWriteBuffer,
                const host::label_t timeStep) const
            {
                for (host::label_t virtualDeviceIndex = 0; virtualDeviceIndex < components_[0].programCtrl().deviceList().size(); virtualDeviceIndex++)
                {
                    hostWriteBuffer.copyFromDevice(
                        constPtr(virtualDeviceIndex),
                        components_[0].mesh(),
                        virtualDeviceIndex);
                }

                if constexpr (TimeType == time::instantaneous)
                {
                    Writer::write(
                        name_,
                        components_[0].mesh(),
                        makeComponentNames<std::vector<std::string>>(name_),
                        hostWriteBuffer.data(),
                        timeStep);
                }
                else
                {
                    Writer::write(
                        name_,
                        components_[0].mesh(),
                        makeComponentNames<std::vector<std::string>>(name_),
                        hostWriteBuffer.data(),
                        timeStep,
                        meanCount());
                }
            }

            /**
             * @brief Get a collection of const device pointers (one per component).
             * @param[in] idx Virtual device index.
             * @return ptrCollection with N const scalar_t*.
             */
            __host__ [[nodiscard]] inline constexpr device::ptrCollection<N, const scalar_t> constPtr(const host::label_t idx) const noexcept
            {
                return makeConstPtrCollection(idx, std::make_index_sequence<N>{});
            }

            /**
             * @brief Get a collection of non-const device pointers (one per component).
             * @param[in] idx Virtual device index.
             * @return ptrCollection with N scalar_t*.
             */
            __host__ [[nodiscard]] inline constexpr device::ptrCollection<N, scalar_t> ptr(const host::label_t idx) noexcept
            {
                return makePtrCollection(idx, std::make_index_sequence<N>{});
            }

        private:
            /**
             * @brief Helper function to generate component names based on the base name and the number of components (N).
             * @tparam ReturnType The type of the returned collection (e.g., std::array<std::string, N> or std::vector<std::string>).
             * @param[in] baseName The base name for the field, used to generate component names.
             * @return A collection of component names corresponding to the field components, following a consistent naming convention based on N.
             */
            template <class ReturnType>
            __host__ [[nodiscard]] static const ReturnType makeComponentNames(const name_t &baseName)
            {
                static_assert(N == 1 || N == 3 || N == 6, "Unsupported component count");

                if constexpr (N == 1)
                {
                    return {baseName};
                }
                else if constexpr (N == 3)
                {
                    return {baseName + "_x", baseName + "_y", baseName + "_z"};
                }
                else if constexpr (N == 6)
                {
                    return {baseName + "_xx", baseName + "_xy", baseName + "_xz", baseName + "_yy", baseName + "_yz", baseName + "_zz"};
                }
                else
                {
                    static_assert(N == 1 || N == 3 || N == 6, "Unsupported component count");
                }
            }

            /**
             * @brief Helper function to create components using pack expansion.
             * @tparam Is... Compile-time indices for pack expansion.
             * @param[in] baseName Base name for the components.
             * @param[in] mesh Reference to the lattice mesh.
             * @param[in] values Initial values for each component (only used if allocate=true).
             * @param[in] programCtrl Reference to the program control object.
             * @param[in] allocate Whether to allocate memory and initialize values.
             * @return std::array of ComponentType with N initialized components.
             */
            template <const host::label_t... Is>
            __host__ [[nodiscard]] static const std::array<ComponentType, N> makeComponents(
                const std::index_sequence<Is...>,
                const name_t &baseName,
                const host::latticeMesh &mesh,
                const std::array<scalar_t, N> values,
                const programControl &programCtrl,
                const bool allocate)
            {
                const std::array<std::string, N> compNames = makeComponentNames<std::array<std::string, N>>(baseName);
                // Use pack expansion to construct each element in-place
                return {ComponentType(baseName, compNames[Is], mesh, values[Is], programCtrl, allocate)...};
            }

            /**
             * @brief Helper function to create components using pack expansion.
             * @tparam Is... Compile-time indices for pack expansion.
             * @param[in] baseName Base name for the components.
             * @param[in] mesh Reference to the lattice mesh.
             * @param[in] programCtrl Reference to the program control object.
             * @param[in] allocate Whether to allocate memory and initialize values.
             * @return std::array of ComponentType with N initialized components.
             */
            template <const host::label_t... Is>
            __host__ [[nodiscard]] static const std::array<ComponentType, N> makeComponents(
                const std::index_sequence<Is...>,
                const name_t &baseName,
                const host::latticeMesh &mesh,
                const programControl &programCtrl,
                const bool allocate)
            {
                const std::array<std::string, N> compNames = makeComponentNames<std::array<std::string, N>>(baseName);
                return {ComponentType(baseName, compNames[Is], mesh, programCtrl, allocate)...};
            }

            /**
             * @brief Get a collection of const device pointers (one per component).
             * @param[in] idx Virtual device index.
             * @return ptrCollection with N const scalar_t*.
             */
            template <const host::label_t... Is>
            __host__ [[nodiscard]] inline constexpr const device::ptrCollection<N, const scalar_t> makeConstPtrCollection(const host::label_t idx, std::index_sequence<Is...>) const noexcept
            {
                return {components_[Is].constPtr(idx)...};
            }

            /**
             * @brief Get a collection of non-const device pointers (one per component).
             * @param[in] idx Virtual device index.
             * @return ptrCollection with N scalar_t*.
             */
            template <const host::label_t... Is>
            __host__ [[nodiscard]] inline constexpr const device::ptrCollection<N, scalar_t> makePtrCollection(const host::label_t idx, std::index_sequence<Is...>) noexcept
            {
                return {components_[Is].ptr(idx)...};
            }
        };

        template <class VelocitySet, const time::type TimeType>
        class scalarField : public fieldBase<VelocitySet, TimeType, 1>
        {
            using Base = fieldBase<VelocitySet, TimeType, 1>;

        public:
            using Base::Base; // Inherit all constructors from fieldBase

            /**
             * @brief Get a mutable reference to the scalar field.
             */
            __host__ [[nodiscard]] inline constexpr Base::ComponentType &self() noexcept { return Base::components_[0]; }

            /**
             * @brief Get a const reference to the scalar field.
             */
            __host__ [[nodiscard]] inline constexpr const Base::ComponentType &self() const noexcept { return Base::components_[0]; }
        };

        template <class VelocitySet, const time::type TimeType>
        class vectorField : public fieldBase<VelocitySet, TimeType, 3>
        {
            using Base = fieldBase<VelocitySet, TimeType, 3>;

        public:
            using Base::Base;

            /**
             * @brief Get a mutable reference to the components of the vector field.
             */
            __host__ [[nodiscard]] inline constexpr Base::ComponentType &x() noexcept { return Base::components_[0]; }
            __host__ [[nodiscard]] inline constexpr Base::ComponentType &y() noexcept { return Base::components_[1]; }
            __host__ [[nodiscard]] inline constexpr Base::ComponentType &z() noexcept { return Base::components_[2]; }

            /**
             * @brief Get a const reference to the components of the vector field.
             */
            __host__ [[nodiscard]] inline constexpr const Base::ComponentType &x() const noexcept { return Base::components_[0]; }
            __host__ [[nodiscard]] inline constexpr const Base::ComponentType &y() const noexcept { return Base::components_[1]; }
            __host__ [[nodiscard]] inline constexpr const Base::ComponentType &z() const noexcept { return Base::components_[2]; }
        };

        template <class VelocitySet, const time::type TimeType>
        class symmetricTensorField : public fieldBase<VelocitySet, TimeType, 6>
        {
            using Base = fieldBase<VelocitySet, TimeType, 6>;

        public:
            using Base::Base;

            /**
             * @brief Get a mutable reference to the components of the tensor field.
             */
            __host__ [[nodiscard]] inline constexpr Base::ComponentType &xx() noexcept { return Base::components_[0]; }
            __host__ [[nodiscard]] inline constexpr Base::ComponentType &xy() noexcept { return Base::components_[1]; }
            __host__ [[nodiscard]] inline constexpr Base::ComponentType &xz() noexcept { return Base::components_[2]; }
            __host__ [[nodiscard]] inline constexpr Base::ComponentType &yy() noexcept { return Base::components_[3]; }
            __host__ [[nodiscard]] inline constexpr Base::ComponentType &yz() noexcept { return Base::components_[4]; }
            __host__ [[nodiscard]] inline constexpr Base::ComponentType &zz() noexcept { return Base::components_[5]; }

            /**
             * @brief Get a const reference to the components of the tensor field.
             */
            __host__ [[nodiscard]] inline constexpr const Base::ComponentType &xx() const noexcept { return Base::components_[0]; }
            __host__ [[nodiscard]] inline constexpr const Base::ComponentType &xy() const noexcept { return Base::components_[1]; }
            __host__ [[nodiscard]] inline constexpr const Base::ComponentType &xz() const noexcept { return Base::components_[2]; }
            __host__ [[nodiscard]] inline constexpr const Base::ComponentType &yy() const noexcept { return Base::components_[3]; }
            __host__ [[nodiscard]] inline constexpr const Base::ComponentType &yz() const noexcept { return Base::components_[4]; }
            __host__ [[nodiscard]] inline constexpr const Base::ComponentType &zz() const noexcept { return Base::components_[5]; }
        };
    }
}

#endif
