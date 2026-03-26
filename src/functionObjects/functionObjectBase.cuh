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
    Base class for LBM function objects, containing common data members.

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_FUNCTIONOBJECTBASE_CUH
#define __MBLBM_FUNCTIONOBJECTBASE_CUH

namespace LBM
{
    namespace functionObjects
    {
        /**
         * @brief Base class for LBM function objects, providing common data members.
         * @tparam VelocitySet The velocity set (D3Q19 or D3Q27)
         * @tparam N The number of spatial components of the function object
         */
        template <class VelocitySet, const host::label_t N>
        class FunctionObjectBase
        {
        protected:
            /**
             * @brief Name of the field and its time-averaged counterpart
             **/
            const name_t name_;
            const name_t nameMean_;

            /**
             * @brief Name of the field's components and their time-averaged counterpart
             **/
            const words_t componentNames_;
            const words_t componentNamesMean_;

            /**
             * @brief Switches to determine whether or not the field is to be calculated
             **/
            const bool calculate_;
            const bool calculateMean_;

            /**
             * @brief Reference to the write buffer
             **/
            host::array<host::PINNED, scalar_t, VelocitySet, time::instantaneous> &hostWriteBuffer_;

            /**
             * @brief Reference to lattice mesh
             **/
            const host::latticeMesh &mesh_;

            /**
             * @brief Device pointer collection
             **/
            const device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> &rho_;
            const device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> &u_;
            const device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> &v_;
            const device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> &w_;
            const device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> &mxx_;
            const device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> &mxy_;
            const device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> &mxz_;
            const device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> &myy_;
            const device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> &myz_;
            const device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> &mzz_;

            /**
             * @brief Stream handler for CUDA operations
             **/
            const streamHandler &streamsLBM_;

            /**
             * @brief Construct the component names from the field name
             **/
            __host__ [[nodiscard]] static inline constexpr const words_t component_names(const name_t &name)
            {
                if constexpr (N == 1)
                {
                    return {name};
                }

                if constexpr (N == 6)
                {
                    return string::catenate(name, {"_xx", "_xy", "_xz", "_yy", "_yz", "_zz"});
                }

                if constexpr (N == 10)
                {
                    return solutionVariableNames;
                }
            }

        public:
            /**
             * @brief Constructs a function object base with common input data.
             * @param[in] hostWriteBuffer Host buffer for writing output data.
             * @param[in] mesh Lattice mesh.
             * @param[in] rho Density field.
             * @param[in] u x‑velocity field.
             * @param[in] v y‑velocity field.
             * @param[in] w z‑velocity field.
             * @param[in] mxx xx‑moment field.
             * @param[in] mxy xy‑moment field.
             * @param[in] mxz xz‑moment field.
             * @param[in] myy yy‑moment field.
             * @param[in] myz yz‑moment field.
             * @param[in] mzz zz‑moment field.
             * @param[in] streamsLBM Stream handler for LBM operations
             */
            __host__ [[nodiscard]] FunctionObjectBase(
                const name_t &name,
                host::array<host::PINNED, scalar_t, VelocitySet, time::instantaneous> &hostWriteBuffer,
                const host::latticeMesh &mesh,
                const device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> &rho,
                const device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> &u,
                const device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> &v,
                const device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> &w,
                const device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> &mxx,
                const device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> &mxy,
                const device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> &mxz,
                const device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> &myy,
                const device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> &myz,
                const device::array<field::FULL_FIELD, scalar_t, VelocitySet, time::instantaneous> &mzz,
                const streamHandler &streamsLBM) noexcept
                : name_(name),
                  nameMean_(name + "Mean"),
                  componentNames_(component_names(name_)),
                  componentNamesMean_(component_names(nameMean_)),
                  calculate_(initialiserSwitch(name_)),
                  calculateMean_(initialiserSwitch(nameMean_)),
                  hostWriteBuffer_(hostWriteBuffer),
                  mesh_(mesh),
                  rho_(rho),
                  u_(u),
                  v_(v),
                  w_(w),
                  mxx_(mxx),
                  mxy_(mxy),
                  mxz_(mxz),
                  myy_(myy),
                  myz_(myz),
                  mzz_(mzz),
                  streamsLBM_(streamsLBM) {}

            /**
             * @brief Check if instantaneous calculation is enabled
             * @return True if instantaneous calculation is enabled
             **/
            __host__ [[nodiscard]] inline constexpr bool doInstantaneous() const noexcept
            {
                return calculate_;
            }

            /**
             * @brief Check if mean calculation is enabled
             * @return True if mean calculation is enabled
             **/
            __host__ [[nodiscard]] inline constexpr bool doMean() const noexcept
            {
                return calculateMean_;
            }
        };
    }
}

#endif // __MBLBM_FUNCTIONOBJECTBASE_CUH