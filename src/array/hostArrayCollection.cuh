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
Authors: Nathan Duggins (Geoenergia Lab, UDESC)

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
    A templated class for allocating collections of arrays on the CPU

Namespace
    LBM::host

SourceFiles
    hostArrayCollection.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_HOSTARRAYCOLLECTION_CUH
#define __MBLBM_HOSTARRAYCOLLECTION_CUH

namespace LBM
{
    namespace host
    {
        /**
         * @class arrayCollection
         * @brief Templated container for multiple field arrays with flexible initialization
         * @tparam T Data type of array elements
         * @tparam cType Constructor type specification
         **/
        template <typename T>
        class arrayCollection
        {
        public:
            __host__ [[nodiscard]] arrayCollection(const name_t &fileName, const words_t &varNames)
                : empty_(!(std::filesystem::exists(fileName))),
                  arr_(initialiseVector(fileName, empty_)),
                  varNames_(varNames) {}

            /**
             * @brief Destructor for the host arrayCollection class
             **/
            ~arrayCollection() {}

            /**
             * @brief Get read-only access to underlying data
             * @return Const reference to data vector
             **/
            __host__ [[nodiscard]] inline constexpr const std::vector<T> &arr() const noexcept
            {
                return arr_;
            }

            /**
             * @brief Get variable names in collection
             * @return Const reference to variable names vector
             **/
            __host__ [[nodiscard]] inline const words_t &varNames() const noexcept
            {
                return varNames_;
            }

            __host__ [[nodiscard]] inline constexpr bool empty() const noexcept
            {
                return empty_;
            }

        private:
            const bool empty_;

            /**
             * @brief The underlying std::vector
             **/
            const std::vector<T> arr_;

            /**
             * @brief Names of the solution variables
             **/
            const words_t varNames_;

            /**
             * @brief Initialize vector from mesh dimensions
             * @param[in] programCtrl The program control object
             * @param[in] mesh The lattice mesh
             * @return Initialized data vector
             * @throws std::runtime_error if indexed files not found
             **/
            __host__ [[nodiscard]] static const std::vector<T> initialiseVector(const name_t &fileName, const bool empty)
            {
                if (empty)
                {
                    return std::vector<T>(); // Return an empty vector if the file was not found
                }
                else
                {
                    return fileIO::readFieldFile<T>(fileName);
                }
            }
        };
    }
}

#endif
