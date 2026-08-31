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
    LBMBin binary file writer

Namespace
    LBM::postProcess

SourceFiles
    LBMBin.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_LBMBIN_CUH
#define __MBLBM_LBMBIN_CUH

namespace LBM
{
    namespace postProcess
    {
        /**
         * @brief Writer class for LBMBin binary files.
         *
         * This class handles writing lattice Boltzmann field data to a custom binary
         * format with a .LBMBin extension. It supports both raw pointer and
         * vector-of-vectors field containers, and can write both instantaneous and
         * time-averaged data
         **/
        class LBMBin : public writer
        {
        public:
            static constexpr const fileSystem::format file_format = fileSystem::BINARY;
            static constexpr const fileSystem::fields::contained has_fields = fileSystem::fields::Yes;
            static constexpr const fileSystem::points::contained has_points = fileSystem::points::No;
            static constexpr const fileSystem::elements::contained has_elements = fileSystem::elements::No;
            static constexpr const fileSystem::offsets::contained has_offsets = fileSystem::offsets::No;

            static constexpr const char *fileExtension = ".LBMBin";
            static constexpr const char *name = "LBMBin";

            __host__ [[nodiscard]] inline consteval LBMBin() {}

            using This = LBMBin;

            /**
             * @brief Write field data to an LBMBin file.
             *
             * This method writes the provided field data (stored either as a raw
             * pointer or as a vector of vectors) along with mesh and system
             * information to a binary file. The data is always written in
             * time-averaged mode (time::timeAverage) regardless of the value of
             * meanCount.
             *
             * @tparam Fields Type of the field container. Can be a raw pointer (e.g.,
             *                const T*) or a std::vector<std::vector<T>>.
             * @param fileName   Base name of the output file (without directory).
             * @param mesh       The host-side lattice mesh.
             * @param varNames   Names of the field variables.
             * @param fields     Container holding the field data.
             * @param timeStep   Current time step (used for directory naming).
             * @param meanCount  Number of samples used in the time average
             *                   (unused in the current implementation).
             **/
            template <typename Fields>
            __host__ static void write(
                const name_t &fileName,
                const host::latticeMesh &mesh,
                const words_t &varNames,
                const Fields &fields,
                const host::label_t timeStep,
                const host::label_t meanCount)
            {
                common_write<time::timeAverage>(fileName, mesh, varNames, fields, timeStep, meanCount);
            }

            template <typename Fields>
            __host__ static void write(
                const name_t &fileName,
                const host::latticeMesh &mesh,
                const words_t &varNames,
                const Fields &fields,
                const host::label_t timeStep)
            {
                common_write<time::instantaneous>(fileName, mesh, varNames, fields, timeStep, 0);
            }

            /**
             * @brief Construct a full output filename from a directory name and a field name.
             *
             * @param dirName    Directory portion (e.g., the time step as a string).
             * @param fieldName  Base name of the field/file.
             * @return A string of the form "timeStep/<dirName>/<fieldName>.LBMBin".
             **/
            __host__ static inline const name_t make_filename(const name_t &dirName, const name_t &fieldName)
            {
                return "timeStep/" + dirName + "/" + fieldName + fileExtension;
            }

            /**
             * @brief Construct a full output filename from a time step and a field name.
             *
             * @param timeStep   Numeric time step (converted to string).
             * @param fieldName  Base name of the field/file.
             * @return A string of the form "timeStep/<timeStep>/<fieldName>.LBMBin".
             **/
            __host__ static inline const name_t make_filename(const host::label_t timeStep, const name_t &fieldName)
            {
                return make_filename(std::to_string(timeStep), fieldName);
            }

        private:
            /**
             * @brief Create the directory structure for a given time step.
             *
             * Ensures that the "timeStep" directory and its subdirectory for the
             * current time step exist. Throws std::runtime_error on failure.
             *
             * @param timeStep Current time step used to name the subdirectory.
             **/
            __host__ static void createTimeStepDirectories(const host::label_t timeStep)
            {
                if (!fileSystem::makeDirectory("timeStep"))
                {
                    throw std::runtime_error("Error: unable to create timeStep directory");
                }
                if (!fileSystem::makeDirectory("timeStep/" + std::to_string(timeStep)))
                {
                    throw std::runtime_error("Error: unable to create directory for time step " + std::to_string(timeStep));
                }
            }

            /**
             * @brief Core writing routine shared by all public write methods.
             *
             * Performs validation, opens the output file, writes system/mesh
             * information, field information, and the actual binary field data.
             *
             * @tparam TimeType  Time type (time::instantaneous or time::timeAverage).
             * @tparam Fields    Type of the field container (raw pointer or vector of vectors).
             * @param fileName   Base name of the output file (without directory).
             * @param mesh       The host-side lattice mesh.
             * @param varNames   Names of the field variables.
             * @param fields     Container holding the field data.
             * @param timeStep   Current time step.
             * @param meanCount  Number of samples for time averaging (only used when
             *                   TimeType is time::timeAverage).
             **/
            template <const time::type TimeType, typename Fields>
            __host__ static void common_write(
                const name_t &fileName,
                const host::latticeMesh &mesh,
                const words_t &varNames,
                const Fields &fields,
                const host::label_t timeStep,
                const host::label_t meanCount)
            {
                const name_t resolvedName = make_filename(timeStep, fileName);
                createTimeStepDirectories(timeStep);

                //  types::assertions::validate<typename std::remove_pointer_t<Fields>>();
                endian::assertions::validate();

                // Check disk space
                writer::diskSpaceAssertion<This>(mesh, varNames, resolvedName);

                std::ofstream out(resolvedName, std::ios::binary);
                if (!out)
                {
                    throw std::runtime_error("Cannot open file: " + resolvedName);
                }

                // Write system and mesh information
                system::print(out);
                mesh.dimensions().print<true>("latticeMesh", out);
                mesh.nDevices().print<true>("deviceDecomposition", out);

                // Write field information
                writeFieldInformation<TimeType>(timeStep, varNames, meanCount, out);

                // Write the actual field data (handles both pointer and vector cases)
                writeFieldData(mesh, fields, varNames, out);
            }

            /**
             * @brief Write the field data block to the output stream.
             *
             * Handles both raw pointer and vector-of-vectors field containers.
             * For raw pointers, the byte size is computed and written as a contiguous
             * block. For vectors, the overload of fileIO::writeBinaryBlock that
             * accepts a vector of vectors is used.
             *
             * @tparam Fields Type of the field container.
             * @param mesh     The host-side lattice mesh (used to determine dimensions).
             * @param fields   The field data to write.
             * @param varNames Names of the variables (used to compute expected size).
             * @param out      Output stream (already opened in binary mode).
             **/
            template <typename Fields>
            __host__ static void writeFieldData(
                const host::latticeMesh &mesh,
                const Fields &fields,
                const words_t &varNames,
                std::ofstream &out)
            {
                const host::label_t nPoints = mesh.size();
                const host::label_t expectedSize = nPoints * varNames.size();

                out << "fieldData" << std::endl;
                out << "{" << std::endl;
                IO::printFields(out, "fieldType", "nonUniform");
                out << IO::whitespace<4>{} << "field[" << expectedSize << "][" << varNames.size() << "][" << mesh.template dimension<axis::Z>() << "][" << mesh.template dimension<axis::Y>() << "][" << mesh.template dimension<axis::X>() << "]" << std::endl;
                out << IO::whitespace<4>{} << "{" << std::endl;

                if constexpr (std::is_pointer_v<Fields>)
                {
                    // Raw pointer: compute byte size and write directly.
                    using T = std::remove_cv_t<std::remove_pointer_t<Fields>>;
                    const host::label_t byteSize = expectedSize * sizeof(T);
                    fileIO::writeBinaryBlock(fields, byteSize, out);
                }
                else
                {
                    // Vector-of-vectors: let the overload handle the writing.
                    fileIO::writeBinaryBlock(fields, out);
                }

                out << std::endl;
                out << IO::whitespace<4>{} << "};" << std::endl;
                out << "};" << std::endl;
            }

            /**
             * @brief Write field metadata (time step, time type, field names, etc.).
             *
             * @tparam TimeType  Time type (instantaneous or time-averaged).
             * @param timeStep   Current time step.
             * @param varNames   Names of the fields.
             * @param meanCount  Number of samples used in averaging (only written when
             *                   TimeType is time::timeAverage).
             * @param out        Output stream.
             **/
            template <const time::type TimeType>
            __host__ static void writeFieldInformation(
                const host::label_t timeStep,
                const words_t &varNames,
                const host::label_t meanCount,
                std::ofstream &out)
            {
                out << "fieldInformation" << std::endl;
                out << "{" << std::endl;

                IO::printFields(out, "timeStep", timeStep);
                IO::printFields(out, "timeType", time::nameString<TimeType>());

                if constexpr (TimeType == time::timeAverage)
                {
                    IO::printFields(out, "meanCount", meanCount);
                }

                IO::printFields(out, "nFields", varNames.size());
                IO::print_container_expanded<4, 8>(out, varNames, "fieldNames");
                out << "};" << std::endl;
                out << std::endl;
            }
        };
    }
}

#endif