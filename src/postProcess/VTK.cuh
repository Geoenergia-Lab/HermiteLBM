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
    VTK base class

Namespace
    LBM::postProcess

SourceFiles
    VTS.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_VTK_CUH
#define __MBLBM_VTK_CUH

namespace LBM
{
    namespace postProcess
    {
        /**
         * @brief Base class for VTK file writers
         * @tparam Structured Whether the grid is structured (StructuredGrid/ImageData) or unstructured (UnstructuredGrid)
         * @tparam Fields Whether the file contains field data
         * @tparam Points Whether the file contains point data
         * @tparam Elements Whether the file contains cell data (elements)
         * @tparam Offsets Whether the file contains cell offsets (required for unstructured grids)
         **/
        template <const bool Structured, const fileSystem::fields::contained Fields, const fileSystem::points::contained Points, const fileSystem::elements::contained Elements, const fileSystem::offsets::contained Offsets>
        class VTK : public writer
        {
        public:
            /**
             * @brief Compile-time constants describing the file format and contents
             **/
            static constexpr const fileSystem::format format = fileSystem::BINARY;
            static constexpr const fileSystem::fields::contained fields = Fields;
            static constexpr const fileSystem::points::contained points = Points;
            static constexpr const fileSystem::elements::contained elements = Elements;
            static constexpr const fileSystem::offsets::contained offsets = Offsets;

            /**
             * @brief Write the VTK file header and data sections
             * @param[in] solutionVars A vector of vectors containing the field data to be written (one vector per variable)
             * @param[out] outFile An open output file stream to which the VTK data will be written
             * @param[in] mesh The lattice mesh containing the grid dimensions and coordinates
             * @param[in] varNames A vector of variable names corresponding to the field data (used for the "Name" attribute in the VTK file)
             * @return True if the file was written successfully, false otherwise
             * @note This function writes the XML header and structure of the VTK file, then appends the binary data blocks for the fields, points, and cells as specified by the template parameters
             **/
            __host__ [[nodiscard]] static bool write(
                const std::vector<std::vector<scalar_t>> &solutionVars,
                std::ofstream &outFile,
                const host::latticeMesh &mesh,
                const words_t &varNames)
            {
                {
                    std::stringstream xml;
                    host::label_t currentOffset = 0;

                    xml << "<?xml " << xmlVersionString << "?>\n";
                    xml << "<VTKFile type=\"" << gridName() << "\" " << xmlVersionString << " byte_order=\"" << ((std::endian::native == std::endian::little) ? "LittleEndian" : "BigEndian") << "\" header_type=\"" << typeName<host::label_t>() << "\">\n";

                    writeGridSection(xml, solutionVars, varNames, mesh, currentOffset);

                    outFile << xml.str();
                }

                // Append the data
                append(solutionVars, outFile, mesh);
                outFile << "</VTKFile>\n";

                outFile.close();

                return outFile.good();
            }

        private:
            /**
             * @brief XML declaration string for the VTK file header
             **/
            static constexpr const char *xmlVersionString = "version=\"1.0\"";

            /**
             * @brief Append the binary data blocks for the fields, points, and cells to the VTK file
             * @param[in] solutionVars A vector of vectors containing the field data to be written
             * @param[out] outFile An open output file stream to which the binary data will be written
             * @param[in] mesh The lattice mesh containing the grid dimensions and coordinates (used for writing the points and cell connectivity data)
             **/
            __host__ static void append(
                const std::vector<std::vector<scalar_t>> &solutionVars,
                std::ofstream &outFile,
                const host::latticeMesh &mesh)
            {
                outFile << "  <AppendedData encoding=\"raw\">_";

                if constexpr (Fields == fileSystem::fields::Yes)
                {
                    fileIO::writeBinaryBlock(solutionVars, outFile);
                }

                if constexpr (Points == fileSystem::points::Yes)
                {
                    const std::vector<scalar_t> points = meshCoordinates<scalar_t>(mesh);

                    fileIO::writeBinaryBlock(points, outFile);
                }

                if constexpr ((Elements == fileSystem::elements::Yes) && (Offsets == fileSystem::offsets::Yes))
                {
                    const std::vector<host::label_t> connectivity = meshConnectivity<false, host::label_t>(mesh);
                    const std::vector<host::label_t> offsets = meshOffsets<host::label_t>(mesh);
                    const std::vector<nodeType_t> types((mesh.dimension<axis::X>() - 1) * (mesh.dimension<axis::Y>() - 1) * (mesh.dimension<axis::Z>() - 1), std::integral_constant<nodeType_t, 12>::value);

                    fileIO::writeBinaryBlock(connectivity, outFile);
                    fileIO::writeBinaryBlock(offsets, outFile);
                    fileIO::writeBinaryBlock(types, outFile);
                }

                outFile << "  </AppendedData>\n";
            }

            /**
             * @brief Write the XML structure for a single Piece section of the VTK file, including the PointData, Points, and Cells sections as specified by the template parameters
             * @param[out] xml A stringstream to which the XML structure will be written
             * @param[in] solutionVars A vector of vectors containing the field data to be written (used for determining the variable names and offsets in the PointData section)
             * @param[in] varNames A vector of variable names corresponding to the field data (used for the "Name" attribute in the PointData section)
             * @param[in] mesh The lattice mesh containing the grid dimensions and coordinates (used for determining the extents in the StructuredGrid section and for writing the Points and Cells sections)
             * @param[in,out] currentOffset A reference to a variable tracking the current byte offset in the appended data section, which is updated as each data block is accounted for in the XML structure
             **/
            __host__ static void writePiece(
                std::stringstream &xml,
                const std::vector<std::vector<scalar_t>> &solutionVars,
                const words_t &varNames,
                const host::latticeMesh &mesh,
                size_t &currentOffset)
            {
                if constexpr (Structured)
                {
                    const host::label_t dimX = mesh.dimension<axis::X>() - 1;
                    const host::label_t dimY = mesh.dimension<axis::Y>() - 1;
                    const host::label_t dimZ = mesh.dimension<axis::Z>() - 1;
                    xml << "    <Piece Extent=\"0 " << dimX << " 0 " << dimY << " 0 " << dimZ << "\">\n";
                }
                else
                {
                    const host::label_t numNodes = mesh.dimension<axis::X>() * mesh.dimension<axis::Y>() * mesh.dimension<axis::Z>();
                    const host::label_t numElements = (mesh.dimension<axis::X>() - 1) * (mesh.dimension<axis::Y>() - 1) * (mesh.dimension<axis::Z>() - 1);
                    xml << "    <Piece NumberOfPoints=\"" << numNodes << "\" NumberOfCells=\"" << numElements << "\">\n";
                }

                // Point data
                writePointData(xml, solutionVars, varNames, currentOffset);

                // Points section
                writePoints(xml, mesh, currentOffset);

                // Cells section
                writeCells(xml, mesh, currentOffset);

                xml << "    </Piece>\n";
            }

            /**
             * @brief Write the opening tag for the grid section of the VTK file, which is either <StructuredGrid>, <ImageData>, or <UnstructuredGrid> depending on the template parameters
             * @param[out] xml A stringstream to which the XML structure will be written
             * @param[in] mesh The lattice mesh containing the grid dimensions and coordinates (used for determining the extents in the StructuredGrid section)
             **/
            __host__ static inline void openGridSection(std::stringstream &xml, const host::latticeMesh &mesh)
            {
                if constexpr (Structured)
                {
                    // Calculate extents - note the -1 for the maximum indices
                    const host::label_t dimX = mesh.dimension<axis::X>() - 1;
                    const host::label_t dimY = mesh.dimension<axis::Y>() - 1;
                    const host::label_t dimZ = mesh.dimension<axis::Z>() - 1;

                    if (Points == fileSystem::points::Yes)
                    {
                        xml << "  <StructuredGrid WholeExtent=\"0 " << dimX << " 0 " << dimY << " 0 " << dimZ << "\">\n";
                    }
                    else
                    {
                        // ImageData coordinates are implicit
                        constexpr host::label_t ox = static_cast<host::label_t>(0);
                        constexpr host::label_t oy = static_cast<host::label_t>(0);
                        constexpr host::label_t oz = static_cast<host::label_t>(0);
                        constexpr host::label_t sx = static_cast<host::label_t>(1);
                        constexpr host::label_t sy = static_cast<host::label_t>(1);
                        constexpr host::label_t sz = static_cast<host::label_t>(1);
                        xml << "  <ImageData WholeExtent=\"0 " << dimX << " 0 " << dimY << " 0 " << dimZ << "\" Origin=\"" << ox << " " << oy << " " << oz << "\" Spacing=\"" << sx << " " << sy << " " << sz << "\">\n";
                    }
                }
                else
                {
                    xml << "  <UnstructuredGrid>\n";
                }
            }

            /**
             * @brief Write the XML structure for the grid section of the VTK file, including the PointData, Points, and Cells sections as specified by the template parameters
             * @param[out] xml A stringstream to which the XML structure will be written
             * @param[in] solutionVars A vector of vectors containing the field data to be written (used for determining the variable names and offsets in the PointData section)
             * @param[in] varNames A vector of variable names corresponding to the field data (used for the "Name" attribute in the PointData section)
             * @param[in] mesh The lattice mesh containing the grid dimensions and coordinates (used for determining the extents in the StructuredGrid section and for writing the Points and Cells sections)
             * @param[in,out] currentOffset A reference to a variable tracking the current byte offset in the appended data section, which is updated as each data block is accounted for in the XML structure
             **/
            __host__ static void writeGridSection(
                std::stringstream &xml,
                const std::vector<std::vector<scalar_t>> &solutionVars,
                const words_t &varNames,
                const host::latticeMesh &mesh,
                size_t &currentOffset)
            {
                openGridSection(xml, mesh);
                writePiece(xml, solutionVars, varNames, mesh, currentOffset);
                xml << "  </" << gridName() << ">\n";
            }

            /**
             * @brief Write the XML structure for the PointData section of the VTK file, including the DataArray entries for each variable in the solutionVars vector
             * @param[out] xml A stringstream to which the XML structure will be written
             * @param[in] solutionVars A vector of vectors containing the field data to be written (used for determining the variable names and offsets in the PointData section)
             * @param[in] varNames A vector of variable names corresponding to the field data (used for the "Name" attribute in the PointData section)
             * @param[in,out] currentOffset A reference to a variable tracking the current byte offset in the appended data section, which is updated as each data block is accounted for in the XML structure
             **/
            __host__ static void writePointData(std::stringstream &xml, const std::vector<std::vector<scalar_t>> &solutionVars, const std::vector<std::string> &varNames, size_t &currentOffset)
            {
                if constexpr (Fields == fileSystem::fields::Yes)
                {
                    xml << "      <PointData Scalars=\"" << (varNames.empty() ? "" : varNames[0]) << "\">\n";
                    for (host::label_t i = 0; i < solutionVars.size(); ++i)
                    {
                        xml << "        <DataArray type=\"" << typeName<scalar_t>() << "\" Name=\"" << varNames[i] << "\" format=\"appended\" offset=\"" << currentOffset << "\"/>\n";
                        currentOffset += sizeof(host::label_t) + solutionVars[i].size() * sizeof(scalar_t);
                    }
                    xml << "      </PointData>\n";
                }
            }

            /**
             * @brief Write the XML structure for the Points section of the VTK file, which contains the coordinates of the points in the grid, if the Points template parameter is set to Yes
             * @param[out] xml A stringstream to which the XML structure will be written
             * @param[in] mesh The lattice mesh containing the grid dimensions and coordinates (used for determining the number of points and their coordinates)
             * @param[in,out] currentOffset A reference to a variable tracking the current byte offset in the appended data section, which is updated as the points data block is accounted for in the XML structure
             **/
            __host__ static void writePoints(std::stringstream &xml, const host::latticeMesh &mesh, size_t &currentOffset)
            {
                if constexpr (Points == fileSystem::points::Yes)
                {
                    const host::label_t nx = mesh.dimension<axis::X>();
                    const host::label_t ny = mesh.dimension<axis::Y>();
                    const host::label_t nz = mesh.dimension<axis::Z>();
                    const host::label_t nPoints = nx * ny * nz * 3;

                    xml << "      <Points>\n";
                    xml << "        <DataArray type=\"" << typeName<scalar_t>() << "\" Name=\"Coordinates\" NumberOfComponents=\"3\" format=\"appended\" offset=\"" << currentOffset << "\"/>\n";
                    xml << "      </Points>\n";
                    currentOffset += sizeof(host::label_t) + nPoints * sizeof(scalar_t);
                }
            }

            /**
             * @brief Write the XML structure for the Cells section of the VTK file, which contains the connectivity and types of the cells in the grid, if the Elements and Offsets template parameters are set to Yes
             * @param[out] xml A stringstream to which the XML structure will be written
             * @param[in] mesh The lattice mesh containing the grid dimensions and coordinates (used for determining the number of cells and their connectivity)
             * @param[in,out] currentOffset A reference to a variable tracking the current byte offset in the appended data section, which is updated as the connectivity, offsets, and types data blocks are accounted for in the XML structure
             **/
            __host__ static void writeCells(std::stringstream &xml, const host::latticeMesh &mesh, size_t &currentOffset)
            {
                if constexpr ((Elements == fileSystem::elements::Yes) && (Offsets == fileSystem::offsets::Yes))
                {
                    const host::label_t nx = mesh.dimension<axis::X>();
                    const host::label_t ny = mesh.dimension<axis::Y>();
                    const host::label_t nz = mesh.dimension<axis::Z>();
                    const host::label_t nElements = (nx - 1) * (ny - 1) * (nz - 1);
                    const host::label_t nConnectivity = nElements * 8;

                    xml << "      <Cells>\n";
                    xml << "        <DataArray type=\"" << typeName<host::label_t>() << "\" Name=\"connectivity\" format=\"appended\" offset=\"" << currentOffset << "\"/>\n";
                    currentOffset += sizeof(host::label_t) + nConnectivity * sizeof(host::label_t);
                    xml << "        <DataArray type=\"" << typeName<host::label_t>() << "\" Name=\"offsets\" format=\"appended\" offset=\"" << currentOffset << "\"/>\n";
                    currentOffset += sizeof(host::label_t) + nElements * sizeof(host::label_t);
                    xml << "        <DataArray type=\"" << typeName<nodeType_t>() << "\" Name=\"types\" format=\"appended\" offset=\"" << currentOffset << "\"/>\n";
                    xml << "      </Cells>\n";
                }
            }

            /**
             * @brief Obtain the name of the grid type based on the template parameters
             * @return A string containing the name of the VTK grid type (e.g. "StructuredGrid", "ImageData", or "UnstructuredGrid")
             **/
            __host__ [[nodiscard]] static inline consteval const char *gridName() noexcept
            {
                if constexpr (Structured)
                {
                    if constexpr (Points == fileSystem::points::Yes)
                    {
                        return "StructuredGrid";
                    }
                    else
                    {
                        return "ImageData";
                    }
                }
                else
                {
                    return "UnstructuredGrid";
                }
            }

            /**
             * @brief Obtain the name of the type that corresponds to the C++ data type
             * @tparam T The C++ data type (e.g. float, int64_t)
             * @return A string containing the name of the VTK type (e.g. "Float32", "Int64")
             **/
            template <typename T>
            __host__ [[nodiscard]] static inline consteval const char *typeName() noexcept
            {
                if constexpr (std::is_same_v<T, float>)
                {
                    return "Float32";
                }
                else if constexpr (std::is_same_v<T, double>)
                {
                    return "Float64";
                }
                else if constexpr (std::is_same_v<T, int32_t>)
                {
                    return "Int32";
                }
                else if constexpr (std::is_same_v<T, uint32_t>)
                {
                    return "UInt32";
                }
                else if constexpr (std::is_same_v<T, int64_t>)
                {
                    return "Int64";
                }
                else if constexpr (std::is_same_v<T, uint64_t>)
                {
                    return "UInt64";
                }
                else if constexpr (std::is_same_v<T, uint8_t>)
                {
                    return "UInt8";
                }
                else if constexpr (std::is_same_v<T, int8_t>)
                {
                    return "Int8";
                }
                else
                {
                    static_assert(std::is_same_v<T, void>, "Unsupported type for getVtkTypeName");
                    return "Unknown";
                }
            }
        };
    }
}

#endif