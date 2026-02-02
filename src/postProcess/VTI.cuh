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
Authors: Nathan Duggins, Breno Gemelgo (Geoenergia Lab, UDESC)

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
    VTI binary file writer

Namespace
    LBM::postProcess

SourceFiles
    VTI.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_VTI_CUH
#define __MBLBM_VTI_CUH

namespace LBM
{
    namespace postProcess
    {
        namespace VTI
        {
            __host__ [[nodiscard]] inline consteval bool hasFields() { return true; }
            __host__ [[nodiscard]] inline consteval bool hasPoints() { return false; }
            __host__ [[nodiscard]] inline consteval bool hasElements() { return false; }
            __host__ [[nodiscard]] inline consteval bool hasOffsets() { return false; }
            __host__ [[nodiscard]] inline consteval const char *fileExtension() { return ".vti"; }

            /**
             * @brief Auxiliary template function that performs the VTI file writing.
             */
            __host__ void VTIWriter(
                const std::vector<std::vector<scalar_t>> &solutionVars,
                std::ofstream &outFile,
                const host::latticeMesh &mesh,
                const std::vector<std::string> &solutionVarNames) noexcept
            {
                const std::size_t numVars = solutionVars.size();

                std::stringstream xml;
                uint64_t currentOffset = 0;

                // Calculate extents - note the -1 for the maximum indices
                const std::size_t dimX = mesh.nx<std::size_t>() - 1;
                const std::size_t dimY = mesh.ny<std::size_t>() - 1;
                const std::size_t dimZ = mesh.nz<std::size_t>() - 1;

                // ImageData coordinates are implicit
                constexpr scalar_t ox = static_cast<scalar_t>(0);
                constexpr scalar_t oy = static_cast<scalar_t>(0);
                constexpr scalar_t oz = static_cast<scalar_t>(0);
                constexpr scalar_t sx = static_cast<scalar_t>(1);
                constexpr scalar_t sy = static_cast<scalar_t>(1);
                constexpr scalar_t sz = static_cast<scalar_t>(1);

                xml << "<?xml version=\"1.0\"?>\n";
                xml << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
                xml << "  <ImageData WholeExtent=\"0 " << dimX << " 0 " << dimY << " 0 " << dimZ
                    << "\" Origin=\"" << ox << " " << oy << " " << oz
                    << "\" Spacing=\"" << sx << " " << sy << " " << sz << "\">\n";
                xml << "    <Piece Extent=\"0 " << dimX << " 0 " << dimY << " 0 " << dimZ << "\">\n";

                // Point data (same as before)
                xml << "      <PointData Scalars=\"" << (solutionVarNames.empty() ? "" : solutionVarNames[0]) << "\">\n";
                for (std::size_t i = 0; i < numVars; ++i)
                {
                    xml << "        <DataArray type=\"" << getVtkTypeName<scalar_t>() << "\" Name=\"" << solutionVarNames[i] << "\" format=\"appended\" offset=\"" << currentOffset << "\"/>\n";
                    currentOffset += sizeof(std::size_t) + solutionVars[i].size() * sizeof(scalar_t);
                }
                xml << "      </PointData>\n";

                // No explicit <Points> section for ImageData (points are implicit via Origin/Spacing)
                // No <Cells> section

                xml << "    </Piece>\n";
                xml << "  </ImageData>\n";
                xml << "  <AppendedData encoding=\"raw\">_";

                outFile << xml.str();

                // Write point data arrays
                for (const auto &varData : solutionVars)
                {
                    writeBinaryBlock(varData, outFile);
                }

                outFile << "</AppendedData>\n";
                outFile << "</VTKFile>\n";

                outFile.close();
            }

            /**
             * @brief Writes solution variables to an ImageData VTI file (.vti)
             * This function checks the mesh size and dispatches to the implementation with
             * the appropriate index type (32-bit or 64-bit).
             */
            __host__ void write(
                const std::vector<std::vector<scalar_t>> &solutionVars,
                const std::string &fileName,
                const host::latticeMesh &mesh,
                const std::vector<std::string> &solutionVarNames)
            {
                const std::size_t numNodes = static_cast<std::size_t>(mesh.nx()) * static_cast<std::size_t>(mesh.ny()) * static_cast<std::size_t>(mesh.nz());
                const std::size_t numVars = solutionVars.size();

                if (numVars != solutionVarNames.size())
                {
                    errorHandler(-1, "Error: The number of solution (" + std::to_string(numVars) + ") does not match the count of variable names (" + std::to_string(solutionVarNames.size()));
                }

                for (std::size_t i = 0; i < numVars; i++)
                {
                    if (solutionVars[i].size() != numNodes)
                    {
                        errorHandler(-1, "Error: The solution variable " + std::to_string(i) + " has " + std::to_string(solutionVars[i].size()) + " elements, expected " + std::to_string(numNodes));
                    }
                }

                std::cout << "vtiWriter:" << std::endl;
                std::cout << "{" << std::endl;
                std::cout << "    fileName: " << directoryPrefix() << "/" << fileName << fileExtension() << ";" << std::endl;

                if (!std::filesystem::is_directory(directoryPrefix()))
                {
                    if (!std::filesystem::create_directory(directoryPrefix()))
                    {
                        std::cout << "    directoryStatus: unable to create directory" << directoryPrefix() << ";" << std::endl;
                        std::cout << "    writeStatus: fail (unable to create directory)" << ";" << std::endl;
                        std::cout << "};" << std::endl;
                        errorHandler(-1, "Error: unable to create directory" + std::string(directoryPrefix()));
                    }
                }
                else
                {
                    std::cout << "    directoryStatus: OK;" << std::endl;
                }

                std::cout << "    fileSize: " << fileSystem::to_mebibytes<double>(fileSystem::expectedDiskUsage<fileSystem::BINARY, hasFields(), hasPoints(), hasElements(), hasOffsets()>(mesh, solutionVars.size())) << " MiB;" << std::endl;

                // Check if there is enough disk space to store the file
                if (!fileSystem::diskSpaceCheck<fileSystem::ASCII, hasFields(), hasPoints(), hasElements(), hasOffsets()>(mesh, solutionVars.size()))
                {
                    std::cout << "    diskSpace: insufficient (" << fileSystem::to_mebibytes<double>(fileSystem::availableDiskSpace()) << " MiB);" << std::endl;
                    std::cout << "    writeStatus: fail (insufficient disk space)" << ";" << std::endl;
                    std::cout << "};" << std::endl;
                    errorHandler(-1, "Error: Insufficient disk space on drive " + fileSystem::diskName());
                }
                else
                {
                    std::cout << "    diskSpace: OK (" << fileSystem::to_mebibytes<double>(fileSystem::availableDiskSpace()) << " MiB);" << std::endl;
                }

                // Check if there is enough disk space to store the file
                fileSystem::diskSpaceAssertion<fileSystem::BINARY, hasFields(), hasPoints(), hasElements(), hasOffsets()>(mesh, solutionVars.size(), fileName);

                const std::string trueFileName(std::string(directoryPrefix()) + "/" + fileName + fileExtension());

                std::ofstream outFile(trueFileName, std::ios::binary);
                if (outFile)
                {
                    std::cout << "    ofstreamStatus: OK;" << std::endl;
                }
                else
                {
                    std::cout << "    ofstreamStatus: Fail" << std::endl;
                    std::cout << "};" << std::endl;
                    errorHandler(-1, "Error opening file: " + trueFileName);
                }

                VTIWriter(solutionVars, outFile, mesh, solutionVarNames);
                std::cout << "    writeStatus: success" << ";" << std::endl;
                std::cout << "};" << std::endl;
                std::cout << std::endl;
            }
        }
    }
}

#endif