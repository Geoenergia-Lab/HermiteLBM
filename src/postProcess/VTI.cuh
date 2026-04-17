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
        class VTI : public writer
        {
        public:
            __host__ [[nodiscard]] static inline consteval fileSystem::format format() noexcept { return fileSystem::BINARY; }
            __host__ [[nodiscard]] static inline consteval fileSystem::fields::contained hasFields() noexcept { return fileSystem::fields::Yes; }
            __host__ [[nodiscard]] static inline consteval fileSystem::points::contained hasPoints() noexcept { return fileSystem::points::No; }
            __host__ [[nodiscard]] static inline consteval fileSystem::elements::contained hasElements() noexcept { return fileSystem::elements::No; }
            __host__ [[nodiscard]] static inline consteval fileSystem::offsets::contained hasOffsets() noexcept { return fileSystem::offsets::No; }
            __host__ [[nodiscard]] static inline consteval const char *fileExtension() noexcept { return ".vti"; }
            __host__ [[nodiscard]] static inline consteval const char *name() noexcept { return "VTI"; }

            __host__ [[nodiscard]] inline consteval VTI(){};

            /**
             * @brief Auxiliary template function that performs the VTI file writing.
             */
            __host__ static bool write(
                const std::vector<std::vector<scalar_t>> &solutionVars,
                std::ofstream &outFile,
                const host::latticeMesh &mesh,
                const words_t &varNames) noexcept
            {
                const host::label_t numVars = solutionVars.size();

                {
                    std::stringstream xml;
                    host::label_t currentOffset = 0;

                    // Calculate extents - note the -1 for the maximum indices
                    const host::label_t dimX = mesh.dimension<axis::X>() - 1;
                    const host::label_t dimY = mesh.dimension<axis::Y>() - 1;
                    const host::label_t dimZ = mesh.dimension<axis::Z>() - 1;

                    // ImageData coordinates are implicit
                    constexpr scalar_t ox = static_cast<scalar_t>(0);
                    constexpr scalar_t oy = static_cast<scalar_t>(0);
                    constexpr scalar_t oz = static_cast<scalar_t>(0);
                    constexpr scalar_t sx = static_cast<scalar_t>(1);
                    constexpr scalar_t sy = static_cast<scalar_t>(1);
                    constexpr scalar_t sz = static_cast<scalar_t>(1);

                    xml << "<?xml version=\"1.0\"?>\n";
                    xml << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
                    xml << "  <ImageData WholeExtent=\"0 " << dimX << " 0 " << dimY << " 0 " << dimZ << "\" Origin=\"" << ox << " " << oy << " " << oz << "\" Spacing=\"" << sx << " " << sy << " " << sz << "\">\n";
                    xml << "    <Piece Extent=\"0 " << dimX << " 0 " << dimY << " 0 " << dimZ << "\">\n";

                    xml << "      <PointData Scalars=\"" << (varNames.empty() ? "" : varNames[0]) << "\">\n";
                    for (host::label_t i = 0; i < numVars; ++i)
                    {
                        xml << "        <DataArray type=\"" << getVtkTypeName<scalar_t>() << "\" Name=\"" << varNames[i] << "\" format=\"appended\" offset=\"" << currentOffset << "\"/>\n";
                        currentOffset += sizeof(host::label_t) + solutionVars[i].size() * sizeof(scalar_t);
                    }
                    xml << "      </PointData>\n";

                    xml << "    </Piece>\n";
                    xml << "  </ImageData>\n";
                    xml << "  <AppendedData encoding=\"raw\">_";

                    outFile << xml.str();
                }

                // Write point data arrays
                for (const auto &varData : solutionVars)
                {
                    fileIO::writeBinaryBlock(varData, outFile);
                }

                outFile << "</AppendedData>\n";
                outFile << "</VTKFile>\n";

                outFile.close();

                return outFile.good();
            }
        };
    }
}

#endif