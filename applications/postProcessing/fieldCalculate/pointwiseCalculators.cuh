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
    Defines per-point calculation operations that output a new field

Namespace
    LBM

SourceFiles
    pointwiseCalculators.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_POINTWISECALCULATORS_CUH
#define __MBLBM_POINTWISECALCULATORS_CUH

namespace LBM
{
    namespace calculator
    {
        /**
         * @brief Writes the magnitude (or squared magnitude) of a tensor field to file.
         * @tparam Squared  If true, computes the squared magnitude; otherwise the regular magnitude.
         * @param[in] variables The arrayCollection containing the fields.
         * @param[in] mesh The lattice mesh.
         * @param[in] timeStep Current simulation time step (for logging).
         * @param[in] fileName Base name for the output file.
         */
        template <const numericalSchemes::sqMode Squared>
        __host__ void magnitudeImpl(
            const host::arrayCollection<scalar_t> &variables,
            const host::latticeMesh &mesh,
            const host::label_t timeStep,
            const name_t &fileName) noexcept
        {
            const std::vector<std::vector<scalar_t>> fields = variables.splitFieldsRaw(mesh);

            std::cout << "Time: " << timeStep << std::endl;
            std::cout << "{" << std::endl;

            const std::vector<scalar_t> result = numericalSchemes::mag<Squared>(fields);
            const std::string suffix = Squared ? "MagSq" : "Mag";
            const std::string outputName = fileName + suffix;

            postProcess::LBMBin::write(outputName, mesh, {fileName}, result.data(), timeStep);
            std::cout << "    Written " << (Squared ? "squared " : "") << "magnitude to " << outputName << std::endl;

            std::cout << "};" << std::endl;
        }

        /**
         * @brief Convenience function - calculates the magnitude
         **/
        __host__ void magnitude(
            const host::arrayCollection<scalar_t> &variables,
            const host::latticeMesh &mesh,
            const host::label_t timeStep,
            [[maybe_unused]] int &status,
            const name_t &fileName) noexcept
        {
            magnitudeImpl<numericalSchemes::NOT_SQUARED>(variables, mesh, timeStep, fileName);
        }

        /**
         * @brief Convenience function - calculates the square of the magnitude
         **/
        __host__ void magnitudeSquared(
            const host::arrayCollection<scalar_t> &variables,
            const host::latticeMesh &mesh,
            const host::label_t timeStep,
            [[maybe_unused]] int &status,
            const name_t &fileName) noexcept
        {
            magnitudeImpl<numericalSchemes::SQUARED>(variables, mesh, timeStep, fileName);
        }
    }
}

#endif