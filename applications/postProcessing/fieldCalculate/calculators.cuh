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
    This file contains function definitions for calculating various properties
    of fields, such as checking for NaN values and calculating spatial means.
    These functions are used in the post-processing stage of the LBM simulation
    to analyze the results.

Namespace
    LBM

SourceFiles
    calculators.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_CALCULATORS_CUH
#define __MBLBM_CALCULATORS_CUH

namespace LBM
{
    namespace calculator
    {
        /**
         * @brief Type definition for a function pointer to a field calculation function
         **/
        using functionType = void (*)(
            const host::arrayCollection<scalar_t> &,
            const host::latticeMesh &,
            const host::label_t,
            int &,
            const std::string &);

        /**
         * @brief Checks if a field contains any NaN values
         * @param[in] field The field to check
         * @return True if the field contains NaN values, false otherwise
         **/
        __host__ [[nodiscard]] inline bool containsNaN(const std::vector<scalar_t> &field) noexcept
        {
            for (const scalar_t &value : field)
            {
                if (std::isnan(value))
                {
                    return true;
                }
            }
            return false;
        }

        /**
         * @brief Checks if any of the fields in the arrayCollection contain NaN values and prints the result
         * @param[in] variables The arrayCollection containing the fields to check
         * @param[in] mesh The lattice mesh
         * @param[in] timeStep The current time step for logging purposes
         * @param[in,out] status The status flag for error handling
         * @param[in] fileName The field name to operate
         **/
        __host__ void containsNaN(
            const host::arrayCollection<scalar_t> &variables,
            const host::latticeMesh &mesh,
            const host::label_t timeStep,
            int &status,
            [[maybe_unused]] const name_t &fileName) noexcept
        {
            // De-interleave the fields
            const std::vector<std::vector<scalar_t>> fields = variables.deinterleaveAoS(mesh);

            std::cout << "Time: " << timeStep << std::endl;
            std::cout << "{" << std::endl;

            host::label_t numberNaNs = 0;

            // Loop over the fields checking for NaN
            for (host::label_t field = 0; field < fields.size(); field++)
            {
                if (containsNaN(fields[field]))
                {
                    std::cout << "    NaN detected in field " << variables.varNames()[field] << std::endl;
                    numberNaNs++;
                }
            }

            if (numberNaNs == 0)
            {
                std::cout << "    No NaN values detected in any field." << std::endl;
            }
            else
            {
                status = 1; // Set status to indicate that NaN values were found
                std::cout << "    Total number of fields with NaN values: " << numberNaNs << std::endl;
            }

            std::cout << "};" << std::endl;
        }
    }
}

#include "reductionCalculators.cuh"
#include "pointwiseCalculators.cuh"

#endif