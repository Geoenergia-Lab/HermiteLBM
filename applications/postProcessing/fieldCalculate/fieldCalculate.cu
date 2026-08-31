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
    Post-processing utility to calculate derived fields from saved moment fields
    Supported calculations: velocity magnitude, velocity divergence, vorticity,
    vorticity magnitude, integrated vorticity

Namespace
    LBM

SourceFiles
    fieldCalculate.cu

\*---------------------------------------------------------------------------*/

#include "fieldCalculate.cuh"

using namespace LBM;

int main(const int argc, const char *const argv[])
{
    const programControl programCtrl(argc, argv);

    // If the calculation type argument is not present, we cannot proceed, so we should print an error message and return
    if (!programCtrl.input().isArgPresent("-calculationType"))
    {
        errorHandler::handle(error::UNSPECIFIED_CALCULATIONTYPE);
        return 0;
    }

    // If the field name argument is not present, we cannot proceed, so we should print an error message and return
    if (!programCtrl.input().isArgPresent("-fieldName"))
    {
        errorHandler::handle(error::UNSPECIFIED_FIELDNAME);
        return 0;
    }

    const host::latticeMesh mesh(programCtrl);

    // If we have supplied a -fieldName argument, replace programCtrl.caseName() with the fieldName
    const name_t fieldName = programCtrl.getArgument("-fieldName");

    // Get the time indices
    const std::vector<host::label_t> fileNameIndices = programCtrl.timeStepIndices();

    // Parse the argument if present, otherwise set to empty string
    const name_t calculationType = programCtrl.getArgument("-calculationType");

    // Get the calculation function
    const std::unordered_map<name_t, calculator::functionType>::const_iterator it = calculators.find(calculationType);

    if (it != calculators.end())
    {
        const calculator::functionType calculation = it->second;

        if (!fileNameIndices.empty())
        {
            int status = 0;
            bool foundField = false;

            for (const host::label_t timeStep : fileNameIndices)
            {
                const name_t fileName = "timeStep/" + std::to_string(timeStep) + "/" + fieldName + ".LBMBin";
                const words_t fieldNames = fileIO::fieldInformation::readFieldNames(fieldName, fileName);

                // Initialise the fields to be processed
                const host::arrayCollection<scalar_t> hostMoments(fileName, fieldNames);

                if (!hostMoments.empty())
                {
                    foundField = true;

                    calculation(hostMoments, mesh, timeStep, status, fieldName);

                    if (!(timeStep == fileNameIndices.back()))
                    {
                        std::cout << std::endl;
                    }
                }
            }

            if (!foundField)
            {
                errorHandler::handle(error::FIELDNAME_NOT_FOUND);
            }
            else
            {
                return status;
            }
        }
        else
        {
            // We don't actually need to throw, we can just print the error message
            errorHandler::handle(error::EMPTY_TIMESTEP_DIRECTORY);
        }
    }
    else
    {
        errorHandler::handle(error::INVALID_CALCULATION_FUNCTION);
    }

    return 0;
}