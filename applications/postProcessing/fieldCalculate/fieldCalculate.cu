/*---------------------------------------------------------------------------*\
|                                                                             |
| HermiteLBM: CUDA-based moment representation Lattice Boltzmann Method       |
| Developed at UDESC - State University of Santa Catarina                     |
| Website: https://www.udesc.br                                               |
| Github: https://github.com/Geoenergia-Lab/cudaLBM                           |
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
        errorHandler::check<throws::NO_THROW>(-1, "Unspecified calculation type. Please provide an argument using the -calculationType argument.");
        return 0;
    }

    // If the field name argument is not present, we cannot proceed, so we should print an error message and return
    if (!programCtrl.input().isArgPresent("-fieldName"))
    {
        errorHandler::check<throws::NO_THROW>(-1, "Unspecified field name. Please provide an argument using the -fieldName argument.");
        return 0;
    }

    const host::latticeMesh mesh(programCtrl);

    // If we have supplied a -fieldName argument, replace programCtrl.caseName() with the fieldName
    const name_t fieldName = programCtrl.getArgument("-fieldName");

    // Get the time indices
    const std::vector<host::label_t> fileNameIndices = programCtrl.timeStepIndices();

    // Parse the argument if present, otherwise set to empty string
    const name_t calculationTypeString = programCtrl.getArgument("-calculationType");

    // Get the calculation function
    const std::unordered_map<name_t, calculator::functionType>::const_iterator it = calculators.find(calculationTypeString);

    if (it != calculators.end())
    {
        const calculator::functionType calculation = it->second;

        if (!fileNameIndices.empty())
        {
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

                    calculation(hostMoments, mesh, timeStep);

                    if (!(timeStep == fileNameIndices.back()))
                    {
                        std::cout << std::endl;
                    }
                }
            }

            if (!foundField)
            {
                errorHandler::check<throws::NO_THROW>(-1, "Specified field name not found in any time step directory.");
            }
        }
        else
        {
            // We don't actually need to throw, we can just print the error message
            errorHandler::check<throws::NO_THROW>(-1, "Empty timeStep directory.");
        }
    }
    else
    {
        errorHandler::check<throws::NO_THROW>(-1, "Invalid calculation function for calculation type: " + calculationTypeString);
    }

    return 0;

    // if (calculationTypeString == "vorticity")
    // {
    //     // Get the conversion type
    //     const name_t conversion = programCtrl.getArgument("-fileType");

    //     // Get the writer function
    //     const std::unordered_map<name_t, postProcess::writerFunction>::const_iterator it = postProcess::writers.find(conversion);

    //     // Get the time indices
    //     const std::vector<host::label_t> fileNameIndices = fileIO::timeIndices(programCtrl.caseName());

    //     if (it != postProcess::writers.end())
    //     {
    //         for (host::label_t timeStep = fileIO::getStartIndex(programCtrl.caseName(), programCtrl); timeStep < fileNameIndices.size(); timeStep++)
    //         {
    //             // Get the file name at the present time step
    //             const name_t fileName = "vorticity_" + std::to_string(fileNameIndices[timeStep]);

    //             const host::arrayCollection<scalar_t, ctorType::MUST_READ> hostMoments(
    //                 programCtrl,
    //                 {"rho", "u", "v", "w", "m_xx", "m_xy", "m_xz", "m_yy", "m_yz", "m_zz"},
    //                 timeStep);

    //             const std::vector<std::vector<scalar_t>> fields = fileIO::deinterleaveAoS(hostMoments.arr(), mesh);

    //             const std::vector<std::vector<scalar_t>> omega = numericalSchemes::derivative::curl<SchemeOrder()>(fields[index::u], fields[index::v], fields[index::w], mesh);
    //             const std::vector<scalar_t> magomega = numericalSchemes::mag(omega[0], omega[1], omega[2]);

    //             const postProcess::writerFunction writer = it->second;

    //             writer({omega[0], omega[1], omega[2], magomega}, fileName, mesh, {"omega_x", "omega_y", "omega_z", "mag[omega]"});

    //             if (timeStep < fileNameIndices.size() - 1)
    //             {
    //                 std::cout << std::endl;
    //             }
    //         }
    //     }
    // }

    // if (calculationTypeString == "div[U]")
    // {
    //     // Get the conversion type
    //     const name_t conversion = programCtrl.getArgument("-fileType");

    //     // Get the writer function
    //     const std::unordered_map<name_t, postProcess::writerFunction>::const_iterator it = postProcess::writers.find(conversion);

    //     // Get the time indices
    //     const std::vector<host::label_t> fileNameIndices = fileIO::timeIndices(programCtrl.caseName());

    //     if (it != postProcess::writers.end())
    //     {
    //         for (host::label_t timeStep = fileIO::getStartIndex(programCtrl.caseName(), programCtrl); timeStep < fileNameIndices.size(); timeStep++)
    //         {
    //             // Get the file name at the present time step
    //             const name_t fileName = "div[U]_" + std::to_string(fileNameIndices[timeStep]);

    //             const host::arrayCollection<scalar_t, ctorType::MUST_READ> hostMoments(
    //                 programCtrl,
    //                 {"rho", "u", "v", "w", "m_xx", "m_xy", "m_xz", "m_yy", "m_yz", "m_zz"},
    //                 timeStep);

    //             const std::vector<std::vector<scalar_t>> fields = fileIO::deinterleaveAoS(hostMoments.arr(), mesh);

    //             const std::vector<scalar_t> divu = numericalSchemes::derivative::div<SchemeOrder()>(fields[index::u], fields[index::v], fields[index::w], mesh);

    //             const postProcess::writerFunction writer = it->second;

    //             writer({divu}, fileName, mesh, {"div[U]"});

    //             if (timeStep < fileNameIndices.size() - 1)
    //             {
    //                 std::cout << std::endl;
    //             }
    //         }
    //     }
    // }

    return 0;
}