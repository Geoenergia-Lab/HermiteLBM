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
    Functions used to handle errors

Namespace
    LBM

SourceFiles
    runTime.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_RUNTIME_CUH
#define __MBLBM_RUNTIME_CUH

namespace LBM
{
    namespace runTime
    {
        typedef enum programStatusEnum : int
        {
            BAD = -1,
            GOOD = 0
        } programStatus;

        namespace error
        {
            typedef enum errorCodeEnum : int
            {
                NO_ERROR,
                INCORRECT_NUMBER_OF_GPUS,
                INVALID_DEVICE_DECOMPOSITION,
                LABEL_T_CAPACITY_EXCEEDED,
                UNSPECIFIED_CALCULATIONTYPE,
                UNSPECIFIED_FILETYPE,
                UNSPECIFIED_FIELDNAME,
                FIELDNAME_NOT_FOUND,
                EMPTY_TIMESTEP_DIRECTORY,
                INVALID_CALCULATION_FUNCTION,
                INVALID_WRITER_FUNCTION
            } code;

            __host__ [[nodiscard]] inline consteval const std::array<const char *, 11> messages() noexcept
            {
                return {
                    "",
                    "Number of GPUs must match the number of devices in the mesh decomposition,",
                    "HermiteLBM currently only supports decomposition in the z axis,",
                    "Mesh size exceeds maximum allowed value of device::label_t,",
                    "Unspecified calculation type. Please provide an argument using the -calculationType argument.",
                    "Unspecified file type. Please provide an argument using the -fileType argument",
                    "Unspecified field name. Please provide an argument using the -fieldName argument.",
                    "Specified field name not found in any time step directory.",
                    "Empty timeStep directory.",
                    "Invalid calculation function."
                    "Invalid writer function"};
            }
        }

        // Single std::atomic storing the status of the program (BAD or GOOD)
        static constinit std::atomic<programStatus> program_status(GOOD);
        static constinit std::atomic<cudaError_t> first_cuda_error(cudaSuccess);
        static constinit std::atomic<int> first_reg_error(0);

        /**
         * @brief Updates the internal allocation status and records the first CUDA error encountered.
         * @param[in] code The CUDA error code to process.
         **/
        __host__ void update_codes(const cudaError_t code) noexcept
        {
            // Record the first error (only if no error has been recorded yet).
            cudaError_t expected_error = cudaSuccess;
            first_cuda_error.compare_exchange_strong(expected_error, code);

            // Make program_status sticky: once BAD, it stays BAD.
            if (code != cudaSuccess)
            {
                programStatus expected_status = GOOD;
                program_status.compare_exchange_strong(expected_status, BAD);
            }
        }

        __host__ void update_codes(const int code) noexcept
        {
            // Record the first error (only if no error has been recorded yet).
            int expected_error = cudaSuccess;
            first_reg_error.compare_exchange_strong(expected_error, code);

            // Make program_status sticky: once BAD, it stays BAD.
            if (code != 0)
            {
                programStatus expected_status = GOOD;
                program_status.compare_exchange_strong(expected_status, BAD);
            }
        }
    }
}

#include "errorHandler.cuh"
#include "signalHandler.cuh"
#include "sysInfo.cuh"

#endif