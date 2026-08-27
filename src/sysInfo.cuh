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
    Compile-time information about the hardware and operating system

Namespace
    LBM, LBM::system

SourceFiles
    sysInfo.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_SYSINFO_CUH
#define __MBLBM_SYSINFO_CUH

namespace LBM
{
    struct system
    {
        /**
         * @brief Supported operating systems (Linux, Windows)
         **/
        typedef enum distroEnum : int64_t
        {
            UNDEFINED = -1,
            LINUX = 0,
            WINDOWS = 1,
        } distroEnum;

        /**
         * @brief Get the name of the operating system
         **/
        __host__ [[nodiscard]] static inline constexpr distroEnum distro() noexcept
        {
#if defined(_WIN32) && !defined(__linux__)
            return WINDOWS;
#elif defined(__linux__) && !defined(_WIN32)
            return LINUX;
#else
            return UNDEFINED;
#endif
        }

        /**
         * @brief Check if the system has more than 1 GPU
         **/
        __host__ [[nodiscard]] static inline consteval bool hasMultiGPU() noexcept
        {
#ifdef HAS_MULTI_GPU
            return HAS_MULTI_GPU;
#else
            return false;
#endif
        }
    };

    /**
     * @brief Assert that the operating system is valid
     **/
    static_assert(!(system::distro() == system::UNDEFINED), "Operating system must be either LINUX or WINDOWS");
}

#endif