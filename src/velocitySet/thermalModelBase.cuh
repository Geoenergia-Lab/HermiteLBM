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
    Definition of the lattice thermal model; weakly compressible or isothermal

Namespace
    LBM

SourceFiles
    thermalModelBase.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_THERMALMODELBASE_CUH
#define __MBLBM_THERMALMODELBASE_CUH

namespace LBM
{
    /**
     * @brief Enumerated type for indexing pointers to halos
     **/
    typedef enum thermalModelEnum : bool
    {
        Thermal = 0,
        Isothermal = 1
    } thermalModel_t;

    template <const thermalModel_t ThermalModel>
    class thermalModelBase
    {
    public:
        static_assert(((ThermalModel == Thermal) || (ThermalModel == Isothermal)), "ThermalModel must be Thermal or Isothermal.");

        /**
         * @brief Get the thermal model of the velocity set
         **/
        __device__ __host__ [[nodiscard]] static inline consteval thermalModel_t thermalModel() noexcept
        {
            return ThermalModel;
        }
    };
}

#endif