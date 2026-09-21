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
    Base type for the boundary condition class

Namespace
    LBM

SourceFiles
    boundaryConditionType.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_BOUNDARYCONDITIONTYPE_CUH
#define __MBLBM_BOUNDARYCONDITIONTYPE_CUH

namespace LBM
{
    /**
     * @brief Type of boundary: either a wall (Dirichlet or Neumann), or a periodic boundary
     **/
    typedef enum boundaryTypeEnum : bool
    {
        WALL = 0,
        PERIODIC = 1
    } boundaryType;

    /**
     * @brief Boundary condition set applies a boundary condition or not
     **/
    typedef enum hasBoundaryConditionTypeEnum : bool
    {
        NO_CONDITION = 0,
        APPLIES_CONDITION = 1
    } hasBoundaryConditionType;

    /**
     * @class boundaryConditionType
     * @brief Base class to determine periodicity of a particular boundary condition setup
     **/
    template <const boundaryType TypeX, const boundaryType TypeY, const boundaryType TypeZ>
    class boundaryConditionType
    {
    public:
        /**
         * @brief Define the normal vector type
         **/
        using NormalVector = normalVector<var3<bool>(TypeX, TypeY, TypeZ)>;

        /**
         * @brief Determine whether or not the boundary conditions are periodc along a particular axis
         * @tparam alpha The axis direction (X, Y or Z)
         **/
        template <const axis::type alpha>
        __device__ __host__ [[nodiscard]] static inline consteval bool periodic() noexcept
        {
            return var3<boundaryType>(TypeX, TypeY, TypeZ).value<alpha>();
        }

        /**
         * @brief Switch determining whether or not the boundary condition actually applies a condition
         **/
        __device__ __host__ [[nodiscard]] static inline consteval hasBoundaryConditionType appliesCondition() noexcept
        {
            return static_cast<hasBoundaryConditionType>((TypeX == APPLIES_CONDITION) || (TypeY == APPLIES_CONDITION) || (TypeZ == APPLIES_CONDITION));
        }
    };
}

#endif