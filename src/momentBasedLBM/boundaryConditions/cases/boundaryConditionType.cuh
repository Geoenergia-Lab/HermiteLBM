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

#include "../regularizedBoundaryCondition/regularizedBC.cuh"
#include "../velocityBoundaryCondition/velocityBC.cuh"

namespace LBM
{
    /**
     * @class boundaryConditionType
     * @brief Base class to determine periodicity of a particular boundary condition setup
     **/
    template <const bool PeriodicX, const bool PeriodicY, const bool PeriodicZ>
    class boundaryConditionType
    {
    public:
        /**
         * @brief Define the normal vector type
         **/
        using NormalVector = normalVector<var3<bool>(PeriodicX, PeriodicY, PeriodicZ)>;

        /**
         * @brief Determine whether or not the boundary conditions are periodc along a particular axis
         * @tparam alpha The axis direction (X, Y or Z)
         **/
        template <const axis::type alpha>
        __device__ __host__ [[nodiscard]] static inline consteval bool periodic() noexcept
        {
            constexpr const var3<bool> result(PeriodicX, PeriodicY, PeriodicZ);
            return result.value<alpha>();
        }

        /**
         * @brief Switch determining whether or not the simulation should save to a file
         **/
        __device__ __host__ [[nodiscard]] static inline consteval bool save() noexcept { return true; }

        /**
         * @brief Generic switch to apply the boundary conditions to the case (WIP)
         * @tparam VelocitySet The velocity set (D3Q19 or D3Q27)
         * @param[in] pop Population density array at current lattice node
         * @param[in] moments Moment array (rho, U, Pi)
         * @param[in] nodeType Orientation of the boundary lattice site
         **/
        template <class VelocitySet>
        __device__ static inline constexpr void apply(
            const thread::array<scalar_t, VelocitySet::Q()> &pop,
            momentsArray &moments,
            const nodeType_t &nodeType) noexcept
        {
            switch (nodeType)
            {
            case normalVectorBase::SOUTH_WEST_BACK():
            {
                noSlipVelocityBC::apply<normalVectorBase::SOUTH_WEST_BACK()>(moments);
                noSlipBC::apply<VelocitySet, normalVectorBase::SOUTH_WEST_BACK()>(moments, pop);

                return;
            }
            case normalVectorBase::SOUTH_WEST_FRONT():
            {
                noSlipVelocityBC::apply<normalVectorBase::SOUTH_WEST_FRONT()>(moments);
                noSlipBC::apply<VelocitySet, normalVectorBase::SOUTH_WEST_FRONT()>(moments, pop);

                return;
            }
            case normalVectorBase::SOUTH_EAST_BACK():
            {
                noSlipVelocityBC::apply<normalVectorBase::SOUTH_EAST_BACK()>(moments);
                noSlipBC::apply<VelocitySet, normalVectorBase::SOUTH_EAST_BACK()>(moments, pop);

                return;
            }
            case normalVectorBase::SOUTH_EAST_FRONT():
            {
                noSlipVelocityBC::apply<normalVectorBase::SOUTH_EAST_FRONT()>(moments);
                noSlipBC::apply<VelocitySet, normalVectorBase::SOUTH_EAST_FRONT()>(moments, pop);

                return;
            }
            case normalVectorBase::SOUTH_WEST():
            {
                noSlipVelocityBC::apply<normalVectorBase::SOUTH_WEST()>(moments);
                noSlipBC::apply<VelocitySet, normalVectorBase::SOUTH_WEST()>(moments, pop);

                return;
            }
            case normalVectorBase::SOUTH_EAST():
            {
                noSlipVelocityBC::apply<normalVectorBase::SOUTH_EAST()>(moments);
                noSlipBC::apply<VelocitySet, normalVectorBase::SOUTH_EAST()>(moments, pop);

                return;
            }
            case normalVectorBase::WEST_BACK():
            {
                noSlipVelocityBC::apply<normalVectorBase::WEST_BACK()>(moments);
                noSlipBC::apply<VelocitySet, normalVectorBase::WEST_BACK()>(moments, pop);

                return;
            }
            case normalVectorBase::WEST_FRONT():
            {
                noSlipVelocityBC::apply<normalVectorBase::WEST_FRONT()>(moments);
                noSlipBC::apply<VelocitySet, normalVectorBase::WEST_FRONT()>(moments, pop);

                return;
            }
            case normalVectorBase::EAST_BACK():
            {
                noSlipVelocityBC::apply<normalVectorBase::EAST_BACK()>(moments);
                noSlipBC::apply<VelocitySet, normalVectorBase::EAST_BACK()>(moments, pop);

                return;
            }
            case normalVectorBase::EAST_FRONT():
            {
                noSlipVelocityBC::apply<normalVectorBase::EAST_FRONT()>(moments);
                noSlipBC::apply<VelocitySet, normalVectorBase::EAST_FRONT()>(moments, pop);

                return;
            }
            case normalVectorBase::SOUTH_BACK():
            {
                noSlipVelocityBC::apply<normalVectorBase::SOUTH_BACK()>(moments);
                noSlipBC::apply<VelocitySet, normalVectorBase::SOUTH_BACK()>(moments, pop);

                return;
            }
            case normalVectorBase::SOUTH_FRONT():
            {
                noSlipVelocityBC::apply<normalVectorBase::SOUTH_FRONT()>(moments);
                noSlipBC::apply<VelocitySet, normalVectorBase::SOUTH_FRONT()>(moments, pop);

                return;
            }
            case normalVectorBase::WEST():
            {
                noSlipVelocityBC::apply<normalVectorBase::WEST()>(moments);
                noSlipBC::apply<VelocitySet, normalVectorBase::WEST()>(moments, pop);

                return;
            }
            case normalVectorBase::EAST():
            {
                noSlipVelocityBC::apply<normalVectorBase::EAST()>(moments);
                noSlipBC::apply<VelocitySet, normalVectorBase::EAST()>(moments, pop);

                return;
            }
            case normalVectorBase::SOUTH():
            {
                noSlipVelocityBC::apply<normalVectorBase::SOUTH()>(moments);
                noSlipBC::apply<VelocitySet, normalVectorBase::SOUTH()>(moments, pop);

                return;
            }
            case normalVectorBase::BACK():
            {
                noSlipVelocityBC::apply<normalVectorBase::BACK()>(moments);
                noSlipBC::apply<VelocitySet, normalVectorBase::BACK()>(moments, pop);

                return;
            }
            case normalVectorBase::FRONT():
            {
                noSlipVelocityBC::apply<normalVectorBase::FRONT()>(moments);
                noSlipBC::apply<VelocitySet, normalVectorBase::FRONT()>(moments, pop);

                return;
            }
            case normalVectorBase::NORTH():
            {
                constantVelocityBC::apply<normalVectorBase::NORTH()>(moments);
                Dirichlet::apply<VelocitySet, normalVectorBase::NORTH()>(moments, pop);

                return;
            }
            case normalVectorBase::NORTH_WEST_BACK():
            {
                constantVelocityBC::apply<normalVectorBase::NORTH_WEST_BACK()>(moments);
                Dirichlet::apply<VelocitySet, normalVectorBase::NORTH_WEST_BACK()>(moments, pop);

                return;
            }
            case normalVectorBase::NORTH_WEST_FRONT():
            {
                constantVelocityBC::apply<normalVectorBase::NORTH_WEST_FRONT()>(moments);
                Dirichlet::apply<VelocitySet, normalVectorBase::NORTH_WEST_FRONT()>(moments, pop);

                return;
            }
            case normalVectorBase::NORTH_EAST_BACK():
            {
                constantVelocityBC::apply<normalVectorBase::NORTH_EAST_BACK()>(moments);
                Dirichlet::apply<VelocitySet, normalVectorBase::NORTH_EAST_BACK()>(moments, pop);

                return;
            }
            case normalVectorBase::NORTH_EAST_FRONT():
            {
                constantVelocityBC::apply<normalVectorBase::NORTH_EAST_FRONT()>(moments);
                Dirichlet::apply<VelocitySet, normalVectorBase::NORTH_EAST_FRONT()>(moments, pop);

                return;
            }
            case normalVectorBase::NORTH_BACK():
            {
                constantVelocityBC::apply<normalVectorBase::NORTH_BACK()>(moments);
                Dirichlet::apply<VelocitySet, normalVectorBase::NORTH_BACK()>(moments, pop);

                return;
            }
            case normalVectorBase::NORTH_FRONT():
            {
                constantVelocityBC::apply<normalVectorBase::NORTH_FRONT()>(moments);
                Dirichlet::apply<VelocitySet, normalVectorBase::NORTH_FRONT()>(moments, pop);

                return;
            }
            case normalVectorBase::NORTH_EAST():
            {
                constantVelocityBC::apply<normalVectorBase::NORTH_EAST()>(moments);
                Dirichlet::apply<VelocitySet, normalVectorBase::NORTH_EAST()>(moments, pop);

                return;
            }
            case normalVectorBase::NORTH_WEST():
            {
                constantVelocityBC::apply<normalVectorBase::NORTH_WEST()>(moments);
                Dirichlet::apply<VelocitySet, normalVectorBase::NORTH_WEST()>(moments, pop);

                return;
            }
            }
        }
    };
}

#endif