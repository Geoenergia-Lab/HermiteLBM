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
         * @brief Place the moments into the shared memory prior to application of the boundary conditions
         * @param[in] moments Moment array (rho, U, Pi)
         * @param[out] sharedBuffer Shared memory array containing moment data
         * @param[in] tid Thread ID within block
         **/
        __device__ static inline constexpr void update_smem(
            const momentsArray &moments,
            blockSharedBuffer &sharedBuffer,
            const device::label_t tid) noexcept
        {
            device::constexpr_for<0, NUMBER_MOMENTS()>(
                [&](const auto moment)
                {
                    const device::label_t ID = tid * label_constant<NUMBER_MOMENTS()>() + label_constant<moment>();
                    sharedBuffer[ID] = moments[moment];
                });

            block::sync();
        }

        /**
         * @brief Apply an equilibrium boundary condition to the second order moments
         * @param[out] moments Moment array (rho, U, Pi)
         **/
        __device__ static inline constexpr void equilibrium_moment(momentsArray &moments) noexcept
        {
            moments[axis::index<axis::X, axis::X>()] = moments[axis::index<axis::X>()] * moments[axis::index<axis::X>()];
            moments[axis::index<axis::X, axis::Y>()] = moments[axis::index<axis::X>()] * moments[axis::index<axis::Y>()];
            moments[axis::index<axis::X, axis::Z>()] = moments[axis::index<axis::X>()] * moments[axis::index<axis::Z>()];
            moments[axis::index<axis::Y, axis::Y>()] = moments[axis::index<axis::Y>()] * moments[axis::index<axis::Y>()];
            moments[axis::index<axis::Y, axis::Z>()] = moments[axis::index<axis::Y>()] * moments[axis::index<axis::Z>()];
            moments[axis::index<axis::Z, axis::Z>()] = moments[axis::index<axis::Z>()] * moments[axis::index<axis::Z>()];
        }

        /**
         * @brief Define the normal vector type
         **/
        using NormalVector = normalVector<var3<bool>(PeriodicX, PeriodicY, PeriodicZ)>;
        using This = boundaryConditionType<PeriodicX, PeriodicY, PeriodicZ>;

        /**
         * @brief Determine whether or not the boundary conditions are periodc along a particular axis
         * @tparam alpha The axis direction (X, Y or Z)
         **/
        template <const axis::type alpha>
        deviceHostFunctionReturnType static inline consteval bool periodic() noexcept
        {
            constexpr const var3<bool> result(PeriodicX, PeriodicY, PeriodicZ);
            return result.value<alpha>();
        }

        /**
         * @brief Switch determining whether or not the simulation should save to a file
         **/
        deviceHostFunctionReturnType static inline consteval bool save() noexcept { return true; }

        /**
         * @brief Get the offset of the interior node based on the boundary node and the normal vector
         * @tparam alpha The axis direction (X, Y or Z)
         * @param[in] boundaryNormal Normal vector information at boundary node
         **/
        template <const axis::type alpha>
        deviceFunctionReturnType static inline constexpr device::label_t boundaryOffset(const NormalVector &boundaryNormal) noexcept
        {
            if constexpr (alpha == axis::X)
            {
                if constexpr (PeriodicX)
                {
                    return 0;
                }
                return boundaryNormal.template isWest<device::label_t>() - boundaryNormal.template isEast<device::label_t>();
            }

            if constexpr (alpha == axis::Y)
            {
                if constexpr (PeriodicY)
                {
                    return 0;
                }
                return boundaryNormal.template isSouth<device::label_t>() - boundaryNormal.template isNorth<device::label_t>();
            }

            if constexpr (alpha == axis::Z)
            {
                if constexpr (PeriodicZ)
                {
                    return 0;
                }
                return boundaryNormal.template isBack<device::label_t>() - boundaryNormal.template isFront<device::label_t>();
            }
        }

        /**
         * @brief Get the thread coordinate of the interior node based on the boundary node and the normal vector
         * @param[in] Tx Three-dimensional thread coordinates
         * @param[in] boundaryNormal Normal vector information at boundary node
         **/
        deviceFunctionReturnType static inline constexpr device::label_t idxInterior(const thread::coordinate &Tx, const NormalVector &boundaryNormal) noexcept
        {
            return block::idx(Tx.value<axis::X>() + boundaryOffset<axis::X>(boundaryNormal), Tx.value<axis::Y>() + boundaryOffset<axis::Y>(boundaryNormal), Tx.value<axis::Z>() + boundaryOffset<axis::Z>(boundaryNormal));
        }

        template <const device::label_t moment>
        deviceFunctionReturnType static inline constexpr device::label_t idxShared(const device::label_t tidInterior) noexcept
        {
            return (tidInterior * NUMBER_MOMENTS<device::label_t>()) + label_constant<moment>();
        }

        /**
         * @brief Generic switch to apply the boundary conditions to the case (WIP)
         * @tparam VelocitySet The velocity set (D3Q19 or D3Q27)
         * @param[in] pop Population density array at current lattice node
         * @param[in] moments Moment array (rho, U, Pi)
         * @param[in] nodeType Orientation of the boundary lattice site
         **/
        template <class VelocitySet>
        __device__ static inline constexpr void apply(
            momentsArray &moments,
            const nodeType_t &nodeType) noexcept
        {
            switch (nodeType)
            {
            case normalVectorBase::SOUTH_WEST_BACK():
            {
                noSlipVelocityBC::apply<normalVectorBase::SOUTH_WEST_BACK()>(moments);
                noSlipBC::apply<VelocitySet, normalVectorBase::SOUTH_WEST_BACK()>(moments);

                return;
            }
            case normalVectorBase::SOUTH_WEST_FRONT():
            {
                noSlipVelocityBC::apply<normalVectorBase::SOUTH_WEST_FRONT()>(moments);
                noSlipBC::apply<VelocitySet, normalVectorBase::SOUTH_WEST_FRONT()>(moments);

                return;
            }
            case normalVectorBase::SOUTH_EAST_BACK():
            {
                noSlipVelocityBC::apply<normalVectorBase::SOUTH_EAST_BACK()>(moments);
                noSlipBC::apply<VelocitySet, normalVectorBase::SOUTH_EAST_BACK()>(moments);

                return;
            }
            case normalVectorBase::SOUTH_EAST_FRONT():
            {
                noSlipVelocityBC::apply<normalVectorBase::SOUTH_EAST_FRONT()>(moments);
                noSlipBC::apply<VelocitySet, normalVectorBase::SOUTH_EAST_FRONT()>(moments);

                return;
            }
            case normalVectorBase::SOUTH_WEST():
            {
                noSlipVelocityBC::apply<normalVectorBase::SOUTH_WEST()>(moments);
                noSlipBC::apply<VelocitySet, normalVectorBase::SOUTH_WEST()>(moments);

                return;
            }
            case normalVectorBase::SOUTH_EAST():
            {
                noSlipVelocityBC::apply<normalVectorBase::SOUTH_EAST()>(moments);
                noSlipBC::apply<VelocitySet, normalVectorBase::SOUTH_EAST()>(moments);

                return;
            }
            case normalVectorBase::WEST_BACK():
            {
                noSlipVelocityBC::apply<normalVectorBase::WEST_BACK()>(moments);
                noSlipBC::apply<VelocitySet, normalVectorBase::WEST_BACK()>(moments);

                return;
            }
            case normalVectorBase::WEST_FRONT():
            {
                noSlipVelocityBC::apply<normalVectorBase::WEST_FRONT()>(moments);
                noSlipBC::apply<VelocitySet, normalVectorBase::WEST_FRONT()>(moments);

                return;
            }
            case normalVectorBase::EAST_BACK():
            {
                noSlipVelocityBC::apply<normalVectorBase::EAST_BACK()>(moments);
                noSlipBC::apply<VelocitySet, normalVectorBase::EAST_BACK()>(moments);

                return;
            }
            case normalVectorBase::EAST_FRONT():
            {
                noSlipVelocityBC::apply<normalVectorBase::EAST_FRONT()>(moments);
                noSlipBC::apply<VelocitySet, normalVectorBase::EAST_FRONT()>(moments);

                return;
            }
            case normalVectorBase::SOUTH_BACK():
            {
                noSlipVelocityBC::apply<normalVectorBase::SOUTH_BACK()>(moments);
                noSlipBC::apply<VelocitySet, normalVectorBase::SOUTH_BACK()>(moments);

                return;
            }
            case normalVectorBase::SOUTH_FRONT():
            {
                noSlipVelocityBC::apply<normalVectorBase::SOUTH_FRONT()>(moments);
                noSlipBC::apply<VelocitySet, normalVectorBase::SOUTH_FRONT()>(moments);

                return;
            }
            case normalVectorBase::WEST():
            {
                noSlipVelocityBC::apply<normalVectorBase::WEST()>(moments);
                noSlipBC::apply<VelocitySet, normalVectorBase::WEST()>(moments);

                return;
            }
            case normalVectorBase::EAST():
            {
                noSlipVelocityBC::apply<normalVectorBase::EAST()>(moments);
                noSlipBC::apply<VelocitySet, normalVectorBase::EAST()>(moments);

                return;
            }
            case normalVectorBase::SOUTH():
            {
                noSlipVelocityBC::apply<normalVectorBase::SOUTH()>(moments);
                noSlipBC::apply<VelocitySet, normalVectorBase::SOUTH()>(moments);

                return;
            }
            case normalVectorBase::BACK():
            {
                noSlipVelocityBC::apply<normalVectorBase::BACK()>(moments);
                noSlipBC::apply<VelocitySet, normalVectorBase::BACK()>(moments);

                return;
            }
            case normalVectorBase::FRONT():
            {
                noSlipVelocityBC::apply<normalVectorBase::FRONT()>(moments);
                noSlipBC::apply<VelocitySet, normalVectorBase::FRONT()>(moments);

                return;
            }
            case normalVectorBase::NORTH():
            {
                constantVelocityBC::apply<normalVectorBase::NORTH()>(moments);
                Dirichlet::apply<VelocitySet, normalVectorBase::NORTH()>(moments);

                return;
            }
            case normalVectorBase::NORTH_WEST_BACK():
            {
                constantVelocityBC::apply<normalVectorBase::NORTH_WEST_BACK()>(moments);
                Dirichlet::apply<VelocitySet, normalVectorBase::NORTH_WEST_BACK()>(moments);

                return;
            }
            case normalVectorBase::NORTH_WEST_FRONT():
            {
                constantVelocityBC::apply<normalVectorBase::NORTH_WEST_FRONT()>(moments);
                Dirichlet::apply<VelocitySet, normalVectorBase::NORTH_WEST_FRONT()>(moments);

                return;
            }
            case normalVectorBase::NORTH_EAST_BACK():
            {
                constantVelocityBC::apply<normalVectorBase::NORTH_EAST_BACK()>(moments);
                Dirichlet::apply<VelocitySet, normalVectorBase::NORTH_EAST_BACK()>(moments);

                return;
            }
            case normalVectorBase::NORTH_EAST_FRONT():
            {
                constantVelocityBC::apply<normalVectorBase::NORTH_EAST_FRONT()>(moments);
                Dirichlet::apply<VelocitySet, normalVectorBase::NORTH_EAST_FRONT()>(moments);

                return;
            }
            case normalVectorBase::NORTH_BACK():
            {
                constantVelocityBC::apply<normalVectorBase::NORTH_BACK()>(moments);
                Dirichlet::apply<VelocitySet, normalVectorBase::NORTH_BACK()>(moments);

                return;
            }
            case normalVectorBase::NORTH_FRONT():
            {
                constantVelocityBC::apply<normalVectorBase::NORTH_FRONT()>(moments);
                Dirichlet::apply<VelocitySet, normalVectorBase::NORTH_FRONT()>(moments);

                return;
            }
            case normalVectorBase::NORTH_EAST():
            {
                constantVelocityBC::apply<normalVectorBase::NORTH_EAST()>(moments);
                Dirichlet::apply<VelocitySet, normalVectorBase::NORTH_EAST()>(moments);

                return;
            }
            case normalVectorBase::NORTH_WEST():
            {
                constantVelocityBC::apply<normalVectorBase::NORTH_WEST()>(moments);
                Dirichlet::apply<VelocitySet, normalVectorBase::NORTH_WEST()>(moments);

                return;
            }
            }
        }
    };
}

#endif