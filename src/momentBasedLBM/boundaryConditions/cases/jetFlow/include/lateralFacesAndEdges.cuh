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
Authors: Nathan Duggins, Breno Gemelgo (Geoenergia Lab, UDESC)

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
    Face and edge definitions along the lateral planes of the jet.
    Periodicity is implemented at halo level.
    See /src/blockHalo/halo.cuh for more information.

SourceFiles
    lateralFacesAndEdges.cuh

    This file is intended to be included directly inside a switch-case block.
    Do NOT use include guards (#ifndef/#define/#endif).

\*---------------------------------------------------------------------------*/

case normalVectorBase::WEST():
{
    if constexpr (lateralNeumann)
    {
        const device::label_t lateralOutlet = block::idx(WestInterior, Tx.value<axis::Y>(), Tx.value<axis::Z>());

        // Classic Neumann
        moments[m_i<1>()] = sharedBuffer[lateralOutlet * (NUMBER_MOMENTS()) + m_i<1>()];
        moments[m_i<2>()] = sharedBuffer[lateralOutlet * (NUMBER_MOMENTS()) + m_i<2>()];
        moments[m_i<3>()] = sharedBuffer[lateralOutlet * (NUMBER_MOMENTS()) + m_i<3>()];

        Dirichlet::apply<VelocitySet, normalVectorBase::WEST()>(moments);

        if constexpr (fixOutletDensity)
        {
            moments[0] = rho0();
        }
    }

    return;
}
case normalVectorBase::EAST():
{
    if constexpr (lateralNeumann)
    {
        const device::label_t lateralOutlet = block::idx(EastInterior, Tx.value<axis::Y>(), Tx.value<axis::Z>());

        // Classic Neumann
        moments[m_i<1>()] = sharedBuffer[lateralOutlet * (NUMBER_MOMENTS()) + m_i<1>()];
        moments[m_i<2>()] = sharedBuffer[lateralOutlet * (NUMBER_MOMENTS()) + m_i<2>()];
        moments[m_i<3>()] = sharedBuffer[lateralOutlet * (NUMBER_MOMENTS()) + m_i<3>()];

        Dirichlet::apply<VelocitySet, normalVectorBase::EAST()>(moments);

        if constexpr (fixOutletDensity)
        {
            moments[0] = rho0();
        }
    }

    return;
}
case normalVectorBase::SOUTH():
{
    if constexpr (lateralNeumann)
    {
        const device::label_t lateralOutlet = block::idx(Tx.value<axis::X>(), SouthInterior, Tx.value<axis::Z>());

        // Classic Neumann
        moments[m_i<1>()] = sharedBuffer[lateralOutlet * (NUMBER_MOMENTS()) + m_i<1>()];
        moments[m_i<2>()] = sharedBuffer[lateralOutlet * (NUMBER_MOMENTS()) + m_i<2>()];
        moments[m_i<3>()] = sharedBuffer[lateralOutlet * (NUMBER_MOMENTS()) + m_i<3>()];

        Dirichlet::apply<VelocitySet, normalVectorBase::SOUTH()>(moments);

        if constexpr (fixOutletDensity)
        {
            moments[0] = rho0();
        }
    }

    return;
}
case normalVectorBase::NORTH():
{
    if constexpr (lateralNeumann)
    {
        const device::label_t lateralOutlet = block::idx(Tx.value<axis::X>(), NorthInterior, Tx.value<axis::Z>());

        // Classic Neumann
        moments[m_i<1>()] = sharedBuffer[lateralOutlet * (NUMBER_MOMENTS()) + m_i<1>()];
        moments[m_i<2>()] = sharedBuffer[lateralOutlet * (NUMBER_MOMENTS()) + m_i<2>()];
        moments[m_i<3>()] = sharedBuffer[lateralOutlet * (NUMBER_MOMENTS()) + m_i<3>()];

        Dirichlet::apply<VelocitySet, normalVectorBase::NORTH()>(moments);

        if constexpr (fixOutletDensity)
        {
            moments[0] = rho0();
        }
    }

    return;
}
case normalVectorBase::SOUTH_WEST():
{
    if constexpr (lateralNeumann)
    {
        const device::label_t lateralOutlet = block::idx(WestInterior, SouthInterior, Tx.value<axis::Z>());

        // Classic Neumann
        moments[m_i<1>()] = sharedBuffer[lateralOutlet * (NUMBER_MOMENTS()) + m_i<1>()];
        moments[m_i<2>()] = sharedBuffer[lateralOutlet * (NUMBER_MOMENTS()) + m_i<2>()];
        moments[m_i<3>()] = sharedBuffer[lateralOutlet * (NUMBER_MOMENTS()) + m_i<3>()];

        Dirichlet::apply<VelocitySet, normalVectorBase::SOUTH_WEST()>(moments);

        if constexpr (fixOutletDensity)
        {
            moments[0] = rho0();
        }
    }

    return;
}
case normalVectorBase::NORTH_WEST():
{
    if constexpr (lateralNeumann)
    {
        const device::label_t lateralOutlet = block::idx(WestInterior, NorthInterior, Tx.value<axis::Z>());

        // Classic Neumann
        moments[m_i<1>()] = sharedBuffer[lateralOutlet * (NUMBER_MOMENTS()) + m_i<1>()];
        moments[m_i<2>()] = sharedBuffer[lateralOutlet * (NUMBER_MOMENTS()) + m_i<2>()];
        moments[m_i<3>()] = sharedBuffer[lateralOutlet * (NUMBER_MOMENTS()) + m_i<3>()];

        Dirichlet::apply<VelocitySet, normalVectorBase::NORTH_WEST()>(moments);

        if constexpr (fixOutletDensity)
        {
            moments[0] = rho0();
        }
    }

    return;
}
case normalVectorBase::SOUTH_EAST():
{
    if constexpr (lateralNeumann)
    {
        const device::label_t lateralOutlet = block::idx(EastInterior, SouthInterior, Tx.value<axis::Z>());

        // Classic Neumann
        moments[m_i<1>()] = sharedBuffer[lateralOutlet * (NUMBER_MOMENTS()) + m_i<1>()];
        moments[m_i<2>()] = sharedBuffer[lateralOutlet * (NUMBER_MOMENTS()) + m_i<2>()];
        moments[m_i<3>()] = sharedBuffer[lateralOutlet * (NUMBER_MOMENTS()) + m_i<3>()];

        Dirichlet::apply<VelocitySet, normalVectorBase::SOUTH_EAST()>(moments);

        if constexpr (fixOutletDensity)
        {
            moments[0] = rho0();
        }
    }

    return;
}
case normalVectorBase::NORTH_EAST():
{
    if constexpr (lateralNeumann)
    {
        const device::label_t lateralOutlet = block::idx(EastInterior, NorthInterior, Tx.value<axis::Z>());

        // Classic Neumann
        moments[m_i<1>()] = sharedBuffer[lateralOutlet * (NUMBER_MOMENTS()) + m_i<1>()];
        moments[m_i<2>()] = sharedBuffer[lateralOutlet * (NUMBER_MOMENTS()) + m_i<2>()];
        moments[m_i<3>()] = sharedBuffer[lateralOutlet * (NUMBER_MOMENTS()) + m_i<3>()];

        Dirichlet::apply<VelocitySet, normalVectorBase::NORTH_EAST()>(moments);

        if constexpr (fixOutletDensity)
        {
            moments[0] = rho0();
        }
    }

    return;
}