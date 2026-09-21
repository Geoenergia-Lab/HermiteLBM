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
Authors: Nathan Duggins, Vinicius Czarnobay, Breno Gemelgo (Geoenergia Lab, UDESC)

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
    Edge and corner definitions along the outlet plane of the jet.

SourceFiles
    outlet.cuh

Notes
    This file is intended to be included directly inside a switch-case block.
    Do NOT use include guards (#ifndef/#define/#endif).

\*---------------------------------------------------------------------------*/

// Faces
case normalVectorBase::FRONT():
{
    Neumann::apply<VelocitySet, normalVectorBase::FRONT()>(moments, incomings, sharedBuffer, tid);

    return;
}

// Edges
case normalVectorBase::WEST_FRONT():
{
    Neumann::apply<VelocitySet, normalVectorBase::WEST_FRONT()>(moments, incomings, sharedBuffer, tid);

    return;
}
case normalVectorBase::EAST_FRONT():
{
    Neumann::apply<VelocitySet, normalVectorBase::EAST_FRONT()>(moments, incomings, sharedBuffer, tid);

    return;
}
case normalVectorBase::SOUTH_FRONT():
{
    Neumann::apply<VelocitySet, normalVectorBase::SOUTH_FRONT()>(moments, incomings, sharedBuffer, tid);

    return;
}
case normalVectorBase::NORTH_FRONT():
{
    Neumann::apply<VelocitySet, normalVectorBase::NORTH_FRONT()>(moments, incomings, sharedBuffer, tid);

    return;
}

// Edges
case normalVectorBase::SOUTH_WEST_FRONT():
{
    Neumann::apply<VelocitySet, normalVectorBase::SOUTH_WEST_FRONT()>(moments, incomings, sharedBuffer, tid);

    return;
}
case normalVectorBase::NORTH_WEST_FRONT():
{
    Neumann::apply<VelocitySet, normalVectorBase::NORTH_WEST_FRONT()>(moments, incomings, sharedBuffer, tid);

    return;
}
case normalVectorBase::SOUTH_EAST_FRONT():
{
    Neumann::apply<VelocitySet, normalVectorBase::SOUTH_EAST_FRONT()>(moments, incomings, sharedBuffer, tid);

    return;
}
case normalVectorBase::NORTH_EAST_FRONT():
{
    Neumann::apply<VelocitySet, normalVectorBase::NORTH_EAST_FRONT()>(moments, incomings, sharedBuffer, tid);

    return;
}