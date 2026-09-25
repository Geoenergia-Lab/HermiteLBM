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
    Header file to avoid repeated definition of the boundary condition function

SourceFiles
    jetBoundaryCondition.cuh

Notes
    This file is intended to be included directly inside a switch-case block.
    Do NOT use include guards (#ifndef/#define/#endif).

\*---------------------------------------------------------------------------*/

constexpr const device::label_t FrontInterior = block::nz<device::label_t>() - 2;

const device::label_t tid = block::idx(Tx.value<axis::X>(), Tx.value<axis::Y>(), FrontInterior);

switch (boundaryNormal.nodeType())
{
// Round inflow + no-slip
case normalVectorBase::BACK():
{
    const scalar_t is_jet = static_cast<scalar_t>(rms_sq(point.value<axis::X, scalar_t>() - center_x(), point.value<axis::Y, scalar_t>() - center_y()) <= r2());

    moments[0] = rho0();

    moments[1] = static_cast<scalar_t>(0);
    moments[2] = static_cast<scalar_t>(0);
    moments[3] = is_jet * device::U_Back[2];

    moments[4] = static_cast<scalar_t>(0);
    moments[5] = static_cast<scalar_t>(0);
    moments[6] = static_cast<scalar_t>(0);
    moments[7] = static_cast<scalar_t>(0);
    moments[8] = static_cast<scalar_t>(0);
    moments[m_i<9>()] = is_jet * device::U_Back[2] * device::U_Back[2];

    return;
}

// Lateral boundary faces and edges
#include "include/lateralFacesAndEdges.cuh"

// Outlet face, edges and corners
#include "include/outlet.cuh"

// Edges and corners on the inlet plane
#include "include/inletEdgesAndCorners.cuh"
}