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
    Implements Guo's forcing scheme for the collision step

Namespace
    LBM

SourceFiles
    GuoForcing.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_GUOFORCING_CUH
#define __MBLBM_GUOFORCING_CUH

namespace LBM
{
    namespace forcing
    {
        class Guo : public forceBase<true>
        {
        public:
            hostFunctionReturnType static inline consteval const char *name() noexcept { return "Guo"; }

            template <const axis::type alpha>
            deviceFunctionReturnType static inline constexpr scalar_t apply() noexcept
            {
                if constexpr (alpha == axis::X)
                {
                    return static_cast<scalar_t>(0);
                }

                if constexpr (alpha == axis::Y)
                {
                    return static_cast<scalar_t>(0);
                }

                if constexpr (alpha == axis::Z)
                {
                    return device::g;
                }
            }

            template <const axis::type alpha, const axis::type beta>
            deviceFunctionReturnType static inline constexpr scalar_t apply(const momentsArray &moments) noexcept
            {
                axis::assertions::validate<alpha, axis::NOT_NULL>();
                axis::assertions::validate<beta, axis::NOT_NULL>();
                if constexpr (alpha == beta)
                {
                    return -(static_cast<scalar_t>(1.5) * apply<alpha>() * moments[axis::index<alpha>()] * (device::omega - static_cast<scalar_t>(2))) / moments[axis::index<axis::NO_DIRECTION>()];
                }
                else
                {
                    return -(static_cast<scalar_t>(1.5) * (apply<alpha>() * moments[axis::index<beta>()] + apply<beta>() * moments[axis::index<alpha>()]) * (device::omega - static_cast<scalar_t>(2))) / moments[axis::index<axis::NO_DIRECTION>()];
                }
            }

        private:
            // Froude number
            deviceHostFunctionReturnType static inline consteval scalar_t Fr() noexcept
            {
                return static_cast<scalar_t>(1);
            }
        };
    }
}

#endif