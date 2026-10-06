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
    Implementation of the moment representation with the D3Q27 velocity set

Namespace
    LBM

SourceFiles
    testExecutable.cu

\*---------------------------------------------------------------------------*/

#include "testExecutable.cuh"

#include "../../../src/momentBasedLBM/momentBasedLBM.cuh"

using namespace LBM;

namespace LBM
{
    __host__ void printBinaryRepresentation(const nodeType_t t)
    {
        std::cout << static_cast<host::label_t>(t) << ":" << std::endl;
        for (nodeType_t i = 0; i < 8; ++i)
        {
            std::cout << ((t >> i) & static_cast<nodeType_t>(1));
        }
        std::cout << std::endl;
    }

    template <const axis::type alpha, const axis::type beta>
    __host__ [[nodiscard]] inline constexpr scalar_t H(const int c_a, const int c_b) noexcept
    {
        return static_cast<scalar_t>(c_a * c_b) - (static_cast<scalar_t>(alpha == beta) / static_cast<scalar_t>(3));
    }

    template <const axis::type alpha, const axis::type beta>
    __host__ [[nodiscard]] inline constexpr bool has_H(const int c_a, const int c_b) noexcept
    {
        return static_cast<bool>((c_a * c_b * 3) - static_cast<int>(alpha == beta));
    }

    template <const nodeType_t Boundary>
    __host__ [[nodiscard]] inline consteval const char *boundaryName() noexcept
    {
        if constexpr (Boundary == normalVectorBase::EAST())
        {
            return "EAST";
        }
        else if constexpr (Boundary == normalVectorBase::WEST())
        {
            return "WEST";
        }
        else if constexpr (Boundary == normalVectorBase::NORTH())
        {
            return "NORTH";
        }
        else if constexpr (Boundary == normalVectorBase::SOUTH())
        {
            return "SOUTH";
        }
        else if constexpr (Boundary == normalVectorBase::FRONT())
        {
            return "FRONT";
        }
        else if constexpr (Boundary == normalVectorBase::BACK())
        {
            return "BACK";
        }
        else
        {
            return "INTERNAL";
        }
    }

    template <const nodeType_t Boundary>
    __host__ void boundaryCheck(const nodeType_t t) noexcept
    {
        std::cout << "Is " << boundaryName<Boundary>() << ": " << normalVectorBase::is<bool, Boundary>(t) << std::endl;
    }

    template <typename T, const nodeType_t Boundary, const device::label_t q_>
    __device__ __host__ [[nodiscard]] static inline constexpr T is_incoming() noexcept
    {
        // boundaryNormal.x > 0  => EAST boundary
        // boundaryNormal.x < 0  => WEST boundary
        const bool cond_x = (normalVectorBase::is<bool, normalVectorBase::EAST()>(Boundary) & VelocitySet::is_negative<axis::X>(integralConstant<device::label_t, q_>())) |
                            (normalVectorBase::is<bool, normalVectorBase::WEST()>(Boundary) & VelocitySet::is_positive<axis::X>(integralConstant<device::label_t, q_>()));

        // boundaryNormal.y > 0  => NORTH boundary
        // boundaryNormal.y < 0  => SOUTH boundary
        const bool cond_y = (normalVectorBase::is<bool, normalVectorBase::NORTH()>(Boundary) & VelocitySet::is_negative<axis::Y>(integralConstant<device::label_t, q_>())) |
                            (normalVectorBase::is<bool, normalVectorBase::SOUTH()>(Boundary) & VelocitySet::is_positive<axis::Y>(integralConstant<device::label_t, q_>()));

        // boundaryNormal.z > 0  => FRONT boundary
        // boundaryNormal.z < 0  => BACK boundary
        const bool cond_z = (normalVectorBase::is<bool, normalVectorBase::FRONT()>(Boundary) & VelocitySet::is_negative<axis::Z>(integralConstant<device::label_t, q_>())) |
                            (normalVectorBase::is<bool, normalVectorBase::BACK()>(Boundary) & VelocitySet::is_positive<axis::Z>(integralConstant<device::label_t, q_>()));

        return static_cast<T>(!(cond_x | cond_y | cond_z));
    }
}

int main()
{
    constexpr const nodeType_t t = normalVectorBase::NORTH_EAST();

    LBM::printBinaryRepresentation(t);

    LBM::printBinaryRepresentation(t & 0x40);

    constexpr const axis::type alpha = axis::X;
    constexpr const axis::type beta = axis::X;

    constexpr const thread::array<int, 27> ca = VelocitySet::template c<int, alpha>();
    constexpr const thread::array<int, 27> cb = VelocitySet::template c<int, beta>();

    // for (host::label_t i = 0; i < 27; ++i)
    // {
    //     std::cout << "H(" << ca[i] << ", " << cb[i] << ", " << i << ") = " << LBM::has_H<alpha, beta>(ca[i], cb[i]) << std::endl;
    // }

    LBM::boundaryCheck<normalVectorBase::WEST()>(t);
    LBM::boundaryCheck<normalVectorBase::EAST()>(t);
    LBM::boundaryCheck<normalVectorBase::NORTH()>(t);
    LBM::boundaryCheck<normalVectorBase::SOUTH()>(t);
    LBM::boundaryCheck<normalVectorBase::BACK()>(t);
    LBM::boundaryCheck<normalVectorBase::FRONT()>(t);

    host::constexpr_for<0, VelocitySet::Q()>(
        [&](const auto i)
        {
            // if (is_incoming<bool, t, i>())
            {
                std::cout << LBM::H<alpha, beta>(ca[i], cb[i]) * is_incoming<scalar_t, t, i>() << std::endl;
                // std::cout << i << std::endl;
            }
        });

    return 0;
}