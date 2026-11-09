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
    A class applying boundary conditions to the jet in cross flow case

Namespace
    LBM

SourceFiles
    JICF.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_JICF_CUH
#define __MBLBM_JICF_CUH

namespace LBM
{
    /**
     * @class JICF
     *
     * @brief Applies boundary conditions for a jet in cross flow configuration (JICF)
     **/
    class JICF : public boundaryConditionType<false, false, false>
    {
    public:
        using Base = boundaryConditionType::This;

        /**
         * @brief Switch determining whether or not the boundary condition actually applies a condition
         **/
        deviceHostFunctionReturnType static inline consteval bool appliesCondition() noexcept
        {
            return true;
        }

        /**
         * @brief Public method to apply the boundary conditions
         * @tparam VelocitySet The velocity set (D3Q19 or D3Q27)
         * @param[out] moments Moment array (rho, U, Pi)
         * @param[in] boundaryNormal Normal vector information at boundary node
         * @param[in] sharedBuffer Shared memory array containing moment data
         * @param[in] Tx Three-dimensional thread coordinates
         * @param[in] point The global point coordinate
         **/
        template <class VelocitySet>
        __device__ static inline constexpr void calculate_moments(
            momentsArray &moments,
            const NormalVector &boundaryNormal,
            const blockSharedBuffer &sharedBuffer,
            const thread::coordinate &Tx,
            const device::pointCoordinate &point) noexcept
        {
            calculate_moments_impl<VelocitySet>(moments, boundaryNormal, sharedBuffer, Tx, point);
        }

    private:
        deviceFunctionReturnType static inline consteval scalar_t R() noexcept
        {
            return static_cast<scalar_t>(3.3);
        }

        /**
         * @brief Calculate moment variables at boundary nodes
         * @tparam VelocitySet The velocity set (D3Q19 or D3Q27)
         * @param[out] moments Moment array (rho, U, Pi)
         * @param[in] boundaryNormal Normal vector information at boundary node
         * @param[in] sharedBuffer Shared memory array containing moment data
         * @param[in] Tx Three-dimensional thread coordinates
         * @param[in] point The global point coordinate
         **/
        template <class VelocitySet>
        __device__ static inline constexpr void calculate_moments_impl(
            momentsArray &moments,
            const NormalVector &boundaryNormal,
            const blockSharedBuffer &sharedBuffer,
            const thread::coordinate &Tx,
            const device::pointCoordinate &point) noexcept
        {
            constexpr const bool InletNeumannRho = false;
            constexpr const bool OutletNeumannRho = true;

            constexpr const bool OutletNeumannPi = true;

            if ((point.value<axis::Z, device::label_t>() == static_cast<device::label_t>(0)) || (point.value<axis::X, device::label_t>() == static_cast<device::label_t>(0)))
            {
                // Do the Blasius velocity profile
                const scalar_t z = point.value<axis::Z, scalar_t>();
                const scalar_t U_x = uBlasius(z) * device::U_Back[2] / R();

                // Do the top hat velocity profile
                const scalar_t x = point.value<axis::X, scalar_t>() - center<axis::X>();
                const scalar_t y = point.value<axis::Y, scalar_t>() - center<axis::Y>();
                const scalar_t r = math::sqrt(math::pow<2>(x) + math::pow<2>(y));
                const scalar_t A = (math::erf((device::L_char - static_cast<scalar_t>(2) * r) / (static_cast<scalar_t>(2) * math::sqrt(device::sigma))) + math::erf((device::L_char + static_cast<scalar_t>(2) * r) / (static_cast<scalar_t>(2) * math::sqrt(device::sigma)))) / (static_cast<scalar_t>(2) * math::erf(device::L_char / (static_cast<scalar_t>(2) * math::sqrt(device::sigma))));
                const scalar_t U_z = A * device::U_Back[2];

                if constexpr (InletNeumannRho)
                {
                    const device::label_t tidInterior = idxInterior(Tx, boundaryNormal);
                    moments[0] = sharedBuffer[idxShared<0>(tidInterior)];
                }
                else
                {
                    moments[0] = rho0();
                }

                moments[1] = U_x;
                moments[2] = static_cast<scalar_t>(0);
                moments[3] = U_z;

                if constexpr (OutletNeumannPi)
                {
                    Base::equilibrium_moment(moments);
                }
            }
            else
            {
                const device::label_t tidInterior = idxInterior(Tx, boundaryNormal);

                if constexpr (OutletNeumannRho)
                {
                    moments[0] = sharedBuffer[idxShared<0>(tidInterior)];
                }
                else
                {
                    moments[0] = rho0();
                }

                moments[1] = sharedBuffer[idxShared<1>(tidInterior)];
                moments[2] = sharedBuffer[idxShared<2>(tidInterior)];
                moments[3] = sharedBuffer[idxShared<3>(tidInterior)];

                if constexpr (OutletNeumannPi)
                {
                    moments[4] = sharedBuffer[idxShared<4>(tidInterior)];
                    moments[5] = sharedBuffer[idxShared<5>(tidInterior)];
                    moments[6] = sharedBuffer[idxShared<6>(tidInterior)];
                    moments[7] = sharedBuffer[idxShared<7>(tidInterior)];
                    moments[8] = sharedBuffer[idxShared<8>(tidInterior)];
                    moments[9] = sharedBuffer[idxShared<9>(tidInterior)];
                }
            }

            if constexpr (!OutletNeumannPi)
            {
                Base::equilibrium_moment(moments);
            }
        }

        template <typename T>
        deviceFunctionReturnType static inline consteval T tauBlasius() noexcept
        {
            return static_cast<T>(0.332057);
        }

        template <typename T>
        deviceFunctionReturnType static inline consteval T alphaBlasius() noexcept
        {
            return static_cast<T>(0.432089);
        }

        template <typename T>
        deviceFunctionReturnType static inline consteval T alphaSqBlasius() noexcept
        {
            return math::constexpr_pow<2, T, alphaBlasius<double>()>();
        }

        template <typename T>
        deviceFunctionReturnType static inline consteval T coeffBlasius() noexcept
        {
            return static_cast<T>(tauBlasius<double>() - ((static_cast<double>(2) * alphaBlasius<double>()) / math::constants::sqrtpi<double>()));
        }

        deviceFunctionReturnType static inline constexpr scalar_t uBlasius(const scalar_t y) noexcept
        {
            return math::erf(alphaBlasius<scalar_t>() * y) + (y * math::exp(-alphaSqBlasius<scalar_t>() * math::pow<2>(y)) * coeffBlasius<scalar_t>());
        }

        template <const axis::type alpha>
        deviceFunctionReturnType static inline constexpr scalar_t center() noexcept
        {
            if constexpr (alpha == axis::X)
            {
                return static_cast<scalar_t>(256);
            }
            return static_cast<scalar_t>(0.5) * static_cast<scalar_t>(device::latticeMesh::nPoints[alpha] - static_cast<device::label_t>(1));
        }

        deviceFunctionReturnType static inline constexpr scalar_t dudx(const scalar_t x, const scalar_t y) noexcept
        {
            return -(x * (math::exp(-math::pow<2>(device::L_char - static_cast<scalar_t>(2) * math::sqrt(math::pow<2>(x) + math::pow<2>(y))) / (static_cast<scalar_t>(4) * device::sigma)) - math::exp(-math::pow<2>(device::L_char + static_cast<scalar_t>(2) * math::sqrt(math::pow<2>(x) + math::pow<2>(y))) / (static_cast<scalar_t>(4) * device::sigma)))) / (math::sqrt(device::sigma) * math::constants::sqrtpi<scalar_t>() * math::erf(device::L_char / (static_cast<scalar_t>(2) * math::sqrt(device::sigma))) * math::sqrt(math::pow<2>(x) + math::pow<2>(y)));
        }

        deviceFunctionReturnType static inline constexpr scalar_t dudy(const scalar_t x, const scalar_t y) noexcept
        {
            return -(y * (math::exp(-math::pow<2>(device::L_char - static_cast<scalar_t>(2) * math::sqrt(math::pow<2>(x) + math::pow<2>(y))) / (static_cast<scalar_t>(4) * device::sigma)) - math::exp(-math::pow<2>(device::L_char + static_cast<scalar_t>(2) * math::sqrt(math::pow<2>(x) + math::pow<2>(y))) / (static_cast<scalar_t>(4) * device::sigma)))) / (math::sqrt(device::sigma) * math::constants::sqrtpi<scalar_t>() * math::erf(device::L_char / (static_cast<scalar_t>(2) * math::sqrt(device::sigma))) * math::sqrt(math::pow<2>(x) + math::pow<2>(y)));
        }
    };
}

#endif