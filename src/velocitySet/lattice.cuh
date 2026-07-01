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
    Definition of lattice speeds and weights for all 3D lattices

Namespace
    LBM

SourceFiles
    lattice.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_LATTICE_CUH
#define __MBLBM_LATTICE_CUH

namespace LBM
{
    template <const host::label_t Q_>
    class lattice
    {
    public:
        static_assert(((Q_ == 19) || (Q_ == 27)), "VelocitySet must be D3Q19 or D3Q27.");

        /**
         * @brief Get weight for stationary component (q=0)
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval T w_0() noexcept
        {
            if constexpr (Q_ == 27)
            {
                return static_cast<T>(static_cast<double>(8) / static_cast<double>(27));
            }

            if constexpr (Q_ == 19)
            {
                return static_cast<T>(static_cast<double>(1) / static_cast<double>(3));
            }
        }

        /**
         * @brief Get weight for orthogonal directions (q=1-6)
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval T w_1() noexcept
        {
            if constexpr (Q_ == 27)
            {
                return static_cast<T>(static_cast<double>(2) / static_cast<double>(27));
            }

            if constexpr (Q_ == 19)
            {
                return static_cast<T>(static_cast<double>(1) / static_cast<double>(18));
            }
        }

        /**
         * @brief Get weight for diagonal directions (q=7-18)
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval T w_2() noexcept
        {
            if constexpr (Q_ == 27)
            {
                return static_cast<T>(static_cast<double>(1) / static_cast<double>(54));
            }

            if constexpr (Q_ == 19)
            {
                return static_cast<T>(static_cast<double>(1) / static_cast<double>(36));
            }
        }

        /**
         * @brief Get weight for corner directions (q=19-26)
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval T w_3() noexcept
        {
            if constexpr (Q_ == 27)
            {
                return static_cast<T>(static_cast<double>(1) / static_cast<double>(216));
            }

            if constexpr (Q_ == 19)
            {
                return static_cast<T>(0);
            }
        }

        /**
         * @brief Get all weights for device computation
         * @return Thread array of 27 weights in D3Q27 order
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval const thread::array<T, Q_> w_q() noexcept
        {
            return make_first_n<T>(w_impl<T>());
        }

        /**
         * @brief Get the lattice speeds as a thread::array
         * @tparam T The underlying data type of the array
         * @tparam alpha The axis (X, Y or Z)
         **/
        template <typename T, const axis::type alpha>
        __device__ __host__ [[nodiscard]] static inline consteval const thread::array<T, Q_> c() noexcept
        {
            return make_first_n<T>(c_base_impl<T, alpha>());
        }

        /**
         * @brief Returns a component of the velocity set along an arbitrary axis
         * @tparam T The type of data to return
         * @tparam alpha The axis (X, Y or Z)
         * @param[in] q The index of the component
         **/
        template <typename T, const axis::type alpha, const device::label_t q_>
        __device__ __host__ [[nodiscard]] static inline consteval T c(const q_i<q_> q) noexcept
        {
            return c<T, alpha>()[q];
        }

        /**
         * @brief Returns the number of components of the velocity set
         **/
        template <typename T = host::label_t>
        __device__ __host__ [[nodiscard]] static inline consteval T Q() noexcept
        {
            return Q_;
        }

        /**
         * @brief Returns the number of components of the velocity set facing any given cardinal direction
         **/
        template <typename T = host::label_t>
        __device__ __host__ [[nodiscard]] static inline consteval T QF() noexcept
        {
            if constexpr (Q_ == 27)
            {
                return 9;
            }

            if constexpr (Q_ == 19)
            {
                return 5;
            }
        }

    private:
        /**
         * @brief Get x-components for all directions (device version)
         * @return Thread array of 27 x-velocity components
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval const thread::array<T, 27> cx_base() noexcept
        {
            return {static_cast<T>(0), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(-1), static_cast<T>(1)};
        }

        /**
         * @brief Get y-components for all directions (device version)
         * @return Thread array of 27 y-velocity components
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval const thread::array<T, 27> cy_base() noexcept
        {
            return {static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(-1), static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(-1), static_cast<T>(1), static_cast<T>(1), static_cast<T>(-1)};
        }

        /**
         * @brief Get z-components for all directions (device version)
         * @return Thread array of 27 z-velocity components
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval const thread::array<T, 27> cz_base() noexcept
        {
            return {static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(-1), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(1), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(-1), static_cast<T>(1), static_cast<T>(1), static_cast<T>(-1), static_cast<T>(1), static_cast<T>(-1)};
        }

        /**
         * @brief Returns the first Q_ elements of an arbitrary array
         * @tparam Fundamental type of the underlying array
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval const thread::array<T, Q_> make_first_n(const thread::array<T, 27> &arr) noexcept
        {
            thread::array<T, Q_> result;
            for (host::label_t i = 0; i < Q_; i++)
            {
                result[i] = arr[i];
            }
            return result;
        }

        /**
         * @brief Fundamental definition of the lattice speeds
         * @tparam The underlying type
         * @tparam alpha The axis direction (X, Y or Z)
         **/
        template <typename T, const axis::type alpha>
        __device__ __host__ [[nodiscard]] static inline consteval const thread::array<T, 27> c_base_impl() noexcept
        {
            axis::assertions::validate<alpha, axis::CAN_BE_NULL>();

            if constexpr (alpha == axis::NO_DIRECTION)
            {
                thread::array<T, 27> result;
                for (host::label_t i = 0; i < 27; i++)
                {
                    result[i] = 1;
                }
                return result;
            }
            if constexpr (alpha == axis::X)
            {
                return cx_base<T>();
            }
            if constexpr (alpha == axis::Y)
            {
                return cy_base<T>();
            }
            if constexpr (alpha == axis::Z)
            {
                return cz_base<T>();
            }
        }

        /**
         * @brief Get all weights for device computation
         * @return Thread array of 27 weights in D3Q27 order
         **/
        template <typename T>
        __device__ __host__ [[nodiscard]] static inline consteval const thread::array<T, 27> w_impl() noexcept
        {
            return {w_0<T>(), w_1<T>(), w_1<T>(), w_1<T>(), w_1<T>(), w_1<T>(), w_1<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_2<T>(), w_3<T>(), w_3<T>(), w_3<T>(), w_3<T>(), w_3<T>(), w_3<T>(), w_3<T>(), w_3<T>()};
        }
    };
}

#endif