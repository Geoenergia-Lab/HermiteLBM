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
    Numerical differentiation schemes

Namespace
    LBM

SourceFiles
    derivativeSchemes.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_DERIVATIVESCHEMES_CUH
#define __MBLBM_DERIVATIVESCHEMES_CUH

namespace LBM
{
    namespace numericalSchemes
    {
        namespace derivative
        {
            __device__ __host__ [[nodiscard]] inline consteval host::label_t maxSchemeOrder() noexcept
            {
                return 8;
            }

            __device__ __host__ [[nodiscard]] inline consteval host::label_t gridPadding(const host::label_t SchemeOrder) noexcept
            {
                return SchemeOrder - 1;
            }

            /**
             * @brief Calculate the centered finite difference for a given numerical scheme order
             * @tparam SchemeOrder The order of the numerical scheme
             * @tparam ReturnType The return type of the function
             * @param[in] padded_line The line of the field to calculate the derivative of
             * @param[in] center The coordinate at which to evaluate the finite difference
             **/
            template <const host::label_t SchemeOrder, typename ReturnType, class PaddedLine>
            __host__ [[nodiscard]] inline constexpr ReturnType finite_difference(
                const PaddedLine &padded_line,
                const host::label_t center) noexcept
            {
                LBM::numericalSchemes::assertions::validate<SchemeOrder, maxSchemeOrder()>();

                constexpr const double d_alpha = static_cast<double>(1);

                if constexpr (SchemeOrder == 2)
                {
                    return static_cast<ReturnType>(
                        (padded_line[center + 1] - padded_line[center - 1]) / (2.0 * d_alpha));
                }

                if constexpr (SchemeOrder == 4)
                {
                    return static_cast<ReturnType>(
                        (2.0 / 3.0 * (padded_line[center + 1] - padded_line[center - 1]) -
                         1.0 / 12.0 * (padded_line[center + 2] - padded_line[center - 2])) /
                        d_alpha);
                }

                if constexpr (SchemeOrder == 6)
                {
                    return static_cast<ReturnType>(
                        (3.0 / 4.0 * (padded_line[center + 1] - padded_line[center - 1]) -
                         3.0 / 20.0 * (padded_line[center + 2] - padded_line[center - 2]) +
                         1.0 / 60.0 * (padded_line[center + 3] - padded_line[center - 3])) /
                        d_alpha);
                }

                if constexpr (SchemeOrder == 8)
                {
                    return static_cast<ReturnType>(
                        (4.0 / 5.0 * (padded_line[center + 1] - padded_line[center - 1]) -
                         1.0 / 5.0 * (padded_line[center + 2] - padded_line[center - 2]) +
                         4.0 / 105.0 * (padded_line[center + 3] - padded_line[center - 3]) -
                         1.0 / 280.0 * (padded_line[center + 4] - padded_line[center - 4])) /
                        d_alpha);
                }

                return static_cast<ReturnType>(0);
            }

            /**
             * @brief Fill a line of the field along an arbitrary axis for a given numerical scheme order
             * @tparam alpha The axis direction (X, Y or Z)
             * @tparam SchemeOrder The order of the numerical scheme
             * @param[in] mesh Reference to the lattice mesh
             * @param[in] padded_line The line of the field to fill from f
             * @param[in] f The field of which the derivative is to be computed
             * @param[in] beta The first orthogonal coordinate to alpha
             * @param[in] gamma The second orthogonal coordinate to beta
             **/
            template <const axis::type alpha, const host::label_t SchemeOrder, typename T, typename ReturnType>
            __host__ void fill_padded_line(
                const host::latticeMesh &mesh,
                std::vector<ReturnType> &padded_line,
                const std::vector<T> &f,
                const host::label_t beta,
                const host::label_t gamma)
            {
                // Fill interior region
                for (host::label_t i = 0; i < mesh.dimension<alpha>(); i++)
                {
                    const host::pointLabel I = axis::to_3d<alpha>(beta, gamma, i);
                    padded_line[gridPadding(SchemeOrder) + i] = static_cast<ReturnType>(f[global::idx(I, mesh.dimension<axis::X>(), mesh.dimension<axis::Y>())]);
                }

                // Set front ghost cells
                for (host::label_t i = 0; i < gridPadding(SchemeOrder); i++)
                {
                    const host::pointLabel I = axis::to_3d<alpha>(beta, gamma, gridPadding(SchemeOrder) - i);
                    padded_line[i] = -static_cast<ReturnType>(f[global::idx(I, mesh.dimension<axis::X>(), mesh.dimension<axis::Y>())]);
                }

                // Set back ghost cells
                for (host::label_t i = 0; i < gridPadding(SchemeOrder); i++)
                {
                    const host::pointLabel I = axis::to_3d<alpha>(beta, gamma, mesh.dimension<alpha>() - static_cast<host::label_t>(2) - i);
                    padded_line[gridPadding(SchemeOrder) + mesh.dimension<alpha>() + i] = -static_cast<ReturnType>(f[global::idx(I, mesh.dimension<axis::X>(), mesh.dimension<axis::Y>())]);
                }
            }

            /**
             * @brief Calculates the derivative of a scalar field in the alpha-direction
             * @tparam alpha The axis direction (X, Y or Z)
             * @return The alpha-derivative of f
             * @param[in] f The field to be differentiated
             * @param[in] mesh The lattice mesh
             **/
            template <const axis::type alpha, const host::label_t SchemeOrder, typename ReturnType, typename T>
            __host__ [[nodiscard]] const std::vector<ReturnType> diff(
                const std::vector<T> &f,
                const host::latticeMesh &mesh)
            {
                LBM::numericalSchemes::assertions::validate<SchemeOrder, maxSchemeOrder()>();

                std::vector<ReturnType> result(f.size(), 0);
                std::vector<double> padded_line(mesh.dimension<alpha>() + static_cast<host::label_t>(2) * gridPadding(SchemeOrder), 0);

                for (host::label_t gamma = 0; gamma < mesh.dimension<axis::orthogonal<alpha, 1>()>(); gamma++)
                {
                    for (host::label_t beta = 0; beta < mesh.dimension<axis::orthogonal<alpha, 0>()>(); beta++)
                    {
                        fill_padded_line<alpha, SchemeOrder>(mesh, padded_line, f, beta, gamma);

                        // Compute derivatives for each point in the alpha direction
                        for (host::label_t i = 0; i < mesh.dimension<alpha>(); i++)
                        {
                            const host::label_t center = gridPadding(SchemeOrder) + i;

                            const host::pointLabel I = axis::to_3d<alpha>(beta, gamma, i);

                            result[global::idx(I, mesh.dimension<axis::X>(), mesh.dimension<axis::Y>())] = finite_difference<SchemeOrder, ReturnType>(padded_line, center);
                        }
                    }
                }

                return result;
            }

            template <const host::label_t SchemeOrder, typename ReturnType, typename T>
            __host__ [[nodiscard]] const std::vector<ReturnType> dfdx(
                const std::vector<T> &f,
                const host::latticeMesh &mesh)
            {
                return diff<axis::X, SchemeOrder, ReturnType>(f, mesh);
            }

            template <const host::label_t SchemeOrder, typename ReturnType, typename T>
            __host__ [[nodiscard]] const std::vector<ReturnType> dfdy(
                const std::vector<T> &f,
                const host::latticeMesh &mesh)
            {
                return diff<axis::Y, SchemeOrder, ReturnType>(f, mesh);
            }

            template <const host::label_t SchemeOrder, typename ReturnType, typename T>
            __host__ [[nodiscard]] const std::vector<ReturnType> dfdz(
                const std::vector<T> &f,
                const host::latticeMesh &mesh)
            {
                return diff<axis::Z, SchemeOrder, ReturnType>(f, mesh);
            }

            template <typename ReturnType, const bool LeftBoundary, const bool RightBoundary>
            __host__ [[nodiscard]] inline const thread::array<const ReturnType, block::nx() * 3> stencil_line(
                const std::vector<scalar_t> &f,
                const host::label_t ty, const host::label_t tz,
                const host::label_t bx, const host::label_t by, const host::label_t bz,
                const host::latticeMesh &mesh) noexcept
            {
                if constexpr (LeftBoundary)
                {
                    return {
                        static_cast<ReturnType>(-f[host::idx(0, ty, tz, bx + 1, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(-f[host::idx(7, ty, tz, bx, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(-f[host::idx(6, ty, tz, bx, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(-f[host::idx(5, ty, tz, bx, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(-f[host::idx(4, ty, tz, bx, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(-f[host::idx(3, ty, tz, bx, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(-f[host::idx(2, ty, tz, bx, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(-f[host::idx(1, ty, tz, bx, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(0, ty, tz, bx, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(1, ty, tz, bx, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(2, ty, tz, bx, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(3, ty, tz, bx, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(4, ty, tz, bx, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(5, ty, tz, bx, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(6, ty, tz, bx, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(7, ty, tz, bx, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(0, ty, tz, bx + 1, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(1, ty, tz, bx + 1, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(2, ty, tz, bx + 1, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(3, ty, tz, bx + 1, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(4, ty, tz, bx + 1, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(5, ty, tz, bx + 1, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(6, ty, tz, bx + 1, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(7, ty, tz, bx + 1, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())])};
                }

                if constexpr (RightBoundary)
                {
                    return {
                        static_cast<ReturnType>(f[host::idx(0, ty, tz, bx - 1, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(1, ty, tz, bx - 1, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(2, ty, tz, bx - 1, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(3, ty, tz, bx - 1, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(4, ty, tz, bx - 1, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(5, ty, tz, bx - 1, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(6, ty, tz, bx - 1, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(7, ty, tz, bx - 1, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(0, ty, tz, bx, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(1, ty, tz, bx, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(2, ty, tz, bx, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(3, ty, tz, bx, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(4, ty, tz, bx, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(5, ty, tz, bx, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(6, ty, tz, bx, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(7, ty, tz, bx, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(-f[host::idx(6, ty, tz, bx, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(-f[host::idx(5, ty, tz, bx, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(-f[host::idx(4, ty, tz, bx, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(-f[host::idx(3, ty, tz, bx, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(-f[host::idx(2, ty, tz, bx, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(-f[host::idx(1, ty, tz, bx, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(-f[host::idx(0, ty, tz, bx, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(-f[host::idx(7, ty, tz, bx - 1, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())])};
                }

                if constexpr ((!LeftBoundary) && (!RightBoundary))
                {
                    return {
                        static_cast<ReturnType>(f[host::idx(0, ty, tz, bx - 1, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(1, ty, tz, bx - 1, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(2, ty, tz, bx - 1, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(3, ty, tz, bx - 1, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(4, ty, tz, bx - 1, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(5, ty, tz, bx - 1, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(6, ty, tz, bx - 1, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(7, ty, tz, bx - 1, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(0, ty, tz, bx, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(1, ty, tz, bx, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(2, ty, tz, bx, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(3, ty, tz, bx, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(4, ty, tz, bx, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(5, ty, tz, bx, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(6, ty, tz, bx, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(7, ty, tz, bx, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(0, ty, tz, bx + 1, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(1, ty, tz, bx + 1, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(2, ty, tz, bx + 1, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(3, ty, tz, bx + 1, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(4, ty, tz, bx + 1, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(5, ty, tz, bx + 1, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(6, ty, tz, bx + 1, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())]),
                        static_cast<ReturnType>(f[host::idx(7, ty, tz, bx + 1, by, bz, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>())])};
                }
            };

            template <typename ReturnType, typename T>
            __host__ [[nodiscard]] const std::vector<ReturnType> dfdx_v2(
                const std::vector<T> &f,
                const host::latticeMesh &mesh)
            {
                std::vector<ReturnType> result(f.size(), 0);

                mesh.nDevices().print("nDevices");

                GPU::forAll(
                    mesh.nDevices(),
                    [&]([[maybe_unused]] const host::label_t GPU_x, [[maybe_unused]] const host::label_t GPU_y, [[maybe_unused]] const host::label_t GPU_z)
                    {
                        // const host::label_t virtualDeviceIndex = GPU::idx(GPU_x, GPU_y, GPU_z, nxGPUs, nyGPUs);

                        for (host::label_t bz = 0; bz < mesh.blocksPerDevice<axis::Z>(); bz++)
                        {
                            for (host::label_t by = 0; by < mesh.blocksPerDevice<axis::Y>(); by++)
                            {
                                for (host::label_t bx = 1; bx < mesh.blocksPerDevice<axis::X>() - 1; bx++)
                                {
                                    const host::blockLabel Bx(bx, by, bz);

                                    for (host::label_t tz = 0; tz < block::nz<host::label_t>(); tz++)
                                    {
                                        for (host::label_t ty = 0; ty < block::ny<host::label_t>(); ty++)
                                        {
                                            // Construct a line that spans the width of the stencil across the entire block x dimension
                                            const thread::array<const double, block::nx() * 3> stencil_array = stencil_line<double, false, false>(f, ty, tz, bx, by, bz, mesh);

                                            for (host::label_t tx = 0; tx < block::nx<host::label_t>(); tx++)
                                            {
                                                const host::threadLabel Tx(tx, ty, tz);
                                                const host::label_t center = host::idx(Tx, Bx, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>());

                                                // Get the finite difference value
                                                result[center] = finite_difference<2, ReturnType>(stencil_array, tx + block::nx<host::label_t>());
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        for (host::label_t bz = 0; bz < mesh.blocksPerDevice<axis::Z>(); bz++)
                        {
                            for (host::label_t by = 0; by < mesh.blocksPerDevice<axis::Y>(); by++)
                            {
                                constexpr const host::label_t bx = 0;
                                const host::blockLabel Bx(bx, by, bz);
                                for (host::label_t tz = 0; tz < block::nz<host::label_t>(); tz++)
                                {
                                    for (host::label_t ty = 0; ty < block::ny<host::label_t>(); ty++)
                                    {
                                        // Construct a line that spans the width of the stencil across the entire block x dimension
                                        const thread::array<const double, block::nx() * 3> stencil_array = stencil_line<double, true, false>(f, ty, tz, bx, by, bz, mesh);

                                        for (host::label_t tx = 0; tx < block::nx(); tx++)
                                        {
                                            const host::threadLabel Tx(tx, ty, tz);
                                            const host::label_t center = host::idx(Tx, Bx, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>());

                                            // Get the finite difference value
                                            result[center] = finite_difference<2, ReturnType>(stencil_array, tx + block::nx());
                                        }
                                    }
                                }
                            }
                        }

                        for (host::label_t bz = 0; bz < mesh.blocksPerDevice<axis::Z>(); bz++)
                        {
                            for (host::label_t by = 0; by < mesh.blocksPerDevice<axis::Y>(); by++)
                            {
                                const host::label_t bx = mesh.blocksPerDevice<axis::X>() - 1;
                                const host::blockLabel Bx(bx, by, bz);

                                for (host::label_t tz = 0; tz < block::nz<host::label_t>(); tz++)
                                {
                                    for (host::label_t ty = 0; ty < block::ny<host::label_t>(); ty++)
                                    {
                                        // Construct a line that spans the width of the stencil across the entire block x dimension
                                        const thread::array<const double, block::nx() * 3> stencil_array = stencil_line<double, false, true>(f, ty, tz, bx, by, bz, mesh);

                                        for (host::label_t tx = 0; tx < block::nx(); tx++)
                                        {
                                            const host::threadLabel Tx(tx, ty, tz);
                                            const host::label_t center = host::idx(Tx, Bx, mesh.blocksPerDevice<axis::X>(), mesh.blocksPerDevice<axis::Y>());

                                            // Get the finite difference value
                                            result[center] = finite_difference<2, ReturnType>(stencil_array, tx + block::nx());
                                        }
                                    }
                                }
                            }
                        }
                    });

                return result;
            }
        }

    }
}

#include "curl.cuh"
#include "div.cuh"

#endif