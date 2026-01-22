/*---------------------------------------------------------------------------*\
|                                                                             |
| cudaLBM: CUDA-based moment representation Lattice Boltzmann Method          |
| Developed at UDESC - State University of Santa Catarina                     |
| Website: https://www.udesc.br                                               |
| Github: https://github.com/geoenergiaUDESC/cudaLBM                          |
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
    This file is part of cudaLBM.

    cudaLBM is free software: you can redistribute it and/or modify it
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
    A single-buffer device halo. Intended for split-kernel pipelines where
    halos are written in a dedicated step and read-only thereafter.

Namespace
    LBM::device

SourceFiles
    haloSingle.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_HALO_SINGLE_CUH
#define __MBLBM_HALO_SINGLE_CUH

namespace LBM
{
    namespace device
    {
        /**
         * @class haloSingle
         * @brief Single-buffer halo container (one face-set)
         * @tparam VelocitySet Velocity set configuration defining lattice structure
         *
         * This class allocates a single halo face-set. It is suitable when halo
         * production and consumption are separated by kernel boundaries
         **/
        template <class VelocitySet, const bool PeriodicX, const bool PeriodicY>
        class haloSingle
        {
        public:
            /**
             * @brief Constructs a single halo face-set from moment data and mesh
             * @param[in] mesh Lattice mesh defining simulation domain
             * @param[in] programCtrl Program control parameters
             **/
            __host__ [[nodiscard]] haloSingle(
                const host::latticeMesh &mesh,
                const programControl &programCtrl) noexcept
                : ghost_(
                      [](const host::latticeMesh &m, const programControl &p)
                      {
                          if constexpr (VelocitySet::Q() == 7)
                          {
                              return makePhaseHalo(m, p);
                          }
                          else
                          {
                              return makeHydroHalo(m, p);
                          }
                      }(mesh, programCtrl)) {}

            /**
             * @brief Default destructor
             **/
            ~haloSingle() = default;

            /**
             * @brief Read-only access to halo faces
             * @return Collection of const pointers to halo faces (x0, x1, y0, y1, z0, z1)
             **/
            __device__ __host__ [[nodiscard]] inline constexpr const device::ptrCollection<6, const scalar_t> ghostConst() const noexcept
            {
                return {ghost_.x0Const(), ghost_.x1Const(), ghost_.y0Const(), ghost_.y1Const(), ghost_.z0Const(), ghost_.z1Const()};
            }

            /**
             * @brief Mutable access to halo faces
             * @return Collection of mutable pointers to halo faces (x0, x1, y0, y1, z0, z1)
             **/
            __device__ __host__ [[nodiscard]] inline constexpr const device::ptrCollection<6, scalar_t> ghost() noexcept
            {
                return {ghost_.x0(), ghost_.x1(), ghost_.y0(), ghost_.y1(), ghost_.z0(), ghost_.z1()};
            }

        private:
            /**
             * @brief The single halo face-set
             **/
            haloFace<VelocitySet> ghost_;

            /**
             * @brief Construct halo regions from hydrodynamic moments only
             **/
            __host__ [[nodiscard]] static inline haloFace<VelocitySet>
            makeHydroHalo(const host::latticeMesh &mesh, const programControl &programCtrl) noexcept
            {
                return haloFace<VelocitySet>(
                    host::array<false, scalar_t, VelocitySet, time::instantaneous>("rho", mesh, programCtrl),
                    host::array<false, scalar_t, VelocitySet, time::instantaneous>("u", mesh, programCtrl),
                    host::array<false, scalar_t, VelocitySet, time::instantaneous>("v", mesh, programCtrl),
                    host::array<false, scalar_t, VelocitySet, time::instantaneous>("w", mesh, programCtrl),
                    host::array<false, scalar_t, VelocitySet, time::instantaneous>("m_xx", mesh, programCtrl),
                    host::array<false, scalar_t, VelocitySet, time::instantaneous>("m_xy", mesh, programCtrl),
                    host::array<false, scalar_t, VelocitySet, time::instantaneous>("m_xz", mesh, programCtrl),
                    host::array<false, scalar_t, VelocitySet, time::instantaneous>("m_yy", mesh, programCtrl),
                    host::array<false, scalar_t, VelocitySet, time::instantaneous>("m_yz", mesh, programCtrl),
                    host::array<false, scalar_t, VelocitySet, time::instantaneous>("m_zz", mesh, programCtrl),
                    mesh);
            }

            /**
             * @brief Construct halo regions from hydrodynamic moments + phase field
             **/
            __host__ [[nodiscard]] static inline haloFace<VelocitySet>
            makePhaseHalo(const host::latticeMesh &mesh, const programControl &programCtrl) noexcept
            {
                auto rho = host::array<false, scalar_t, VelocitySet, time::instantaneous>("rho", mesh, programCtrl);
                auto u = host::array<false, scalar_t, VelocitySet, time::instantaneous>("u", mesh, programCtrl);
                auto v = host::array<false, scalar_t, VelocitySet, time::instantaneous>("v", mesh, programCtrl);
                auto w = host::array<false, scalar_t, VelocitySet, time::instantaneous>("w", mesh, programCtrl);
                auto m_xx = host::array<false, scalar_t, VelocitySet, time::instantaneous>("m_xx", mesh, programCtrl);
                auto m_xy = host::array<false, scalar_t, VelocitySet, time::instantaneous>("m_xy", mesh, programCtrl);
                auto m_xz = host::array<false, scalar_t, VelocitySet, time::instantaneous>("m_xz", mesh, programCtrl);
                auto m_yy = host::array<false, scalar_t, VelocitySet, time::instantaneous>("m_yy", mesh, programCtrl);
                auto m_yz = host::array<false, scalar_t, VelocitySet, time::instantaneous>("m_yz", mesh, programCtrl);
                auto m_zz = host::array<false, scalar_t, VelocitySet, time::instantaneous>("m_zz", mesh, programCtrl);
                auto phi = host::array<false, scalar_t, D3Q7, time::instantaneous>("phi", mesh, programCtrl);

                return haloFace<VelocitySet>(
                    rho,
                    u,
                    v,
                    w,
                    m_xx,
                    m_xy,
                    m_xz,
                    m_yy,
                    m_yz,
                    m_zz,
                    mesh,
                    &phi);
            }
        };
    }
}

#endif
