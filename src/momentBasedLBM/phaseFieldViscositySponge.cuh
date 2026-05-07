/*---------------------------------------------------------------------------*\
|                                                                             |
| cudaLBM: CUDA-based moment representation Lattice Boltzmann Method          |
| Developed at UDESC - State University of Santa Catarina                     |
| Website: https://www.udesc.br                                               |
| Github: https://github.com/geoenergiaUDESC/cudaLBM                          |
|                                                                             |
\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_PHASEFIELDVISCOSITYSPONGE_CUH
#define __MBLBM_PHASEFIELDVISCOSITYSPONGE_CUH

#include "../LBMIncludes.cuh"
#include "../typedefs/typedefs.cuh"

namespace LBM
{
    namespace phaseFieldSponge
    {
        __device__ __host__ [[nodiscard]] inline consteval scalar_t K_gain() noexcept
        {
            return static_cast<scalar_t>(100);
        }

        __device__ [[nodiscard]] inline device::label_t sponge_cells() noexcept
        {
            return device::ny / static_cast<device::label_t>(8);
        }

        __device__ [[nodiscard]] inline scalar_t sponge_width() noexcept
        {
            return static_cast<scalar_t>(static_cast<scalar_t>(sponge_cells()) / static_cast<scalar_t>(device::ny - static_cast<device::label_t>(1)));
        }

        __device__ [[nodiscard]] inline scalar_t y_start() noexcept
        {
            return static_cast<scalar_t>(static_cast<scalar_t>(device::ny - static_cast<device::label_t>(1) - sponge_cells()) / static_cast<scalar_t>(device::ny - static_cast<device::label_t>(1)));
        }

        __device__ [[nodiscard]] inline scalar_t inv_ny_m1() noexcept
        {
            return static_cast<scalar_t>(static_cast<scalar_t>(1) / static_cast<scalar_t>(device::ny - static_cast<device::label_t>(1)));
        }

        __device__ [[nodiscard]] inline scalar_t inv_sponge() noexcept
        {
            return static_cast<scalar_t>(static_cast<scalar_t>(1) / static_cast<scalar_t>(sponge_width()));
        }

        __device__ [[nodiscard]] inline scalar_t ramp_ymax(const device::label_t yGlobal) noexcept
        {
            const scalar_t yn = static_cast<scalar_t>(yGlobal) * inv_ny_m1();
            scalar_t s = (yn - y_start()) * inv_sponge();
            s = (s < static_cast<scalar_t>(0)) ? static_cast<scalar_t>(0) : s;
            s = (s > static_cast<scalar_t>(1)) ? static_cast<scalar_t>(1) : s;
            return s * s * s * (s * (s * static_cast<scalar_t>(6) - static_cast<scalar_t>(15)) + static_cast<scalar_t>(10));
        }

        __device__ [[nodiscard]] inline scalar_t tau_local(const scalar_t phi) noexcept
        {
            return (static_cast<scalar_t>(1) - phi) * device::tau_A + phi * device::tau_B;
        }

        __device__ [[nodiscard]] inline scalar_t tau_ymax(const scalar_t phi) noexcept
        {
            const scalar_t tau = tau_local(phi);
            return static_cast<scalar_t>(0.5) + (tau - static_cast<scalar_t>(0.5)) * (static_cast<scalar_t>(1) + K_gain());
        }

        template <const bool ApplySponge>
        __device__ [[nodiscard]] inline scalar_t tau(const scalar_t phi, const device::label_t yGlobal) noexcept
        {
            const scalar_t tauPhi = tau_local(phi);

            if constexpr (ApplySponge)
            {
                const scalar_t r = ramp_ymax(yGlobal);
                return tauPhi + r * (tau_ymax(phi) - tauPhi);
            }

            (void)yGlobal;
            return tauPhi;
        }

        template <const bool ApplySponge>
        __device__ [[nodiscard]] inline scalar_t omega(const scalar_t phi, const device::label_t yGlobal) noexcept
        {
            return static_cast<scalar_t>(1) / tau<ApplySponge>(phi, yGlobal);
        }
    }
}

#endif
