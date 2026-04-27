/*---------------------------------------------------------------------------*\
|                                                                             |
| cudaLBM: CUDA-based moment representation Lattice Boltzmann Method          |
| Developed at UDESC - State University of Santa Catarina                     |
| Website: https://www.udesc.br                                               |
| Github: https://github.com/geoenergiaUDESC/cudaLBM                          |
|                                                                             |
\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_JETWHITENOISE_CUH
#define __MBLBM_JETWHITENOISE_CUH

namespace LBM
{
    namespace jetWhiteNoise
    {
        __device__ [[nodiscard]] static inline constexpr uint32_t hash32(uint32_t x) noexcept
        {
            x ^= x >> 16;
            x *= 0x7FEB352Du;
            x ^= x >> 15;
            x *= 0x846CA68Bu;
            x ^= x >> 16;

            return x;
        }

        __device__ [[nodiscard]] static inline constexpr scalar_t uniform01(const uint32_t seed) noexcept
        {
            constexpr scalar_t inv2_32 = static_cast<scalar_t>(2.3283064365386963e-10);

            return (static_cast<scalar_t>(seed) + static_cast<scalar_t>(0.5)) * inv2_32;
        }

        __device__ [[nodiscard]] static inline scalar_t box_muller(
            scalar_t rrx,
            const scalar_t rry) noexcept
        {
            rrx = fmaxf(rrx, static_cast<scalar_t>(1e-12));
            const scalar_t r = sqrtf(-static_cast<scalar_t>(2) * logf(rrx));
            const scalar_t theta = static_cast<scalar_t>(6.283185307179586476925286766559) * rry;

            return r * cosf(theta);
        }

        template <uint32_t SALT = 0u>
        __device__ [[nodiscard]] static inline scalar_t white_noise(
            const device::label_t x,
            const device::label_t y,
            const device::label_t t) noexcept
        {
            const uint32_t base = (0x9E3779B9u ^ SALT) ^
                                  static_cast<uint32_t>(x) ^
                                  (static_cast<uint32_t>(y) * 0x85EBCA6Bu) ^
                                  (static_cast<uint32_t>(t) * 0xC2B2AE35u);

            const scalar_t rrx = uniform01(hash32(base));
            const scalar_t rry = uniform01(hash32(base ^ 0x68BC21EBu));

            return box_muller(rrx, rry);
        }
    }
}

#endif
