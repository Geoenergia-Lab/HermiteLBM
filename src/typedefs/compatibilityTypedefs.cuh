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
    A list of typedefs used throughout the HermiteLBM source code

Namespace
    LBM

SourceFiles
    compatibilityTypedefs.cuh

\*---------------------------------------------------------------------------*/

#ifndef __MBLBM_COMPATIBILITYTYPEDEFS_CUH
#define __MBLBM_COMPATIBILITYTYPEDEFS_CUH

namespace LBM
{

#define GPU_COMPILER_ERROR_MSG "GPU compiler not found"

    /**
     * @brief Supported GPU brands for the build
     **/
    typedef enum deviceBrandEnum : int64_t
    {
        UNDEFINED = -1,
        NVIDIA = 0,
        AMD = 1,
    } deviceBrandEnum;

    struct deviceType
    {
#if defined(__HIP__)
        static constexpr const deviceBrandEnum v = AMD;
#elif defined(__CUDACC__)
        static constexpr const deviceBrandenum v = NVIDIA;
#else
        static constexpr const deviceBrandEnum v = UNDEFINED;
#endif
    };

#if defined(__HIP__)
#define devicehostFunctionReturnType [[nodiscard]] __device__ __host__
#define deviceFunctionReturnType [[nodiscard]] __device__
#define hostFunctionReturnType [[nodiscard]] __host__

    using deviceError_t = hipError_t;
    using deviceStream_t = hipStream_t;
    using deviceEvent_t = hipEvent_t;
    using deviceMemcpyKind = hipMemcpyKind;
    using deviceProp_t = hipDeviceProp_t;
    using ptrAttributes_t = hipPointerAttribute_t;
    using deviceFuncCache_t = hipFuncCache_t;

    typedef enum cachePreferenceTypeEnum : int
    {
        PREFER_NONE = hipFuncCachePreferNone,
        PREFER_SHARED = hipFuncCachePreferShared,
        PREFER_L1 = hipFuncCachePreferL1
    } cachePreferenceType;

    typedef enum deviceMemcpyTypeEnum : int
    {
        memcpyHostToHost = hipMemcpyHostToHost,
        memcpyDeviceToDevice = hipMemcpyDeviceToDevice,
        memcpyHostToDevice = hipMemcpyHostToDevice,
        memcpyDeviceToHost = hipMemcpyDeviceToHost
    } deviceMemcpyType_t;

    struct errorCodes
    {
        static constexpr const deviceError_t deviceSuccess = hipSuccess;
        static constexpr const deviceError_t deviceErrorMemoryAllocation = hipErrorOutOfMemory;
    };

    struct memoryType
    {
        using T = hipMemoryType;
        static constexpr const T unregistered = hipMemoryTypeUnregistered;
        static constexpr const T host = hipMemoryTypeHost;
        static constexpr const T device = hipMemoryTypeDevice;
        static constexpr const T managed = hipMemoryTypeManaged;
    };

#elif defined(__CUDACC__)

#define devicehostFunctionReturnType __device__ __host__ [[nodiscard]]
#define deviceFunctionReturnType __device__ [[nodiscard]]
#define hostFunctionReturnType __host__ [[nodiscard]]

    using deviceError_t = cudaError_t;
    using deviceStream_t = cudaStream_t;
    using deviceEvent_t = cudaEvent_t;
    using deviceMemcpyKind = cudaMemcpyKind;
    using deviceProp_t = cudaDeviceProp;
    using ptrAttributes_t = cudaPointerAttributes;
    using deviceFuncCache_t = cudaFuncCache;

    typedef enum cachePreferenceTypeEnum : int
    {
        PREFER_NONE = cudaFuncCachePreferNone,
        PREFER_SHARED = cudaFuncCachePreferShared,
        PREFER_L1 = cudaFuncCachePreferL1
    } cachePreferenceType;

    typedef enum deviceMemcpyTypeEnum : int
    {
        memcpyHostToHost = cudaMemcpyHostToHost,
        memcpyDeviceToDevice = cudaMemcpyDeviceToDevice,
        memcpyHostToDevice = cudaMemcpyHostToDevice,
        memcpyDeviceToHost = cudaMemcpyDeviceToHost
    } deviceMemcpyType_t;

    struct errorCodes
    {
        static constexpr const deviceError_t deviceSuccess = cudaSuccess;
        static constexpr const deviceError_t deviceErrorMemoryAllocation = cudaErrorMemoryAllocation;
    };

    struct memoryType
    {
        using T = cudaMemoryType;
        static constexpr const T unregistered = cudaMemoryTypeUnregistered;
        static constexpr const T host = cudaMemoryTypeHost;
        static constexpr const T device = cudaMemoryTypeDevice;
        static constexpr const T managed = cudaMemoryTypeManaged;
    };

#else
    static_assert(false, GPU_COMPILER_ERROR_MSG);

#endif

    namespace API
    {
        hostFunctionReturnType inline deviceError_t getDeviceCount(int *count) noexcept
        {
#if defined(__HIP__)
            return hipGetDeviceCount(count);
#elif defined(__CUDACC__)
            return cudaGetDeviceCount(count);
#else
            static_assert(false, GPU_COMPILER_ERROR_MSG);
            return 0;
#endif
        }

        hostFunctionReturnType inline deviceError_t getDeviceProperties(deviceProp_t *props, int deviceID) noexcept
        {
#if defined(__HIP__)
            return hipGetDeviceProperties(props, deviceID);
#elif defined(__CUDACC__)
            return cudaGetDeviceProperties(props, deviceID);
#else
            static_assert(false, GPU_COMPILER_ERROR_MSG);
            return 0;
#endif
        }

        hostFunctionReturnType inline deviceError_t getDevice(int *deviceID) noexcept
        {
#if defined(__HIP__)
            return hipGetDevice(deviceID);
#elif defined(__CUDACC__)
            return cudaGetDevice(deviceID);
#else
            static_assert(false, GPU_COMPILER_ERROR_MSG);
            return 0;
#endif
        }

        hostFunctionReturnType inline deviceError_t setDevice(const int deviceID) noexcept
        {
#if defined(__HIP__)
            return hipSetDevice(deviceID);
#elif defined(__CUDACC__)
            return cudaSetDevice(deviceID);
#else
            static_assert(false, GPU_COMPILER_ERROR_MSG);
            return 0;
#endif
        }

        hostFunctionReturnType inline deviceError_t deviceSynchronize() noexcept
        {
#if defined(__HIP__)
            return hipDeviceSynchronize();
#elif defined(__CUDACC__)
            return cudaDeviceSynchronize();
#else
            static_assert(false, GPU_COMPILER_ERROR_MSG);
            return 0;
#endif
        }

        namespace stream
        {
            hostFunctionReturnType inline deviceError_t create(deviceStream_t *stream) noexcept
            {
#if defined(__HIP__)
                return hipStreamCreate(stream);
#elif defined(__CUDACC__)
                return cudaStreamCreate(stream);
#else
                static_assert(false, GPU_COMPILER_ERROR_MSG);
                return 0;
#endif
            }

            hostFunctionReturnType inline deviceError_t destroy(deviceStream_t stream) noexcept
            {
#if defined(__HIP__)
                return hipStreamDestroy(stream);
#elif defined(__CUDACC__)
                return cudaStreamDestroy(stream);
#else
                static_assert(false, GPU_COMPILER_ERROR_MSG);
                return 0;
#endif
            }

            hostFunctionReturnType inline deviceError_t synchronize(deviceStream_t stream) noexcept
            {
#if defined(__HIP__)
                return hipStreamSynchronize(stream);
#elif defined(__CUDACC__)
                return cudaStreamSynchronize(stream);
#else
                static_assert(false, GPU_COMPILER_ERROR_MSG);
                return 0;
#endif
            }
        }

        template <typename T>
        hostFunctionReturnType inline deviceError_t memcpy(T *const dst, const T *const src, const size_t count, const deviceMemcpyType_t kind) noexcept
        {
#if defined(__HIP__)
            return hipMemcpy(dst, src, count, static_cast<deviceMemcpyKind>(kind));
#elif defined(__CUDACC__)
            return cudaMemcpy(dst, src, count, static_cast<deviceMemcpyKind>(kind));
#else
            static_assert(false, GPU_COMPILER_ERROR_MSG);
            return 0;
#endif
        }

        template <typename Symbol, typename T>
        hostFunctionReturnType inline deviceError_t memcpyToSymbol(const Symbol &symbol, const T *const src, const size_t count, const size_t offset, const deviceMemcpyType_t kind) noexcept
        {
#if defined(__HIP__)
            return hipMemcpyToSymbol(HIP_SYMBOL(symbol), src, count, offset, static_cast<deviceMemcpyKind>(kind));
#elif defined(__CUDACC__)
            return cudaMemcpyToSymbol(symbol, src, count, offset, static_cast<deviceMemcpyKind>(kind));
#else
            static_assert(false, GPU_COMPILER_ERROR_MSG);
            return 0;
#endif
        }

        hostFunctionReturnType inline deviceError_t memGetInfo(size_t *free, size_t *total) noexcept
        {
#if defined(__HIP__)
            return hipMemGetInfo(free, total);
#elif defined(__CUDACC__)
            return cudaMemGetInfo(free, total);
#else
            static_assert(false, GPU_COMPILER_ERROR_MSG);
            return 0;
#endif
        }

        template <typename T>
        hostFunctionReturnType inline deviceError_t pointerGetAttributes(ptrAttributes_t *attributes, const T *ptr) noexcept
        {
#if defined(__HIP__)
            return hipPointerGetAttributes(attributes, ptr);
#elif defined(__CUDACC__)
            return cudaPointerGetAttributes(attributes, ptr);
#else
            static_assert(false, GPU_COMPILER_ERROR_MSG);
            return 0;
#endif
        }

        hostFunctionReturnType inline const char *getErrorString(const deviceError_t error) noexcept
        {
#if defined(__HIP__)
            return hipGetErrorString(error);
#elif defined(__CUDACC__)
            return cudaGetErrorString(error);
#else
            static_assert(false, GPU_COMPILER_ERROR_MSG);
            return 0;
#endif
        }

        template <typename Kernel, typename = std::enable_if_t<std::is_function_v<Kernel>>>
        hostFunctionReturnType inline deviceError_t funcSetCacheConfig(Kernel *func, const deviceFuncCache_t config) noexcept
        {
#if defined(__HIP__)
            return hipFuncSetCacheConfig(reinterpret_cast<void *>(func), config);
#elif defined(__CUDACC__)
            return cudaFuncSetCacheConfig(func, config);
#else
            static_assert(false, GPU_COMPILER_ERROR_MSG);
            return 0;
#endif
        }

        template <typename Kernel>
        hostFunctionReturnType inline deviceError_t funcSetMaxDynamicSharedMemorySize(Kernel *func, const int bytes) noexcept
        {
#if defined(__HIP__)
            return hipFuncSetAttribute(reinterpret_cast<void *>(func), hipFuncAttributeMaxDynamicSharedMemorySize, bytes);
#elif defined(__CUDACC__)
            return cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
#else
            static_assert(false, GPU_COMPILER_ERROR_MSG);
            return 0;
#endif
        }

        namespace device
        {
            template <typename T>
            hostFunctionReturnType inline deviceError_t free(T *const ptr) noexcept
            {
#if defined(__HIP__)
                return hipFree(ptr);
#elif defined(__CUDACC__)
                return cudaFree(ptr);
#else
                static_assert(false, GPU_COMPILER_ERROR_MSG);
                return 0;
#endif
            }

            template <typename T>
            hostFunctionReturnType inline deviceError_t malloc(T **ptr, const size_t size) noexcept
            {
#if defined(__HIP__)
                return hipMalloc(ptr, size);
#elif defined(__CUDACC__)
                return cudaMalloc(ptr, size);
#else
                static_assert(false, GPU_COMPILER_ERROR_MSG);
                return 0;
#endif
            }
        }

        namespace host
        {
            template <typename T>
            hostFunctionReturnType inline deviceError_t free(T *const ptr) noexcept
            {
#if defined(__HIP__)
                return hipHostFree(ptr);
#elif defined(__CUDACC__)
                return cudaFreeHost(ptr);
#else
                static_assert(false, GPU_COMPILER_ERROR_MSG);
                return 0;
#endif
            }

            template <typename T>
            hostFunctionReturnType inline deviceError_t malloc(T **ptr, const size_t size) noexcept
            {
#if defined(__HIP__)
                return hipHostMalloc(ptr, size);
#elif defined(__CUDACC__)
                return cudaMallocHost(ptr, size);
#else
                static_assert(false, GPU_COMPILER_ERROR_MSG);
                return 0;
#endif
            }
        }

        template <typename T>
        hostFunctionReturnType inline deviceError_t memcpyPeerAsync(T *const dst, const int dstDeviceId, const T *const src, const int srcDeviceId, const size_t count, const deviceStream_t stream) noexcept
        {
#if defined(__HIP__)
            return hipMemcpyPeerAsync(dst, dstDeviceId, src, srcDeviceId, count, stream);
#elif defined(__CUDACC__)
            return cudaMemcpyPeerAsync(dst, dstDeviceId, src, srcDeviceId, count, stream);
#else
            static_assert(false, GPU_COMPILER_ERROR_MSG);
            return 0;
#endif
        }

        template <typename T>
        hostFunctionReturnType inline deviceError_t memcpyAsync(T *const dst, const T *const src, const size_t count, const deviceMemcpyType_t kind, const deviceStream_t stream) noexcept
        {
#if defined(__HIP__)
            return hipMemcpyAsync(dst, src, count, static_cast<deviceMemcpyKind>(kind), stream);
#elif defined(__CUDACC__)
            return cudaMemcpyAsync(dst, src, count, static_cast<deviceMemcpyKind>(kind), stream);
#else
            static_assert(false, GPU_COMPILER_ERROR_MSG);
            return 0;
#endif
        }
    }
}

#endif