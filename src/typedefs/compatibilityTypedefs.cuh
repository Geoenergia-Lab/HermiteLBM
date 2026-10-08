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
#if defined(__HIP__)
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

    static constexpr const deviceError_t deviceSuccess = hipSuccess;
    static constexpr const deviceError_t deviceErrorMemoryAllocation = hipErrorOutOfMemory;

    namespace device
    {
        // enum memoryType
        // {
        //     memoryTypeUnregistered = hipMemoryTypeUnregistered,
        //     memoryTypeHost = hipMemoryTypeHost,
        //     memoryTypeDevice = hipMemoryTypeDevice,
        //     memoryTypeManaged = hipMemoryTypeManaged
        // };

        static constexpr const hipMemoryType memoryTypeUnregistered = hipMemoryTypeUnregistered;
        static constexpr const hipMemoryType memoryTypeHost = hipMemoryTypeHost;
        static constexpr const hipMemoryType memoryTypeDevice = hipMemoryTypeDevice;
        static constexpr const hipMemoryType memoryTypeManaged = hipMemoryTypeManaged;

        namespace API
        {
            [[nodiscard]] __host__ inline deviceError_t getDeviceCount(int *count) noexcept
            {
                return hipGetDeviceCount(count);
            }

            [[nodiscard]] __host__ inline deviceError_t getDeviceProperties(deviceProp_t *props, int deviceID) noexcept
            {
                return hipGetDeviceProperties(props, deviceID);
            }

            [[nodiscard]] __host__ inline deviceError_t getDevice(int *deviceID) noexcept
            {
                return hipGetDevice(deviceID);
            }

            [[nodiscard]] __host__ inline deviceError_t setDevice(const int deviceID) noexcept
            {
                return hipSetDevice(deviceID);
            }

            [[nodiscard]] __host__ inline deviceError_t deviceSynchronize() noexcept
            {
                return hipDeviceSynchronize();
            }

            [[nodiscard]] __host__ inline deviceError_t streamCreate(deviceStream_t *stream) noexcept
            {
                return hipStreamCreate(stream);
            }

            [[nodiscard]] __host__ inline deviceError_t streamDestroy(deviceStream_t stream) noexcept
            {
                return hipStreamDestroy(stream);
            }

            [[nodiscard]] __host__ inline deviceError_t streamSynchronize(deviceStream_t stream) noexcept
            {
                return hipStreamSynchronize(stream);
            }

            template <typename T>
            [[nodiscard]] __host__ inline deviceError_t memcpy(T *const dst, const T *src, const size_t count, const deviceMemcpyType_t kind) noexcept
            {
                return hipMemcpy(dst, src, count, static_cast<hipMemcpyKind>(kind));
            }

            template <typename T>
            [[nodiscard]] __host__ inline deviceError_t memcpyToSymbol(
                const T &symbol, const void *src, const size_t count,
                const size_t offset,
                const deviceMemcpyType_t kind) noexcept
            {
                return hipMemcpyToSymbol(HIP_SYMBOL(symbol), src, count, offset, static_cast<hipMemcpyKind>(kind));
            }

            [[nodiscard]] __host__ inline deviceError_t memGetInfo(size_t *free, size_t *total) noexcept
            {
                return hipMemGetInfo(free, total);
            }

            [[nodiscard]] __host__ inline deviceError_t pointerGetAttributes(hipPointerAttribute_t *attributes, const void *ptr) noexcept
            {
                return hipPointerGetAttributes(attributes, ptr);
            }

            [[nodiscard]] __host__ inline constexpr const char *getErrorString(const deviceError_t error) noexcept
            {
                return hipGetErrorString(error);
            }

            template <typename Kernel, typename = std::enable_if_t<std::is_function_v<Kernel>>>
            [[nodiscard]] __host__ inline deviceError_t funcSetCacheConfig(Kernel *func, const hipFuncCache_t config) noexcept
            {
                return hipFuncSetCacheConfig(
                    reinterpret_cast<const void *>(func),
                    config);
            }

            template <typename Kernel>
            [[nodiscard]] __host__ inline deviceError_t funcSetMaxDynamicSharedMemorySize(Kernel *func, const int bytes) noexcept
            {
                return hipFuncSetAttribute(reinterpret_cast<const void *>(func), hipFuncAttributeMaxDynamicSharedMemorySize, bytes);
            }

            template <typename T>
            [[nodiscard]] __host__ inline deviceError_t hostFree(T *const ptr) noexcept
            {
                return hipHostFree(ptr);
            }

            template <typename T>
            [[nodiscard]] __host__ inline deviceError_t deviceFree(T *const ptr) noexcept
            {
                return hipFree(ptr);
            }

            template <typename T>
            [[nodiscard]] __host__ inline deviceError_t deviceMalloc(T **ptr, const size_t size) noexcept
            {
                return hipMalloc(ptr, size);
            }

            template <typename T>
            [[nodiscard]] __host__ inline deviceError_t mallocHost(T **ptr, const size_t size) noexcept
            {
                return hipHostMalloc(ptr, size, hipHostMallocDefault);
            }

            template <typename T>
            [[nodiscard]] __host__ inline deviceError_t
            memcpyPeerAsync(T *const dst, const int dstDeviceId, const T *const src, const int srcDeviceId, const size_t count, const deviceStream_t stream) noexcept
            {
                return hipMemcpyPeerAsync(dst, dstDeviceId, src, srcDeviceId, count, stream);
            }

            template <typename T>
            [[nodiscard]] __host__ inline deviceError_t memcpyAsync(T *const dst, const T *const src, const size_t count, const deviceMemcpyType_t kind, const hipStream_t stream) noexcept
            {
                return hipMemcpyAsync(dst, src, count, static_cast<hipMemcpyKind>(kind), stream);
            }
        }
    }

#else
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

    static constexpr const deviceError_t deviceSuccess = cudaSuccess;
    static constexpr const deviceError_t deviceErrorMemoryAllocation = cudaErrorMemoryAllocation;

    namespace device
    {
        // static constexpr const memoryType_t memoryTypeUnregistered = cudaMemoryTypeUnregistered;
        // static constexpr const memoryType_t memoryTypeHost = cudaMemoryTypeHost;
        // static constexpr const memoryType_t memoryTypeDevice = cudaMemoryTypeDevice;
        // static constexpr const memoryType_t memoryTypeManaged = cudaMemoryTypeManaged;

        static constexpr const cudaMemoryType memoryTypeUnregistered = cudaMemoryTypeUnregistered;
        static constexpr const cudaMemoryType memoryTypeHost = cudaMemoryTypeHost;
        static constexpr const cudaMemoryType memoryTypeDevice = cudaMemoryTypeDevice;
        static constexpr const cudaMemoryType memoryTypeManaged = cudaMemoryTypeManaged;

        // enum memoryType
        // {
        //     memoryTypeUnregistered = cudaMemoryTypeUnregistered,
        //     memoryTypeHost = cudaMemoryTypeHost,
        //     memoryTypeDevice = cudaMemoryTypeDevice,
        //     memoryTypeManaged = cudaMemoryTypeManaged
        // };

        namespace API
        {
            [[nodiscard]] __host__ inline deviceError_t getDeviceCount(int *count) noexcept
            {
                return cudaGetDeviceCount(count);
            }

            [[nodiscard]] __host__ inline deviceError_t getDeviceProperties(deviceProp_t *props, int deviceID) noexcept
            {
                return cudaGetDeviceProperties(props, deviceID);
            }

            [[nodiscard]] __host__ inline deviceError_t getDevice(int *deviceID) noexcept
            {
                return cudaGetDevice(deviceID);
            }

            [[nodiscard]] __host__ inline deviceError_t setDevice(const int deviceID) noexcept
            {
                return cudaSetDevice(deviceID);
            }

            [[nodiscard]] __host__ inline deviceError_t deviceSynchronize() noexcept
            {
                return cudaDeviceSynchronize();
            }

            [[nodiscard]] __host__ inline deviceError_t streamCreate(deviceStream_t *stream) noexcept
            {
                return cudaStreamCreate(stream);
            }

            [[nodiscard]] __host__ inline deviceError_t streamDestroy(deviceStream_t stream) noexcept
            {
                return cudaStreamDestroy(stream);
            }

            [[nodiscard]] __host__ inline deviceError_t streamSynchronize(deviceStream_t stream) noexcept
            {
                return cudaStreamSynchronize(stream);
            }

            template <typename T>
            [[nodiscard]] __host__ inline deviceError_t memcpy(T *const dst, const T *src, const size_t count, const deviceMemcpyType_t kind) noexcept
            {
                return cudaMemcpy(dst, src, count, kind);
            }

            template <typename T>
            [[nodiscard]] __host__ inline deviceError_t memcpyToSymbol(const T &symbol, const void *src, const size_t count, const size_t offset = 0, const deviceMemcpyKind kind = deviceMemcpyHostToDevice) noexcept
            {
                return cudaMemcpyToSymbol(symbol, src, count, offset, kind);
            }

            [[nodiscard]] __host__ inline deviceError_t memGetInfo(const size_t *free, const size_t *total) noexcept
            {
                return cudaMemGetInfo(free, total);
            }

            template <typename T>
            [[nodiscard]] __host__ inline deviceError_t memcpyAsync(T *dst, const T *src, const size_t count, const deviceMemcpyKind kind, const hipStream_t stream) noexcept
            {
                return cudaMemcpyAsync(dst, src, count, static_cast<cudaMemcpyKind>(kind), stream);
            }

            [[nodiscard]] __host__ inline deviceError_t pointerGetAttributes(devicePointerAttribute_t *attributes, const void *ptr) noexcept
            {
                return cudaPointerGetAttributes(attributes, ptr);
            }

            [[nodiscard]] __host__ inline constexpr const char *getErrorString(const deviceError_t error) noexcept
            {
                return cudaGetErrorString(error);
            }

            [[nodiscard]] __host__ inline deviceError_t funcSetCacheConfig(const void *func, const hipFuncCache_t config) noexcept
            {
                return cudaFuncSetCacheConfig(func, config);
            }

            [[nodiscard]] __host__ inline deviceError_t funcSetMaxDynamicSharedMemorySize(const void *func, const int bytes) noexcept
            {
                return cudaFuncSetAttribute(
                    func,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    bytes);
            }

            template <typename T>
            [[nodiscard]] __host__ inline deviceError_t hostFree(T *const ptr) noexcept
            {
                return cudaFreeHost(ptr);
            }

            template <typename T>
            [[nodiscard]] __host__ inline deviceError_t deviceFree(T *const ptr) noexcept
            {
                return cudaFree(ptr);
            }

            template <typename T>
            [[nodiscard]] __host__ inline deviceError_t deviceMalloc(T **ptr, const size_t size) noexcept
            {
                return cudaMalloc(ptr, size);
            }

            template <typename T>
            [[nodiscard]] __host__ inline deviceError_t mallocHost(T **ptr, const size_t size) noexcept
            {
                return cudaMallocHost(ptr, size, hipHostMallocDefault);
            }

            template <typename T>
            [[nodiscard]] __host__ inline deviceError_t memcpyPeerAsync(T *const dst, const int dstDeviceId, const T *const src, const int srcDeviceId, const size_t count, const deviceStream_t stream) noexcept
            {
                return cudaMemcpyPeerAsync(dst, dstDeviceId, src, srcDeviceId, count, stream);
            }
        }
    }
#endif
}

#endif