@echo off
:: --------------------------------------------------------------------------- ::
:: common.bat - Shared CUDA/C++ flags for Windows                              ::
:: --------------------------------------------------------------------------- ::

set "NVCXX=nvcc"

:: Include hardware info if translated to a batch script
if exist "%HERMITELBM_INCLUDE_DIR%\hardware.bat" (
    call "%HERMITELBM_INCLUDE_DIR%\hardware.bat"
)

:: CUDA Compiler Flags
set "NVCXX_STANDARD=-std c++20"
set "NVCXX_OPTFLAGS=-O3 --restrict -extra-device-vectorization --maxrregcount=128"
set "NVCXX_MFLAGS=--m64"
set "NVCXX_WFLAGS=--Werror cross-execution-space-call,reorder,default-stream-launch,missing-launch-bounds,ext-lambda-captures-this"

:: MSVC Host Compiler Flags (Replaces GCC flags)
:: /O2 = Max speed, /W4 = Warning level 4, /WX = Warnings as errors
set "NVCXX_COMPILER_WFLAGS=-Xcompiler /O2,/W4,/WX,/diagnostics:caret"

:: Defines
set "NVCXX_DFLAGS=-DSCALAR_PRECISION=32 -DLABEL_SIZE=32 -DUSE_SMEM_HALO"

:: Final combined flags
set "NVCXX_FLAGS=%NVCXX_STANDARD% %NVCXX_OPTFLAGS% %NVCXX_MFLAGS% %NVCXX_ALL_ARCHFLAGS% %NVCXX_WFLAGS% %NVCXX_COMPILER_WFLAGS% %NVCXX_DFLAGS% %HAS_MULTI_GPU%"