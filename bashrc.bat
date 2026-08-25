@echo off
:: --------------------------------------------------------------------------- ::
::                                                                             ::
::  HermiteLBM: CUDA-based moment representation Lattice Boltzmann Method      ::
::  Developed at UDESC - State University of Santa Catarina                    ::
::  Website: https://www.udesc.br                                              ::
::  Github: https://github.com/Geoenergia-Lab/HermiteLBM                       ::
::                                                                             ::
:: --------------------------------------------------------------------------- ::

:: Route doskey function calls to the appropriate label
if "%~1"=="cleanCase" goto :cleanCase
if "%~1"=="printHeader" goto :printHeader
if "%~1"=="printEnd" goto :printEnd
if "%~1"=="notFound" goto :notFound
if "%~1"=="skip_header_footer" goto :skip_header_footer
if "%~1"=="profileRoofline" goto :profileRoofline

:: --------------------------------------------------------------------------- ::
::  USER-DEFINED ENVIRONMENT VARIABLES                                         ::
:: --------------------------------------------------------------------------- ::

:: CUDA version (major and minor) - Note: your previous logs showed 13.3 on your system
set "HERMITELBM_CUDA_VERSION_MAJOR=13"
set "HERMITELBM_CUDA_VERSION_MINOR=0"

:: Architecture detection mode: "Automatic" or "Manual"
set "HERMITELBM_ARCHITECTURE_DETECTION=Automatic"

:: Manual architecture version (used only if HERMITELBM_ARCHITECTURE_DETECTION="Manual")
set "HERMITELBM_ARCHITECTURE_VERSION=89"

:: --------------------------------------------------------------------------- ::
::  AUTOMATIC SETUP - DO NOT MODIFY BELOW UNLESS YOU KNOW WHAT YOU ARE DOING   ::
:: --------------------------------------------------------------------------- ::

:: Project root directory (where this bat file lives)
set "HERMITELBM_PROJECT_DIR=%~dp0"
:: Remove trailing backslash
set "HERMITELBM_PROJECT_DIR=%HERMITELBM_PROJECT_DIR:~0,-1%"

:: Build tree structure
set "HERMITELBM_BUILD_DIR=%HERMITELBM_PROJECT_DIR%\build"
set "HERMITELBM_BIN_DIR=%HERMITELBM_BUILD_DIR%\bin"
set "HERMITELBM_INCLUDE_DIR=%HERMITELBM_BUILD_DIR%\include"

:: Distro detection is skipped on Windows. Assigning a static identifier.
set "HERMITELBM_DISTRO=windows"

:: --------------------------------------------------------------------------- ::
::  CUDA Toolkit                                                               ::
:: --------------------------------------------------------------------------- ::

set "HERMITELBM_CUDA_DIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v%HERMITELBM_CUDA_VERSION_MAJOR%.%HERMITELBM_CUDA_VERSION_MINOR%"

:: Fallback: Check if nvcc is available
where nvcc >nul 2>&1
if %errorlevel% neq 0 (
    if not exist "%HERMITELBM_CUDA_DIR%\bin\nvcc.exe" (
        echo Error: nvcc not found. Ensure CUDA is installed and in your PATH. 1>&2
        exit /b 1
    )
)

set "PATH=%HERMITELBM_CUDA_DIR%\bin;%PATH%"
:: LD_LIBRARY_PATH/LIBRARY_PATH translated to Windows MSVC LIB environment
set "LIB=%HERMITELBM_CUDA_DIR%\lib\x64;%LIB%"

:: --------------------------------------------------------------------------- ::
::  UCX (Unified Communication X)                                              ::
:: --------------------------------------------------------------------------- ::

set "HERMITELBM_UCX_DIR=%HERMITELBM_BUILD_DIR%\ucx"
set "PATH=%HERMITELBM_UCX_DIR%\bin;%PATH%"
set "LIB=%HERMITELBM_UCX_DIR%\lib;%LIB%"

:: --------------------------------------------------------------------------- ::
::  OpenMPI                                                                    ::
:: --------------------------------------------------------------------------- ::

set "HERMITELBM_MPI_DIR=%HERMITELBM_BUILD_DIR%\OpenMPI"
set "PATH=%HERMITELBM_MPI_DIR%\bin;%PATH%"
set "LIB=%HERMITELBM_MPI_DIR%\lib;%LIB%"
:: C_INCLUDE_PATH / CPLUS_INCLUDE_PATH translated to Windows MSVC INCLUDE environment
set "INCLUDE=%HERMITELBM_MPI_DIR%\include;%INCLUDE%"

:: --------------------------------------------------------------------------- ::
::  Add project executables to PATH                                            ::
:: --------------------------------------------------------------------------- ::

set "PATH=%HERMITELBM_BIN_DIR%;%PATH%"

:: --------------------------------------------------------------------------- ::
::  Utility Functions (Simulated via DOSKEY)                                   ::
:: --------------------------------------------------------------------------- ::

doskey cleanCase=call "%~f0" cleanCase
doskey printHeader=call "%~f0" printHeader
doskey printEnd=call "%~f0" printEnd
doskey notFound=call "%~f0" notFound $*
doskey profileRoofline=call "%~f0" profileRoofline $*

exit /b 0

:: =========================================================================== ::
:: FUNCTION IMPLEMENTATIONS                                                    ::
:: =========================================================================== ::

:cleanCase
if exist "programControl" (
    if exist "timeStep" rmdir /s /q "timeStep"
    if exist "postProcess" rmdir /s /q "postProcess"
    exit /b 0
) else (
    exit /b 1
)

:printHeader
echo /*---------------------------------------------------------------------------*\ 
echo ^|                                                                           ^|
echo ^| HermiteLBM: CUDA-based moment representation Lattice Boltzmann Method       ^|
echo ^| Developed at UDESC - State University of Santa Catarina                     ^|
echo ^| Website: https://www.udesc.br                                             ^|
echo ^| Github: https://github.com/Geoenergia-Lab/HermiteLBM                      ^|
echo ^|                                                                           ^|
echo \*---------------------------------------------------------------------------*/
echo.
exit /b 0

:printEnd
echo.
echo End
echo.
exit /b 0

:notFound
setlocal
set "ret=1"
:notFoundLoop
:: Shift drops the function name %1, looking at arguments %2 onward
if "%~2"=="" goto :notFoundDone
where "%~2" >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: '%~2' not found. Ensure environment is set up. 1>&2
    set "ret=0"
)
shift
goto :notFoundLoop
:notFoundDone
endlocal & exit /b %ret%

:skip_header_footer
if /I "%SKIP_HEADER_AND_FOOTER%"=="true" exit /b 0
if /I "%SKIP_HEADER_AND_FOOTER%"=="1" exit /b 0
if /I "%SKIP_HEADER_AND_FOOTER%"=="yes" exit /b 0
if /I "%SKIP_HEADER_AND_FOOTER%"=="on" exit /b 0
exit /b 1

:profileRoofline
setlocal
:: Map to a .bat script on Windows instead of .sh
set "script_path=%HERMITELBM_PROJECT_DIR%\roofline.bat"
if not exist "%script_path%" (
    echo ERROR: Profiling script not found at %script_path%
    exit /b 1
)
:: Call the external bat file passing arguments %2 through %9
call "%script_path%" %2 %3 %4 %5 %6 %7 %8 %9
endlocal
exit /b 0