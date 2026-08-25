@echo off

:: --------------------------------------------------------------------------- ::
::  Route doskey function calls to the appropriate label                       ::
:: --------------------------------------------------------------------------- ::
if "%~1"=="cleanCase" goto :cleanCase
if "%~1"=="printHeader" goto :printHeader
if "%~1"=="printEnd" goto :printEnd
if "%~1"=="notFound" goto :notFound
if "%~1"=="skip_header_footer" goto :skip_header_footer
if "%~1"=="profileRoofline" goto :profileRoofline

:: --------------------------------------------------------------------------- ::
::  USER-DEFINED ENVIRONMENT VARIABLES                                         ::
:: --------------------------------------------------------------------------- ::

:: CUDA version (major and minor)
set "HERMITELBM_CUDA_VERSION_MAJOR=13"
set "HERMITELBM_CUDA_VERSION_MINOR=1"

:: Architecture detection mode: "Automatic" or "Manual"
set "HERMITELBM_ARCHITECTURE_DETECTION=Automatic"

:: Manual architecture version (used only if HERMITELBM_ARCHITECTURE_DETECTION="Manual")
set "HERMITELBM_ARCHITECTURE_VERSION=89"

:: --------------------------------------------------------------------------- ::
::  AUTOMATIC SETUP - DO NOT MODIFY BELOW UNLESS YOU KNOW WHAT YOU ARE DOING   ::
:: --------------------------------------------------------------------------- ::

call :printHeader

:: Project root directory (where this bat file lives)
set "HERMITELBM_PROJECT_DIR=%~dp0"
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

set "HERMITELBM_CUDA_FOUND=0"
set "CUDA_DIR_SUFFIX=%HERMITELBM_CUDA_VERSION_MAJOR%.%HERMITELBM_CUDA_VERSION_MINOR%"

:: 1) Try the default CUDA directory first
set "DEFAULT_CUDA_DIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_DIR_SUFFIX%"
if exist "%DEFAULT_CUDA_DIR%\bin\nvcc.exe" (
    set "HERMITELBM_CUDA_DIR=%DEFAULT_CUDA_DIR%"
    set "PATH=%HERMITELBM_CUDA_DIR%\bin;%PATH%"
    set "LIB=%HERMITELBM_CUDA_DIR%\lib\x64;%LIB%"
    set "HERMITELBM_CUDA_FOUND=1"
    goto :cuda_done
)

:: 2) If not in default location, fall back to whatever nvcc is in PATH
where nvcc >nul 2>&1
if %errorlevel% equ 0 (
    for /f "delims=" %%i in ('where nvcc') do (
        set "NVCC_PATH=%%i"
        goto :nvcc_found
    )
)

:nvcc_found
if defined NVCC_PATH (
    :: Derive CUDA directory: one level up from bin
    for %%F in ("%NVCC_PATH%") do set "NVCC_BIN_DIR=%%~dpF"
    set "HERMITELBM_CUDA_DIR=%NVCC_BIN_DIR%.."
    :: Normalize path (remove trailing backslash if any)
    if "%HERMITELBM_CUDA_DIR:~-1%"=="\" set "HERMITELBM_CUDA_DIR=%HERMITELBM_CUDA_DIR:~0,-1%"
    if exist "%HERMITELBM_CUDA_DIR%\bin\nvcc.exe" (
        set "PATH=%HERMITELBM_CUDA_DIR%\bin;%PATH%"
        set "LIB=%HERMITELBM_CUDA_DIR%\lib\x64;%LIB%"
        set "HERMITELBM_CUDA_FOUND=1"
    ) else (
        echo Warning: CUDA directory %HERMITELBM_CUDA_DIR% not found. Skipping CUDA paths. 1>&2
        set "HERMITELBM_CUDA_FOUND=0"
    )
) else (
    echo Warning: nvcc not found. CUDA environment will not be configured. 1>&2
    set "HERMITELBM_CUDA_FOUND=0"
)

:cuda_done

:: --------------------------------------------------------------------------- ::
::  Print detected environment                                                 ::
:: --------------------------------------------------------------------------- ::

echo HermiteLBM
echo {
if "%HERMITELBM_CUDA_FOUND%"=="1" (
    echo     CUDA version: %HERMITELBM_CUDA_VERSION_MAJOR%.%HERMITELBM_CUDA_VERSION_MINOR%
    echo     CUDA directory: %HERMITELBM_CUDA_DIR%
) else (
    echo     CUDA version: Not found
)
echo     Distro: %HERMITELBM_DISTRO%
echo     Architecture detection: %HERMITELBM_ARCHITECTURE_DETECTION%
echo     Project directory: %HERMITELBM_PROJECT_DIR%
echo };
echo.

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
:: C_INCLUDE_PATH / CPLUS_INCLUDE_PATH translated to Windows MSVC INCLUDE
set "INCLUDE=%HERMITELBM_MPI_DIR%\include;%INCLUDE%"

:: --------------------------------------------------------------------------- ::
::  Add project executables to PATH                                            ::
:: --------------------------------------------------------------------------- ::

set "PATH=%HERMITELBM_BIN_DIR%;%PATH%"

:: --------------------------------------------------------------------------- ::
::  Define DOSKEY aliases for utility functions                                ::
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
echo ^|                                                                             ^|
echo ^| HermiteLBM: CUDA-based moment representation Lattice Boltzmann Method       ^|
echo ^| Developed at UDESC - State University of Santa Catarina                     ^|
echo ^| Website: https://www.udesc.br                                               ^|
echo ^| Github: https://github.com/Geoenergia-Lab/HermiteLBM                        ^|
echo ^|                                                                             ^|
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
set "script_path=%HERMITELBM_PROJECT_DIR%\roofline.bat"
if not exist "%script_path%" (
    echo ERROR: Profiling script not found at %script_path%
    exit /b 1
)
:: Pass all arguments after the function name
shift
call "%script_path%" %*
endlocal
exit /b 0