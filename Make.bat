@echo off
:: --------------------------------------------------------------------------- ::
::                                                                             ::
::  HermiteLBM: CUDA-based moment representation Lattice Boltzmann Method      ::
::  Developed at UDESC - State University of Santa Catarina                    ::
::  Website: https://www.udesc.br                                              ::
::  Github: https://github.com/Geoenergia-Lab/HermiteLBM                       ::
::                                                                             ::
:: --------------------------------------------------------------------------- ::

:: --------------------------------------------------------------------------- ::
::  Top-level Build Script (make.bat)                                          ::
:: --------------------------------------------------------------------------- ::

:: 1. Check if required environment variables are set
if "%HERMITELBM_BUILD_DIR%"=="" (
    echo Error: HERMITELBM_BUILD_DIR is not set. Please run "bashrc" in the project directory first.
    exit /b 1
)
if "%HERMITELBM_BIN_DIR%"=="" (
    echo Error: HERMITELBM_BIN_DIR is not set. Please run "bashrc" in the project directory first.
    exit /b 1
)
if "%HERMITELBM_INCLUDE_DIR%"=="" (
    echo Error: HERMITELBM_INCLUDE_DIR is not set. Please run "bashrc" in the project directory first.
    exit /b 1
)
if "%HERMITELBM_CUDA_DIR%"=="" (
    echo Error: HERMITELBM_CUDA_DIR is not set. Please run "bashrc" in the project directory first.
    exit /b 1
)

:: 2. Define Subdirectories (using spaces to separate them)
set "TOOL_SUBDIRS=applications\computeVersion applications\postProcessing\fieldConvert applications\postProcessing\fieldCalculate"
set "GPU_SUBDIRS=applications\solvers\momentBasedD3Q19 applications\solvers\momentBasedD3Q27 applications\solvers\isothermalD3Q19 applications\solvers\isothermalD3Q27"

:: 3. Target Routing
set "TARGET=%~1"
if "%TARGET%"=="" set "TARGET=all"

if /I "%TARGET%"=="all" goto :all
if /I "%TARGET%"=="directories" goto :directories
if /I "%TARGET%"=="clean" goto :clean
if /I "%TARGET%"=="install" goto :install
if /I "%TARGET%"=="uninstall" goto :uninstall

echo Error: Unknown target '%TARGET%'
exit /b 1

:: =========================================================================== ::
:: TARGET IMPLEMENTATIONS                                                      ::
:: =========================================================================== ::

:directories
if not exist "%HERMITELBM_BUILD_DIR%" mkdir "%HERMITELBM_BUILD_DIR%"
if not exist "%HERMITELBM_BIN_DIR%" mkdir "%HERMITELBM_BIN_DIR%"
if not exist "%HERMITELBM_INCLUDE_DIR%" mkdir "%HERMITELBM_INCLUDE_DIR%"
exit /b 0


:hardware_info
call :directories
echo Building computeVersion to generate hardware.info...
pushd "applications\computeVersion"
call make.bat install
popd
:: Run computeVersion.exe to generate hardware.info
computeVersion.exe
exit /b %errorlevel%


:all
call :directories

:: Compile tool subdirectories
for %%D in (%TOOL_SUBDIRS%) do (
    echo.
    echo Entering %%D
    pushd "%%D"
    call make.bat
    popd
)

:: Generate hardware.info before building GPU solvers
call :hardware_info

:: Compile GPU applications
for %%D in (%GPU_SUBDIRS%) do (
    echo.
    echo Entering %%D
    pushd "%%D"
    call make.bat
    popd
)
exit /b 0


:install
call :directories
call :hardware_info

for %%D in (%TOOL_SUBDIRS% %GPU_SUBDIRS%) do (
    echo.
    echo Installing %%D
    pushd "%%D"
    call make.bat install
    popd
)
exit /b 0


:clean
for %%D in (%TOOL_SUBDIRS% %GPU_SUBDIRS%) do (
    echo.
    echo Cleaning %%D
    pushd "%%D"
    call make.bat clean
    popd
)
if exist "%HERMITELBM_BUILD_DIR%" rmdir /s /q "%HERMITELBM_BUILD_DIR%"
exit /b 0


:uninstall
for %%D in (%TOOL_SUBDIRS% %GPU_SUBDIRS%) do (
    echo.
    echo Uninstalling %%D
    pushd "%%D"
    call make.bat uninstall
    popd
)
if exist "%HERMITELBM_BIN_DIR%" rmdir /s /q "%HERMITELBM_BIN_DIR%"
if exist "%HERMITELBM_INCLUDE_DIR%" rmdir /s /q "%HERMITELBM_INCLUDE_DIR%"
exit /b 0