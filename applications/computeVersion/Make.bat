@echo off
:: --------------------------------------------------------------------------- ::
:: make.bat - Build script for computeVersion                                  ::
:: --------------------------------------------------------------------------- ::

:: 1. Load root variables
call "%HERMITELBM_PROJECT_DIR%\common.bat"

:: 2. Set application specific variables
set "EXECUTABLE=computeVersion.exe"
set "SOURCE=computeVersion.cu"

:: 3. Call the shared application targets script and pass along any arguments
call "%HERMITELBM_PROJECT_DIR%\applications\common.bat" %*