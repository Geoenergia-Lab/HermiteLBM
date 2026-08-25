@echo off
:: --------------------------------------------------------------------------- ::
:: make.bat - Build script for fieldCalculate                                  ::
:: --------------------------------------------------------------------------- ::

:: 1. Load root variables
call "%HERMITELBM_PROJECT_DIR%\common.bat"

:: 2. Set application specific variables
set "EXECUTABLE=fieldCalculate.exe"
set "SOURCE=fieldCalculate.cu"

:: 3. Call the shared application targets script and pass along any arguments
call "%HERMITELBM_PROJECT_DIR%\applications\common.bat" %*