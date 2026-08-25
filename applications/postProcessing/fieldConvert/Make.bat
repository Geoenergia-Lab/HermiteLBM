@echo off
:: --------------------------------------------------------------------------- ::
:: make.bat - Build script for fieldConvert                                    ::
:: --------------------------------------------------------------------------- ::

:: 1. Load root variables
call "%HERMITELBM_PROJECT_DIR%\common.bat"

:: 2. Set application specific variables
set "EXECUTABLE=fieldConvert.exe"
set "SOURCE=fieldConvert.cu"

:: 3. Call the shared application targets script and pass along any arguments
call "%HERMITELBM_PROJECT_DIR%\applications\common.bat" %*