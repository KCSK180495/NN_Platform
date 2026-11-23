@echo off
setlocal
pushd "%~dp0" || exit /b 1
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -X utf8 main.py
) else (
    python -X utf8 main.py
)
set "app_exit=%errorlevel%"
if not "%app_exit%"=="0" (
    echo.
    echo NN Studio could not start. See START_HERE.txt for setup instructions.
    pause
)
popd
exit /b %app_exit%
