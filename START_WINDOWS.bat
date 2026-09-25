@echo off
setlocal
cd /d "%~dp0"

echo.
echo ==========================================
echo   XARM VISION - CLASSROOM STARTUP
echo ==========================================
echo.

REM --------------------------------------------------
REM Find Python 3.11
REM --------------------------------------------------

set "PYTHON_CMD="

py -3.11 --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=py -3.11"
    goto python_found
)

python --version >nul 2>&1
if not errorlevel 1 (
    python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3,11) else 1)" >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_CMD=python"
        goto python_found
    )
)

echo ERROR: Python 3.11 was not found.
echo.
echo Please install Python 3.11 first.
echo.
echo IMPORTANT:
echo During Python installation, select:
echo     Add Python to PATH
echo.
pause
exit /b 1


:python_found

echo Python 3.11 found.
%PYTHON_CMD% --version
echo.

REM --------------------------------------------------
REM Create virtual environment if needed
REM --------------------------------------------------

if not exist ".venv\Scripts\python.exe" (
    echo Creating classroom Python environment...
    %PYTHON_CMD% -m venv .venv

    if errorlevel 1 (
        echo.
        echo ERROR: Could not create the Python environment.
        pause
        exit /b 1
    )
)

echo Python environment ready.
echo.

REM --------------------------------------------------
REM Install / verify required packages
REM --------------------------------------------------

echo Checking required Python packages...
".venv\Scripts\python.exe" -m pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Package installation failed.
    echo Check the Internet connection and try again.
    pause
    exit /b 1
)

echo.
echo Required packages are ready.
echo.

REM --------------------------------------------------
REM Start student setup menu
REM --------------------------------------------------

echo Starting XArm Vision Student Setup...
echo.

".venv\Scripts\python.exe" student_setup.py

if errorlevel 1 (
    echo.
    echo The setup program ended with an error.
    pause
)

endlocal