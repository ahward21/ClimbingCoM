@echo off
title ClimbAnalytics - Self-Hosted Server

echo.
echo ============================================================
echo   ClimbAnalytics - Self-Hosted Web Server
echo   Center of Mass Trajectory Analysis
echo ============================================================
echo.

REM Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again.
    pause
    exit /b 1
)

REM Activate virtual environment if exists
if exist "venv_cuda\Scripts\activate.bat" (
    echo Activating CUDA virtual environment...
    call venv_cuda\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Warning: No virtual environment found, using system Python
)

REM Check for required packages
echo.
echo Checking dependencies...
python -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo Installing Flask...
    pip install flask flask-cors
)

REM Create required directories
if not exist "uploads" mkdir uploads
if not exist "outputs" mkdir outputs

echo.
echo ============================================================
echo   FEATURES:
echo   - Single Video Analysis
echo   - Multi-Camera 3D Reconstruction  
echo   - Batch Processing
echo   - Pose-Only (Anonymized) Mode
echo   - CoM Trajectory Visualization
echo   - Speed-Based Coloring
echo   - JSON/CSV/Video Export
echo ============================================================
echo.
echo   Starting server...
echo.
echo   Open your browser to: http://localhost:5000
echo.
echo   Press Ctrl+C to stop the server.
echo.

python server.py

echo.
echo Server stopped.
pause
