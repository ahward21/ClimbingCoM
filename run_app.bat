@echo off
echo ========================================
echo   Climbing Pose Analysis - Starting...
echo ========================================
echo.

REM Change to the script's directory
cd /d "%~dp0"

REM Check if virtual environment exists
if not exist "venv_cuda\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please create it first with: python -m venv venv_cuda
    pause
    exit /b 1
)

REM Activate virtual environment
call venv_cuda\Scripts\activate.bat

REM Run the Gradio app
echo Starting Gradio app...
echo.
python app.py

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo ERROR: Application exited with an error.
    pause
)
