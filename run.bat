@echo off
SETLOCAL EnableDelayedExpansion

echo [UltraQuant] Starting setup and execution...

:: 1) venv check/create
if not exist "venv" (
    echo [UltraQuant] Creating virtual environment...
    python -m venv venv
    if !errorlevel! neq 0 (
        echo [Error] Python is not installed or not in PATH.
        pause
        exit /b 1
    )
)

:: 2) activate venv and install dependencies
echo [UltraQuant] Activating virtual environment...
call venv\Scripts\activate

echo [UltraQuant] Checking/Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

:: yfinance fallback
pip show yfinance >nul 2>&1
if !errorlevel! neq 0 (
    echo [UltraQuant] Installing yfinance...
    pip install yfinance
)

:: 3) select run mode
echo.
echo [UltraQuant] Select run mode:
echo   1. Console mode ^(ultra_quant.py^)
echo   2. GUI mode ^(gui_bridge.py^)
set "RUN_MODE="
set /p RUN_MODE=Enter 1 or 2 [default: 1]: 
if "!RUN_MODE!"=="" set "RUN_MODE=1"

if "!RUN_MODE!"=="2" (
    echo [UltraQuant] Running GUI...
    python gui_bridge.py
) else (
    if not "!RUN_MODE!"=="1" (
        echo [UltraQuant] Invalid selection. Running default console mode.
    )
    echo [UltraQuant] Running UltraQuant...
    python ultra_quant.py
)

echo [UltraQuant] Execution finished.
pause
