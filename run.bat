@echo off
SETLOCAL EnableDelayedExpansion

echo [UltraQuant] Starting setup and execution...

:: 1. 가상환경 확인 및 생성
if not exist "venv" (
    echo [UltraQuant] Creating virtual environment...
    python -m venv venv
    if !errorlevel! neq 0 (
        echo [Error] Python is not installed or not in PATH.
        pause
        exit /b 1
    )
)

:: 2. 가상환경 활성화 및 패키지 설치
echo [UltraQuant] Activating virtual environment...
call venv\Scripts\activate

echo [UltraQuant] Checking/Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

:: 3. 시스템 실행 (기본적으로 ultra_quant.py 실행)
echo [UltraQuant] Running UltraQuant...
python ultra_quant.py

echo [UltraQuant] Execution finished.
pause
