#!/bin/bash

echo "[UltraQuant] Starting setup and execution..."

# 1. 가상환경 확인 및 생성
if [ ! -d "venv" ]; then
    echo "[UltraQuant] Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "[Error] Python3 is not installed or not in PATH."
        exit 1
    fi
fi

# 2. 가상환경 활성화 및 패키지 설치
echo "[UltraQuant] Activating virtual environment..."
source venv/bin/activate

echo "[UltraQuant] Checking/Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 3. 시스템 실행
echo "[UltraQuant] Running UltraQuant..."
python3 ultra_quant.py

echo "[UltraQuant] Execution finished."
