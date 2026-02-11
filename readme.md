# UltraQuant

UltraQuant는 백테스트 엔진(`ultra_quant.py`)과 전략 연구 스위트(`strategy.py`)를 결합한 Python 기반 퀀트 트레이딩 프로젝트입니다.

## 문서 안내
- [설치 가이드](docs/INSTALL.md)
- [실행 가이드](docs/RUN_GUIDE.md)
- [전략 가이드](docs/STRATEGY_GUIDE.md)
- [GUI 가이드](docs/GUI_GUIDE.md)
- [아키텍처 개요](docs/ARCHITECTURE.md)

## 빠른 시작
```bat
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python ultra_quant.py
```

## 테스트
```bat
python -m unittest discover -s tests -p "test_*.py"
```

## 파일 구성
- `ultra_quant.py`: 백테스트/실행 엔진
- `strategy.py`: 전략 시그널, 최적화, 분석 도구
- `gui_bridge.py`: GUI 브리지
- `tests/test_core.py`: 핵심 테스트

## 인코딩 안내 (Windows 한글 깨짐)
- 원인: PowerShell 기본 코드페이지(CP949)와 UTF-8 문서 인코딩 불일치
- 해결: PowerShell에서 `chcp 65001` 실행 후 다시 열기
- 권장 설정: VS Code/터미널/깃 인코딩을 UTF-8로 통일
