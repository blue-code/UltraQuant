# 설치 가이드

## 1. 가상환경 생성
```bat
python -m venv .venv
.\.venv\Scripts\activate
```

## 2. 의존성 설치
```bat
pip install -r requirements.txt
```

## 3. 선택 패키지 설치
```bat
pip install numba ray cryptography pyotp streamlit
pip install alpaca-trade-api ccxt FinanceDataReader ibapi
```

## 4. 설치 확인
```bat
python -c "import numpy, pandas; print('ok')"
```

## 5. 환경 변수 확인
- `.env.example`를 기준으로 `.env`를 준비
- API 키/시크릿은 코드에 하드코딩하지 않고 `.env` 또는 런타임 주입 사용
- 실거래 전 `paper` 또는 모의 환경 기본값 확인

## 설치 체크리스트
- [ ] `python --version` 확인(프로젝트에서 사용하는 버전과 일치)
- [ ] `.venv` 활성화 후 `pip install -r requirements.txt` 완료
- [ ] `python ultra_quant.py` 최소 1회 실행 확인
- [ ] `python -m unittest discover -s tests -p "test_*.py"` 통과 확인

## 인코딩 주의
- 원인: Windows 콘솔 인코딩과 UTF-8 텍스트 인코딩 불일치
- 해결: `chcp 65001`
- 권장 설정: Git, 에디터, 터미널 모두 UTF-8 사용

## Git UTF-8 권장 설정
```bat
git config --global i18n.commitEncoding utf-8
git config --global i18n.logOutputEncoding utf-8
git config --global core.quotepath false
```
