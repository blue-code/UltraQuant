# 실행 가이드

## 1. 기본 실행
```bat
python ultra_quant.py
```

## 2. 전략 분석 실행
```bat
python strategy.py
```

## 3. GUI 실행
```bat
python gui_bridge.py
```

## 4. 배치 실행
- Windows: `run.bat`
- Unix: `bash run.sh`

## 5. 테스트
```bat
python -m unittest discover -s tests -p "test_*.py"
```

## 실행 시나리오
1. 로직 검증: `python strategy.py`
2. 엔진 검증: `python ultra_quant.py`
3. UI 검증: `python gui_bridge.py`
4. 전체 테스트: `python -m unittest discover -s tests -p "test_*.py"`

## 운영 체크포인트
- 기본 실행은 포트를 사용하지 않는 로컬 스크립트 형태
- Streamlit 확장 시 기본 포트 `8501` 충돌 확인 필요
- 장시간 실행 전 동일 심볼/동일 계정 중복 프로세스 확인
- 예외 발생 시 입력 데이터 기간, 결측치, 거래 비용 설정부터 점검
