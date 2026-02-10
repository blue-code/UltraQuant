# 🚀 UltraQuant

> **ML Quant Trading - Ultra Edition**  
> 고성능 백테스팅 + 보안 + 멀티마켓 지원 퀀트 트레이딩 시스템

이제 진짜 최종 완전체입니다. 속도, 보안, 글로벌 확장성까지 모두 담았습니다.

---

## 📦 전체 설치 가이드

```bash
# ===== 기본 패키지 =====
pip install numpy pandas yfinance scikit-learn matplotlib seaborn plotly

# ===== 고성능 =====
pip install numba          # JIT 컴파일 (100배 속도 향상)
pip install ray            # 분산 처리

# ===== 보안 =====
pip install cryptography   # 암호화
pip install pyotp          # 2FA

# ===== 브로커 =====
pip install alpaca-trade-api   # Alpaca (미국 주식)
pip install ccxt                # 암호화폐 (100+ 거래소)
pip install FinanceDataReader   # 한국 주식 데이터

# ===== 대시보드 =====
pip install streamlit

# ===== 선택사항 =====
pip install ibapi              # IBKR
```

---

## 🏎️ 성능 및 리스크 분석 (UltraQuant Core)

| 구현 | 설명 | 향상률/효과 |
|------|------|------|
| Pure Python | 일반 루프 기반 백테스팅 | 1x |
| **Numba JIT** | JIT 컴파일 (루프 최적화) | 100x+ |
| **Monte Carlo** | 5,000회 시뮬레이션으로 VaR/CVaR 측정 | 리스크 관리 강화 |
| **Walk-Forward** | 전진 분석을 통한 파라미터 과적합 방지 | 실전 신뢰도 향상 |

---

## 🔐 보안 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                    보안 레이어                           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  [사용자 입력]                                          │
│       ↓                                                 │
│  ┌─────────────┐                                       │
│  │ 마스터 PW   │ ──→ PBKDF2 (480,000 iterations)       │
│  └─────────────┘           ↓                           │
│                       AES-256 Key                       │
│                            ↓                            │
│  ┌─────────────────────────────────────────┐           │
│  │         Fernet 암호화 저장소             │           │
│  │  • API Keys                             │           │
│  │  • Secrets                              │           │
│  │  • Config                               │           │
│  └─────────────────────────────────────────┘           │
│                            ↑                            │
│                       복호화                            │
│                            │                            │
│  ┌─────────────┐     ┌─────────────┐                   │
│  │  2FA TOTP   │ ──→ │   Access    │                   │
│  └─────────────┘     └─────────────┘                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 🌍 멀티 마켓 지원

| 마켓 | 브로커/API | 기능 |
|------|-----------|------|
| 🇺🇸 미국 주식 | Alpaca, IBKR | 실시간 체결, 마진, 숏 |
| 🇰🇷 한국 주식 | 키움증권 | 실시간 체결 |
| ₿ 암호화폐 | CCXT (100+ 거래소) | Spot, Futures, Margin |
| 💱 외환 | OANDA | 70+ 통화쌍 |
| 📈 선물 | IBKR | 지수, 원자재 |

---

## 🎯 사용 예시

### 고성능 백테스팅 (`ultra_quant.py`)

```python
backtester = FastBacktester(use_numba=True, use_ray=True)

# 파라미터 스윕 (81개 조합)
param_grid = {
    'lookback': [30, 50, 100],
    'sma_short': [10, 20, 30],
    'sma_long': [50, 100, 200],
    'target_vol': [0.10, 0.15, 0.20]
}

results = backtester.run_parameter_sweep(prices, param_grid)
```

### 보안 API 키 저장

```python
security = SecureConfigManager()
security.setup_encryption()  # 마스터 PW 입력
security.setup_2fa()         # Google Authenticator

security.store_api_key('binance', 'api_key_xxx', 'secret_xxx')
```

### 멀티 마켓 트레이딩

```python
broker = UniversalBroker()
broker.connect_us_stock(api_key, secret, paper=True)
broker.connect_crypto('binance', api_key, secret)

# 주문
order = Order(
    symbol='BTC/USDT',
    side='buy',
    quantity=0.01,
    market_type=MarketType.CRYPTO
)
broker.place_order(order)
```

---

## 🧪 전략 분석 스위트 (`strategy.py`)

12가지 이상의 클래식 및 최신 퀀트 전략을 제공하며, 백테스팅 및 파라미터 최적화 도구를 포함합니다.

### 📊 포함된 전략 목록

| # | 전략 | 유형 | 설명 |
|---|------|------|------|
| 1 | **Turtle Trading** | 추세추종 | 돈치안 채널 돌파 전략 (Richard Dennis) |
| 2 | **Momentum** | 모멘텀 | 6개월 수익률 + 장기 추세 필터 |
| 3 | **SuperTrend** | 추세추종 | ATR 기반 변동성 추세 지표 |
| 4 | **RSI 2** | 평균회귀 | 단기 과매수/과매도 + 장기 추세 필터 (Larry Connors) |
| 5 | **Bollinger Reversion** | 평균회귀 | 볼린저 밴드 역추세 매매 |
| 6 | **Williams %R** | 평균회귀 | 과매수/과매도 구간 활용 |
| 7 | **Dual Thrust** | 변동성 | 전일 고저차 기반 데이트레이딩 |
| 8 | **Volatility Breakout** | 변동성 | 변동성 돌파 전략 (Larry Williams) |
| 9 | **MA Cross** | 추세추종 | 이동평균선 골든/데드크로스 |
| 10 | **ML Ensemble** | 머신러닝 | RF Classifier 기반 방향성 확률 예측 |
| 11 | **Regime Switching** | 하이브리드 | 변동성 기반 추세/평균회귀 자동 전환 |
| 12 | **Liquidity Sweep** | 시장 구조 | 전일 고저점 이탈 후 복귀(SMC) 패턴 |

### 🔧 분석 및 최적화 도구 기능

*   **백테스팅 엔진**: 수수료, 슬리피지, 공매도 등을 고려한 정밀 시뮬레이션
*   **성과 지표**: Sharpe Ratio, Sortino Ratio, MDD, 승률, Profit Factor 등 10+ 지표 자동 계산
*   **시각화**: 자산 곡선(Equity Curve), 수익률 분포, 리스크-리턴 산점도, 승률 차트 등 제공
*   **Walk-Forward Optimizer**: 슬라이딩 윈도우 기반 전진 분석으로 과적합(Curve Fitting) 방지
*   **Genetic Algorithm (DE)**: 차분 진화 알고리즘을 통한 비선형 파라미터 공간의 초고속 최적화
*   **Monte Carlo Simulator**: 5,000회 이상의 경로 시뮬레이션으로 VaR/CVaR 리스크 측정

---

## 💡 UltraQuant 시너지 활용 가이드

본 시스템은 전략 개발(`strategy.py`)과 실행 엔진(`ultra_quant.py`)이 분리되어 시너지를 내도록 설계되었습니다.

### 🔄 통합 워크플로우: 연구에서 실전까지

1.  **전략 선정 (`strategy.py`)**: 12종의 내장 전략 중 하나(예: ML Ensemble)를 선택하거나 자신만의 `signal_func`를 정의합니다.
2.  **전략 최적화 (`strategy.py`)**: `WalkForwardOptimizer`를 사용하여 과거 데이터에서의 과적합을 방지하고, `DifferentialEvolutionOptimizer`로 최적의 파라미터 조합을 찾습니다.
3.  **초고속 검증 (`ultra_quant.py`)**: 최적화된 파라미터를 `FastBacktester`에 넣어 Numba/Ray 가속을 통해 수만 번의 시뮬레이션을 순식간에 완료합니다.
4.  **리스크 스트레스 테스트 (`ultra_quant.py`)**: `MonteCarloSimulator`를 통해 최악의 시장 상황(VaR, CVaR)에서도 계좌가 견딜 수 있는지 검증합니다.
5.  **보안 접속 및 실행 (`ultra_quant.py`)**: `SecureConfigManager`로 API 키를 안전하게 로드하고, `UniversalBroker`를 통해 멀티 마켓에 주문을 전송합니다.

### 🛠️ 결합 코드 예시

```python
from strategy import StrategySignals, WalkForwardOptimizer
from ultra_quant import FastBacktester, MonteCarloSimulator, SecureConfigManager, UniversalBroker

# 1. 전략 최적화 (Brain)
wfo = WalkForwardOptimizer(data)
best_params = wfo.run_wfa(StrategySignals.ml_ensemble_signals, param_grid)

# 2. 고성능 검증 (Engine)
fast_bt = FastBacktester(use_numba=True)
results = fast_bt.run_single_backtest(data['Close'].values, best_params)

# 3. 리스크 분석 (Shield)
mc = MonteCarloSimulator()
risk_stats = mc.analyze_risk(mc.run_simulation(results['daily_returns'], 100000))

# 4. 실전 투입 (Execution)
security = SecureConfigManager()
security.setup_encryption() # 마스터 PW 인증
broker = UniversalBroker()
broker.connect_us_stock(*security.get_api_key('alpaca'))
```

---

**ULTRA EDITION** 기능 요약:
- ⚡ Numba + Ray로 100~500배 속도 향상
- 🔐 AES-256 암호화 + 2FA 보안
- 🌍 주식, 암호화폐, 외환 통합 지원
- 🧪 12+ 퀀트 전략 및 정밀 분석 도구 포함
---

## 🚀 실행 가이드 (최신)

### 1) 가장 쉬운 실행: `run.bat` 하나로 시작

Windows에서는 아래처럼 실행하면 `venv` 생성/활성화, 의존성 설치 후 실행 모드를 선택할 수 있습니다.

```bat
run.bat
```

실행 중 메뉴:

- `1`: 콘솔 모드 (`ultra_quant.py`)
- `2`: GUI 모드 (`gui_bridge.py`)
- 엔터: 기본값 `1`

### 2) GUI로 전략 테스트 후 즉시 적용

GUI 모드(`2`)를 선택하면 `strategy.py`와 `ultra_quant.py`를 연결한 브리지 화면이 열립니다.

작업 순서:

1. 심볼 입력 (예: `SPY`, `AAPL`)
2. 기간 선택 (예: `1y`, `2y`)
3. 전략 선택 (`Turtle`, `RSI2`, `Momentum`, `ML Ensemble`, `Regime Switching`, `Liquidity Sweep`)
4. 전략 파라미터(JSON) 수정
5. `1) strategy.py로 테스트` 클릭
6. 결과 확인 후 `2) ultra_quant.py에 즉시 적용` 클릭

결과 창에 수익률/샤프비율/MDD/트레이드 수와 함께, `FastBacktester`에 적용된 매핑 파라미터가 출력됩니다.

### 3) 수동 실행 (선택)

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python ultra_quant.py
python gui_bridge.py
```

### 4) 테스트 실행

```bat
python -m unittest discover -s tests -p "test_*.py"
```

### 5) 자주 발생하는 문제

- `ModuleNotFoundError`: 가상환경 활성화 여부 확인 후 `pip install -r requirements.txt` 재실행
- GUI가 안 뜸: `python gui_bridge.py`를 직접 실행해 오류 메시지 확인
- 한글 깨짐: PowerShell에서 `chcp 65001` 실행 후 다시 확인

---

## GUI 상세 사용법 (전략 브리지 최신)

### 핵심 기능 요약
- `gui_bridge.py`는 `strategy.py` 전략 백테스트와 `ultra_quant.py` 적용을 한 화면에서 연결합니다.
- 단일 전략 테스트와 **전체 전략 일괄 테스트**를 모두 지원합니다.
- 심볼은 텍스트 입력이 아니라 **셀렉트박스**에서 선택합니다.

### 실행 방법
```bat
run.bat
```
- 메뉴에서 `2`를 선택하면 GUI가 실행됩니다.

또는 수동 실행:
```bat
venv\Scripts\activate
python gui_bridge.py
```

### 화면 사용 순서
1. 심볼 선택
2. 기간 선택 (`1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `max`)
3. 전략 선택
4. 필요 시 JSON 파라미터 수정 + `Rolling 검증(1년 창/3개월 스텝)` 체크 여부 선택
5. 아래 중 하나 실행
   - `1) strategy.py로 테스트`: 선택한 전략만 실행
   - `1-A) 전체 전략 일괄 테스트`: 등록된 모든 전략 실행 후 최종 점수(최근+Rolling 반영) 순 정렬
   - `1-B) 전체 전략×기간(1mo~5y) 일괄 테스트`: 기간별 전체 전략 실행 후 최종 점수(최근+Rolling 반영) 순 정렬
6. `2) ultra_quant.py에 즉시 적용`: 마지막 테스트 대상(일괄 테스트 시 최고 최종점수 전략)을 바로 적용

### symbols.json으로 심볼 목록 관리
GUI 심볼 목록은 루트의 `symbols.json`을 우선 사용합니다.

예시:
```json
{
  "symbols": ["SPY", "QQQ", "AAPL", "BTC-USD", "EURUSD=X"]
}
```

- 파일이 없거나 형식이 잘못되면 내부 기본 목록으로 자동 폴백됩니다.
- 심볼을 바꾸면 불러오는 OHLC 데이터가 달라지므로, 백테스트 결과도 함께 달라집니다.

### SPY 의미
- `SPY`는 **SPDR S&P 500 ETF** 티커입니다.
- 미국 대형주 지수(S&P 500)를 추종하므로, 전략 검증의 기본 벤치마크 심볼로 자주 사용됩니다.

---

## 신규 전략 추가 가이드 (StrategySignals)

`strategy.py`의 `StrategySignals`에 다음 전략이 추가되었습니다.

### 1) Turtle + Momentum Confirm
- 함수: `turtle_momentum_confirm_signals(params)`
- 목적: 단순 돌파의 휩쏘를 줄이기 위해, 채널 돌파와 중기 모멘텀 동시 확인
- 주요 파라미터:
  - `entry_period` (기본 20)
  - `exit_period` (기본 10)
  - `momentum_lookback` (기본 60)
  - `momentum_threshold` (기본 0.01)

### 2) RSI2 + Bollinger Reversion
- 함수: `rsi2_bollinger_reversion_signals(params)`
- 목적: RSI2 단독 신호의 과진입을 줄이기 위해 볼린저 밴드 이탈 조건을 추가
- 주요 파라미터:
  - `rsi_period` (기본 2)
  - `oversold` / `overbought` (기본 8 / 92)
  - `bb_window` / `bb_std` (기본 20 / 2.0)
  - `trend_period` (기본 100)

### 3) Regime + Liquidity Sweep
- 함수: `regime_liquidity_sweep_signals(params)`
- 목적: 유동성 스윕 신호를 변동성 레짐 필터와 결합해 가짜 신호를 완화
- 주요 파라미터:
  - `vol_lookback` (기본 20)
  - `regime_threshold` (기본 1.2)
  - `sweep_lookback` (기본 20)
  - `confirm_momentum` (기본 10)

### 4) Adaptive Fractal Regime (완전 신규)
- 함수: `adaptive_fractal_regime_signals(params)`
- 목적: 장세를 자동 분류해
  - 추세장: Donchian 돌파 추종
  - 횡보장: z-score 평균회귀
  로 분기하고, ATR/실현변동성으로 신호 강도를 자동 조절
- 주요 파라미터:
  - `trend_lookback` (기본 55)
  - `mean_window` (기본 20)
  - `z_entry` (기본 1.6)
  - `chop_window` / `chop_threshold` (기본 14 / 58)
  - `atr_period` (기본 14)
  - `target_daily_vol` (기본 0.012)

### 간단 실행 예시
```python
from strategy import StrategyBacktester, StrategySignals

sig = StrategySignals.adaptive_fractal_regime_signals({
    'trend_lookback': 55,
    'mean_window': 20,
    'z_entry': 1.6,
    'chop_window': 14,
    'chop_threshold': 58,
})

bt = StrategyBacktester()
result = bt.run_backtest(df, sig, strategy_name='adaptive_fractal_regime')
print(result.metrics)
```

---

## 백테스트 엔진 보완 사항 (최신)

### 1) 체결/비용 모델 고도화
`StrategyBacktester`에 거래 비용 모델이 확장되었습니다.

- 지원 항목:
  - `open_cost`, `close_cost`
  - `min_cost`
  - `impact_cost`
  - `trade_unit` (구조 확장)
  - `volume_limit_ratio` (일별 거래량 기반 체결 상한)
- 사용 예시:
```python
bt = StrategyBacktester(
    open_cost=0.001,
    close_cost=0.0015,
    min_cost=1.0,
    impact_cost=0.0002,
    volume_limit_ratio=0.1,
)
```

### 2) 벤치마크/초과수익 리포트 추가
`run_backtest()` 결과에 `report` DataFrame이 포함됩니다.

- 주요 컬럼:
  - `return`, `cost`, `bench`, `turnover`
  - `excess_return_wo_cost`, `excess_return_w_cost`
- `metrics`에 아래 항목이 함께 추가됩니다.
  - `Excess Return (wo Cost)`
  - `Excess Return (w Cost)`
  - `Risk (Excess wo Cost)`
  - `Risk (Excess w Cost)`

### 3) Rolling OOS 자동화
`RollingOOSRunner`가 추가되어 롤링 학습/검증을 자동 실행할 수 있습니다.

```python
runner = RollingOOSRunner(df)
out = runner.run(
    signal_factory=StrategySignals.momentum_signals,
    param_grid={'lookback': [40, 60], 'sma_filter': [120, 160]},
    train_size=350,
    test_size=80,
    step_size=40,
)
```

- 반환값:
  - `windows`: 윈도우별 OOS 결과
  - `report`: 통합 리포트
  - `aggregate`: 평균 OOS 성능 및 리스크 요약

### 4) 데이터 헬스체크 추가
`DataHealthCheckerLite`를 통해 백테스트 전 데이터 무결성을 검사합니다.

- 체크 항목:
  - 필수 컬럼(OHLC) 누락
  - 인덱스 정렬/중복
  - 결측치
  - 비정상 급등락(기본 임계치)
- 기본 동작: `StrategyBacktester` 실행 시 자동 검증 (`run_health_check=True`)

### Windows 인코딩 호환성
`ultra_quant.py`의 경고 출력 문구를 ASCII 형태(`[WARN] ...`)로 정리했습니다.

- 목적: CP949 콘솔 환경에서 import 시 발생하던 UnicodeEncodeError 방지
- 기존 실행/사용법은 동일합니다.

---

## 전략 목록 및 GUI 기능 (현재 기준)

### 현재 지원 전략 수
현재 `StrategySignals` + GUI 연동 기준으로 총 **20개 전략**을 지원합니다.

### 전략 목록 (20)
1. Turtle
2. RSI2
3. Momentum
4. SuperTrend
5. Bollinger Reversion
6. Williams %R
7. Dual Thrust
8. Volatility Breakout
9. MA Cross
10. ML Ensemble
11. Regime Switching
12. Liquidity Sweep
13. Adaptive EMA+ADX
14. ATR Breakout VolTarget
15. Z-Score Mean Reversion
16. MACD Regime
17. Turtle + Momentum Confirm
18. RSI2 + Bollinger Reversion
19. Regime + Liquidity Sweep
20. Adaptive Fractal Regime

### GUI 일괄 테스트 모드
GUI(`gui_bridge.py`)의 실행 버튼은 아래 3가지를 지원합니다.

- `1) strategy.py로 테스트`
  - 선택 전략 1개만 실행
- `1-A) 전체 전략 일괄 테스트`
  - 선택한 단일 기간에서 전체 전략 실행
  - 최근 구간(3mo, 6mo) 재평가 점수와 Rolling 검증을 함께 반영
  - 상위 10개 결과를 표시
- `1-B) 전체 전략×기간(1mo~5y) 일괄 테스트`
  - 기간 `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y` 각각에 대해 전체 전략 실행
  - 전체 결과를 최종 점수 기준으로 정렬(최근 가중 + Rolling 포함)
  - 상위 20개 결과를 표시
  - 최고 최종점수 결과를 `2) ultra_quant.py에 즉시 적용` 대상으로 자동 설정

### 점수 체계 (정렬 기준)
GUI의 전체 전략 정렬은 Sharpe 단일 지표가 아니라 아래 3단계 점수를 사용합니다.

1. 기본 복합점수(Base)
   - Return: `tanh` 포화 적용(과도한 고수익 단일 케이스 영향 완화)
   - MDD: 0~20% 구간 제곱 페널티 + 20% 초과 구간 로그 페널티
   - `Base = 0.6*Sharpe + 0.4*ReturnEffect - 0.3*MDDPenalty`
2. 최근 가중점수(Recent Blend)
   - 최근 `3mo`, `6mo` 평균 점수를 사용
   - `Blend = 0.4*Base + 0.6*Recent`
   - 최근 게이트: `recent_score >= 0`, `recent_trades >= 1`
3. Rolling 일관성 보정(Final)
   - 옵션: `Rolling 검증(1년 창/3개월 스텝)` 체크박스 (기본 ON)
   - Rolling 게이트: 최소 윈도우 수/승률 기준 충족 시 통과
   - `Final = 0.75*Blend + 0.25*RollingScore` (Rolling ON일 때)

### 기간 선택
GUI 기간 콤보박스는 현재 아래 값을 지원합니다.

- `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `max`
