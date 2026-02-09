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