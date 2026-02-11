# 전략 가이드

## 핵심 구성
- `StrategySignals`: 전략별 시그널 생성
- `StrategyBacktester`: 전략 백테스트 실행
- `WalkForwardOptimizer`: 전진 분석 기반 파라미터 검증
- `DifferentialEvolutionOptimizer`: 파라미터 탐색 최적화
- `RollingOOSRunner`: 구간별 OOS 자동 평가

## 대표 전략군
- 추세추종: Turtle, MA Cross, SuperTrend
- 모멘텀: Momentum, Turtle + Momentum Confirm
- 평균회귀: RSI2, Bollinger Reversion, RSI2 + Bollinger Reversion
- 변동성/시장구조: Volatility Breakout, Dual Thrust, Liquidity Sweep
- 하이브리드/적응형: Regime Switching, Regime + Liquidity Sweep, Adaptive Fractal Regime
- ML 계열: ML Ensemble

## 자주 쓰는 전략 기본 파라미터
| 전략 | 주요 파라미터 | 권장 시작값 |
|---|---|---|
| Momentum | `lookback`, `sma_filter` | `60`, `120` |
| Turtle | `entry_period`, `exit_period` | `20`, `10` |
| RSI2 | `rsi_period`, `oversold`, `overbought` | `2`, `8`, `92` |
| SuperTrend | `atr_period`, `multiplier` | `10`, `3.0` |
| Volatility Breakout | `k`, `lookback` | `0.5`, `20` |

## 예시: 단일 전략 테스트
```python
from strategy import StrategyBacktester, StrategySignals

sig = StrategySignals.momentum_signals({'lookback': 60, 'sma_filter': 120})
bt = StrategyBacktester()
result = bt.run_backtest(df, sig, strategy_name='momentum')
print(result.metrics)
```

## 예시: 롤링 OOS 검증
```python
from strategy import RollingOOSRunner, StrategySignals

runner = RollingOOSRunner(df)
out = runner.run(
    signal_factory=StrategySignals.momentum_signals,
    param_grid={'lookback': [40, 60], 'sma_filter': [100, 120]},
    train_size=350,
    test_size=80,
    step_size=40,
)
print(out['aggregate'])
```

## 검증 포인트
- 수익률 단일 지표 대신 Sharpe, MDD, 거래 횟수 동시 확인
- OOS 구간 성능과 최근 구간(3~6개월) 성능 분리 확인
- 수수료/슬리피지 반영 여부를 고정 조건으로 관리
- 거래 횟수 과소 전략(예: 전체 기간 5회 이하)은 재검증 후보로 분류
