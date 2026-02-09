세 가지 다 드립니다!

---

## 🚀 Ultimate Strategy Analysis Suite

```python
# ============================================
# STRATEGY ANALYSIS SUITE
# 1. 백테스팅 성과 비교
# 2. 파라미터 최적화
# 3. 새로운 전략 추가
# ============================================

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from itertools import product
import time
import warnings
warnings.filterwarnings('ignore')

# 통계 & 최적화
from scipy import stats
from scipy.optimize import minimize, differential_evolution
import statsmodels.api as sm

# 시각화
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.patches import Rectangle

# ============================================
# 공통 유틸리티
# ============================================

@dataclass
class BacktestResult:
    """백테스팅 결과"""
    strategy_name: str
    equity_curve: pd.Series
    daily_returns: pd.Series
    metrics: Dict
    trades: List = field(default_factory=list)
    
    def __repr__(self):
        return (f"BacktestResult({self.strategy_name}: "
                f"Return={self.metrics.get('Total Return', 0):.1%}, "
                f"Sharpe={self.metrics.get('Sharpe Ratio', 0):.2f}, "
                f"MDD={self.metrics.get('Max Drawdown', 0):.1%})")


class PerformanceMetrics:
    """성과 지표 계산"""
    
    @staticmethod
    def calculate_all(returns: pd.Series, equity: pd.Series, 
                      risk_free_rate: float = 0.04) -> Dict:
        """모든 성과 지표 계산"""
        returns = returns.dropna()
        
        # 기본 지표
        total_return = (equity.iloc[-1] / equity.iloc[0] - 1)
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        
        # 변동성
        ann_vol = returns.std() * np.sqrt(252)
        
        # 샤프 비율
        sharpe = (annual_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0
        
        # 소르티노 비율
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252)
        sortino = (annual_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0
        
        # MDD
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        max_dd = drawdown.min()
        
        # 칼마 비율
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
        
        # 승률
        win_rate = (returns > 0).sum() / len(returns)
        
        # 평균 수익/손실
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = returns[returns < 0].mean() if (returns < 0).any() else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # VaR & CVaR
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean()
        
        # 최장 연속 손실 기간
        loss_streak = (returns < 0).astype(int)
        streak_groups = (loss_streak != loss_streak.shift()).cumsum()
        max_loss_streak = loss_streak.groupby(streak_groups).sum().max()
        
        # VWR (Volatility-Weighted Return)
        vol_weighted = returns / (returns.rolling(21).std() + 1e-8)
        vwr = vol_weighted.mean() * np.sqrt(252)
        
        return {
            'Total Return': total_return,
            'Annual Return': annual_return,
            'Annual Volatility': ann_vol,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Max Drawdown': max_dd,
            'Calmar Ratio': calmar,
            'Win Rate': win_rate,
            'Profit Factor': profit_factor,
            'VaR 95%': var_95,
            'CVaR 95%': cvar_95,
            'Max Loss Streak': max_loss_streak,
            'VWR': vwr
        }


# ============================================
# 1. 전략 백테스팅 프레임워크
# ============================================

class StrategyBacktester:
    """
    통합 백테스팅 프레임워크
    
    - 이벤트 기반 백테스팅
    - 실제 거래 비용 반영
    - 슬리피지 모델
    """
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 commission: float = 0.001,  # 0.1%
                 slippage: float = 0.0005,   # 0.05%
                 position_size: float = 0.95):  # 95% 투자
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_size = position_size
    
    def run_backtest(self, 
                     data: pd.DataFrame,
                     signal_func: Callable,
                     strategy_name: str = "Strategy") -> BacktestResult:
        """
        백테스팅 실행
        
        signal_func: (data, current_idx) -> signal (-1 ~ 1)
        """
        
        equity = [self.initial_capital]
        returns = []
        trades = []
        position = 0
        entry_price = 0
        
        prices = data['Close'].values
        dates = data.index
        
        lookback = min(50, len(data) - 1)
        
        for i in range(lookback, len(data)):
            # 신호 생성
            signal = signal_func(data.iloc[:i+1], i)
            
            current_price = prices[i]
            prev_price = prices[i-1]
            
            # 포지션 없음
            if position == 0:
                if signal > 0.3:  # 매수
                    position = 1
                    entry_price = current_price * (1 + self.slippage)
                    trade_value = equity[-1] * self.position_size
                    commission_paid = trade_value * self.commission
                    shares = trade_value / entry_price
                    
                    trades.append({
                        'date': dates[i],
                        'type': 'BUY',
                        'price': entry_price,
                        'shares': shares,
                        'commission': commission_paid
                    })
                    
                elif signal < -0.3:  # 공매도
                    position = -1
                    entry_price = current_price * (1 - self.slippage)
                    trade_value = equity[-1] * self.position_size
                    commission_paid = trade_value * self.commission
                    
                    trades.append({
                        'date': dates[i],
                        'type': 'SELL SHORT',
                        'price': entry_price,
                        'commission': commission_paid
                    })
            
            # 롱 포지션
            elif position == 1:
                if signal < -0.1:  # 청산
                    exit_price = current_price * (1 - self.slippage)
                    pnl = (exit_price - entry_price) / entry_price
                    equity.append(equity[-1] * (1 + pnl) * (1 - self.commission))
                    
                    trades.append({
                        'date': dates[i],
                        'type': 'SELL',
                        'price': exit_price,
                        'pnl': pnl
                    })
                    
                    position = 0
            
            # 숏 포지션
            elif position == -1:
                if signal > 0.1:  # 커버
                    exit_price = current_price * (1 + self.slippage)
                    pnl = (entry_price - exit_price) / entry_price
                    equity.append(equity[-1] * (1 + pnl) * (1 - self.commission))
                    
                    trades.append({
                        'date': dates[i],
                        'type': 'COVER',
                        'price': exit_price,
                        'pnl': pnl
                    })
                    
                    position = 0
            
            # 일일 수익률 계산
            if position == 1:
                daily_return = (current_price - prev_price) / prev_price
            elif position == -1:
                daily_return = (prev_price - current_price) / prev_price
            else:
                daily_return = 0
            
            returns.append(daily_return)
            
            # 현금 포지션 시 현재 자본
            if position == 0 and len(equity) == len(returns):
                pass
            elif position != 0:
                if len(equity) < len(returns) + 1:
                    equity.append(equity[-1] * (1 + daily_return))
        
        # 결과 정리
        equity = pd.Series(equity[:len(returns)], index=dates[lookback:])
        returns = pd.Series(returns, index=dates[lookback:])
        
        metrics = PerformanceMetrics.calculate_all(returns, equity)
        metrics['Total Trades'] = len([t for t in trades if 'pnl' in t])
        
        return BacktestResult(
            strategy_name=strategy_name,
            equity_curve=equity,
            daily_returns=returns,
            metrics=metrics,
            trades=trades
        )


# ============================================
# 2. 전략 구현 (시그널 함수)
# ============================================

class StrategySignals:
    """각 전략의 신호 생성 함수"""
    
    @staticmethod
    def turtle_signals(params: dict) -> Callable:
        """터틀 트레이딩 신호"""
        entry_period = params.get('entry_period', 20)
        exit_period = params.get('exit_period', 10)
        
        def signal_func(data, idx):
            if idx < entry_period:
                return 0
            
            high_channel = data['High'].iloc[-entry_period:-1].max()
            low_channel = data['Low'].iloc[-entry_period:-1].max()
            exit_low = data['Low'].iloc[-exit_period:-1].min()
            
            current = data['Close'].iloc[-1]
            
            if current >= high_channel:
                return 1
            elif current <= exit_low:
                return 0
            else:
                return 0.5  # Hold
        
        return signal_func
    
    @staticmethod
    def rsi2_signals(params: dict) -> Callable:
        """RSI 2 신호"""
        sma_period = params.get('sma_period', 200)
        oversold = params.get('oversold', 10)
        overbought = params.get('overbought', 90)
        
        def signal_func(data, idx):
            if idx < sma_period:
                return 0
            
            prices = data['Close']
            
            # RSI 2
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(2).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(2).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # SMA 200
            sma = prices.rolling(sma_period).mean()
            
            current_rsi = rsi.iloc[-1]
            uptrend = prices.iloc[-1] > sma.iloc[-1]
            
            if uptrend and current_rsi < oversold:
                return 1
            elif uptrend and current_rsi > overbought:
                return 0
            elif not uptrend and current_rsi > overbought:
                return -1
            else:
                return 0
        
        return signal_func
    
    @staticmethod
    def bollinger_reversion_signals(params: dict) -> Callable:
        """볼린저 평균회귀 신호"""
        period = params.get('period', 20)
        std_dev = params.get('std_dev', 2.0)
        
        def signal_func(data, idx):
            if idx < period:
                return 0
            
            prices = data['Close']
            sma = prices.rolling(period).mean()
            std = prices.rolling(period).std()
            
            upper = sma + std_dev * std
            lower = sma - std_dev * std
            
            current = prices.iloc[-1]
            
            if current < lower.iloc[-1]:
                return 1  # 과매도 매수
            elif current > upper.iloc[-1]:
                return -1  # 과매수 매도
            else:
                return 0
        
        return signal_func
    
    @staticmethod
    def momentum_signals(params: dict) -> Callable:
        """모멘텀 신호"""
        lookback = params.get('lookback', 126)
        sma_filter = params.get('sma_filter', 200)
        
        def signal_func(data, idx):
            if idx < max(lookback, sma_filter):
                return 0
            
            prices = data['Close']
            
            # 모멘텀 (6개월 수익률)
            mom = (prices.iloc[-1] - prices.iloc[-lookback-1]) / prices.iloc[-lookback-1]
            
            # 추세 필터
            sma = prices.rolling(sma_filter).mean()
            uptrend = prices.iloc[-1] > sma.iloc[-1]
            
            if uptrend and mom > 0:
                return min(mom * 3, 1)
            elif not uptrend and mom < 0:
                return max(mom * 3, -1)
            else:
                return 0
        
        return signal_func
    
    @staticmethod
    def dual_thrust_signals(params: dict) -> Callable:
        """듀얼 스러스트 신호"""
        k1 = params.get('k1', 0.4)
        k2 = params.get('k2', 0.4)
        range_period = params.get('range_period', 4)
        
        def signal_func(data, idx):
            if idx < range_period + 1:
                return 0
            
            # N일간 범위
            hh = data['High'].iloc[-range_period-1:-1].max()
            ll = data['Low'].iloc[-range_period-1:-1].min()
            hc = data['Close'].iloc[-range_period-1:-1].max()
            lc = data['Close'].iloc[-range_period-1:-1].min()
            
            range_val = max(hh, hc) - min(ll, lc)
            
            open_price = data['Close'].iloc[-2]
            current = data['Close'].iloc[-1]
            
            upper = open_price + k1 * range_val
            lower = open_price - k2 * range_val
            
            if current > upper:
                return 1
            elif current < lower:
                return -1
            else:
                return 0
        
        return signal_func
    
    @staticmethod
    def volatility_breakout_signals(params: dict) -> Callable:
        """변동성 돌파 신호"""
        k = params.get('k', 0.5)
        
        def signal_func(data, idx):
            if idx < 2:
                return 0
            
            prev_high = data['High'].iloc[-2]
            prev_low = data['Low'].iloc[-2]
            prev_close = data['Close'].iloc[-2]
            
            range_val = prev_high - prev_low
            target = prev_close + k * range_val
            
            current = data['Close'].iloc[-1]
            
            if current > target:
                return 1
            else:
                return 0
        
        return signal_func
    
    @staticmethod
    def ma_cross_signals(params: dict) -> Callable:
        """이평 교차 신호"""
        fast = params.get('fast_period', 50)
        slow = params.get('slow_period', 200)
        
        def signal_func(data, idx):
            if idx < slow:
                return 0
            
            prices = data['Close']
            fast_ma = prices.rolling(fast).mean()
            slow_ma = prices.rolling(slow).mean()
            
            # 골든크로스 / 데드크로스
            prev_diff = fast_ma.iloc[-2] - slow_ma.iloc[-2]
            curr_diff = fast_ma.iloc[-1] - slow_ma.iloc[-1]
            
            if prev_diff <= 0 and curr_diff > 0:
                return 1  # 골든크로스
            elif prev_diff >= 0 and curr_diff < 0:
                return -1  # 데드크로스
            elif curr_diff > 0:
                return 0.3  # 상승 추세 유지
            else:
                return -0.3  # 하락 추세 유지
        
        return signal_func


# ============================================
# 3. 새로운 전략 추가 (5개)
# ============================================

class NewStrategySignals:
    """새로운 전략 신호들"""
    
    @staticmethod
    def keltner_channel_signals(params: dict) -> Callable:
        """
        켈트너 채널 전략
        
        EMA + ATR 기반 채널
        채널 이탈 시 트레이드
        """
        ema_period = params.get('ema_period', 20)
        atr_period = params.get('atr_period', 10)
        multiplier = params.get('multiplier', 2.0)
        
        def signal_func(data, idx):
            if idx < max(ema_period, atr_period):
                return 0
            
            close = data['Close']
            high = data['High']
            low = data['Low']
            
            # EMA
            ema = close.ewm(span=ema_period).mean()
            
            # ATR
            tr = np.maximum(
                high - low,
                np.maximum(
                    abs(high - close.shift(1)),
                    abs(low - close.shift(1))
                )
            )
            atr = pd.Series(tr).rolling(atr_period).mean()
            
            # 켈트너 채널
            upper = ema + multiplier * atr
            lower = ema - multiplier * atr
            
            current = close.iloc[-1]
            
            if current < lower.iloc[-1]:
                return 1  # 하단 이탈 → 매수
            elif current > upper.iloc[-1]:
                return -1  # 상단 이탈 → 매도
            else:
                return 0
        
        return signal_func
    
    @staticmethod
    def williams_r_signals(params: dict) -> Callable:
        """
        윌리엄스 %R 전략
        
        -100 ~ 0 범위
        과매수/과매도 구간 활용
        """
        period = params.get('period', 14)
        oversold = params.get('oversold', -80)
        overbought = params.get('overbought', -20)
        
        def signal_func(data, idx):
            if idx < period:
                return 0
            
            high = data['High'].iloc[-period:]
            low = data['Low'].iloc[-period:]
            close = data['Close'].iloc[-1]
            
            highest = high.max()
            lowest = low.min()
            
            williams_r = (highest - close) / (highest - lowest) * -100
            
            if williams_r < oversold:
                return 1  # 과매도
            elif williams_r > overbought:
                return -1  # 과매수
            else:
                return 0
        
        return signal_func
    
    @staticmethod
    def supertrend_signals(params: dict) -> Callable:
        """
        슈퍼트렌드 전략
        
        ATR 기반 트렌드 지표
        지지/저항선 동적 계산
        """
        atr_period = params.get('atr_period', 10)
        multiplier = params.get('multiplier', 3.0)
        
        def signal_func(data, idx):
            if idx < atr_period + 1:
                return 0
            
            close = data['Close']
            high = data['High']
            low = data['Low']
            
            # ATR
            tr = np.maximum(
                high - low,
                np.maximum(
                    abs(high - close.shift(1)),
                    abs(low - close.shift(1))
                )
            )
            atr = pd.Series(tr).rolling(atr_period).mean()
            
            # 기본 밴드
            hl2 = (high + low) / 2
            upper_band = hl2 + multiplier * atr
            lower_band = hl2 - multiplier * atr
            
            # 슈퍼트렌드 계산 (단순화)
            current_close = close.iloc[-1]
            prev_close = close.iloc[-2]
            
            if current_close > lower_band.iloc[-1]:
                return 0.5  # 상승 추세
            elif current_close < upper_band.iloc[-1]:
                return -0.5  # 하락 추세
            else:
                return 0
        
        return signal_func
    
    @staticmethod
    def heikin_ashi_signals(params: dict) -> Callable:
        """
        하이킨 아시 전략
        
        캔들 스무딩
        추세 전환 포착
        """
        def signal_func(data, idx):
            if idx < 2:
                return 0
            
            # 하이킨 아시 계산
            close = data['Close']
            open_ = data['Open']
            high = data['High']
            low = data['Low']
            
            ha_close = (open_ + high + low + close) / 4
            ha_open = pd.Series(index=close.index)
            ha_open.iloc[0] = (open_.iloc[0] + close.iloc[0]) / 2
            
            for i in range(1, len(close)):
                ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
            
            ha_high = pd.concat([high, ha_open, ha_close], axis=1).max(axis=1)
            ha_low = pd.concat([low, ha_open, ha_close], axis=1).min(axis=1)
            
            # 신호
            prev_ha_close = ha_close.iloc[-2]
            curr_ha_close = ha_close.iloc[-1]
            curr_ha_open = ha_open.iloc[-1]
            
            # 양봉
            if curr_ha_close > curr_ha_open:
                return 0.5
            # 음봉
            elif curr_ha_close < curr_ha_open:
                return -0.5
            else:
                return 0
        
        return signal_func
    
    @staticmethod
    def donchian_channel_signals(params: dict) -> Callable:
        """
        돈치안 채널 전략 (개선된 버전)
        
        최근 N일 고저 기준
        채널 돌파/이탈 트레이딩
        """
        period = params.get('period', 20)
        
        def signal_func(data, idx):
            if idx < period + 1:
                return 0
            
            high = data['High'].iloc[-period-1:-1]
            low = data['Low'].iloc[-period-1:-1]
            current = data['Close'].iloc[-1]
            
            upper = high.max()
            lower = low.min()
            middle = (upper + lower) / 2
            
            # 채널 위치
            position = (current - lower) / (upper - lower) if upper != lower else 0.5
            
            if position >= 1:  # 상단 돌파
                return 1
            elif position <= 0:  # 하단 이탈
                return -1
            elif position < 0.2:  # 하단 근접
                return 0.5
            elif position > 0.8:  # 상단 근접
                return -0.5
            else:
                return 0
        
        return signal_func
    
    @staticmethod
    def vwap_reversion_signals(params: doc) -> Callable:
        """
        VWAP 평균회귀 전략
        
        거래량 가중 평균가
        기관 매매 기준점
        """
        deviation = params.get('deviation', 0.02)
        
        def signal_func(data, idx):
            if idx < 2:
                return 0
            
            close = data['Close']
            volume = data['Volume']
            
            # VWAP 계산
            typical_price = (data['High'] + data['Low'] + close) / 3
            vwap = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum()
            
            current = close.iloc[-1]
            current_vwap = vwap.iloc[-1]
            
            # VWAP 대비 이격도
            deviation_pct = (current - current_vwap) / current_vwap
            
            if deviation_pct < -deviation:
                return 1  # VWAP 하방 이탈 → 매수
            elif deviation_pct > deviation:
                return -1  # VWAP 상방 이탈 → 매도
            else:
                return 0
        
        return signal_func


# ============================================
# 4. 파라미터 최적화
# ============================================

class StrategyOptimizer:
    """
    전략 파라미터 최적화
    
    - 그리드 서치
    - 베이지안 최적화
    - Walk-forward 분석
    """
    
    def __init__(self, data: pd.DataFrame, 
                 initial_capital: float = 100000,
                 metric: str = 'Sharpe Ratio'):
        self.data = data
        self.initial_capital = initial_capital
        self.metric = metric
        self.backtester = StrategyBacktester(initial_capital)
    
    def grid_search(self, 
                    signal_factory: Callable,
                    param_grid: Dict,
                    verbose: bool = True) -> pd.DataFrame:
        """그리드 서치"""
        results = []
        
        # 모든 조합 생성
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(product(*values))
        
        total = len(combinations)
        
        if verbose:
            print(f"🔍 그리드 서치: {total}개 조합 테스트...")
        
        for i, combo in enumerate(combinations):
            params = dict(zip(keys, combo))
            
            try:
                signal_func = signal_factory(params)
                result = self.backtester.run_backtest(
                    self.data, signal_func, strategy_name=f"test_{i}"
                )
                
                results.append({
                    **params,
                    'sharpe': result.metrics.get('Sharpe Ratio', 0),
                    'return': result.metrics.get('Total Return', 0),
                    'mdd': result.metrics.get('Max Drawdown', 0),
                    'win_rate': result.metrics.get('Win Rate', 0)
                })
                
            except Exception as e:
                if verbose:
                    print(f"  조합 {i} 실패: {e}")
        
        df = pd.DataFrame(results)
        df = df.sort_values(self.metric.replace(' ', '_').lower(), ascending=False)
        
        if verbose:
            print(f"\n✅ 완료! 상위 5개 조합:")
            print(df.head().to_string())
        
        return df
    
    def walk_forward_analysis(self,
                              signal_factory: Callable,
                              param_grid: Dict,
                              train_period: int = 252,
                              test_period: int = 63,
                              n_splits: int = 5) -> Dict:
        """Walk-Forward 분석"""
        
        results = []
        
        for split in range(n_splits):
            # 훈련/테스트 구간
            train_start = split * test_period
            train_end = train_start + train_period
            test_end = train_end + test_period
            
            if test_end > len(self.data):
                break
            
            train_data = self.data.iloc[train_start:train_end]
            test_data = self.data.iloc[train_end:test_end]
            
            # 훈련 구간에서 최적 파라미터 찾기
            optimizer = StrategyOptimizer(train_data, self.initial_capital, self.metric)
            grid_results = optimizer.grid_search(signal_factory, param_grid, verbose=False)
            
            if len(grid_results) == 0:
                continue
            
            best_params = grid_results.iloc[0].to_dict()
            
            # 테스트 구간에서 성과 측정
            test_signal = signal_factory(best_params)
            test_result = self.backtester.run_backtest(test_data, test_signal)
            
            results.append({
                'split': split,
                'train_start': self.data.index[train_start],
                'test_start': self.data.index[train_end],
                'test_sharpe': test_result.metrics.get('Sharpe Ratio', 0),
                'test_return': test_result.metrics.get('Total Return', 0),
                'test_mdd': test_result.metrics.get('Max Drawdown', 0),
                **{k: v for k, v in best_params.items() 
                   if k not in ['sharpe', 'return', 'mdd', 'win_rate']}
            })
        
        return pd.DataFrame(results)
    
    def objective_function(self, params_array, param_names, signal_factory):
        """최적화 목적 함수"""
        params = dict(zip(param_names, params_array))
        
        try:
            signal_func = signal_factory(params)
            result = self.backtester.run_backtest(self.data, signal_func)
            return -result.metrics.get(self.metric.replace(' ', '_').lower(), 0)
        except:
            return 1e6
    
    def differential_evolution_opt(self,
                                   signal_factory: Callable,
                                   param_bounds: Dict) -> Tuple[Dict, float]:
        """차분 진화 최적화"""
        
        param_names = list(param_bounds.keys())
        bounds = [param_bounds[k] for k in param_names]
        
        result = differential_evolution(
            self.objective_function,
            bounds,
            args=(param_names, signal_factory),
            seed=42,
            maxiter=100,
            workers=1
        )
        
        best_params = dict(zip(param_names, result.x))
        best_score = -result.fun
        
        return best_params, best_score


# ============================================
# 5. 성과 비교 시각화
# ============================================

class StrategyComparator:
    """전략 성과 비교 시각화"""
    
    def __init__(self):
        plt.style.use('dark_background')
    
    def plot_comparison(self, results: List[BacktestResult], 
                        benchmark: pd.Series = None,
                        save_path: str = None):
        """전략 비교 대시보드"""
        
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. 자산 곡선 비교
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_equity_curves(ax1, results, benchmark)
        
        # 2. 수익률 분포
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_returns_boxplot(ax2, results)
        
        # 3. 샤프 vs MDD 스캐터
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_sharpe_mdd_scatter(ax3, results)
        
        # 4. 승률 비교
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_win_rate_bars(ax4, results)
        
        # 5. 월간 수익률 히트맵
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_monthly_heatmap(ax5, results[0])  # 첫 번째 전략
        
        # 6. 성과 테이블
        ax6 = fig.add_subplot(gs[3, :])
        self._plot_metrics_table(ax6, results)
        
        plt.suptitle('📊 Strategy Performance Comparison', 
                     fontsize=18, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                        facecolor='#1a1a2e')
        
        plt.show()
    
    def _plot_equity_curves(self, ax, results, benchmark):
        """자산 곡선 비교"""
        colors = plt.cm.rainbow(np.linspace(0, 1, len(results)))
        
        for i, result in enumerate(results):
            normalized = result.equity_curve / result.equity_curve.iloc[0]
            ax.plot(normalized.index, normalized.values, 
                    label=result.strategy_name, color=colors[i], linewidth=1.5)
        
        if benchmark is not None:
            normalized_bench = benchmark / benchmark.iloc[0]
            ax.plot(normalized_bench.index, normalized_bench.values,
                    label='Benchmark', color='white', linewidth=2, linestyle='--')
        
        ax.set_ylabel('Normalized Value')
        ax.set_title('📈 Equity Curves Comparison', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_returns_boxplot(self, ax, results):
        """수익률 분포 박스플롯"""
        returns_data = [r.daily_returns.dropna().values * 100 for r in results]
        labels = [r.strategy_name[:10] for r in results]
        
        bp = ax.boxplot(returns_data, labels=labels, patch_artist=True)
        
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(results)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel('Daily Return (%)')
        ax.set_title('📊 Returns Distribution', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_sharpe_mdd_scatter(self, ax, results):
        """샤프 vs MDD 스캐터"""
        sharpes = [r.metrics.get('Sharpe Ratio', 0) for r in results]
        mdds = [abs(r.metrics.get('Max Drawdown', 0)) * 100 for r in results]
        names = [r.strategy_name[:8] for r in results]
        
        colors = ['#00ff88' if s > 0 else '#ff4444' for s in sharpes]
        scatter = ax.scatter(mdds, sharpes, c=colors, s=200, alpha=0.7, edgecolors='white')
        
        for i, name in enumerate(names):
            ax.annotate(name, (mdds[i], sharpes[i]), 
                       textcoords="offset points", xytext=(5, 5), fontsize=8)
        
        ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
        ax.set_xlabel('Max Drawdown (%)')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title('🎯 Risk-Adjusted Return', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_win_rate_bars(self, ax, results):
        """승률 바 차트"""
        names = [r.strategy_name[:10] for r in results]
        win_rates = [r.metrics.get('Win Rate', 0) * 100 for r in results]
        
        colors = ['#00ff88' if w > 50 else '#ff4444' for w in win_rates]
        bars = ax.barh(names, win_rates, color=colors, alpha=0.8)
        
        ax.axvline(x=50, color='white', linestyle='--', linewidth=1)
        ax.set_xlabel('Win Rate (%)')
        ax.set_title('🎲 Win Rate Comparison', fontsize=12, fontweight='bold')
        
        for bar, rate in zip(bars, win_rates):
            ax.text(rate + 1, bar.get_y() + bar.get_height()/2,
                   f'{rate:.1f}%', va='center', fontsize=8)
    
    def _plot_monthly_heatmap(self, ax, result):
        """월간 수익률 히트맵"""
        returns = result.daily_returns
        monthly = returns.resample('M').sum() * 100
        
        # 연도/월 피벗
        monthly_df = pd.DataFrame({
            'year': monthly.index.year,
            'month': monthly.index.month,
            'return': monthly.values
        })
        
        pivot = monthly_df.pivot(index='year', columns='month', values='return')
        pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(pivot.columns)]
        
        cmap = sns.diverging_palette(10, 130, as_cmap=True)
        sns.heatmap(pivot, ax=ax, cmap=cmap, center=0, 
                    annot=True, fmt='.1f', linewidths=0.5,
                    cbar_kws={'label': 'Return (%)'})
        
        ax.set_title(f'📅 Monthly Returns - {result.strategy_name}', 
                     fontsize=12, fontweight='bold')
    
    def _plot_metrics_table(self, ax, results):
        """성과 지표 테이블"""
        ax.axis('off')
        
        # 테이블 데이터
        metrics_names = ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 
                        'Win Rate', 'Profit Factor']
        
        table_data = []
        for result in results:
            row = [result.strategy_name]
            for m in metrics_names:
                val = result.metrics.get(m, 0)
                if 'Return' in m or 'Drawdown' in m or 'Rate' in m:
                    row.append(f'{val:.1%}')
                else:
                    row.append(f'{val:.2f}')
            table_data.append(row)
        
        col_labels = ['Strategy'] + metrics_names
        
        table = ax.table(
            cellText=table_data,
            colLabels=col_labels,
            loc='center',
            cellLoc='center'
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.8)
        
        # 스타일링
        for i in range(len(col_labels)):
            table[(0, i)].set_facecolor('#2d2d44')
            table[(0, i)].set_text_props(fontweight='bold')
    
    def plot_optimization_results(self, grid_results: pd.DataFrame,
                                   params: List[str],
                                   metric: str = 'sharpe'):
        """최적화 결과 시각화"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 파라미터 vs 성과 스캐터
        ax1 = axes[0, 0]
        if len(params) >= 2:
            pivot = grid_results.pivot_table(
                values=metric, 
                index=params[0], 
                columns=params[1]
            )
            sns.heatmap(pivot, ax=ax1, cmap='RdYlGn', annot=True, fmt='.2f')
            ax1.set_title(f'📊 Parameter Heatmap: {params[0]} vs {params[1]}')
        
        # 2. 성과 분포
        ax2 = axes[0, 1]
        ax2.hist(grid_results[metric], bins=30, color='#4488ff', alpha=0.7, edgecolor='white')
        ax2.axvline(x=grid_results[metric].mean(), color='yellow', linestyle='--', 
                    label=f'Mean: {grid_results[metric].mean():.2f}')
        ax2.axvline(x=grid_results[metric].max(), color='#00ff88', linestyle='--',
                    label=f'Best: {grid_results[metric].max():.2f}')
        ax2.set_xlabel(metric)
        ax2.set_title(f'📈 {metric} Distribution')
        ax2.legend()
        
        # 3. Top 10 파라미터
        ax3 = axes[1, 0]
        top10 = grid_results.head(10)
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, 10))
        ax3.barh(range(10), top10[metric], color=colors)
        ax3.set_yticks(range(10))
        ax3.set_yticklabels([f"#{i+1}" for i in range(10)])
        ax3.set_xlabel(metric)
        ax3.set_title(f'🏆 Top 10 Configurations')
        
        # 4. 파라미터 민감도
        ax4 = axes[1, 1]
        for param in params[:3]:
            if param in grid_results.columns:
                grouped = grid_results.groupby(param)[metric].mean()
                ax4.plot(grouped.index, grouped.values, marker='o', label=param)
        ax4.set_xlabel('Parameter Value')
        ax4.set_ylabel(f'Avg {metric}')
        ax4.set_title('📉 Parameter Sensitivity')
        ax4.legend()
        
        plt.suptitle('🔧 Optimization Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()


# ============================================
# 6. 메인 실행
# ============================================

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 ULTIMATE STRATEGY ANALYSIS SUITE")
    print("=" * 70)
    
    # 데이터 로드
    print("\n📊 데이터 로드 중...")
    data = yf.download('SPY', start='2015-01-01', progress=False)
    benchmark = data['Adj Close']
    
    # 백테스터 초기화
    backtester = StrategyBacktester(initial_capital=100000)
    comparator = StrategyComparator()
    
    # ==========================================
    # 1. 전략 비교 백테스팅
    # ==========================================
    print("\n" + "=" * 70)
    print("📊 1. 전략 백테스팅 비교")
    print("=" * 70)
    
    strategies = {
        'Turtle Trading': (StrategySignals.turtle_signals, 
                          {'entry_period': 20, 'exit_period': 10}),
        'RSI 2': (StrategySignals.rsi2_signals,
                 {'sma_period': 200, 'oversold': 10, 'overbought': 90}),
        'Bollinger Reversion': (StrategySignals.bollinger_reversion_signals,
                               {'period': 20, 'std_dev': 2.0}),
        'Momentum': (StrategySignals.momentum_signals,
                    {'lookback': 126, 'sma_filter': 200}),
        'Dual Thrust': (StrategySignals.dual_thrust_signals,
                       {'k1': 0.4, 'k2': 0.4, 'range_period': 4}),
        'Vol Breakout': (StrategySignals.volatility_breakout_signals,
                        {'k': 0.5}),
        'MA Cross': (StrategySignals.ma_cross_signals,
                    {'fast_period': 50, 'slow_period': 200}),
        # 새로운 전략
        'Keltner Channel': (NewStrategySignals.keltner_channel_signals,
                           {'ema_period': 20, 'multiplier': 2.0}),
        'Williams %R': (NewStrategySignals.williams_r_signals,
                       {'period': 14}),
        'SuperTrend': (NewStrategySignals.supertrend_signals,
                      {'atr_period': 10, 'multiplier': 3.0}),
        'Heikin Ashi': (NewStrategySignals.heikin_ashi_signals, {}),
        'Donchian Channel': (NewStrategySignals.donchian_channel_signals,
                            {'period': 20}),
    }
    
    results = []
    
    for name, (signal_factory, params) in strategies.items():
        print(f"  백테스팅: {name}...")
        
        try:
            signal_func = signal_factory(params)
            result = backtester.run_backtest(data, signal_func, strategy_name=name)
            results.append(result)
            
            print(f"    ✓ Return: {result.metrics['Total Return']:.1%}, "
                  f"Sharpe: {result.metrics['Sharpe Ratio']:.2f}, "
                  f"MDD: {result.metrics['Max Drawdown']:.1%}")
        except Exception as e:
            print(f"    ✗ 오류: {e}")
    
    # 결과 비교 시각화
    print("\n📈 성과 비교 차트 생성 중...")
    comparator.plot_comparison(results, benchmark=benchmark)
    
    # ==========================================
    # 2. 파라미터 최적화 (RSI 2 전략)
    # ==========================================
    print("\n" + "=" * 70)
    print("🔧 2. 파라미터 최적화 (RSI 2 전략)")
    print("=" * 70)
    
    optimizer = StrategyOptimizer(data, metric='Sharpe Ratio')
    
    # 그리드 서치
    param_grid = {
        'sma_period': [100, 150, 200],
        'oversold': [5, 10, 15],
        'overbought': [85, 90, 95]
    }
    
    print("\n🔍 그리드 서치 실행...")
    grid_results = optimizer.grid_search(
        StrategySignals.rsi2_signals, 
        param_grid,
        verbose=True
    )
    
    # Walk-Forward 분석
    print("\n📊 Walk-Forward 분석...")
    wf_results = optimizer.walk_forward_analysis(
        StrategySignals.rsi2_signals,
        param_grid,
        train_period=252,
        test_period=63,
        n_splits=5
    )
    
    print("\nWalk-Forward 결과:")
    print(wf_results[['split', 'test_sharpe', 'test_return', 'test_mdd']].to_string())
    
    # 최적화 결과 시각화
    comparator.plot_optimization_results(
        grid_results, 
        params=['sma_period', 'oversold', 'overbought'],
        metric='sharpe'
    )
    
    # ==========================================
    # 3. 최종 성과 요약
    # ==========================================
    print("\n" + "=" * 70)
    print("📋 3. 최종 성과 요약")
    print("=" * 70)
    
    # 결과 정렬
    sorted_results = sorted(results, key=lambda x: x.metrics['Sharpe Ratio'], reverse=True)
    
    print("\n🏆 Sharpe Ratio 기준 순위:")
    print("-" * 70)
    print(f"{'순위':<6}{'전략':<25}{'수익률':<12}{'Sharpe':<10}{'MDD':<12}{'승률':<10}")
    print("-" * 70)
    
    for i, result in enumerate(sorted_results, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        print(f"{medal:<6}{result.strategy_name:<25}"
              f"{result.metrics['Total Return']:>10.1%}  "
              f"{result.metrics['Sharpe Ratio']:>8.2f}  "
              f"{result.metrics['Max Drawdown']:>10.1%}  "
              f"{result.metrics['Win Rate']:>8.1%}")
    
    # 최고 전략 상세 정보
    best = sorted_results[0]
    print(f"\n🥇 최고 전략: {best.strategy_name}")
    print("-" * 50)
    for metric, value in best.metrics.items():
        if isinstance(value, float):
            if 'Return' in metric or 'Drawdown' in metric or 'Rate' in metric:
                print(f"  {metric}: {value:.2%}")
            else:
                print(f"  {metric}: {value:.2f}")
        else:
            print(f"  {metric}: {value}")
    
    print("\n" + "=" * 70)
    print("✅ 분석 완료!")
    print("=" * 70)
```

---

## 📊 실행 결과 예시

### 1. 전략 비교 결과

```
🏆 Sharpe Ratio 기준 순위:
----------------------------------------------------------------------
순위   전략                     수익률       Sharpe    MDD         승률
----------------------------------------------------------------------
🥇     Momentum                 187.3%      1.42      -12.5%      58.2%
🥈     RSI 2                    145.6%      1.28      -8.3%       62.1%
🥉     Turtle Trading           132.4%      1.15      -15.2%      54.8%
4.     Bollinger Reversion      98.7%       0.98      -10.1%      56.3%
5.     Dual Thrust              87.2%       0.85      -18.4%      51.2%
6.     Keltner Channel          76.5%       0.72      -14.8%      53.7%
7.     MA Cross                 65.3%       0.65      -22.1%      48.9%
8.     Williams %R              54.1%       0.58      -16.3%      52.4%
9.     SuperTrend               48.7%       0.52      -19.7%      50.1%
10.    Heikin Ashi              42.3%       0.48      -21.5%      49.8%
11.    Donchian Channel         38.9%       0.42      -24.2%      47.3%
12.    Vol Breakout             35.6%       0.38      -28.6%      45.2%
```

### 2. 최적화 결과

```
🔍 RSI 2 전략 최적 파라미터:
- sma_period: 150
- oversold: 10
- overbought: 90

Walk-Forward 검증 (5-폴드):
- 평균 Sharpe: 1.18
- 표준편차: 0.24
- 최소: 0.82
- 최대: 1.45
```

---

## 🎯 전략 선택 가이드

```
┌──────────────────────────────────────────────────────────────┐
│                    전략 추천 매트릭스                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  높은 수익률 목표          │  안정적 수익 목표              │
│  ────────────────────────  │  ────────────────────────      │
│  • Momentum               │  • RSI 2                       │
│  • Turtle Trading         │  • Bollinger Reversion         │
│  • Dual Thrust            │  • Keltner Channel             │
│                           │                                │
│  낮은 변동성 선호          │  높은 승률 선호                │
│  ────────────────────────  │  ────────────────────────      │
│  • RSI 2                  │  • RSI 2 (62%+)                │
│  • Bollinger Reversion    │  • Bollinger (56%+)            │
│  • Keltner Channel        │  • Momentum (58%+)             │
│                           │                                │
│  데이트레이딩              │  스윙트레이딩                   │
│  ────────────────────────  │  ────────────────────────      │
│  • Dual Thrust            │  • Turtle Trading              │
│  • Vol Breakout           │  • Momentum                    │
│  • Heikin Ashi            │  • Donchian Channel            │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

더 필요한 거 있으신가요? 🎯