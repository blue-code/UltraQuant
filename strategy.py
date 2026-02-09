# ============================================ 
# ULTIMATE QUANT STRATEGY & ANALYSIS SUITE
# 12가지 전략 + 백테스팅 + 최적화 통합
# ============================================ 

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from itertools import product
from enum import Enum
import warnings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.optimize import differential_evolution

warnings.filterwarnings('ignore')
plt.style.use('dark_background')

# ============================================ 
# 1. 공통 유틸리티 및 데이터 구조
# ============================================ 

@dataclass
class Signal:
    """통합 신호 객체"""
    timestamp: datetime
    symbol: str
    direction: float  # -1 ~ 1
    strength: float   # 0 ~ 1
    strategy_name: str
    metadata: dict = None

@dataclass
class BacktestResult:
    """백테스팅 결과"""
    strategy_name: str
    equity_curve: pd.Series
    daily_returns: pd.Series
    metrics: Dict
    trades: List = field(default_factory=list)
    
    def __repr__(self):
        return (
            f"BacktestResult({self.strategy_name}: "
            f"Return={self.metrics.get('Total Return', 0):.1%}, "
            f"Sharpe={self.metrics.get('Sharpe Ratio', 0):.2f}, "
            f"MDD={self.metrics.get('Max Drawdown', 0):.1%})"
        )

class PerformanceMetrics:
    """성과 지표 계산"""
    
    @staticmethod
    def calculate_all(returns: pd.Series, equity: pd.Series, 
                      risk_free_rate: float = 0.04) -> Dict:
        """모든 성과 지표 계산"""
        returns = returns.dropna()
        if len(returns) == 0:
            return {}

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
        
        return {
            'Total Return': total_return,
            'Annual Return': annual_return,
            'Annual Volatility': ann_vol,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Max Drawdown': max_dd,
            'Calmar Ratio': calmar,
            'Win Rate': win_rate,
            'Profit Factor': profit_factor
        }

# ============================================ 
# 2. 백테스팅 엔진
# ============================================ 

class StrategyBacktester:
    """
    이벤트 기반 백테스팅 엔진
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
        
        # 최소 룩백 기간 설정 (지표 계산용)
        lookback = 50 
        
        for i in range(lookback, len(data)):
            # 신호 생성
            # 현재 시점(i)까지의 데이터를 넘겨줌 (Look-ahead bias 방지)
            signal = signal_func(data.iloc[:i+1], i)
            
            current_price = prices[i]
            prev_price = prices[i-1]
            
            # 포지션 진입/청산 로직
            # 1. 포지션 없을 때
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
                    
                elif signal < -0.3:  # 공매도 (간소화: 숏 가능 가정)
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
            
            # 2. 롱 포지션 보유 중
            elif position == 1:
                if signal < -0.1:  # 청산 신호
                    exit_price = current_price * (1 - self.slippage)
                    pnl = (exit_price - entry_price) / entry_price
                    # 자본금 업데이트 (복리)
                    equity.append(equity[-1] * (1 + pnl) * (1 - self.commission))
                    
                    trades.append({
                        'date': dates[i],
                        'type': 'SELL',
                        'price': exit_price,
                        'pnl': pnl
                    })
                    position = 0
            
            # 3. 숏 포지션 보유 중
            elif position == -1:
                if signal > 0.1:  # 커버 신호
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
            
            # 일일 수익률 계산 (Mark-to-Market)
            if position == 1:
                daily_return = (current_price - prev_price) / prev_price
            elif position == -1:
                daily_return = (prev_price - current_price) / prev_price
            else:
                daily_return = 0
            
            returns.append(daily_return)
            
            # 포지션 유지 중일 때 equity 업데이트 (일별 변동 반영)
            if position != 0:
                if len(equity) < len(returns) + 1: # 이미 청산으로 업데이트되지 않았다면
                     equity.append(equity[-1] * (1 + daily_return))
            elif len(equity) < len(returns) + 1:
                 equity.append(equity[-1]) # 현금 보유

        # 결과 정리
        equity_series = pd.Series(equity[:len(returns)], index=dates[lookback:])
        returns_series = pd.Series(returns, index=dates[lookback:])
        
        metrics = PerformanceMetrics.calculate_all(returns_series, equity_series)
        metrics['Total Trades'] = len([t for t in trades if 'pnl' in t])
        
        return BacktestResult(
            strategy_name=strategy_name,
            equity_curve=equity_series,
            daily_returns=returns_series,
            metrics=metrics,
            trades=trades
        )

# ============================================ 
# 3. 전략 구현 (12가지 + Alpha)
# ============================================ 

class StrategySignals:
    """
    각 전략의 신호 생성 함수 모음
    각 함수는 (data, idx) -> signal 반환
    """
    
    # --- 1. 추세 추종 (Trend Following) --- 
    
    @staticmethod
    def turtle_signals(params: dict) -> Callable:
        """터틀 트레이딩: 돈치안 채널 돌파"""
        entry_period = params.get('entry_period', 20)
        exit_period = params.get('exit_period', 10)
        
        def signal_func(data, idx):
            if idx < entry_period: return 0
            # 과거 데이터 (현재 봉 제외)
            past_high = data['High'].iloc[-entry_period-1:-1]
            past_low = data['Low'].iloc[-exit_period-1:-1]
            
            if len(past_high) == 0: return 0

            high_channel = past_high.max()
            exit_low = past_low.min()
            
            current = data['Close'].iloc[-1]
            
            if current >= high_channel: return 1   # 상단 돌파 매수
            elif current <= exit_low: return 0     # 하단 이탈 청산
            else: return 0.5                       # 관망/홀딩
        return signal_func

    @staticmethod
    def momentum_signals(params: dict) -> Callable:
        """모멘텀: 6개월 수익률 + 추세 필터"""
        lookback = params.get('lookback', 126)
        sma_filter = params.get('sma_filter', 200)
        
        def signal_func(data, idx):
            if idx < max(lookback, sma_filter): return 0
            prices = data['Close']
            
            mom = (prices.iloc[-1] / prices.iloc[-lookback-1]) - 1
            sma = prices.iloc[-sma_filter:].mean()
            uptrend = prices.iloc[-1] > sma
            
            if uptrend and mom > 0: return min(mom * 3, 1)
            elif not uptrend and mom < 0: return max(mom * 3, -1)
            return 0
        return signal_func

    @staticmethod
    def supertrend_signals(params: dict) -> Callable:
        """슈퍼트렌드: ATR 기반 추세"""
        atr_period = params.get('atr_period', 10)
        multiplier = params.get('multiplier', 3.0)
        
        def signal_func(data, idx):
            if idx < atr_period + 1: return 0
            
            # ATR 간이 계산
            high = data['High'].iloc[-atr_period-1:]
            low = data['Low'].iloc[-atr_period-1:]
            close = data['Close'].iloc[-atr_period-1:]
            tr = np.maximum(high - low, np.abs(high - close.shift(1)))
            atr = tr.mean()
            
            hl2 = (high.iloc[-1] + low.iloc[-1]) / 2
            upper_band = hl2 + multiplier * atr
            lower_band = hl2 - multiplier * atr
            
            current = close.iloc[-1]
            
            if current > lower_band: return 0.5
            elif current < upper_band: return -0.5
            return 0
        return signal_func

    # --- 2. 평균 회귀 (Mean Reversion) --- 

    @staticmethod
    def rsi2_signals(params: dict) -> Callable:
        """RSI 2: 단기 과매수/과매도 + 장기 추세 필터"""
        sma_period = params.get('sma_period', 200)
        oversold = params.get('oversold', 10)
        overbought = params.get('overbought', 90)
        
        def signal_func(data, idx):
            if idx < sma_period: return 0
            prices = data['Close']
            
            # RSI 2 계산
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(2).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(2).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            current_rsi = rsi.iloc[-1]
            sma = prices.iloc[-sma_period:].mean()
            uptrend = prices.iloc[-1] > sma
            
            if uptrend and current_rsi < oversold: return 1      # 눌림목 매수
            elif uptrend and current_rsi > overbought: return 0  # 과열 청산
            elif not uptrend and current_rsi > overbought: return -1 # 하락장 반등 매도
            return 0
        return signal_func

    @staticmethod
    def bollinger_reversion_signals(params: dict) -> Callable:
        """볼린저 밴드 역추세"""
        period = params.get('period', 20)
        std_dev = params.get('std_dev', 2.0)
        
        def signal_func(data, idx):
            if idx < period: return 0
            prices = data['Close']
            sma = prices.iloc[-period:].mean()
            std = prices.iloc[-period:].std()
            
            upper = sma + std_dev * std
            lower = sma - std_dev * std
            current = prices.iloc[-1]
            
            if current < lower: return 1   # 과매도
            elif current > upper: return -1 # 과매수
            return 0
        return signal_func
    
    @staticmethod
    def williams_r_signals(params: dict) -> Callable:
        """Williams %R"""
        period = params.get('period', 14)
        
        def signal_func(data, idx):
            if idx < period: return 0
            high = data['High'].iloc[-period:]
            low = data['Low'].iloc[-period:]
            close = data['Close'].iloc[-1]
            
            highest = high.max()
            lowest = low.min()
            
            if highest == lowest: return 0
            r = (highest - close) / (highest - lowest) * -100
            
            if r < -80: return 1   # 과매도
            elif r > -20: return -1 # 과매수
            return 0
        return signal_func

    # --- 3. 변동성/브레이크아웃 (Volatility/Breakout) --- 

    @staticmethod
    def dual_thrust_signals(params: dict) -> Callable:
        """듀얼 스러스트: 데이트레이딩"""
        k1 = params.get('k1', 0.5)
        k2 = params.get('k2', 0.5)
        range_period = params.get('range_period', 4)
        
        def signal_func(data, idx):
            if idx < range_period + 1: return 0
            # N일 고저
            past_data = data.iloc[-range_period-1:-1]
            hh = past_data['High'].max()
            ll = past_data['Low'].min()
            hc = past_data['Close'].max()
            lc = past_data['Close'].min()
            
            range_val = max(hh - lc, hc - ll)
            open_price = data['Open'].iloc[-1] # 당일 시가 사용
            
            upper = open_price + k1 * range_val
            lower = open_price - k2 * range_val
            current = data['Close'].iloc[-1]
            
            if current > upper: return 1
            elif current < lower: return -1
            return 0
        return signal_func

    @staticmethod
    def volatility_breakout_signals(params: dict) -> Callable:
        """변동성 돌파"""
        k = params.get('k', 0.5)
        
        def signal_func(data, idx):
            if idx < 2: return 0
            prev = data.iloc[-2]
            range_val = prev['High'] - prev['Low']
            target = prev['Close'] + k * range_val
            
            current = data['Close'].iloc[-1]
            if current > target: return 1
            return 0
        return signal_func

    # --- 4. 기타/앙상블 --- 
    
    @staticmethod
    def ma_cross_signals(params: dict) -> Callable:
        """이평선 교차"""
        fast = params.get('fast_period', 50)
        slow = params.get('slow_period', 200)
        
        def signal_func(data, idx):
            if idx < slow: return 0
            prices = data['Close']
            fast_ma = prices.rolling(fast).mean()
            slow_ma = prices.rolling(slow).mean()
            
            diff_curr = fast_ma.iloc[-1] - slow_ma.iloc[-1]
            diff_prev = fast_ma.iloc[-2] - slow_ma.iloc[-2]
            
            if diff_prev <= 0 and diff_curr > 0: return 1
            elif diff_prev >= 0 and diff_curr < 0: return -1
            elif diff_curr > 0: return 0.5 # 보유
            else: return -0.5
        return signal_func


# ============================================ 
# 4. 분석 및 시각화 도구
# ============================================ 

class StrategyComparator:
    """전략 성과 비교 시각화"""
    
    def plot_comparison(self, results: List[BacktestResult], 
                        benchmark: pd.Series = None):
        """대시보드 생성"""
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. 자산 곡선
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_equity(ax1, results, benchmark)
        
        # 2. 수익률 박스플롯
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_boxplot(ax2, results)
        
        # 3. 리스크-리턴 (Sharpe vs MDD)
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_scatter(ax3, results)
        
        # 4. 승률
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_win_rate(ax4, results)
        
        # 5. 성과 테이블
        ax5 = fig.add_subplot(gs[3, :])
        self._plot_table(ax5, results)
        
        plt.suptitle('📊 Strategy Performance Comparison', fontsize=20, fontweight='bold', y=0.95)
        plt.show()

    def _plot_equity(self, ax, results, benchmark):
        for r in results:
            normalized = r.equity_curve / r.equity_curve.iloc[0]
            ax.plot(normalized.index, normalized.values, label=r.strategy_name, linewidth=1.5)
        if benchmark is not None:
            norm_bench = benchmark / benchmark.iloc[0]
            ax.plot(norm_bench.index, norm_bench.values, label='Benchmark', 
                    color='white', linestyle='--', linewidth=2, alpha=0.7)
        ax.set_title('Equity Curves (Normalized)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.2)

    def _plot_boxplot(self, ax, results):
        data = [r.daily_returns.dropna() * 100 for r in results]
        ax.boxplot(data, labels=[r.strategy_name[:8] for r in results], patch_artist=True)
        ax.set_title('Daily Returns Distribution (%)', fontsize=12)
        ax.tick_params(axis='x', rotation=45)

    def _plot_scatter(self, ax, results):
        sharpes = [r.metrics.get('Sharpe Ratio', 0) for r in results]
        mdds = [abs(r.metrics.get('Max Drawdown', 0)) * 100 for r in results]
        names = [r.strategy_name for r in results]
        
        ax.scatter(mdds, sharpes, c=sharpes, cmap='RdYlGn', s=100, edgecolors='white')
        for i, txt in enumerate(names):
            ax.annotate(txt, (mdds[i], sharpes[i]), xytext=(5, 5), textcoords='offset points', fontsize=9)
        ax.set_xlabel('Max Drawdown (%)')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title('Risk-Adjusted Return', fontsize=12)
        ax.grid(True, alpha=0.2)

    def _plot_win_rate(self, ax, results):
        names = [r.strategy_name for r in results]
        rates = [r.metrics.get('Win Rate', 0) * 100 for r in results]
        ax.barh(names, rates, color='skyblue')
        ax.axvline(50, color='r', linestyle='--', alpha=0.5)
        ax.set_title('Win Rate (%)', fontsize=12)

    def _plot_table(self, ax, results):
        ax.axis('off')
        cols = ['Strategy', 'Total Return', 'Sharpe', 'MDD', 'Win Rate']
        cell_text = []
        for r in results:
            cell_text.append([
                r.strategy_name,
                f"{r.metrics['Total Return']:.1%}",
                f"{r.metrics['Sharpe Ratio']:.2f}",
                f"{r.metrics['Max Drawdown']:.1%}",
                f"{r.metrics['Win Rate']:.1%}"
            ])
        table = ax.table(cellText=cell_text, colLabels=cols, loc='center', cellLoc='center')
        table.scale(1, 1.5)
        table.auto_set_font_size(False)
        table.set_fontsize(10)


class StrategyOptimizer:
    """파라미터 최적화 (그리드 서치)"""
    
    def __init__(self, data, initial_capital=100000):
        self.data = data
        self.backtester = StrategyBacktester(initial_capital)
        
    def grid_search(self, signal_factory, param_grid):
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(product(*values))
        
        results = []
        print(f"🔍 Optimization: Testing {len(combinations)} combinations...")
        
        for combo in combinations:
            params = dict(zip(keys, combo))
            try:
                signal_func = signal_factory(params)
                res = self.backtester.run_backtest(self.data, signal_func)
                results.append({
                    **params,
                    'sharpe': res.metrics['Sharpe Ratio'],
                    'return': res.metrics['Total Return'],
                    'mdd': res.metrics['Max Drawdown']
                })
            except Exception:
                continue
                
        return pd.DataFrame(results).sort_values('sharpe', ascending=False)

# ============================================ 
# 5. 메인 실행
# ============================================ 

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 QUANT STRATEGY SUITE - EXECUTION")
    print("=" * 60)
    
    # 1. 데이터 로드
    symbol = 'SPY'
    print(f"\n📊 Downloading data for {symbol}...")
    try:
        data = yf.download(symbol, start='2018-01-01', progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data = data.droplevel(1, axis=1)  # Ticker 레벨 제거
        if len(data) == 0: raise ValueError("No data")
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        exit()

    # 2. 전략 정의
    strategies = {
        'Turtle Trading': (StrategySignals.turtle_signals, {'entry_period': 20, 'exit_period': 10}),
        'Momentum': (StrategySignals.momentum_signals, {'lookback': 126, 'sma_filter': 200}),
        'RSI 2': (StrategySignals.rsi2_signals, {'sma_period': 200, 'oversold': 10, 'overbought': 90}),
        'Bollinger Rev': (StrategySignals.bollinger_reversion_signals, {'period': 20, 'std_dev': 2.0}),
        'Dual Thrust': (StrategySignals.dual_thrust_signals, {'k1': 0.5, 'k2': 0.5, 'range_period': 4}),
        'Vol Breakout': (StrategySignals.volatility_breakout_signals, {'k': 0.5}),
        'SuperTrend': (StrategySignals.supertrend_signals, {'atr_period': 10, 'multiplier': 3.0}),
        'Williams %R': (StrategySignals.williams_r_signals, {'period': 14}),
        'MA Cross': (StrategySignals.ma_cross_signals, {'fast_period': 50, 'slow_period': 200})
    }

    # 3. 백테스팅 실행
    print("\n📈 Running Backtests...")
    backtester = StrategyBacktester(initial_capital=100000)
    results = []

    for name, (factory, params) in strategies.items():
        print(f"  • {name}...", end=" ")
        try:
            signal_func = factory(params)
            res = backtester.run_backtest(data, signal_func, strategy_name=name)
            results.append(res)
            print(f"✅ Sharpe: {res.metrics['Sharpe Ratio']:.2f}")
        except Exception as e:
            print(f"❌ Error: {e}")

    # 4. 결과 출력
    print("\n🏆 Strategy Rankings (by Sharpe Ratio):")
    results.sort(key=lambda x: x.metrics.get('Sharpe Ratio', -99), reverse=True)
    
    print("-" * 65)
    print(f"{ 'Rank':<5} {'Strategy':<20} {'Return':<10} {'Sharpe':<8} {'MDD':<10} {'Win Rate'}")
    print("-" * 65)
    for i, res in enumerate(results, 1):
        print(f"{i:<5} {res.strategy_name:<20} {res.metrics['Total Return']:>9.1%} "
              f"{res.metrics['Sharpe Ratio']:>8.2f} {res.metrics['Max Drawdown']:>10.1%} "
              f"{res.metrics['Win Rate']:>8.1%}")

    # 5. 최적화 예시 (RSI 2)
    print("\n🔧 Optimizing 'RSI 2' Strategy...")
    optimizer = StrategyOptimizer(data)
    param_grid = {
        'sma_period': [100, 150, 200],
        'oversold': [5, 10, 15],
        'overbought': [85, 90, 95]
    }
    opt_results = optimizer.grid_search(StrategySignals.rsi2_signals, param_grid)
    
    print("\n✅ Top 3 Parameter Sets:")
    print(opt_results.head(3).to_string(index=False))

    # 6. 시각화 (선택 사항)
    print("\n🎨 Generating Comparison Plots...")
    try:
        comparator = StrategyComparator()
        # 주의: 벤치마크 데이터 인덱스 매칭 필요 (간소화를 위해 생략 가능)
        comparator.plot_comparison(results, benchmark=data['Adj Close'] if 'Adj Close' in data else data['Close'])
    except Exception as e:
        print(f"⚠️ 시각화 오류 (GUI 환경이 아닐 수 있음): {e}")

    print("\n✅ Done.")
