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

        total_return = (equity.iloc[-1] / equity.iloc[0] - 1)
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = (annual_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0
        
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252)
        sortino = (annual_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0
        
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        max_dd = drawdown.min()
        
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
        win_rate = (returns > 0).sum() / len(returns)
        
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
    """이벤트 기반 백테스팅 엔진"""
    
    def __init__(self, initial_capital: float = 100000, commission: float = 0.001, 
                 slippage: float = 0.0005, position_size: float = 0.95):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_size = position_size
    
    def run_backtest(self, data: pd.DataFrame, signal_func: Callable, 
                     strategy_name: str = "Strategy") -> BacktestResult:
        equity = [self.initial_capital]
        returns = []
        trades = []
        position = 0
        entry_price = 0
        
        prices = data['Close'].values
        dates = data.index
        lookback = 50 
        
        for i in range(lookback, len(data)):
            signal = signal_func(data.iloc[:i+1], i)
            current_price = prices[i]
            prev_price = prices[i-1]
            
            if position == 0:
                if signal > 0.3:
                    position = 1
                    entry_price = current_price * (1 + self.slippage)
                    trade_value = equity[-1] * self.position_size
                    shares = trade_value / entry_price
                    trades.append({'date': dates[i], 'type': 'BUY', 'price': entry_price, 'shares': shares})
                elif signal < -0.3:
                    position = -1
                    entry_price = current_price * (1 - self.slippage)
                    trades.append({'date': dates[i], 'type': 'SELL SHORT', 'price': entry_price})
            
            elif position == 1:
                if signal < -0.1:
                    exit_price = current_price * (1 - self.slippage)
                    pnl = (exit_price - entry_price) / entry_price
                    equity.append(equity[-1] * (1 + pnl) * (1 - self.commission))
                    trades.append({'date': dates[i], 'type': 'SELL', 'price': exit_price, 'pnl': pnl})
                    position = 0
            
            elif position == -1:
                if signal > 0.1:
                    exit_price = current_price * (1 + self.slippage)
                    pnl = (entry_price - exit_price) / entry_price
                    equity.append(equity[-1] * (1 + pnl) * (1 - self.commission))
                    trades.append({'date': dates[i], 'type': 'COVER', 'price': exit_price, 'pnl': pnl})
                    position = 0
            
            daily_return = 0
            if position == 1: daily_return = (current_price - prev_price) / prev_price
            elif position == -1: daily_return = (prev_price - current_price) / prev_price
            returns.append(daily_return)
            
            if len(equity) < len(returns) + 1:
                if position != 0: equity.append(equity[-1] * (1 + daily_return))
                else: equity.append(equity[-1])

        equity_series = pd.Series(equity[:len(returns)], index=dates[lookback:])
        returns_series = pd.Series(returns, index=dates[lookback:])
        metrics = PerformanceMetrics.calculate_all(returns_series, equity_series)
        
        return BacktestResult(strategy_name=strategy_name, equity_curve=equity_series, 
                              daily_returns=returns_series, metrics=metrics, trades=trades)

# ============================================ 
# 3. 전략 구현
# ============================================ 

class StrategySignals:
    @staticmethod
    def turtle_signals(params: dict) -> Callable:
        entry_period = int(params.get('entry_period', 20))
        exit_period = int(params.get('exit_period', 10))
        def signal_func(data, idx):
            if idx < entry_period: return 0
            past_high = data['High'].iloc[-entry_period-1:-1]
            past_low = data['Low'].iloc[-exit_period-1:-1]
            if len(past_high) == 0: return 0
            high_channel = past_high.max()
            exit_low = past_low.min()
            current = data['Close'].iloc[-1]
            if current >= high_channel: return 1
            elif current <= exit_low: return 0
            else: return 0.5
        return signal_func

    @staticmethod
    def rsi2_signals(params: dict) -> Callable:
        sma_period = int(params.get('sma_period', 200))
        oversold = float(params.get('oversold', 10))
        overbought = float(params.get('overbought', 90))
        def signal_func(data, idx):
            if idx < sma_period: return 0
            prices = data['Close']
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(2).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(2).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            sma = prices.iloc[-sma_period:].mean()
            if prices.iloc[-1] > sma and current_rsi < oversold: return 1
            elif prices.iloc[-1] > sma and current_rsi > overbought: return 0
            return 0
        return signal_func

    @staticmethod
    def momentum_signals(params: dict) -> Callable:
        lookback = int(params.get('lookback', 126))
        sma_filter = int(params.get('sma_filter', 200))
        def signal_func(data, idx):
            if idx < max(lookback, sma_filter): return 0
            prices = data['Close']
            mom = (prices.iloc[-1] / prices.iloc[-lookback-1]) - 1
            sma = prices.iloc[-sma_filter:].mean()
            if prices.iloc[-1] > sma and mom > 0: return 1
            return 0
        return signal_func

    # --- 5. 최신/고급 전략 (ML & Structure) ---

    @staticmethod
    def ml_ensemble_signals(params: dict) -> Callable:
        """
        ML 앙상블 전략: RandomForest를 이용한 방향성 예측
        학습 효율을 위해 20봉마다 재학습 수행
        """
        from sklearn.ensemble import RandomForestClassifier
        lookback = int(params.get('lookback', 252))
        retrain_freq = 20
        model_cache = {'model': None, 'last_train_idx': -1}
        
        def signal_func(data, idx):
            if idx < lookback + 50: return 0
            
            # 피처 생성
            df = data.copy()
            df['RSI'] = df['Close'].diff().rolling(14).apply(lambda x: np.sum(x[x>0]) / (np.sum(np.abs(x))+1e-9))
            df['ROC'] = df['Close'].pct_change(10)
            df['Vol'] = df['Close'].rolling(20).std()
            df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
            
            features = ['RSI', 'ROC', 'Vol']
            
            # 주기적 재학습
            if model_cache['model'] is None or (idx - model_cache['last_train_idx']) >= retrain_freq:
                train_df = df.iloc[idx-lookback:idx].dropna()
                if len(train_df) < 50: return 0
                
                model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42) # 가볍게 설정
                model.fit(train_df[features], train_df['Target'])
                model_cache['model'] = model
                model_cache['last_train_idx'] = idx
            
            current_feat = df.iloc[[idx]][features].fillna(0)
            prob = model_cache['model'].predict_proba(current_feat)[0][1]
            
            if prob > 0.55: return 1
            elif prob < 0.45: return -1
            return 0
        return signal_func

    @staticmethod
    def regime_switching_signals(params: dict) -> Callable:
        """
        시장 국면 전환 전략: 변동성 국면에 따라 추세추종/평균회귀 자동 스위칭
        """
        vol_lookback = int(params.get('vol_lookback', 20))
        threshold = float(params.get('threshold', 1.5))
        
        def signal_func(data, idx):
            if idx < vol_lookback + 50: return 0
            
            returns = data['Close'].pct_change()
            curr_vol = returns.iloc[-vol_lookback:].std()
            avg_vol = returns.rolling(100).std().iloc[-1]
            
            # 1. 고변동성 국면 (추세 발생 가능성) -> 모멘텀
            if curr_vol > avg_vol * threshold:
                mom = data['Close'].iloc[-1] / data['Close'].iloc[-20] - 1
                return 1 if mom > 0 else -1
            
            # 2. 저변동성 국면 (박스권) -> RSI 역추세
            else:
                prices = data['Close']
                rsi = (prices.diff().rolling(14).apply(lambda x: np.sum(x[x>0]) / (np.sum(np.abs(x))+1e-9))).iloc[-1]
                if rsi < 0.3: return 1
                elif rsi > 0.7: return -1
            return 0
        return signal_func

    @staticmethod
    def liquidity_sweep_signals(params: dict) -> Callable:
        """
        유동성 스윕(Liquidity Sweep): 전일 고/저점 돌파 후 복귀 포착 (SMC)
        """
        lookback = int(params.get('lookback', 20))
        
        def signal_func(data, idx):
            if idx < 2: return 0
            
            # 전일(또는 이전 구간) 고점/저점
            prev_high = data['High'].iloc[-lookback-1:-1].max()
            prev_low = data['Low'].iloc[-lookback-1:-1].min()
            
            curr_high = data['High'].iloc[-1]
            curr_low = data['Low'].iloc[-1]
            curr_close = data['Close'].iloc[-1]
            
            # 1. Bullish Sweep: 저점 이탈 후 위로 말아올릴 때
            if data['Low'].iloc[-1] < prev_low and curr_close > prev_low:
                return 1
            
            # 2. Bearish Sweep: 고점 돌파 후 아래로 꺾일 때
            if data['High'].iloc[-1] > prev_high and curr_close < prev_high:
                return -1
                
            return 0
        return signal_func

# ============================================ 
# 4. 최적화 도구
# ============================================ 

class StrategyOptimizer:
    def __init__(self, data: pd.DataFrame, initial_capital: float = 100000):
        self.data = data
        self.backtester = StrategyBacktester(initial_capital)
        
    def grid_search(self, signal_factory: Callable, param_grid: dict) -> pd.DataFrame:
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(product(*values))
        results = []
        for combo in combinations:
            params = dict(zip(keys, combo))
            try:
                signal_func = signal_factory(params)
                res = self.backtester.run_backtest(self.data, signal_func)
                results.append({**params, 'sharpe': res.metrics['Sharpe Ratio'], 
                                'return': res.metrics['Total Return'], 'mdd': res.metrics['Max Drawdown']})
            except: continue
        return pd.DataFrame(results).sort_values('sharpe', ascending=False)

class WalkForwardOptimizer:
    def __init__(self, data: pd.DataFrame, initial_capital: float = 100000):
        self.data = data
        self.backtester = StrategyBacktester(initial_capital)

    def run_wfa(self, signal_factory: Callable, param_grid: dict, 
                n_windows: int = 5, train_ratio: float = 0.7) -> List[dict]:
        total_len = len(self.data)
        window_size = total_len // n_windows
        results = []
        for i in range(n_windows):
            start_idx = i * (window_size // 2)
            train_end = start_idx + int(window_size * train_ratio)
            test_end = min(start_idx + window_size, total_len)
            if test_end <= train_end: break
            train_data = self.data.iloc[start_idx:train_end]
            test_data = self.data.iloc[train_end:test_end]
            optimizer = StrategyOptimizer(train_data)
            opt_df = optimizer.grid_search(signal_factory, param_grid)
            if opt_df.empty: continue
            best_params = opt_df.iloc[0].drop(['sharpe', 'return', 'mdd']).to_dict()
            signal_func = signal_factory(best_params)
            test_res = self.backtester.run_backtest(test_data, signal_func)
            results.append({'window': i, 'best_params': best_params, 
                            'oos_sharpe': test_res.metrics['Sharpe Ratio']})
        return results

class DifferentialEvolutionOptimizer:
    def __init__(self, data: pd.DataFrame, initial_capital: float = 100000):
        self.data = data
        self.backtester = StrategyBacktester(initial_capital)

    def optimize(self, signal_factory: Callable, bounds: List[Tuple[float, float]], 
                 param_names: List[str]) -> dict:
        def objective(x):
            params = dict(zip(param_names, x))
            for k in params:
                if 'period' in k or 'lookback' in k: params[k] = int(params[k])
            try:
                signal_func = signal_factory(params)
                res = self.backtester.run_backtest(self.data, signal_func)
                return -res.metrics['Sharpe Ratio']
            except: return 0
        result = differential_evolution(objective, bounds, maxiter=20, popsize=10)
        best_params = dict(zip(param_names, result.x))
        for k in best_params:
            if 'period' in k or 'lookback' in k: best_params[k] = int(best_params[k])
        return {'best_params': best_params, 'best_sharpe': -result.fun}

# ============================================ 
# 5. 메인 실행 (생략 가능)
# ============================================ 
if __name__ == "__main__":
    pass