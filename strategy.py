# ============================================ 
# ULTIMATE QUANT STRATEGY & ANALYSIS SUITE
# 12媛吏 ?꾨왂 + 諛깊뀒?ㅽ똿 + 理쒖쟻???듯빀
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
# 1. 怨듯넻 ?좏떥由ы떚 諛??곗씠??援ъ“
# ============================================ 

@dataclass
class Signal:
    """?듯빀 ?좏샇 媛앹껜"""
    timestamp: datetime
    symbol: str
    direction: float  # -1 ~ 1
    strength: float   # 0 ~ 1
    strategy_name: str
    metadata: dict = None

@dataclass
class BacktestResult:
    """諛깊뀒?ㅽ똿 寃곌낵"""
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
    """?깃낵 吏??怨꾩궛"""
    
    @staticmethod
    def calculate_all(returns: pd.Series, equity: pd.Series, 
                      risk_free_rate: float = 0.04) -> Dict:
        """紐⑤뱺 ?깃낵 吏??怨꾩궛"""
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
# 2. 諛깊뀒?ㅽ똿 ?붿쭊
# ============================================ 

class StrategyBacktester:
    """?대깽??湲곕컲 諛깊뀒?ㅽ똿 ?붿쭊"""
    
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
# 3. ?꾨왂 援ы쁽
# ============================================ 

class StrategySignals:
    @staticmethod
    def _atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
        high = data['High']
        low = data['Low']
        close = data['Close']
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs()
            ],
            axis=1
        ).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def _adx(data: pd.DataFrame, period: int = 14) -> pd.Series:
        high = data['High']
        low = data['Low']

        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        plus_dm = pd.Series(plus_dm, index=data.index)
        minus_dm = pd.Series(minus_dm, index=data.index)
        atr = StrategySignals._atr(data, period).replace(0, np.nan)

        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
        return dx.rolling(period).mean()

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

    # --- 5. 理쒖떊/怨좉툒 ?꾨왂 (ML & Structure) ---

    @staticmethod
    def ml_ensemble_signals(params: dict) -> Callable:
        """
        ML ?숈긽釉??꾨왂: RandomForest瑜??댁슜??諛⑺뼢???덉륫
        ?숈뒿 ?⑥쑉???꾪빐 20遊됰쭏???ы븰???섑뻾
        """
        from sklearn.ensemble import RandomForestClassifier
        lookback = int(params.get('lookback', 252))
        retrain_freq = 20
        model_cache = {'model': None, 'last_train_idx': -1}
        
        def signal_func(data, idx):
            if idx < lookback + 50: return 0
            
            # ?쇱쿂 ?앹꽦
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
                
                model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42) # 媛蹂띻쾶 ?ㅼ젙
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
        ?쒖옣 援?㈃ ?꾪솚 ?꾨왂: 蹂?숈꽦 援?㈃???곕씪 異붿꽭異붿쥌/?됯퇏?뚭? ?먮룞 ?ㅼ쐞移?        """
        vol_lookback = int(params.get('vol_lookback', 20))
        threshold = float(params.get('threshold', 1.5))
        
        def signal_func(data, idx):
            if idx < vol_lookback + 50: return 0
            
            returns = data['Close'].pct_change()
            curr_vol = returns.iloc[-vol_lookback:].std()
            avg_vol = returns.rolling(100).std().iloc[-1]
            
            # 1. 怨좊??숈꽦 援?㈃ (異붿꽭 諛쒖깮 媛?μ꽦) -> 紐⑤찘?
            if curr_vol > avg_vol * threshold:
                mom = data['Close'].iloc[-1] / data['Close'].iloc[-20] - 1
                return 1 if mom > 0 else -1
            
            # 2. ?蹂?숈꽦 援?㈃ (諛뺤뒪沅? -> RSI ??텛??            else:
                prices = data['Close']
                rsi = (prices.diff().rolling(14).apply(lambda x: np.sum(x[x>0]) / (np.sum(np.abs(x))+1e-9))).iloc[-1]
                if rsi < 0.3: return 1
                elif rsi > 0.7: return -1
            return 0
        return signal_func

    @staticmethod
    def liquidity_sweep_signals(params: dict) -> Callable:
        """
        ?좊룞???ㅼ쐲(Liquidity Sweep): ?꾩씪 怨?????뚰뙆 ??蹂듦? ?ъ갑 (SMC)
        """
        lookback = int(params.get('lookback', 20))
        
        def signal_func(data, idx):
            if idx < 2: return 0
            
            # 이전 구간의 고점/저점
            prev_high = data['High'].iloc[-lookback-1:-1].max()
            prev_low = data['Low'].iloc[-lookback-1:-1].min()
            
            curr_high = data['High'].iloc[-1]
            curr_low = data['Low'].iloc[-1]
            curr_close = data['Close'].iloc[-1]
            
            # Bullish sweep
            if data['Low'].iloc[-1] < prev_low and curr_close > prev_low:
                return 1
            
            # Bearish sweep
            if data['High'].iloc[-1] > prev_high and curr_close < prev_high:
                return -1
                
            return 0
        return signal_func

    @staticmethod
    def adaptive_ema_adx_signals(params: dict) -> Callable:
        """
        理쒖떊??異붿꽭 異붿쥌:
        EMA 援먯감 + ADX 媛뺣룄 ?꾪꽣濡??〓낫???몄씠利덈? 以꾩엫
        """
        fast_period = int(params.get('fast_period', 20))
        slow_period = int(params.get('slow_period', 100))
        adx_period = int(params.get('adx_period', 14))
        adx_threshold = float(params.get('adx_threshold', 20))

        def signal_func(data, idx):
            if idx < max(slow_period, adx_period * 2):
                return 0

            close = data['Close']
            ema_fast = close.ewm(span=fast_period, adjust=False).mean()
            ema_slow = close.ewm(span=slow_period, adjust=False).mean()
            adx = StrategySignals._adx(data, adx_period)

            trend_up = ema_fast.iloc[-1] > ema_slow.iloc[-1]
            trend_down = ema_fast.iloc[-1] < ema_slow.iloc[-1]

            adx_val = float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 0.0
            if adx_val < adx_threshold:
                return 0

            # ADX媛 ?믪쓣?섎줉 ?좏샇 媛뺣룄瑜??믪뿬 吏꾩엯 ?꾧퀎移?0.3) ?듦낵瑜?蹂댁옣
            strength = float(np.clip((adx_val - adx_threshold) / 20.0, 0.35, 1.0))
            if trend_up:
                return strength
            if trend_down:
                return -strength
            return 0

        return signal_func

    @staticmethod
    def atr_breakout_vol_target_signals(params: dict) -> Callable:
        """
        理쒖떊???뚰뙆 ?꾨왂:
        Donchian ?뚰뙆 + ATR ?꾪꽣 + 蹂?숈꽦 ?寃뚰똿 ?ъ???媛뺣룄
        """
        lookback = int(params.get('lookback', 55))
        atr_period = int(params.get('atr_period', 20))
        atr_filter = float(params.get('atr_filter', 0.8))
        vol_lookback = int(params.get('vol_lookback', 20))
        target_daily_vol = float(params.get('target_daily_vol', 0.012))

        def signal_func(data, idx):
            min_lb = max(lookback + 2, atr_period + 2, vol_lookback + 2)
            if idx < min_lb:
                return 0

            close = data['Close']
            high = data['High']
            low = data['Low']

            upper = high.iloc[-lookback-1:-1].max()
            lower = low.iloc[-lookback-1:-1].min()
            atr = StrategySignals._atr(data, atr_period)
            curr_atr = float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else 0.0
            atr_med = float(atr.iloc[-(atr_period * 2):].median()) if len(atr) >= atr_period * 2 else curr_atr

            # ?덈Т 議곗슜???μ? ?쒖쇅
            if atr_med <= 0 or curr_atr < atr_med * atr_filter:
                return 0

            realized_vol = float(close.pct_change().rolling(vol_lookback).std().iloc[-1])
            if np.isnan(realized_vol) or realized_vol <= 0:
                vol_scale = 0.5
            else:
                vol_scale = float(np.clip(target_daily_vol / realized_vol, 0.35, 1.0))

            curr_close = close.iloc[-1]
            if curr_close > upper:
                return vol_scale
            if curr_close < lower:
                return -vol_scale
            return 0

        return signal_func

    @staticmethod
    def zscore_mean_reversion_signals(params: dict) -> Callable:
        """
        理쒖떊???됯퇏?뚭?:
        Z-Score 湲곕컲 怨쇰ℓ??怨쇰ℓ??+ ?κ린 異붿꽭 ?꾪꽣 寃고빀
        """
        window = int(params.get('window', 30))
        entry_z = float(params.get('entry_z', 2.0))
        exit_z = float(params.get('exit_z', 0.5))
        trend_period = int(params.get('trend_period', 100))

        def signal_func(data, idx):
            if idx < max(window + 2, trend_period + 2):
                return 0

            close = data['Close']
            mean = close.rolling(window).mean()
            std = close.rolling(window).std().replace(0, np.nan)
            z = ((close - mean) / std).fillna(0)

            long_sma = close.rolling(trend_period).mean()
            curr_close = close.iloc[-1]
            curr_z = float(z.iloc[-1])

            # 추세 방향별 평균회귀 진입
            if curr_close > long_sma.iloc[-1]:
                if curr_z <= -entry_z:
                    return 1
                if curr_z >= -exit_z:
                    return 0
            else:
                if curr_z >= entry_z:
                    return -1
                if curr_z <= exit_z:
                    return 0
            return 0

        return signal_func

    @staticmethod
    def macd_regime_signals(params: dict) -> Callable:
        """
        理쒖떊???덉쭚 ?꾨왂:
        MACD ?덉뒪?좉렇??諛⑺뼢 + 蹂?숈꽦 ?덉쭚 ?꾪꽣
        """
        fast = int(params.get('fast', 12))
        slow = int(params.get('slow', 26))
        signal_period = int(params.get('signal_period', 9))
        vol_window = int(params.get('vol_window', 20))
        high_vol_threshold = float(params.get('high_vol_threshold', 1.2))

        def signal_func(data, idx):
            if idx < max(slow + signal_period + 5, vol_window + 5):
                return 0

            close = data['Close']
            ema_fast = close.ewm(span=fast, adjust=False).mean()
            ema_slow = close.ewm(span=slow, adjust=False).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal_period, adjust=False).mean()
            hist = macd - macd_signal

            returns = close.pct_change()
            curr_vol = returns.iloc[-vol_window:].std()
            avg_vol = returns.rolling(100).std().iloc[-1]
            if np.isnan(curr_vol) or np.isnan(avg_vol) or avg_vol <= 0:
                return 0

            # 怨좊????덉쭚?먯꽌??蹂댁닔?곸쑝濡?媛뺥븳 ?덉뒪?좉렇?⑤쭔 梨꾪깮
            if curr_vol > avg_vol * high_vol_threshold:
                if hist.iloc[-1] > 0 and hist.iloc[-1] > hist.iloc[-2]:
                    return 0.7
                if hist.iloc[-1] < 0 and hist.iloc[-1] < hist.iloc[-2]:
                    return -0.7
                return 0

            # ?蹂???덉쭚?먯꽌??援먯감 以묒떖
            if hist.iloc[-1] > 0 and hist.iloc[-2] <= 0:
                return 1
            if hist.iloc[-1] < 0 and hist.iloc[-2] >= 0:
                return -1
            return 0

        return signal_func


    @staticmethod
    def turtle_momentum_confirm_signals(params: dict) -> Callable:
        """
        Turtle 변형:
        채널 돌파 신호를 기본으로, 중기 모멘텀 동조일 때만 진입
        """
        entry_period = int(params.get('entry_period', 20))
        exit_period = int(params.get('exit_period', 10))
        momentum_lookback = int(params.get('momentum_lookback', 60))
        momentum_threshold = float(params.get('momentum_threshold', 0.01))

        def signal_func(data, idx):
            min_lb = max(entry_period + 2, exit_period + 2, momentum_lookback + 2)
            if idx < min_lb:
                return 0

            close = data['Close']
            high = data['High']
            low = data['Low']
            curr_close = close.iloc[-1]

            upper = high.iloc[-entry_period-1:-1].max()
            lower = low.iloc[-exit_period-1:-1].min()
            mom = (curr_close / close.iloc[-momentum_lookback-1]) - 1

            if curr_close > upper and mom > momentum_threshold:
                return 1
            if curr_close < lower and mom < -momentum_threshold:
                return -1

            if abs(mom) < momentum_threshold * 0.5:
                return 0
            return 0.35 if mom > 0 else -0.35

        return signal_func

    @staticmethod
    def rsi2_bollinger_reversion_signals(params: dict) -> Callable:
        """
        RSI2 변형:
        RSI2 과매수/과매도와 볼린저 밴드 이탈을 동시에 만족할 때만 반전 진입
        """
        rsi_period = int(params.get('rsi_period', 2))
        oversold = float(params.get('oversold', 8))
        overbought = float(params.get('overbought', 92))
        bb_window = int(params.get('bb_window', 20))
        bb_std = float(params.get('bb_std', 2.0))
        trend_period = int(params.get('trend_period', 100))

        def signal_func(data, idx):
            min_lb = max(trend_period + 2, bb_window + 2, rsi_period + 3)
            if idx < min_lb:
                return 0

            close = data['Close']
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
            rs = gain / (loss + 1e-12)
            rsi = 100 - (100 / (1 + rs))

            mid = close.rolling(bb_window).mean()
            sigma = close.rolling(bb_window).std()
            upper = mid + bb_std * sigma
            lower = mid - bb_std * sigma
            long_sma = close.rolling(trend_period).mean()

            curr_close = close.iloc[-1]
            curr_rsi = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0

            if curr_close > long_sma.iloc[-1]:
                if curr_close < lower.iloc[-1] and curr_rsi < oversold:
                    return 1
                if curr_close >= mid.iloc[-1]:
                    return 0
            else:
                if curr_close > upper.iloc[-1] and curr_rsi > overbought:
                    return -1
                if curr_close <= mid.iloc[-1]:
                    return 0

            return 0

        return signal_func

    @staticmethod
    def regime_liquidity_sweep_signals(params: dict) -> Callable:
        """
        Regime 변형:
        변동성 레짐 필터 위에 Liquidity Sweep 트리거를 올린 하이브리드 전략
        """
        vol_lookback = int(params.get('vol_lookback', 20))
        regime_threshold = float(params.get('regime_threshold', 1.2))
        sweep_lookback = int(params.get('sweep_lookback', 20))
        confirm_momentum = int(params.get('confirm_momentum', 10))

        def signal_func(data, idx):
            min_lb = max(vol_lookback + 5, sweep_lookback + 5, confirm_momentum + 5, 120)
            if idx < min_lb:
                return 0

            close = data['Close']
            high = data['High']
            low = data['Low']

            returns = close.pct_change()
            curr_vol = returns.iloc[-vol_lookback:].std()
            base_vol = returns.rolling(100).std().iloc[-1]
            if np.isnan(curr_vol) or np.isnan(base_vol) or base_vol <= 0:
                return 0

            high_regime = curr_vol > base_vol * regime_threshold

            prev_high = high.iloc[-sweep_lookback-1:-1].max()
            prev_low = low.iloc[-sweep_lookback-1:-1].min()
            curr_high = high.iloc[-1]
            curr_low = low.iloc[-1]
            curr_close = close.iloc[-1]

            mom = (curr_close / close.iloc[-confirm_momentum-1]) - 1

            bullish_sweep = (curr_low < prev_low) and (curr_close > prev_low)
            bearish_sweep = (curr_high > prev_high) and (curr_close < prev_high)

            if bullish_sweep:
                return 0.7 if high_regime and mom > 0 else 0.45
            if bearish_sweep:
                return -0.7 if high_regime and mom < 0 else -0.45
            return 0

        return signal_func

    @staticmethod
    def adaptive_fractal_regime_signals(params: dict) -> Callable:
        """
        신규 전략:
        - 추세장에서는 Donchian 돌파 추종
        - 횡보장에서는 z-score 평균회귀
        - ATR 기반으로 신호 강도를 조절해 과도한 레버리지성 진입 완화
        """
        trend_lookback = int(params.get('trend_lookback', 55))
        mean_window = int(params.get('mean_window', 20))
        z_entry = float(params.get('z_entry', 1.6))
        chop_window = int(params.get('chop_window', 14))
        chop_threshold = float(params.get('chop_threshold', 58))
        atr_period = int(params.get('atr_period', 14))
        target_daily_vol = float(params.get('target_daily_vol', 0.012))

        def signal_func(data, idx):
            min_lb = max(trend_lookback + 5, mean_window + 5, chop_window + atr_period + 5)
            if idx < min_lb:
                return 0

            close = data['Close']
            high = data['High']
            low = data['Low']

            # Choppiness Index: 높을수록 횡보, 낮을수록 추세
            tr = pd.concat(
                [
                    (high - low),
                    (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs(),
                ],
                axis=1,
            ).max(axis=1)
            tr_sum = tr.rolling(chop_window).sum()
            hh = high.rolling(chop_window).max()
            ll = low.rolling(chop_window).min()
            denom = (hh - ll).replace(0, np.nan)
            chop = 100 * np.log10(tr_sum / denom) / np.log10(chop_window)
            curr_chop = float(chop.iloc[-1]) if not np.isnan(chop.iloc[-1]) else 50.0

            # 변동성 타겟 기반 강도 조절
            realized_vol = float(close.pct_change().rolling(20).std().iloc[-1])
            if np.isnan(realized_vol) or realized_vol <= 0:
                strength = 0.5
            else:
                strength = float(np.clip(target_daily_vol / realized_vol, 0.35, 1.0))

            # ATR이 너무 낮으면 신호를 약화
            atr = StrategySignals._atr(data, atr_period)
            curr_atr = float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else 0.0
            atr_med = float(atr.iloc[-(atr_period * 2):].median()) if len(atr) >= atr_period * 2 else curr_atr
            if atr_med > 0 and curr_atr < atr_med * 0.7:
                strength *= 0.7

            curr_close = close.iloc[-1]

            # 1) 추세장: 돌파 추종
            if curr_chop < chop_threshold:
                upper = high.iloc[-trend_lookback-1:-1].max()
                lower = low.iloc[-trend_lookback-1:-1].min()
                if curr_close > upper:
                    return strength
                if curr_close < lower:
                    return -strength
                return 0

            # 2) 횡보장: 평균회귀
            mean = close.rolling(mean_window).mean()
            std = close.rolling(mean_window).std().replace(0, np.nan)
            z = ((close - mean) / std).fillna(0)
            curr_z = float(z.iloc[-1])
            if curr_z <= -z_entry:
                return strength
            if curr_z >= z_entry:
                return -strength
            return 0

        return signal_func
# ============================================ 
# 4. 理쒖쟻???꾧뎄
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
# 5. 硫붿씤 ?ㅽ뻾 (?앸왂 媛??
# ============================================ 
if __name__ == "__main__":
    pass


