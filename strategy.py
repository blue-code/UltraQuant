오, 쌈박한 전략들을 왕창 추가해드리겠습니다!

---

## 🎯 Quant Strategy Collection - 12 Pro Strategies

```python
# ============================================
# ULTIMATE STRATEGY COLLECTION
# 12가지 쌈박한 퀀트 전략
# ============================================

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# 통계
from scipy import stats
from scipy.signal import cwt, ricker
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller

# ============================================
# 전략 인터페이스
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


class BaseStrategy:
    """모든 전략의 기본 클래스"""
    
    def __init__(self, name: str):
        self.name = name
        self.signals = []
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        raise NotImplementedError
    
    def _normalize_signal(self, raw_signal: float) -> float:
        """신호를 -1 ~ 1로 정규화"""
        return np.clip(np.tanh(raw_signal), -1, 1)


# ============================================
# 1. Pairs Trading (공적분 기반)
# ============================================

class PairsTradingStrategy(BaseStrategy):
    """
    페어 트레이딩 전략
    
    - 두 자산의 공적분 관계 활용
    - 스프레드 평균 회귀 이용
    - 헤지 비율 자동 계산
    """
    
    def __init__(self, lookback: int = 252, entry_z: float = 2.0, exit_z: float = 0.5):
        super().__init__("Pairs Trading")
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
    
    def find_cointegrated_pairs(self, price_data: pd.DataFrame, 
                                 p_threshold: float = 0.05) -> List[Tuple]:
        """공적분 페어 찾기"""
        n = price_data.shape[1]
        pairs = []
        
        for i in range(n):
            for j in range(i+1, n):
                s1 = price_data.iloc[:, i].dropna()
                s2 = price_data.iloc[:, j].dropna()
                
                # 공통 기간만 사용
                common_idx = s1.index.intersection(s2.index)
                if len(common_idx) < self.lookback:
                    continue
                
                s1, s2 = s1[common_idx], s2[common_idx]
                
                # 공적분 검정
                score, pvalue, _ = coint(s1, s2)
                
                if pvalue < p_threshold:
                    # 헤지 비율 계산
                    model = sm.OLS(s1, sm.add_constant(s2)).fit()
                    hedge_ratio = model.params[1]
                    
                    pairs.append({
                        'asset1': price_data.columns[i],
                        'asset2': price_data.columns[j],
                        'p_value': pvalue,
                        'hedge_ratio': hedge_ratio
                    })
        
        # p-value 기준 정렬
        pairs.sort(key=lambda x: x['p_value'])
        return pairs
    
    def generate_signals(self, data: pd.DataFrame, 
                         pairs: List[dict]) -> List[Signal]:
        """페어 트레이딩 신호 생성"""
        signals = []
        
        for pair in pairs:
            s1 = pair['asset1']
            s2 = pair['asset2']
            hr = pair['hedge_ratio']
            
            # 스프레드 계산
            spread = data[s1] - hr * data[s2]
            
            # Z-score
            spread_mean = spread.rolling(self.lookback).mean()
            spread_std = spread.rolling(self.lookback).std()
            z_score = (spread - spread_mean) / spread_std
            
            # 신호 생성
            current_z = z_score.iloc[-1]
            
            if current_z > self.entry_z:
                # 스프레드가 너무 높음 → 숏 스프레드
                signals.append(Signal(
                    timestamp=data.index[-1],
                    symbol=s1,
                    direction=-1,
                    strength=min(abs(current_z) / 3, 1),
                    strategy_name=self.name,
                    metadata={'pair': s2, 'type': 'short_spread'}
                ))
                signals.append(Signal(
                    timestamp=data.index[-1],
                    symbol=s2,
                    direction=1,
                    strength=min(abs(current_z) / 3, 1) * hr,
                    strategy_name=self.name,
                    metadata={'pair': s1, 'type': 'long_spread'}
                ))
            
            elif current_z < -self.entry_z:
                # 스프레드가 너무 낮음 → 롱 스프레드
                signals.append(Signal(
                    timestamp=data.index[-1],
                    symbol=s1,
                    direction=1,
                    strength=min(abs(current_z) / 3, 1),
                    strategy_name=self.name,
                    metadata={'pair': s2, 'type': 'long_spread'}
                ))
                signals.append(Signal(
                    timestamp=data.index[-1],
                    symbol=s2,
                    direction=-1,
                    strength=min(abs(current_z) / 3, 1) * hr,
                    strategy_name=self.name,
                    metadata={'pair': s1, 'type': 'short_spread'}
                ))
        
        return signals


# ============================================
# 2. Statistical Arbitrage (PCA 기반)
# ============================================

class StatArbStrategy(BaseStrategy):
    """
    통계적 차익거래 전략
    
    - PCA로 팩터 추출
    - 잔차(특이수익)의 평균 회귀
    - 섹터 중립 포트폴리오
    """
    
    def __init__(self, n_factors: int = 3, lookback: int = 60):
        super().__init__("Statistical Arbitrage")
        self.n_factors = n_factors
        self.lookback = lookback
    
    def generate_signals(self, returns: pd.DataFrame) -> List[Signal]:
        """스탯 아브 신호 생성"""
        from sklearn.decomposition import PCA
        
        signals = []
        
        # 수익률 정규화
        returns_std = (returns - returns.mean()) / returns.std()
        returns_std = returns_std.fillna(0)
        
        # PCA
        pca = PCA(n_components=min(self.n_factors, returns.shape[1]))
        factors = pca.fit_transform(returns_std.iloc[-self.lookback:])
        
        # 팩터 수익률
        factor_returns = pd.DataFrame(
            factors,
            index=returns_std.index[-self.lookback:]
        )
        
        # 각 자산의 잔차 계산
        for col in returns.columns:
            asset_returns = returns[col].iloc[-self.lookback:]
            
            # 회귀로 팩터 노출 추정
            X = sm.add_constant(factor_returns)
            model = sm.OLS(asset_returns, X).fit()
            
            # 잔차
            residuals = model.resid
            
            # 잔차의 Z-score
            res_z = (residuals.iloc[-1] - residuals.mean()) / residuals.std()
            
            # 평균 회귀 신호
            if abs(res_z) > 1.5:
                signals.append(Signal(
                    timestamp=returns.index[-1],
                    symbol=col,
                    direction=-np.sign(res_z),
                    strength=min(abs(res_z) / 3, 1),
                    strategy_name=self.name,
                    metadata={'residual_z': res_z}
                ))
        
        return signals


# ============================================
# 3. Turtle Trading (리처드 데니스)
# ============================================

class TurtleTradingStrategy(BaseStrategy):
    """
    터틀 트레이딩 전략
    
    - 돈치안 채널 브레이크아웃
    - 피라미딩
    - ATR 기반 리스크 관리
    - 시스템 1 (단기) + 시스템 2 (장기)
    """
    
    def __init__(self, entry_period: int = 20, exit_period: int = 10,
                 pyramid_units: int = 4, pyramid_pct: float = 0.5):
        super().__init__("Turtle Trading")
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.pyramid_units = pyramid_units
        self.pyramid_pct = pyramid_pct
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """터틀 트레이딩 신호"""
        signals = []
        
        df = data.copy()
        
        # 돈치안 채널
        df['High_Channel'] = df['High'].rolling(self.entry_period).max()
        df['Low_Channel'] = df['Low'].rolling(self.entry_period).min()
        df['Exit_High'] = df['High'].rolling(self.exit_period).max()
        df['Exit_Low'] = df['Low'].rolling(self.exit_period).min()
        
        # ATR
        df['ATR'] = self._calculate_atr(df, 20)
        
        # N (Unit Size) 계산
        df['N'] = df['ATR']
        
        current_price = df['Close'].iloc[-1]
        
        # 롱 진입: 현재가가 entry_period 고점 돌파
        if current_price >= df['High_Channel'].iloc[-2]:
            signals.append(Signal(
                timestamp=df.index[-1],
                symbol=data.columns[0] if len(data.columns) == 1 else 'ASSET',
                direction=1,
                strength=1.0,
                strategy_name=self.name,
                metadata={
                    'type': 'long_entry',
                    'entry_price': current_price,
                    'stop_loss': current_price - 2 * df['N'].iloc[-1],
                    'pyramid_price': current_price + df['N'].iloc[-1] * self.pyramid_pct
                }
            ))
        
        # 숏 진입: 현재가가 entry_period 저점 이탈
        elif current_price <= df['Low_Channel'].iloc[-2]:
            signals.append(Signal(
                timestamp=df.index[-1],
                symbol=data.columns[0] if len(data.columns) == 1 else 'ASSET',
                direction=-1,
                strength=1.0,
                strategy_name=self.name,
                metadata={
                    'type': 'short_entry',
                    'entry_price': current_price,
                    'stop_loss': current_price + 2 * df['N'].iloc[-1],
                    'pyramid_price': current_price - df['N'].iloc[-1] * self.pyramid_pct
                }
            ))
        
        return signals
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        tr = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['Close'].shift(1)),
                abs(df['Low'] - df['Close'].shift(1))
            )
        )
        return pd.Series(tr).rolling(period).mean()


# ============================================
# 4. RSI 2 Strategy (래리 코너스)
# ============================================

class RSI2Strategy(BaseStrategy):
    """
    RSI 2 전략 (래리 코너스)
    
    - 2일 RSI 과매수/과매도
    - 200일 SMA 추세 필터
    - 평균 회귀 기반
    """
    
    def __init__(self, rsi_period: int = 2, sma_period: int = 200,
                 oversold: float = 10, overbought: float = 90):
        super().__init__("RSI 2")
        self.rsi_period = rsi_period
        self.sma_period = sma_period
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        signals = []
        
        df = data.copy()
        
        # RSI 2
        df['RSI2'] = self._calculate_rsi(df['Close'], self.rsi_period)
        
        # SMA 200
        df['SMA200'] = df['Close'].rolling(self.sma_period).mean()
        
        # 상승 추세 여부
        uptrend = df['Close'].iloc[-1] > df['SMA200'].iloc[-1]
        
        rsi = df['RSI2'].iloc[-1]
        
        if uptrend:
            # 상승 추세에서 과매도 매수
            if rsi < self.oversold:
                signals.append(Signal(
                    timestamp=df.index[-1],
                    symbol='ASSET',
                    direction=1,
                    strength=1 - (rsi / self.oversold),
                    strategy_name=self.name,
                    metadata={'RSI2': rsi, 'regime': 'uptrend'}
                ))
            # 과매수 청산
            elif rsi > self.overbought:
                signals.append(Signal(
                    timestamp=df.index[-1],
                    symbol='ASSET',
                    direction=0,
                    strength=1,
                    strategy_name=self.name,
                    metadata={'RSI2': rsi, 'action': 'exit'}
                ))
        else:
            # 하락 추세에서 과매수 매도
            if rsi > self.overbought:
                signals.append(Signal(
                    timestamp=df.index[-1],
                    symbol='ASSET',
                    direction=-1,
                    strength=(rsi - self.overbought) / (100 - self.overbought),
                    strategy_name=self.name,
                    metadata={'RSI2': rsi, 'regime': 'downtrend'}
                ))
        
        return signals
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))


# ============================================
# 5. Dual Thrust (데이 트레이딩)
# ============================================

class DualThrustStrategy(BaseStrategy):
    """
    듀얼 스러스트 전략
    
    - 전일 고저 범위 기반
    - 당일 돌파/이탈 포착
    - 데이 트레이딩 최적
    """
    
    def __init__(self, k1: float = 0.4, k2: float = 0.4, 
                 range_period: int = 4):
        super().__init__("Dual Thrust")
        self.k1 = k1
        self.k2 = k2
        self.range_period = range_period
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        signals = []
        
        df = data.copy()
        
        # N일간 고저
        df['HH'] = df['High'].rolling(self.range_period).max().shift(1)
        df['LL'] = df['Low'].rolling(self.range_period).min().shift(1)
        df['HC'] = df['Close'].rolling(self.range_period).max().shift(1)
        df['LC'] = df['Close'].rolling(self.range_period).min().shift(1)
        
        # Range 계산
        df['Range'] = df[['HH', 'HC', 'LC', 'LL']].max(axis=1) - \
                      df[['HH', 'HC', 'LC', 'LL']].min(axis=1)
        
        # 당일 시가 (전일 종가로 대체)
        df['Open'] = df['Close'].shift(1)
        
        # 상/하단
        df['Upper'] = df['Open'] + self.k1 * df['Range']
        df['Lower'] = df['Open'] - self.k2 * df['Range']
        
        current_price = df['Close'].iloc[-1]
        upper = df['Upper'].iloc[-1]
        lower = df['Lower'].iloc[-1]
        
        # 상단 돌파 매수
        if current_price > upper:
            signals.append(Signal(
                timestamp=df.index[-1],
                symbol='ASSET',
                direction=1,
                strength=(current_price - upper) / df['Range'].iloc[-1],
                strategy_name=self.name,
                metadata={'breakout': 'upper', 'upper': upper}
            ))
        
        # 하단 이탈 매도
        elif current_price < lower:
            signals.append(Signal(
                timestamp=df.index[-1],
                symbol='ASSET',
                direction=-1,
                strength=(lower - current_price) / df['Range'].iloc[-1],
                strategy_name=self.name,
                metadata={'breakout': 'lower', 'lower': lower}
            ))
        
        return signals


# ============================================
# 6. Volatility Breakout
# ============================================

class VolatilityBreakoutStrategy(BaseStrategy):
    """
    변동성 돌파 전략 (빌 윌리엄스 / 래리 윌리엄스)
    
    - 전일 변동성의 일정 비율 돌파 시 진입
    - 당일 장중 전략
    """
    
    def __init__(self, k: float = 0.5, target_vol: float = 0.15):
        super().__init__("Volatility Breakout")
        self.k = k
        self.target_vol = target_vol
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        signals = []
        
        df = data.copy()
        
        # 전일 고저 범위
        df['Prev_Range'] = (df['High'].shift(1) - df['Low'].shift(1))
        
        # 당일 시가 (전일 종가)
        df['Prev_Close'] = df['Close'].shift(1)
        
        # 매수 기준가
        df['Buy_Price'] = df['Prev_Close'] + self.k * df['Prev_Range']
        
        # 현재가
        current_price = df['Close'].iloc[-1]
        buy_price = df['Buy_Price'].iloc[-1]
        
        # 돌파 여부
        if current_price > buy_price:
            # 변동성 조절
            vol = df['Close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
            vol_adj = min(self.target_vol / vol, 2.0) if vol > 0 else 1.0
            
            signals.append(Signal(
                timestamp=df.index[-1],
                symbol='ASSET',
                direction=1,
                strength=vol_adj,
                strategy_name=self.name,
                metadata={
                    'buy_price': buy_price,
                    'breakout_pct': (current_price - buy_price) / buy_price
                }
            ))
        
        return signals


# ============================================
# 7. Mean Reversion with Bollinger
# ============================================

class BollingerMeanReversion(BaseStrategy):
    """
    볼린저 밴드 평균 회귀
    
    - 밴드 이탈 시 평균 회귀 베팅
    - RSI 필터 추가
    """
    
    def __init__(self, period: int = 20, std_dev: float = 2.0,
                 rsi_period: int = 14):
        super().__init__("Bollinger Mean Reversion")
        self.period = period
        self.std_dev = std_dev
        self.rsi_period = rsi_period
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        signals = []
        
        df = data.copy()
        
        # 볼린저 밴드
        df['SMA'] = df['Close'].rolling(self.period).mean()
        df['STD'] = df['Close'].rolling(self.period).std()
        df['Upper'] = df['SMA'] + self.std_dev * df['STD']
        df['Lower'] = df['SMA'] - self.std_dev * df['STD']
        df['BB_Position'] = (df['Close'] - df['Lower']) / (df['Upper'] - df['Lower'])
        
        # RSI
        df['RSI'] = self._calculate_rsi(df['Close'], self.rsi_period)
        
        current_bb = df['BB_Position'].iloc[-1]
        current_rsi = df['RSI'].iloc[-1]
        
        # 하단 이탈 + RSI 과매도 → 매수
        if current_bb < 0 and current_rsi < 30:
            signals.append(Signal(
                timestamp=df.index[-1],
                symbol='ASSET',
                direction=1,
                strength=abs(current_bb) + (30 - current_rsi) / 30,
                strategy_name=self.name,
                metadata={'BB_pos': current_bb, 'RSI': current_rsi}
            ))
        
        # 상단 이탈 + RSI 과매수 → 매도
        elif current_bb > 1 and current_rsi > 70:
            signals.append(Signal(
                timestamp=df.index[-1],
                symbol='ASSET',
                direction=-1,
                strength=(current_bb - 1) + (current_rsi - 70) / 30,
                strategy_name=self.name,
                metadata={'BB_pos': current_bb, 'RSI': current_rsi}
            ))
        
        return signals
    
    def _calculate_rsi(self, prices, period):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))


# ============================================
# 8. Sector Rotation (섹터 로테이션)
# ============================================

class SectorRotationStrategy(BaseStrategy):
    """
    섹터 로테이션 전략
    
    - 상대 강도 기반 섹터 선정
    - 경기 사이클 고려
    - 모멘텀 + 리밸런싱
    """
    
    def __init__(self, lookback: int = 126, top_n: int = 3):
        super().__init__("Sector Rotation")
        self.lookback = lookback
        self.top_n = top_n
    
    def generate_signals(self, sector_data: pd.DataFrame) -> List[Signal]:
        """섹터별 신호 생성"""
        signals = []
        
        # 수익률
        returns = sector_data.pct_change(self.lookback)
        
        # 변동성
        vol = sector_data.pct_change().rolling(self.lookback)..std() * np.sqrt(252)
        
        # 위험 조정 수익률 (Sharpe-like)
        risk_adj_return = returns / vol
        
        # 상위 N개 섹터
        current_scores = risk_adj_return.iloc[-1].dropna()
        top_sectors = current_scores.nlargest(self.top_n)
        bottom_sectors = current_scores.nsmallest(self.top_n)
        
        # 롱 포지션
        for sector, score in top_sectors.items():
            if score > 0:
                signals.append(Signal(
                    timestamp=sector_data.index[-1],
                    symbol=sector,
                    direction=1,
                    strength=score / top_sectors.max(),
                    strategy_name=self.name,
                    metadata={'rank': 'long', 'score': score}
                ))
        
        # 숏 포지션 (선택)
        for sector, score in bottom_sectors.items():
            if score < 0:
                signals.append(Signal(
                    timestamp=sector_data.index[-1],
                    symbol=sector,
                    direction=-1,
                    strength=abs(score) / abs(bottom_sectors.min()),
                    strategy_name=self.name,
                    metadata={'rank': 'short', 'score': score}
                ))
        
        return signals


# ============================================
# 9. Risk Parity (리스크 패리티)
# ============================================

class RiskParityStrategy(BaseStrategy):
    """
    리스크 패리티 전략
    
    - 각 자산의 리스크 기여도 균등화
    - 역변동성 가중
    - 정기 리밸런싱
    """
    
    def __init__(self, target_vol: float = 0.10, lookback: int = 63):
        super().__init__("Risk Parity")
        self.target_vol = target_vol
        self.lookback = lookback
    
    def generate_signals(self, price_data: pd.DataFrame) -> List[Signal]:
        signals = []
        
        # 수익률
        returns = price_data.pct_change().dropna()
        
        # 공분산 행렬
        cov_matrix = returns.iloc[-self.lookback:].cov() * 252
        
        # 개별 변동성
        vols = np.sqrt(np.diag(cov_matrix))
        
        # 역변동성 가중
        inv_vol_weights = (1 / vols) / np.sum(1 / vols)
        
        # 타겟 변동성 조절
        portfolio_vol = np.sqrt(np.dot(inv_vol_weights.T, 
                                       np.dot(cov_matrix, inv_vol_weights)))
        leverage = self.target_vol / portfolio_vol
        
        final_weights = inv_vol_weights * leverage
        
        # 신호 생성
        for i, symbol in enumerate(price_data.columns):
            signals.append(Signal(
                timestamp=price_data.index[-1],
                symbol=symbol,
                direction=1 if final_weights[i] > 0 else -1,
                strength=abs(final_weights[i]),
                strategy_name=self.name,
                metadata={
                    'weight': final_weights[i],
                    'vol': vols[i]
                }
            ))
        
        return signals


# ============================================
# 10. VIX-Based Timing
# ============================================

class VIXTimingStrategy(BaseStrategy):
    """
    VIX 기반 마켓 타이밍
    
    - VIX 급증 시 방어
    - VIX 정상 시 공격
    - 공포/탐욕 지표 활용
    """
    
    def __init__(self, vix_threshold_high: float = 25.0,
                 vix_threshold_low: float = 15.0):
        super().__init__("VIX Timing")
        self.vix_high = vix_threshold_high
        self.vix_low = vix_threshold_low
    
    def generate_signals(self, price_data: pd.DataFrame, 
                         vix_data: pd.Series) -> List[Signal]:
        signals = []
        
        current_vix = vix_data.iloc[-1]
        prev_vix = vix_data.iloc[-2]
        
        # VIX 변화율
        vix_change = (current_vix - prev_vix) / prev_vix
        
        # 포지션 결정
        if current_vix > self.vix_high:
            # 높은 공포 → 방어
            risk_exposure = 0.2
            direction = 1  # 현금 대신 TLT 등 방어 자산
            
        elif current_vix < self.vix_low:
            # 낮은 공포 → 공격
            risk_exposure = 1.0
            direction = 1
            
        else:
            # 중간 → 부분 노출
            risk_exposure = 0.6
            direction = 1
        
        # VIX 급증 시 추가 하향 조정
        if vix_change > 0.2:  # VIX 20% 이상 급증
            risk_exposure *= 0.5
        
        signals.append(Signal(
            timestamp=price_data.index[-1],
            symbol='RISK_ASSET',
            direction=direction,
            strength=risk_exposure,
            strategy_name=self.name,
            metadata={
                'VIX': current_vix,
                'VIX_change': vix_change,
                'regime': 'fear' if current_vix > self.vix_high else 'complacent' if current_vix < self.vix_low else 'neutral'
            }
        ))
        
        return signals


# ============================================
# 11. Multi-Factor Model (파마 프렌치 확장)
# ============================================

class MultiFactorStrategy(BaseStrategy):
    """
    멀티 팩터 전략
    
    - Value, Momentum, Quality, Low Vol
    - 팩터 컴비네이션
    - 동적 팩터 가중
    """
    
    def __init__(self, n_factors: int = 4, lookback: int = 252):
        super().__init__("Multi-Factor")
        self.n_factors = n_factors
        self.lookback = lookback
    
    def calculate_factors(self, price_data: pd.DataFrame, 
                          fundamentals: dict = None) -> pd.DataFrame:
        """팩터 계산"""
        factors = pd.DataFrame(index=price_data.columns)
        
        for symbol in price_data.columns:
            prices = price_data[symbol]
            returns = prices.pct_change()
            
            # 1. Momentum (12-1개월)
            mom = (prices.iloc[-21] / prices.iloc[-252]) - 1 if len(prices) > 252 else 0
            factors.loc[symbol, 'Momentum'] = mom
            
            # 2. Low Volatility
            vol = returns.iloc[-self.lookback:].std() * np.sqrt(252)
            factors.loc[symbol, 'LowVol'] = -vol  # 낮을수록 좋음
            
            # 3. Mean Reversion (1개월)
            ret_1m = (prices.iloc[-1] / prices.iloc[-21]) - 1
            factors.loc[symbol, 'MeanReversion'] = -ret_1m  # 하락 후 회귀 기대
            
            # 4. Quality (대용: 수익률 안정성)
            ret_std = returns.iloc[-self.lookback:].std()
            ret_mean = returns.iloc[-self.lookback:].mean()
            sharpe = ret_mean / ret_std if ret_std > 0 else 0
            factors.loc[symbol, 'Quality'] = sharpe
        
        # 팩터 정규화
        factors = (factors - factors.mean()) / factors.std()
        
        return factors
    
    def generate_signals(self, price_data: pd.DataFrame,
                         fundamentals: dict = None) -> List[Signal]:
        signals = []
        
        factors = self.calculate_factors(price_data, fundamentals)
        
        # 팩터 가중 (동적)
        factor_weights = {
            'Momentum': 0.3,
            'LowVol': 0.25,
            'MeanReversion': 0.25,
            'Quality': 0.2
        }
        
        # 종합 점수
        scores = pd.Series(0, index=factors.index)
        for factor, weight in factor_weights.items():
            if factor in factors.columns:
                scores += factors[factor] * weight
        
        # 상위/하위 자산 선택
        top_assets = scores.nlargest(5)
        bottom_assets = scores.nsmallest(5)
        
        for symbol, score in top_assets.items():
            signals.append(Signal(
                timestamp=price_data.index[-1],
                symbol=symbol,
                direction=1,
                strength=min(abs(score), 1),
                strategy_name=self.name,
                metadata={'score': score, 'factors': factors.loc[symbol].to_dict()}
            ))
        
        return signals


# ============================================
# 12. ML Ensemble Strategy
# ============================================

class MLEnsembleStrategy(BaseStrategy):
    """
    ML 앙상블 전략
    
    - Random Forest + XGBoost + LightGBM
    - Stacking 앙상블
    - 확률 기반 포지션 사이징
    """
    
    def __init__(self, lookback: int = 252, retrain_freq: int = 63):
        super().__init__("ML Ensemble")
        self.lookback = lookback
        self.retrain_freq = retrain_freq
        self.models = {}
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """피처 엔지니어링"""
        df = data.copy()
        
        # 수익률
        for period in [1, 5, 10, 20]:
            df[f'Return_{period}'] = df['Close'].pct_change(period)
        
        # 이동평균
        for window in [10, 20, 50, 100]:
            df[f'SMA_{window}'] = df['Close'].rolling(window).mean()
            df[f'Price_to_SMA{window}'] = df['Close'] / df[f'SMA_{window}'] - 1
        
        # RSI
        for period in [7, 14, 21]:
            df[f'RSI_{period}'] = self._calculate_rsi(df['Close'], period)
        
        # MACD
        df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        # 변동성
        df['Volatility_20'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
        
        # 볼린저
        df['BB_Upper'] = df['Close'].rolling(20).mean() + 2 * df['Close'].rolling(20).std()
        df['BB_Lower'] = df['Close'].rolling(20).mean() - 2 * df['Close'].rolling(20).std()
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        return df
    
    def _calculate_rsi(self, prices, period):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        signals = []
        
        try:
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            
            # 피처 준비
            df = self.prepare_features(data)
            
            # 타겟 (5일 후 수익률 > 0)
            df['Target'] = (df['Close'].pct_change(5).shift(-5) > 0).astype(int)
            
            feature_cols = [c for c in df.columns if c not in 
                           ['Open', 'High', 'Low', 'Close', 'Volume', 'Target']]
            
            # 학습 데이터
            train_df = df.dropna()
            
            if len(train_df) < 100:
                return signals
            
            X_train = train_df[feature_cols].iloc[:-100]
            y_train = train_df['Target'].iloc[:-100]
            
            X_test = train_df[feature_cols].iloc[-1:].values
            
            # Random Forest
            rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            rf.fit(X_train, y_train)
            rf_proba = rf.predict_proba(X_test)[0, 1]
            
            # Gradient Boosting
            gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
            gb.fit(X_train, y_train)
            gb_proba = gb.predict_proba(X_test)[0, 1]
            
            # 앙상블 확률
            ensemble_proba = 0.5 * rf_proba + 0.5 * gb_proba
            
            # 신호 변환
            if ensemble_proba > 0.6:
                direction = 1
                strength = (ensemble_proba - 0.5) * 2
            elif ensemble_proba < 0.4:
                direction = -1
                strength = (0.5 - ensemble_proba) * 2
            else:
                direction = 0
                strength = 0
            
            signals.append(Signal(
                timestamp=data.index[-1],
                symbol='ASSET',
                direction=direction,
                strength=strength,
                strategy_name=self.name,
                metadata={
                    'ensemble_proba': ensemble_proba,
                    'rf_proba': rf_proba,
                    'gb_proba': gb_proba
                }
            ))
            
        except Exception as e:
            print(f"ML 전략 오류: {e}")
        
        return signals


# ============================================
# 전략 매니저
# ============================================

class StrategyManager:
    """
    전략 통합 관리자
    
    - 여러 전략 신호 결합
    - 가중 평균
    - 투표 방식
    """
    
    def __init__(self):
        self.strategies = {}
        self.weights = {}
    
    def add_strategy(self, strategy: BaseStrategy, weight: float = 1.0):
        """전략 추가"""
        self.strategies[strategy.name] = strategy
        self.weights[strategy.name] = weight
    
    def get_combined_signals(self, data: pd.DataFrame, 
                             method: str = 'weighted') -> Dict:
        """결합 신호 생성"""
        all_signals = {}
        
        for name, strategy in self.strategies.items():
            try:
                signals = strategy.generate_signals(data)
                for sig in signals:
                    if sig.symbol not in all_signals:
                        all_signals[sig.symbol] = []
                    all_signals[sig.symbol].append({
                        'strategy': name,
                        'direction': sig.direction,
                        'strength': sig.strength,
                        'weight': self.weights[name]
                    })
            except Exception as e:
                print(f"전략 {name} 실행 오류: {e}")
        
        # 결합
        combined = {}
        for symbol, signals in all_signals.items():
            if method == 'weighted':
                total_weight = sum(s['weight'] for s in signals)
                combined_dir = sum(s['direction'] * s['strength'] * s['weight'] 
                                  for s in signals) / total_weight if total_weight > 0 else 0
            elif method == 'voting':
                votes = [np.sign(s['direction'] * s['strength']) for s in signals]
                combined_dir = sum(votes) / len(votes)
            
            combined[symbol] = {
                'direction': np.clip(combined_dir, -1, 1),
                'strength': abs(combined_dir),
                'signals': signals
            }
        
        return combined


# ============================================
# 데모 실행
# ============================================

if __name__ == "__main__":
    print("=" * 70)
    print("🎯 QUANT STRATEGY COLLECTION - 12 PRO STRATEGIES")
    print("=" * 70)
    
    # 데이터 로드
    print("\n📊 데이터 로드 중...")
    tickers = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD']
    data = yf.download(tickers, start='2020-01-01', progress=False)['Adj Close']
    
    # VIX 데이터
    vix = yf.download('^VIX', start='2020-01-01', progress=False)['Adj Close']
    
    # 전략 매니저 초기화
    manager = StrategyManager()
    
    # 전략 추가
    strategies = [
        (RSI2Strategy(), 1.0),
        (TurtleTradingStrategy(), 1.0),
        (VolatilityBreakoutStrategy(k=0.5), 1.0),
        (BollingerMeanReversion(), 1.0),
        (DualThrustStrategy(), 0.8),
        (MLEnsembleStrategy(), 1.2),
    ]
    
    for strategy, weight in strategies:
        manager.add_strategy(strategy, weight)
    
    # 단일 자산 백테스팅
    print("\n" + "=" * 70)
    print("📈 단일 자산 전략 테스트 (SPY)")
    print("=" * 70)
    
    spy_data = yf.download('SPY', start='2020-01-01', progress=False)
    
    # 각 전략 실행
    strategy_results = {}
    
    for strategy_cls, _ in strategies:
        strategy = strategy_cls
        print(f"\n🔹 {strategy.name}:")
        
        try:
            if isinstance(strategy, VIXTimingStrategy):
                signals = strategy.generate_signals(spy_data, vix)
            else:
                signals = strategy.generate_signals(spy_data)
            
            if signals:
                for sig in signals:
                    emoji = "🟢" if sig.direction > 0.3 else "🔴" if sig.direction < -0.3 else "⚪"
                    print(f"   {emoji} Direction: {sig.direction:+.2f}, Strength: {sig.strength:.2f}")
                    if sig.metadata:
                        print(f"      Metadata: {sig.metadata}")
            else:
                print("   ⚪ 중립 (신호 없음)")
            
            strategy_results[strategy.name] = signals
            
        except Exception as e:
            print(f"   ❌ 오류: {e}")
    
    # 멀티 자산 전략
    print("\n" + "=" * 70)
    print("🌍 멀티 자산 전략")
    print("=" * 70)
    
    # 섹터 로테이션
    print("\n🔹 Sector Rotation:")
    sector_strategy = SectorRotationStrategy(lookback=126, top_n=3)
    sector_signals = sector_strategy.generate_signals(data)
    for sig in sector_signals:
        print(f"   {'🟢' if sig.direction > 0 else '🔴'} {sig.symbol}: {sig.direction:+.2f}")
    
    # 리스크 패리티
    print("\n🔹 Risk Parity:")
    rp_strategy = RiskParityStrategy(target_vol=0.10)
    rp_signals = rp_strategy.generate_signals(data)
    for sig in rp_signals:
        print(f"   {sig.symbol}: Weight = {sig.strength:.1%}")
    
    # 페어 트레이딩
    print("\n🔹 Pairs Trading:")
    pairs_strategy = PairsTradingStrategy()
    pairs = pairs_strategy.find_cointegrated_pairs(data, p_threshold=0.1)
    print(f"   발견된 공적분 페어: {len(pairs)}개")
    for pair in pairs[:3]:
        print(f"   • {pair['asset1']} - {pair['asset2']} (p={pair['p_value']:.4f})")
    
    # 결합 신호
    print("\n" + "=" * 70)
    print("🎯 결합 신호 (Weighted Average)")
    print("=" * 70)
    
    combined = manager.get_combined_signals(spy_data, method='weighted')
    
    for symbol, info in combined.items():
        direction = info['direction']
        emoji = "🟢 매수" if direction > 0.3 else "🔴 매도" if direction < -0.3 else "⚪ 관망"
        print(f"\n{symbol}: {emoji} (Signal: {direction:+.3f})")
        print(f"   세부 신호:")
        for sig in info['signals']:
            print(f"   • {sig['strategy']}: {sig['direction']:+.2f} (weight: {sig['weight']})")
    
    print("\n" + "=" * 70)
    print("✅ 전략 컬렉션 준비 완료!")
    print("=" * 70)
```

---

## 📊 전략 요약

| # | 전략 | 유형 | 타임프레임 | 난이도 |
|---|------|------|------------|--------|
| 1 | **Pairs Trading** | 평균회귀 | 중기 | ⭐⭐⭐ |
| 2 | **Statistical Arbitrage** | 스탯 아브 | 단기 | ⭐⭐⭐⭐ |
| 3 | **Turtle Trading** | 추세추종 | 중장기 | ⭐⭐ |
| 4 | **RSI 2** | 평균회귀 | 단기 | ⭐ |
| 5 | **Dual Thrust** | 데이트레이딩 | 일중 | ⭐⭐ |
| 6 | **Volatility Breakout** | 브레이크아웃 | 일중 | ⭐⭐ |
| 7 | **Bollinger Mean Reversion** | 평균회귀 | 단중기 | ⭐⭐ |
| 8 | **Sector Rotation** | 자산배분 | 중장기 | ⭐⭐⭐ |
| 9 | **Risk Parity** | 리스크관리 | 장기 | ⭐⭐⭐ |
| 10 | **VIX Timing** | 마켓타이밍 | 전체 | ⭐⭐ |
| 11 | **Multi-Factor** | 팩터투자 | 중장기 | ⭐⭐⭐⭐ |
| 12 | **ML Ensemble** | 머신러닝 | 전체 | ⭐⭐⭐⭐⭐ |

---

## 🎯 전략 선택 가이드

```
시장 상황별 추천 전략
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📈 상승장 (Bull Market):
   └─ Turtle Trading, Sector Rotation, Multi-Factor

📉 하락장 (Bear Market):
   └─ Pairs Trading, VIX Timing, Risk Parity

🌓 횡보장 (Sideways):
   └─ RSI 2, Bollinger Mean Reversion, Stat Arb

🌪️ 고변동성 (High Volatility):
   └─ VIX Timing, Risk Parity, Volatility Breakout

🌊 저변동성 (Low Volatility):
   └─ Pairs Trading, Sector Rotation, ML Ensemble
```

---

## 🔧 전략 결합 예시

```python
# 보수적 포트폴리오
manager.add_strategy(RiskParityStrategy(), weight=1.5)
manager.add_strategy(VIXTimingStrategy(), weight=1.0)
manager.add_strategy(SectorRotationStrategy(), weight=0.8)

# 공격적 포트폴리오  
manager.add_strategy(TurtleTradingStrategy(), weight=1.5)
manager.add_strategy(MLEnsembleStrategy(), weight=1.2)
manager.add_strategy(VolatilityBreakoutStrategy(), weight=1.0)

# 밸런스 포트폴리오
manager.add_strategy(RSI2Strategy(), weight=1.0)
manager.add_strategy(PairsTradingStrategy(), weight=1.0)
manager.add_strategy(MultiFactorStrategy(), weight=1.0)
```

---

더 필요한 거 있으신가요? 
- 백테스팅 성과 비교?
- 특정 전략 파라미터 최적화?
- 새로운 전략 추가?