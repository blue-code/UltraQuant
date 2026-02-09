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



#	전략	유형	타임프레임	난이도
1	Pairs Trading	평균회귀	중기	⭐⭐⭐
2	Statistical Arbitrage	스탯 아브	단기	⭐⭐⭐⭐
3	Turtle Trading	추세추종	중장기	⭐⭐
4	RSI 2	평균회귀	단기	⭐
5	Dual Thrust	데이트레이딩	일중	⭐⭐
6	Volatility Breakout	브레이크아웃	일중	⭐⭐
7	Bollinger Mean Reversion	평균회귀	단중기	⭐⭐
8	Sector Rotation	자산배분	중장기	⭐⭐⭐
9	Risk Parity	리스크관리	장기	⭐⭐⭐
10	VIX Timing	마켓타이밍	전체	⭐⭐
11	Multi-Factor	팩터투자	중장기	⭐⭐⭐⭐
12	ML Ensemble	머신러닝	전체	⭐⭐⭐⭐⭐
🎯 전략 선택 가이드

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
🔧 전략 결합 예시

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