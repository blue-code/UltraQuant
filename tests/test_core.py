import unittest
import numpy as np
import pandas as pd
from strategy import StrategyBacktester, StrategySignals
from ultra_quant import FastBacktester

class TestUltraQuant(unittest.TestCase):
    def setUp(self):
        # 가상의 가격 데이터 생성 (Sine Wave + Noise)
        t = np.linspace(0, 100, 1000)
        price = 100 + 10 * np.sin(t) + np.random.normal(0, 1, 1000)
        self.prices = pd.Series(price, index=pd.date_range('2023-01-01', periods=1000, freq='D'))
        self.df = pd.DataFrame({'Close': price, 'High': price+1, 'Low': price-1, 'Open': price}, index=self.prices.index)

    def test_fast_backtester(self):
        """FastBacktester 기본 동작 테스트"""
        backtester = FastBacktester(use_numba=False, use_ray=False) # CI 환경 고려하여 Numba/Ray 비활성화 가능
        params = {'lookback': 20, 'sma_short': 10, 'sma_long': 50, 'target_vol': 0.15}
        result = backtester.run_single_backtest(self.prices.values, params)
        self.assertIn('sharpe_ratio', result)
        self.assertIn('total_return', result)

    def test_strategy_backtester(self):
        """StrategyBacktester 기본 동작 테스트"""
        backtester = StrategyBacktester()
        params = {'entry_period': 20, 'exit_period': 10}
        signal_func = StrategySignals.turtle_signals(params)
        result = backtester.run_backtest(self.df, signal_func)
        self.assertIsNotNone(result.metrics)
        self.assertTrue(len(result.equity_curve) > 0)

    def test_monte_carlo(self):
        """Monte Carlo 시뮬레이션 테스트"""
        from ultra_quant import MonteCarloSimulator
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        mc = MonteCarloSimulator(n_simulations=100, horizon=50)
        paths = mc.run_simulation(returns, 100000)
        stats = mc.analyze_risk(paths)
        self.assertEqual(paths.shape, (100, 50))
        self.assertIn('var_95', stats)

    def test_walk_forward(self):
        """Walk-Forward Analysis 테스트"""
        from strategy import WalkForwardOptimizer, StrategySignals
        wfo = WalkForwardOptimizer(self.df)
        param_grid = {'entry_period': [10, 20], 'exit_period': [5, 10]}
        results = wfo.run_wfa(StrategySignals.turtle_signals, param_grid, n_windows=2)
        self.assertTrue(len(results) > 0)
        self.assertIn('oos_sharpe', results[0])

    def test_advanced_strategies(self):
        """최신 고급 전략 테스트 (ML, Regime, Liquidity)"""
        from strategy import StrategyBacktester, StrategySignals
        backtester = StrategyBacktester()
        
        # 1. ML Ensemble
        res_ml = backtester.run_backtest(self.df, StrategySignals.ml_ensemble_signals({'lookback': 100}))
        self.assertIn('Total Return', res_ml.metrics)
        
        # 2. Regime Switching
        res_regime = backtester.run_backtest(self.df, StrategySignals.regime_switching_signals({}))
        self.assertIn('Total Return', res_regime.metrics)
        
        # 3. Liquidity Sweep
        res_sweep = backtester.run_backtest(self.df, StrategySignals.liquidity_sweep_signals({}))
        self.assertIn('Total Return', res_sweep.metrics)

if __name__ == '__main__':
    unittest.main()
