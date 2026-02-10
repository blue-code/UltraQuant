import unittest
import numpy as np
import pandas as pd
from strategy import StrategyBacktester, StrategySignals
from ultra_quant import FastBacktester


class TestUltraQuant(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        t = np.linspace(0, 100, 1000)
        price = 100 + 10 * np.sin(t) + np.random.normal(0, 1, 1000)
        self.prices = pd.Series(price, index=pd.date_range('2023-01-01', periods=1000, freq='D'))
        self.df = pd.DataFrame(
            {'Close': price, 'High': price + 1, 'Low': price - 1, 'Open': price},
            index=self.prices.index,
        )

    def test_fast_backtester(self):
        backtester = FastBacktester(use_numba=False, use_ray=False)
        params = {'lookback': 20, 'sma_short': 10, 'sma_long': 50, 'target_vol': 0.15}
        result = backtester.run_single_backtest(self.prices.values, params)
        self.assertIn('sharpe_ratio', result)
        self.assertIn('total_return', result)

    def test_strategy_backtester(self):
        backtester = StrategyBacktester()
        params = {'entry_period': 20, 'exit_period': 10}
        signal_func = StrategySignals.turtle_signals(params)
        result = backtester.run_backtest(self.df, signal_func)
        self.assertIsNotNone(result.metrics)
        self.assertTrue(len(result.equity_curve) > 0)

    def test_monte_carlo(self):
        from ultra_quant import MonteCarloSimulator

        returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        mc = MonteCarloSimulator(n_simulations=100, horizon=50)
        paths = mc.run_simulation(returns, 100000)
        stats = mc.analyze_risk(paths)
        self.assertEqual(paths.shape, (100, 50))
        self.assertIn('var_95', stats)

    def test_walk_forward(self):
        from strategy import WalkForwardOptimizer

        wfo = WalkForwardOptimizer(self.df)
        param_grid = {'entry_period': [10, 20], 'exit_period': [5, 10]}
        results = wfo.run_wfa(StrategySignals.turtle_signals, param_grid, n_windows=2)
        self.assertTrue(len(results) > 0)
        self.assertIn('oos_sharpe', results[0])

    def test_advanced_strategies(self):
        backtester = StrategyBacktester()

        res_ml = backtester.run_backtest(self.df, StrategySignals.ml_ensemble_signals({'lookback': 100}))
        self.assertIn('Total Return', res_ml.metrics)

        res_regime = backtester.run_backtest(self.df, StrategySignals.regime_switching_signals({}))
        self.assertIn('Total Return', res_regime.metrics)

        res_sweep = backtester.run_backtest(self.df, StrategySignals.liquidity_sweep_signals({}))
        self.assertIn('Total Return', res_sweep.metrics)

        res_adx = backtester.run_backtest(
            self.df,
            StrategySignals.adaptive_ema_adx_signals({'fast_period': 20, 'slow_period': 80, 'adx_period': 14}),
        )
        self.assertIn('Total Return', res_adx.metrics)

        res_atr = backtester.run_backtest(
            self.df,
            StrategySignals.atr_breakout_vol_target_signals({'lookback': 40, 'atr_period': 14, 'vol_lookback': 20}),
        )
        self.assertIn('Total Return', res_atr.metrics)

        res_zscore = backtester.run_backtest(
            self.df,
            StrategySignals.zscore_mean_reversion_signals({'window': 25, 'entry_z': 1.8, 'trend_period': 80}),
        )
        self.assertIn('Total Return', res_zscore.metrics)

        res_macd = backtester.run_backtest(
            self.df,
            StrategySignals.macd_regime_signals({'fast': 12, 'slow': 26, 'signal_period': 9}),
        )
        self.assertIn('Total Return', res_macd.metrics)


if __name__ == '__main__':
    unittest.main()
