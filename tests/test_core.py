import unittest
import numpy as np
import pandas as pd
from strategy import (
    StrategyBacktester,
    StrategySignals,
    RollingOOSRunner,
    DataHealthCheckerLite,
)


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
        try:
            from ultra_quant import FastBacktester
        except Exception as exc:
            self.skipTest(f"ultra_quant import 실패: {exc}")
            return

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
        try:
            from ultra_quant import MonteCarloSimulator
        except Exception as exc:
            self.skipTest(f"ultra_quant import 실패: {exc}")
            return

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

        res_supertrend = backtester.run_backtest(
            self.df,
            StrategySignals.supertrend_signals({'atr_period': 10, 'multiplier': 3.0}),
        )
        self.assertIn('Total Return', res_supertrend.metrics)

        res_bb = backtester.run_backtest(
            self.df,
            StrategySignals.bollinger_reversion_signals({'window': 20, 'num_std': 2.0}),
        )
        self.assertIn('Total Return', res_bb.metrics)

        res_wr = backtester.run_backtest(
            self.df,
            StrategySignals.williams_r_signals({'lookback': 14, 'oversold': -80, 'overbought': -20}),
        )
        self.assertIn('Total Return', res_wr.metrics)

        res_dual = backtester.run_backtest(
            self.df,
            StrategySignals.dual_thrust_signals({'lookback': 4, 'k1': 0.5, 'k2': 0.5}),
        )
        self.assertIn('Total Return', res_dual.metrics)

        res_vb = backtester.run_backtest(
            self.df,
            StrategySignals.volatility_breakout_signals({'lookback': 20, 'k': 0.5, 'atr_period': 14}),
        )
        self.assertIn('Total Return', res_vb.metrics)

        res_ma = backtester.run_backtest(
            self.df,
            StrategySignals.ma_cross_signals({'short_window': 20, 'long_window': 60}),
        )
        self.assertIn('Total Return', res_ma.metrics)

    def test_backtest_report_and_excess_metrics(self):
        backtester = StrategyBacktester(open_cost=0.001, close_cost=0.0015, min_cost=1.0, impact_cost=0.0002)
        result = backtester.run_backtest(self.df, StrategySignals.momentum_signals({'lookback': 80, 'sma_filter': 120}))
        self.assertIsNotNone(result.report)
        self.assertIn('bench', result.report.columns)
        self.assertIn('excess_return_w_cost', result.report.columns)
        self.assertIn('Excess Return (w Cost)', result.metrics)
        self.assertIn('Risk (Excess wo Cost)', result.metrics)

    def test_data_health_checker(self):
        bad_df = self.df.copy()
        bad_df.loc[bad_df.index[10], 'Close'] = np.nan
        issues = DataHealthCheckerLite.check(bad_df)
        self.assertIn('missing_data', issues)
        with self.assertRaises(ValueError):
            DataHealthCheckerLite.validate_or_raise(bad_df)

    def test_rolling_oos_runner(self):
        runner = RollingOOSRunner(self.df, initial_capital=100000)
        param_grid = {
            'lookback': [40, 60],
            'sma_filter': [120, 160],
        }
        out = runner.run(
            signal_factory=StrategySignals.momentum_signals,
            param_grid=param_grid,
            train_size=350,
            test_size=80,
            step_size=40,
            trunc_days=1,
        )
        self.assertIn('windows', out)
        self.assertIn('aggregate', out)
        self.assertTrue(len(out['windows']) > 0)


if __name__ == '__main__':
    unittest.main()
