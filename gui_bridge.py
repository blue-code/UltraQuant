import json
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import pandas as pd
import yfinance as yf

from strategy import StrategyBacktester, StrategySignals
from ultra_quant import FastBacktester


STRATEGY_MAP = {
    "Turtle": StrategySignals.turtle_signals,
    "RSI2": StrategySignals.rsi2_signals,
    "Momentum": StrategySignals.momentum_signals,
    "ML Ensemble": StrategySignals.ml_ensemble_signals,
    "Regime Switching": StrategySignals.regime_switching_signals,
    "Liquidity Sweep": StrategySignals.liquidity_sweep_signals,
    "Adaptive EMA+ADX": StrategySignals.adaptive_ema_adx_signals,
    "ATR Breakout VolTarget": StrategySignals.atr_breakout_vol_target_signals,
    "Z-Score Mean Reversion": StrategySignals.zscore_mean_reversion_signals,
    "MACD Regime": StrategySignals.macd_regime_signals,
}

DEFAULT_PARAMS = {
    "Turtle": {"entry_period": 20, "exit_period": 10},
    "RSI2": {"sma_period": 200, "oversold": 10, "overbought": 90},
    "Momentum": {"lookback": 126, "sma_filter": 200},
    "ML Ensemble": {"lookback": 252},
    "Regime Switching": {"vol_lookback": 20, "threshold": 1.5},
    "Liquidity Sweep": {"lookback": 20},
    "Adaptive EMA+ADX": {"fast_period": 20, "slow_period": 100, "adx_period": 14, "adx_threshold": 20},
    "ATR Breakout VolTarget": {
        "lookback": 55,
        "atr_period": 20,
        "atr_filter": 0.8,
        "vol_lookback": 20,
        "target_daily_vol": 0.012,
    },
    "Z-Score Mean Reversion": {"window": 30, "entry_z": 2.0, "exit_z": 0.5, "trend_period": 100},
    "MACD Regime": {"fast": 12, "slow": 26, "signal_period": 9, "vol_window": 20, "high_vol_threshold": 1.2},
}

DEFAULT_SYMBOLS = (
    "SPY",
    "QQQ",
    "DIA",
    "IWM",
    "AAPL",
    "MSFT",
    "NVDA",
    "TSLA",
    "BTC-USD",
    "ETH-USD",
    "GLD",
    "TLT",
    "EURUSD=X",
    "JPY=X",
)


def load_symbol_choices(file_path: str = "symbols.json") -> tuple[str, ...]:
    path = Path(file_path)
    if not path.exists():
        return DEFAULT_SYMBOLS

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            symbols = payload.get("symbols", [])
        else:
            symbols = payload

        cleaned = []
        for item in symbols:
            if isinstance(item, str):
                s = item.strip().upper()
                if s:
                    cleaned.append(s)

        # 순서 유지 + 중복 제거
        unique = tuple(dict.fromkeys(cleaned))
        return unique if unique else DEFAULT_SYMBOLS
    except Exception:
        return DEFAULT_SYMBOLS


class StrategyBridgeApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("UltraQuant 전략 브리지")
        self.root.geometry("920x720")

        self.symbol_choices = load_symbol_choices()

        self.last_df = None
        self.last_strategy_params = None
        self.last_strategy_name = None

        self._build_ui()
        self._on_strategy_changed()

    def _build_ui(self):
        container = ttk.Frame(self.root, padding=12)
        container.pack(fill=tk.BOTH, expand=True)

        top = ttk.LabelFrame(container, text="입력", padding=10)
        top.pack(fill=tk.X)

        ttk.Label(top, text="심볼").grid(row=0, column=0, sticky=tk.W, padx=4, pady=4)
        default_symbol = self.symbol_choices[0] if self.symbol_choices else "SPY"
        self.symbol_var = tk.StringVar(value=default_symbol)
        symbol_combo = ttk.Combobox(top, textvariable=self.symbol_var, state="readonly", width=14)
        symbol_combo["values"] = self.symbol_choices
        symbol_combo.grid(row=0, column=1, sticky=tk.W, padx=4, pady=4)

        ttk.Label(top, text="기간").grid(row=0, column=2, sticky=tk.W, padx=4, pady=4)
        self.period_var = tk.StringVar(value="2y")
        period_combo = ttk.Combobox(top, textvariable=self.period_var, state="readonly", width=10)
        period_combo["values"] = ("6mo", "1y", "2y", "5y", "max")
        period_combo.grid(row=0, column=3, sticky=tk.W, padx=4, pady=4)

        ttk.Label(top, text="전략").grid(row=0, column=4, sticky=tk.W, padx=4, pady=4)
        self.strategy_var = tk.StringVar(value="Turtle")
        strategy_combo = ttk.Combobox(top, textvariable=self.strategy_var, state="readonly", width=22)
        strategy_combo["values"] = tuple(STRATEGY_MAP.keys())
        strategy_combo.grid(row=0, column=5, sticky=tk.W, padx=4, pady=4)
        strategy_combo.bind("<<ComboboxSelected>>", lambda _: self._on_strategy_changed())

        params_frame = ttk.LabelFrame(container, text="전략 파라미터 (JSON)", padding=10)
        params_frame.pack(fill=tk.BOTH, expand=False, pady=10)
        self.params_text = tk.Text(params_frame, height=8, wrap=tk.WORD)
        self.params_text.pack(fill=tk.BOTH, expand=True)

        actions = ttk.Frame(container)
        actions.pack(fill=tk.X, pady=8)

        self.test_button = ttk.Button(actions, text="1) strategy.py로 테스트", command=self._run_strategy_test)
        self.test_button.pack(side=tk.LEFT, padx=4)

        self.test_all_button = ttk.Button(actions, text="1-A) 전체 전략 일괄 테스트", command=self._run_all_strategies)
        self.test_all_button.pack(side=tk.LEFT, padx=4)

        self.apply_button = ttk.Button(
            actions,
            text="2) ultra_quant.py에 즉시 적용",
            command=self._apply_to_ultra_quant,
            state=tk.DISABLED,
        )
        self.apply_button.pack(side=tk.LEFT, padx=4)

        output_frame = ttk.LabelFrame(container, text="결과", padding=10)
        output_frame.pack(fill=tk.BOTH, expand=True)
        self.output_text = tk.Text(output_frame, wrap=tk.WORD)
        self.output_text.pack(fill=tk.BOTH, expand=True)

    def _on_strategy_changed(self):
        strategy_name = self.strategy_var.get()
        defaults = DEFAULT_PARAMS.get(strategy_name, {})
        self.params_text.delete("1.0", tk.END)
        self.params_text.insert(tk.END, json.dumps(defaults, ensure_ascii=False, indent=2))

    def _append_output(self, text: str):
        self.output_text.insert(tk.END, text + "\n")
        self.output_text.see(tk.END)

    def _parse_params(self):
        raw = self.params_text.get("1.0", tk.END).strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("JSON 루트는 객체(dict)여야 합니다.")
            return parsed
        except Exception as exc:
            raise ValueError(f"파라미터 JSON 파싱 실패: {exc}") from exc

    @staticmethod
    def _load_data(symbol: str, period: str) -> pd.DataFrame:
        df = yf.download(symbol, period=period, progress=False)
        if df.empty:
            raise ValueError(f"데이터가 없습니다: {symbol} ({period})")
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(1, axis=1)
        required = {"Open", "High", "Low", "Close"}
        if not required.issubset(set(df.columns)):
            raise ValueError("OHLC 컬럼이 부족합니다.")
        return df

    def _set_running(self, running: bool):
        state = tk.DISABLED if running else tk.NORMAL
        self.test_button.config(state=state)
        self.test_all_button.config(state=state)
        if running:
            self.apply_button.config(state=tk.DISABLED)
        elif self.last_df is not None and self.last_strategy_params is not None:
            self.apply_button.config(state=tk.NORMAL)

    def _run_strategy_test(self):
        self._set_running(True)

        def worker():
            try:
                symbol = self.symbol_var.get().strip().upper()
                period = self.period_var.get().strip()
                strategy_name = self.strategy_var.get().strip()
                params = self._parse_params()

                df = self._load_data(symbol, period)
                signal_factory = STRATEGY_MAP[strategy_name]
                signal_func = signal_factory(params)

                backtester = StrategyBacktester()
                result = backtester.run_backtest(df, signal_func, strategy_name=strategy_name)

                self.last_df = df
                self.last_strategy_params = params
                self.last_strategy_name = strategy_name

                metrics = result.metrics
                lines = [
                    "",
                    f"[전략 테스트 완료] {strategy_name} | {symbol} | {period}",
                    f"- Total Return: {metrics.get('Total Return', 0):.2%}",
                    f"- Sharpe Ratio: {metrics.get('Sharpe Ratio', 0):.3f}",
                    f"- Max Drawdown: {metrics.get('Max Drawdown', 0):.2%}",
                    f"- Win Rate: {metrics.get('Win Rate', 0):.2%}",
                    f"- Trades: {len(result.trades)}",
                ]
                self.root.after(0, lambda: self._append_output("\n".join(lines)))
                self.root.after(0, lambda: self.apply_button.config(state=tk.NORMAL))
            except Exception as exc:
                self.root.after(0, lambda: messagebox.showerror("오류", str(exc)))
            finally:
                self.root.after(0, lambda: self._set_running(False))

        threading.Thread(target=worker, daemon=True).start()

    def _run_all_strategies(self):
        self._set_running(True)

        def worker():
            try:
                symbol = self.symbol_var.get().strip().upper()
                period = self.period_var.get().strip()
                selected_strategy = self.strategy_var.get().strip()
                selected_params = self._parse_params()

                df = self._load_data(symbol, period)
                backtester = StrategyBacktester()
                rows = []

                for strategy_name, signal_factory in STRATEGY_MAP.items():
                    try:
                        params = DEFAULT_PARAMS.get(strategy_name, {}).copy()
                        if strategy_name == selected_strategy:
                            params = selected_params

                        signal_func = signal_factory(params)
                        result = backtester.run_backtest(df, signal_func, strategy_name=strategy_name)
                        metrics = result.metrics
                        rows.append(
                            {
                                "name": strategy_name,
                                "params": params,
                                "sharpe": float(metrics.get("Sharpe Ratio", 0.0)),
                                "ret": float(metrics.get("Total Return", 0.0)),
                                "mdd": float(metrics.get("Max Drawdown", 0.0)),
                                "trades": len(result.trades),
                            }
                        )
                    except Exception as strategy_exc:
                        rows.append({
                            "name": strategy_name,
                            "params": DEFAULT_PARAMS.get(strategy_name, {}),
                            "error": str(strategy_exc),
                        })

                ok_rows = [r for r in rows if "error" not in r]
                if not ok_rows:
                    raise RuntimeError("전체 전략 테스트 실패: 성공한 전략이 없습니다.")

                ok_rows.sort(key=lambda x: x["sharpe"], reverse=True)
                best = ok_rows[0]

                self.last_df = df
                self.last_strategy_name = best["name"]
                self.last_strategy_params = best["params"]

                out_lines = [
                    "",
                    f"[전체 전략 테스트 완료] {symbol} | {period}",
                    "Sharpe 기준 상위 결과:",
                ]
                for idx, row in enumerate(ok_rows[:10], start=1):
                    out_lines.append(
                        f"{idx}. {row['name']} | Sharpe {row['sharpe']:.3f} | Return {row['ret']:.2%} | "
                        f"MDD {row['mdd']:.2%} | Trades {row['trades']}"
                    )

                failed = [r for r in rows if "error" in r]
                if failed:
                    out_lines.append("")
                    out_lines.append(f"실패 전략: {len(failed)}개")
                    for row in failed[:3]:
                        out_lines.append(f"- {row['name']}: {row['error']}")

                out_lines.append("")
                out_lines.append(f"즉시 적용 대상(최고 Sharpe): {best['name']}")
                out_lines.append(f"파라미터: {json.dumps(best['params'], ensure_ascii=False)}")

                self.root.after(0, lambda: self._append_output("\n".join(out_lines)))
                self.root.after(0, lambda: self.apply_button.config(state=tk.NORMAL))
            except Exception as exc:
                self.root.after(0, lambda: messagebox.showerror("오류", str(exc)))
            finally:
                self.root.after(0, lambda: self._set_running(False))

        threading.Thread(target=worker, daemon=True).start()

    def _map_strategy_to_fast_params(self, strategy_name: str, params: dict) -> dict:
        mapped = {"lookback": 50, "sma_short": 20, "sma_long": 100, "target_vol": 0.15}

        if strategy_name == "Turtle":
            entry = int(params.get("entry_period", 20))
            exit_p = int(params.get("exit_period", 10))
            mapped["lookback"] = max(10, entry)
            mapped["sma_short"] = max(5, exit_p)
            mapped["sma_long"] = max(mapped["sma_short"] + 5, entry * 2)
        elif strategy_name == "RSI2":
            sma_period = int(params.get("sma_period", 200))
            mapped["lookback"] = max(20, sma_period // 4)
            mapped["sma_short"] = max(5, mapped["lookback"] // 2)
            mapped["sma_long"] = max(mapped["sma_short"] + 10, sma_period)
        elif strategy_name == "Momentum":
            lookback = int(params.get("lookback", 126))
            sma_filter = int(params.get("sma_filter", 200))
            mapped["lookback"] = max(10, lookback)
            mapped["sma_short"] = max(5, lookback // 3)
            mapped["sma_long"] = max(mapped["sma_short"] + 10, sma_filter)
        elif strategy_name == "ML Ensemble":
            lookback = int(params.get("lookback", 252))
            mapped["lookback"] = max(20, lookback // 3)
            mapped["sma_short"] = max(10, mapped["lookback"] // 2)
            mapped["sma_long"] = max(mapped["sma_short"] + 20, lookback // 2)
        elif strategy_name == "Regime Switching":
            vol_lookback = int(params.get("vol_lookback", 20))
            mapped["lookback"] = max(10, vol_lookback * 2)
            mapped["sma_short"] = max(5, vol_lookback)
            mapped["sma_long"] = max(mapped["sma_short"] + 10, vol_lookback * 5)
        elif strategy_name == "Liquidity Sweep":
            lookback = int(params.get("lookback", 20))
            mapped["lookback"] = max(10, lookback)
            mapped["sma_short"] = 10
            mapped["sma_long"] = 50
        elif strategy_name == "Adaptive EMA+ADX":
            fast_period = int(params.get("fast_period", 20))
            slow_period = int(params.get("slow_period", 100))
            mapped["lookback"] = max(20, slow_period // 2)
            mapped["sma_short"] = max(5, fast_period)
            mapped["sma_long"] = max(mapped["sma_short"] + 10, slow_period)
        elif strategy_name == "ATR Breakout VolTarget":
            lookback = int(params.get("lookback", 55))
            atr_period = int(params.get("atr_period", 20))
            mapped["lookback"] = max(20, lookback)
            mapped["sma_short"] = max(5, atr_period // 2)
            mapped["sma_long"] = max(mapped["sma_short"] + 10, lookback)
            mapped["target_vol"] = float(params.get("target_daily_vol", 0.012)) * 16.0
        elif strategy_name == "Z-Score Mean Reversion":
            window = int(params.get("window", 30))
            trend_period = int(params.get("trend_period", 100))
            mapped["lookback"] = max(10, window)
            mapped["sma_short"] = max(5, window // 2)
            mapped["sma_long"] = max(mapped["sma_short"] + 10, trend_period)
        elif strategy_name == "MACD Regime":
            fast = int(params.get("fast", 12))
            slow = int(params.get("slow", 26))
            mapped["lookback"] = max(10, slow)
            mapped["sma_short"] = max(5, fast)
            mapped["sma_long"] = max(mapped["sma_short"] + 5, slow)

        if mapped["sma_short"] >= mapped["sma_long"]:
            mapped["sma_long"] = mapped["sma_short"] + 5
        return mapped

    def _apply_to_ultra_quant(self):
        if self.last_df is None or self.last_strategy_params is None:
            messagebox.showinfo("안내", "먼저 전략 테스트를 실행해 주세요.")
            return

        try:
            fast_params = self._map_strategy_to_fast_params(self.last_strategy_name, self.last_strategy_params)
            prices = self.last_df["Close"].values.astype("float64")

            engine = FastBacktester(use_numba=False, use_ray=False)
            result = engine.run_single_backtest(prices, fast_params)

            lines = [
                "",
                f"[ultra_quant 적용 완료] 기준 전략: {self.last_strategy_name}",
                f"- 매핑 파라미터: {json.dumps(fast_params, ensure_ascii=False)}",
                f"- Final Value: {result.get('final_value', 0):,.2f}",
                f"- Total Return: {result.get('total_return', 0):.2f}%",
                f"- Sharpe Ratio: {result.get('sharpe_ratio', 0):.3f}",
                f"- Max Drawdown: {result.get('max_drawdown', 0):.2f}%",
            ]
            self._append_output("\n".join(lines))
        except Exception as exc:
            messagebox.showerror("오류", str(exc))


def main():
    root = tk.Tk()
    app = StrategyBridgeApp(root)
    app._append_output("UltraQuant 전략 브리지 준비 완료")
    app._append_output("1) 전략 테스트 또는 1-A) 전체 전략 테스트 후 2) 즉시 적용을 실행하세요.")
    app._append_output("심볼 목록은 symbols.json 파일을 우선 사용합니다.")
    root.mainloop()


if __name__ == "__main__":
    main()
