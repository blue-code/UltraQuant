import json
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np
import pandas as pd
import yfinance as yf

from strategy import StrategyBacktester, StrategySignals
from ultra_quant import FastBacktester


STRATEGY_MAP = {
    "Turtle": StrategySignals.turtle_signals,
    "RSI2": StrategySignals.rsi2_signals,
    "Momentum": StrategySignals.momentum_signals,
    "SuperTrend": StrategySignals.supertrend_signals,
    "Bollinger Reversion": StrategySignals.bollinger_reversion_signals,
    "Williams %R": StrategySignals.williams_r_signals,
    "Dual Thrust": StrategySignals.dual_thrust_signals,
    "Volatility Breakout": StrategySignals.volatility_breakout_signals,
    "MA Cross": StrategySignals.ma_cross_signals,
    "ML Ensemble": StrategySignals.ml_ensemble_signals,
    "Regime Switching": StrategySignals.regime_switching_signals,
    "Liquidity Sweep": StrategySignals.liquidity_sweep_signals,
    "Adaptive EMA+ADX": StrategySignals.adaptive_ema_adx_signals,
    "ATR Breakout VolTarget": StrategySignals.atr_breakout_vol_target_signals,
    "Z-Score Mean Reversion": StrategySignals.zscore_mean_reversion_signals,
    "MACD Regime": StrategySignals.macd_regime_signals,
    "Turtle + Momentum Confirm": StrategySignals.turtle_momentum_confirm_signals,
    "RSI2 + Bollinger Reversion": StrategySignals.rsi2_bollinger_reversion_signals,
    "Regime + Liquidity Sweep": StrategySignals.regime_liquidity_sweep_signals,
    "Adaptive Fractal Regime": StrategySignals.adaptive_fractal_regime_signals,
}

DEFAULT_PARAMS = {
    "Turtle": {"entry_period": 20, "exit_period": 10},
    "RSI2": {"sma_period": 200, "oversold": 10, "overbought": 90},
    "Momentum": {"lookback": 126, "sma_filter": 200},
    "SuperTrend": {"atr_period": 10, "multiplier": 3.0},
    "Bollinger Reversion": {"window": 20, "num_std": 2.0, "trend_period": 120},
    "Williams %R": {"lookback": 14, "oversold": -80, "overbought": -20, "trend_period": 80},
    "Dual Thrust": {"lookback": 4, "k1": 0.5, "k2": 0.5},
    "Volatility Breakout": {"lookback": 20, "k": 0.5, "atr_period": 14},
    "MA Cross": {"short_window": 20, "long_window": 60},
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
    "Turtle + Momentum Confirm": {
        "entry_period": 20,
        "exit_period": 10,
        "momentum_lookback": 60,
        "momentum_threshold": 0.01,
    },
    "RSI2 + Bollinger Reversion": {
        "rsi_period": 2,
        "oversold": 8,
        "overbought": 92,
        "bb_window": 20,
        "bb_std": 2.0,
        "trend_period": 100,
    },
    "Regime + Liquidity Sweep": {
        "vol_lookback": 20,
        "regime_threshold": 1.2,
        "sweep_lookback": 20,
        "confirm_momentum": 10,
    },
    "Adaptive Fractal Regime": {
        "trend_lookback": 55,
        "mean_window": 20,
        "z_entry": 1.6,
        "chop_window": 14,
        "chop_threshold": 58,
        "atr_period": 14,
        "target_daily_vol": 0.012,
    },
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

# 최근 적합도 반영 설정값
RECENT_PERIODS = ("3mo", "6mo")
BASE_WEIGHT = 0.4
RECENT_WEIGHT = 0.6
RECENT_GATE_MIN_SCORE = 0.0
RECENT_GATE_MIN_TRADES = 1
FINAL_BASE_RECENT_WEIGHT = 0.75
FINAL_ROLLING_WEIGHT = 0.25
ROLLING_WINDOW_DAYS = 252
ROLLING_STEP_DAYS = 63
ROLLING_GATE_MIN_WINDOWS = 3
ROLLING_GATE_MIN_WIN_RATIO = 0.5


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
        period_combo["values"] = ("1mo", "3mo", "6mo", "1y", "2y", "5y", "max")
        period_combo.grid(row=0, column=3, sticky=tk.W, padx=4, pady=4)

        ttk.Label(top, text="전략").grid(row=0, column=4, sticky=tk.W, padx=4, pady=4)
        self.strategy_var = tk.StringVar(value="Turtle")
        strategy_combo = ttk.Combobox(top, textvariable=self.strategy_var, state="readonly", width=22)
        strategy_combo["values"] = tuple(STRATEGY_MAP.keys())
        strategy_combo.grid(row=0, column=5, sticky=tk.W, padx=4, pady=4)
        strategy_combo.bind("<<ComboboxSelected>>", lambda _: self._on_strategy_changed())

        self.use_rolling_var = tk.BooleanVar(value=True)
        rolling_check = ttk.Checkbutton(
            top,
            text="Rolling 검증(1년 창/3개월 스텝)",
            variable=self.use_rolling_var,
        )
        rolling_check.grid(row=1, column=0, columnspan=6, sticky=tk.W, padx=4, pady=4)

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

        self.test_all_periods_button = ttk.Button(
            actions,
            text="1-B) 전체 전략×기간(1mo~5y) 일괄 테스트",
            command=self._run_all_strategies_all_periods,
        )
        self.test_all_periods_button.pack(side=tk.LEFT, padx=4)

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

    @staticmethod
    def _composite_score(sharpe: float, total_return: float, max_drawdown: float) -> float:
        # 개선된 복합 점수:
        # - Return: tanh 반감(포화) 처리로 극단값 영향 완화
        # - MDD: 0~20% 구간 제곱 페널티 + 20% 초과 구간 로그 페널티
        ret_pct = total_return * 100.0
        mdd_pct = abs(max_drawdown) * 100.0

        # Return 반감기(포화): +/-20% 부근부터 기여 증가율 완만
        ret_effect = 20.0 * np.tanh(ret_pct / 20.0)

        # MDD 비선형 페널티
        capped = min(mdd_pct, 20.0)
        mdd_quad = (capped * capped) / 20.0
        mdd_tail = 5.0 * np.log1p(max(0.0, mdd_pct - 20.0))
        mdd_penalty = mdd_quad + mdd_tail

        return (0.6 * sharpe) + (0.4 * ret_effect) - (0.3 * mdd_penalty)

    @staticmethod
    def _recent_summary(recent_rows: list[dict]) -> dict:
        if not recent_rows:
            return {
                "recent_score": float("-inf"),
                "recent_trades": 0,
                "recent_ok": False,
                "recent_note": "recent 데이터 없음",
            }

        recent_score = float(np.mean([r["score"] for r in recent_rows]))
        recent_trades = int(sum(int(r.get("trades", 0)) for r in recent_rows))
        recent_ok = (recent_score >= RECENT_GATE_MIN_SCORE) and (recent_trades >= RECENT_GATE_MIN_TRADES)
        note = (
            "PASS"
            if recent_ok
            else f"FAIL(score<{RECENT_GATE_MIN_SCORE:.2f} 또는 trades<{RECENT_GATE_MIN_TRADES})"
        )
        return {
            "recent_score": recent_score,
            "recent_trades": recent_trades,
            "recent_ok": recent_ok,
            "recent_note": note,
        }

    @staticmethod
    def _blended_score(base_score: float, recent_score: float) -> float:
        if np.isfinite(recent_score):
            return (BASE_WEIGHT * base_score) + (RECENT_WEIGHT * recent_score)
        return base_score

    @staticmethod
    def _final_score(blended_score: float, rolling_score: float, use_rolling: bool) -> float:
        if use_rolling and np.isfinite(rolling_score):
            return (FINAL_BASE_RECENT_WEIGHT * blended_score) + (FINAL_ROLLING_WEIGHT * rolling_score)
        return blended_score

    def _rolling_window_stats(self, df: pd.DataFrame, signal_factory, params: dict, strategy_name: str) -> dict:
        if len(df) < ROLLING_WINDOW_DAYS:
            return {
                "rolling_score": float("-inf"),
                "rolling_ok": False,
                "rolling_note": "window 부족",
                "rolling_windows": 0,
                "rolling_win_ratio": 0.0,
                "rolling_sharpe_median": 0.0,
                "rolling_mdd_worst": 0.0,
                "rolling_score_std": 0.0,
            }

        backtester = StrategyBacktester()
        rows = []
        for start in range(0, len(df) - ROLLING_WINDOW_DAYS + 1, ROLLING_STEP_DAYS):
            chunk = df.iloc[start : start + ROLLING_WINDOW_DAYS]
            try:
                signal_func = signal_factory(params)
                result = backtester.run_backtest(chunk, signal_func, strategy_name=f"{strategy_name}_roll")
                m = result.metrics
                sharpe = float(m.get("Sharpe Ratio", 0.0))
                ret = float(m.get("Total Return", 0.0))
                mdd = float(m.get("Max Drawdown", 0.0))
                rows.append({
                    "score": self._composite_score(sharpe, ret, mdd),
                    "ret": ret,
                    "sharpe": sharpe,
                    "mdd": mdd,
                })
            except Exception:
                continue

        if not rows:
            return {
                "rolling_score": float("-inf"),
                "rolling_ok": False,
                "rolling_note": "rolling 계산 실패",
                "rolling_windows": 0,
                "rolling_win_ratio": 0.0,
                "rolling_sharpe_median": 0.0,
                "rolling_mdd_worst": 0.0,
                "rolling_score_std": 0.0,
            }

        wins = sum(1 for r in rows if r["ret"] > 0.0)
        win_ratio = wins / len(rows)
        sharpe_median = float(np.median([r["sharpe"] for r in rows]))
        mdd_worst = float(min(r["mdd"] for r in rows))
        score_std = float(np.std([r["score"] for r in rows]))

        # 일관성 점수: 승률/중앙 Sharpe 보상, 최악 MDD/변동성 페널티
        rolling_score = (3.0 * win_ratio) + (0.4 * sharpe_median) - (0.12 * abs(mdd_worst) * 100.0) - (0.1 * score_std)
        rolling_ok = (len(rows) >= ROLLING_GATE_MIN_WINDOWS) and (win_ratio >= ROLLING_GATE_MIN_WIN_RATIO)
        rolling_note = (
            "PASS"
            if rolling_ok
            else f"FAIL(windows<{ROLLING_GATE_MIN_WINDOWS} 또는 win_ratio<{ROLLING_GATE_MIN_WIN_RATIO:.2f})"
        )

        return {
            "rolling_score": rolling_score,
            "rolling_ok": rolling_ok,
            "rolling_note": rolling_note,
            "rolling_windows": len(rows),
            "rolling_win_ratio": win_ratio,
            "rolling_sharpe_median": sharpe_median,
            "rolling_mdd_worst": mdd_worst,
            "rolling_score_std": score_std,
        }

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
        self.test_all_periods_button.config(state=state)
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

    def _run_all_strategies_all_periods(self):
        self._set_running(True)

        def worker():
            try:
                symbol = self.symbol_var.get().strip().upper()
                selected_strategy = self.strategy_var.get().strip()
                selected_params = self._parse_params()
                periods = ("1mo", "3mo", "6mo", "1y", "2y", "5y")
                use_rolling = bool(self.use_rolling_var.get())

                backtester = StrategyBacktester()
                rows = []
                best_payload = None
                strategy_params_map = {}
                for strategy_name in STRATEGY_MAP.keys():
                    params = DEFAULT_PARAMS.get(strategy_name, {}).copy()
                    if strategy_name == selected_strategy:
                        params = selected_params
                    strategy_params_map[strategy_name] = params

                for period in periods:
                    try:
                        df = self._load_data(symbol, period)
                    except Exception as data_exc:
                        rows.append({"period": period, "name": "(data)", "error": str(data_exc)})
                        continue

                    for strategy_name, signal_factory in STRATEGY_MAP.items():
                        try:
                            params = strategy_params_map[strategy_name]
                            signal_func = signal_factory(params)
                            result = backtester.run_backtest(df, signal_func, strategy_name=strategy_name)
                            metrics = result.metrics

                            row = {
                                "period": period,
                                "name": strategy_name,
                                "params": params,
                                "df": df,
                                "sharpe": float(metrics.get("Sharpe Ratio", 0.0)),
                                "ret": float(metrics.get("Total Return", 0.0)),
                                "mdd": float(metrics.get("Max Drawdown", 0.0)),
                                "trades": len(result.trades),
                            }
                            row["score"] = self._composite_score(row["sharpe"], row["ret"], row["mdd"])
                            rows.append(row)

                            if best_payload is None or row["score"] > best_payload["score"]:
                                best_payload = row
                        except Exception as strategy_exc:
                            rows.append({"period": period, "name": strategy_name, "error": str(strategy_exc)})

                ok_rows = [r for r in rows if "error" not in r]
                if not ok_rows or best_payload is None:
                    raise RuntimeError("전체 전략×기간 테스트 실패: 성공한 결과가 없습니다.")

                rolling_map = {}
                if use_rolling:
                    try:
                        rolling_df = self._load_data(symbol, "5y")
                    except Exception:
                        rolling_df = None
                    if rolling_df is not None:
                        for strategy_name, signal_factory in STRATEGY_MAP.items():
                            rolling_map[strategy_name] = self._rolling_window_stats(
                                rolling_df,
                                signal_factory,
                                strategy_params_map[strategy_name],
                                strategy_name,
                            )

                # 전략별 최근(3mo/6mo) 요약 생성 후 점수 보정
                recent_map = {}
                for strategy_name in STRATEGY_MAP.keys():
                    recent_rows = [
                        r for r in ok_rows if r["name"] == strategy_name and r.get("period") in RECENT_PERIODS
                    ]
                    recent_map[strategy_name] = self._recent_summary(recent_rows)

                for row in ok_rows:
                    summary = recent_map.get(row["name"], {})
                    row["recent_score"] = summary.get("recent_score", float("-inf"))
                    row["recent_trades"] = summary.get("recent_trades", 0)
                    row["recent_ok"] = summary.get("recent_ok", False)
                    row["gate_note"] = summary.get("recent_note", "unknown")
                    row["blend_score"] = self._blended_score(row["score"], row["recent_score"])
                    roll = rolling_map.get(row["name"], {})
                    row["rolling_score"] = roll.get("rolling_score", float("-inf"))
                    row["rolling_ok"] = roll.get("rolling_ok", not use_rolling)
                    row["rolling_note"] = roll.get("rolling_note", "SKIP")
                    row["rolling_windows"] = roll.get("rolling_windows", 0)
                    row["rolling_win_ratio"] = roll.get("rolling_win_ratio", 0.0)
                    row["final_score"] = self._final_score(row["blend_score"], row["rolling_score"], use_rolling)
                    row["overall_ok"] = bool(row["recent_ok"]) and bool(row["rolling_ok"])

                passed_rows = [r for r in ok_rows if r.get("overall_ok")]
                if not passed_rows:
                    raise RuntimeError("전체 전략×기간 테스트 실패: 최근/롤링 게이트를 통과한 전략이 없습니다.")

                passed_rows.sort(key=lambda x: x["final_score"], reverse=True)
                best_payload = passed_rows[0]

                self.last_df = best_payload["df"]
                self.last_strategy_name = best_payload["name"]
                self.last_strategy_params = best_payload["params"]

                out_lines = [
                    "",
                    f"[전체 전략×기간 테스트 완료] {symbol} | periods={','.join(periods)}",
                    (
                        "최종 점수 기준 상위 결과 "
                        f"(기본 {int(BASE_WEIGHT*100)}% + 최근 {int(RECENT_WEIGHT*100)}%, "
                        f"최근 게이트 score>={RECENT_GATE_MIN_SCORE:.2f}, trades>={RECENT_GATE_MIN_TRADES}, "
                        f"Rolling={'ON' if use_rolling else 'OFF'})"
                    ),
                ]
                for idx, row in enumerate(passed_rows[:20], start=1):
                    out_lines.append(
                        f"{idx}. [{row['period']}] {row['name']} | Final {row['final_score']:.3f} | "
                        f"Base {row['score']:.3f} | Recent {row['recent_score']:.3f} | "
                        f"Rolling {row['rolling_score']:.3f} | Sharpe {row['sharpe']:.3f} | "
                        f"Return {row['ret']:.2%} | MDD {row['mdd']:.2%} | Trades {row['trades']}"
                    )

                failed = [r for r in rows if "error" in r]
                if failed:
                    out_lines.append("")
                    out_lines.append(f"실패 항목: {len(failed)}개")
                    for row in failed[:5]:
                        out_lines.append(f"- [{row.get('period', '-')}] {row.get('name', '-')}: {row['error']}")

                gated_out = [r for r in ok_rows if not r.get("overall_ok")]
                if gated_out:
                    out_lines.append("")
                    out_lines.append(f"게이트 제외 항목: {len(gated_out)}개")
                    for row in gated_out[:5]:
                        out_lines.append(
                            f"- [{row['period']}] {row['name']}: Final {row['final_score']:.3f}, "
                            f"Recent {row['recent_score']:.3f}({row['gate_note']}), "
                            f"Rolling {row['rolling_score']:.3f}({row['rolling_note']})"
                        )

                out_lines.append("")
                out_lines.append(f"즉시 적용 대상(최고 최종점수): [{best_payload['period']}] {best_payload['name']}")
                out_lines.append(f"파라미터: {json.dumps(best_payload['params'], ensure_ascii=False)}")

                self.root.after(0, lambda: self._append_output("\n".join(out_lines)))
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
                use_rolling = bool(self.use_rolling_var.get())

                df = self._load_data(symbol, period)
                backtester = StrategyBacktester()
                rows = []
                recent_df_map = {}
                for rp in RECENT_PERIODS:
                    try:
                        recent_df_map[rp] = self._load_data(symbol, rp)
                    except Exception:
                        continue
                rolling_df = None
                if use_rolling:
                    try:
                        rolling_df = self._load_data(symbol, "5y")
                    except Exception:
                        rolling_df = None

                for strategy_name, signal_factory in STRATEGY_MAP.items():
                    try:
                        params = DEFAULT_PARAMS.get(strategy_name, {}).copy()
                        if strategy_name == selected_strategy:
                            params = selected_params

                        signal_func = signal_factory(params)
                        result = backtester.run_backtest(df, signal_func, strategy_name=strategy_name)
                        metrics = result.metrics
                        row = {
                            "name": strategy_name,
                            "params": params,
                            "sharpe": float(metrics.get("Sharpe Ratio", 0.0)),
                            "ret": float(metrics.get("Total Return", 0.0)),
                            "mdd": float(metrics.get("Max Drawdown", 0.0)),
                            "trades": len(result.trades),
                        }
                        row["score"] = self._composite_score(row["sharpe"], row["ret"], row["mdd"])

                        recent_rows = []
                        for rp, recent_df in recent_df_map.items():
                            try:
                                recent_signal = signal_factory(params)
                                recent_result = backtester.run_backtest(
                                    recent_df, recent_signal, strategy_name=f"{strategy_name}_{rp}"
                                )
                                recent_metrics = recent_result.metrics
                                recent_rows.append(
                                    {
                                        "period": rp,
                                        "score": self._composite_score(
                                            float(recent_metrics.get("Sharpe Ratio", 0.0)),
                                            float(recent_metrics.get("Total Return", 0.0)),
                                            float(recent_metrics.get("Max Drawdown", 0.0)),
                                        ),
                                        "trades": len(recent_result.trades),
                                    }
                                )
                            except Exception:
                                continue

                        recent_info = self._recent_summary(recent_rows)
                        row["recent_score"] = recent_info["recent_score"]
                        row["recent_trades"] = recent_info["recent_trades"]
                        row["recent_ok"] = recent_info["recent_ok"]
                        row["gate_note"] = recent_info["recent_note"]
                        row["blend_score"] = self._blended_score(row["score"], row["recent_score"])

                        if use_rolling:
                            if rolling_df is not None:
                                roll = self._rolling_window_stats(rolling_df, signal_factory, params, strategy_name)
                            else:
                                roll = {
                                    "rolling_score": float("-inf"),
                                    "rolling_ok": False,
                                    "rolling_note": "rolling 데이터 없음",
                                }
                        else:
                            roll = {"rolling_score": float("-inf"), "rolling_ok": True, "rolling_note": "SKIP"}

                        row["rolling_score"] = roll.get("rolling_score", float("-inf"))
                        row["rolling_ok"] = roll.get("rolling_ok", not use_rolling)
                        row["rolling_note"] = roll.get("rolling_note", "SKIP")
                        row["final_score"] = self._final_score(row["blend_score"], row["rolling_score"], use_rolling)
                        row["overall_ok"] = bool(row["recent_ok"]) and bool(row["rolling_ok"])
                        rows.append(row)
                    except Exception as strategy_exc:
                        rows.append({
                            "name": strategy_name,
                            "params": DEFAULT_PARAMS.get(strategy_name, {}),
                            "error": str(strategy_exc),
                        })

                ok_rows = [r for r in rows if "error" not in r]
                if not ok_rows:
                    raise RuntimeError("전체 전략 테스트 실패: 성공한 전략이 없습니다.")

                passed_rows = [r for r in ok_rows if r.get("overall_ok")]
                if not passed_rows:
                    raise RuntimeError("전체 전략 테스트 실패: 최근/롤링 게이트를 통과한 전략이 없습니다.")

                passed_rows.sort(key=lambda x: x["final_score"], reverse=True)
                best = passed_rows[0]

                self.last_df = df
                self.last_strategy_name = best["name"]
                self.last_strategy_params = best["params"]

                out_lines = [
                    "",
                    f"[전체 전략 테스트 완료] {symbol} | {period}",
                    (
                        "최종 점수 기준 상위 결과 "
                        f"(기본 {int(BASE_WEIGHT*100)}% + 최근 {int(RECENT_WEIGHT*100)}%, "
                        f"최근 게이트 score>={RECENT_GATE_MIN_SCORE:.2f}, trades>={RECENT_GATE_MIN_TRADES}, "
                        f"Rolling={'ON' if use_rolling else 'OFF'})"
                    ),
                ]
                for idx, row in enumerate(passed_rows[:10], start=1):
                    out_lines.append(
                        f"{idx}. {row['name']} | Final {row['final_score']:.3f} | Base {row['score']:.3f} | "
                        f"Recent {row['recent_score']:.3f} | Rolling {row['rolling_score']:.3f} | "
                        f"Sharpe {row['sharpe']:.3f} | Return {row['ret']:.2%} | "
                        f"MDD {row['mdd']:.2%} | Trades {row['trades']}"
                    )

                failed = [r for r in rows if "error" in r]
                if failed:
                    out_lines.append("")
                    out_lines.append(f"실패 전략: {len(failed)}개")
                    for row in failed[:3]:
                        out_lines.append(f"- {row['name']}: {row['error']}")

                gated_out = [r for r in ok_rows if not r.get("overall_ok")]
                if gated_out:
                    out_lines.append("")
                    out_lines.append(f"게이트 제외 전략: {len(gated_out)}개")
                    for row in gated_out[:5]:
                        out_lines.append(
                            f"- {row['name']}: Final {row['final_score']:.3f}, "
                            f"Recent {row['recent_score']:.3f}({row['gate_note']}), "
                            f"Rolling {row['rolling_score']:.3f}({row['rolling_note']})"
                        )

                out_lines.append("")
                out_lines.append(f"즉시 적용 대상(최고 최종점수): {best['name']}")
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
        elif strategy_name == "SuperTrend":
            atr_period = int(params.get("atr_period", 10))
            mapped["lookback"] = max(10, atr_period * 2)
            mapped["sma_short"] = max(5, atr_period)
            mapped["sma_long"] = max(mapped["sma_short"] + 10, atr_period * 6)
        elif strategy_name == "Bollinger Reversion":
            window = int(params.get("window", 20))
            trend_period = int(params.get("trend_period", 120))
            mapped["lookback"] = max(10, window)
            mapped["sma_short"] = max(5, window // 2)
            mapped["sma_long"] = max(mapped["sma_short"] + 10, trend_period)
        elif strategy_name == "Williams %R":
            lookback = int(params.get("lookback", 14))
            trend_period = int(params.get("trend_period", 80))
            mapped["lookback"] = max(10, lookback)
            mapped["sma_short"] = max(5, lookback // 2)
            mapped["sma_long"] = max(mapped["sma_short"] + 10, trend_period)
        elif strategy_name == "Dual Thrust":
            lookback = int(params.get("lookback", 4))
            mapped["lookback"] = max(10, lookback * 5)
            mapped["sma_short"] = max(5, lookback * 2)
            mapped["sma_long"] = max(mapped["sma_short"] + 10, lookback * 10)
        elif strategy_name == "Volatility Breakout":
            lookback = int(params.get("lookback", 20))
            atr_period = int(params.get("atr_period", 14))
            mapped["lookback"] = max(10, lookback)
            mapped["sma_short"] = max(5, atr_period)
            mapped["sma_long"] = max(mapped["sma_short"] + 10, lookback * 2)
        elif strategy_name == "MA Cross":
            short_window = int(params.get("short_window", 20))
            long_window = int(params.get("long_window", 60))
            mapped["lookback"] = max(10, long_window)
            mapped["sma_short"] = max(5, short_window)
            mapped["sma_long"] = max(mapped["sma_short"] + 5, long_window)
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
        elif strategy_name == "Turtle + Momentum Confirm":
            entry = int(params.get("entry_period", 20))
            momentum_lookback = int(params.get("momentum_lookback", 60))
            mapped["lookback"] = max(10, entry)
            mapped["sma_short"] = max(5, entry // 2)
            mapped["sma_long"] = max(mapped["sma_short"] + 10, momentum_lookback)
        elif strategy_name == "RSI2 + Bollinger Reversion":
            bb_window = int(params.get("bb_window", 20))
            trend_period = int(params.get("trend_period", 100))
            mapped["lookback"] = max(10, bb_window)
            mapped["sma_short"] = max(5, bb_window // 2)
            mapped["sma_long"] = max(mapped["sma_short"] + 10, trend_period)
        elif strategy_name == "Regime + Liquidity Sweep":
            sweep_lookback = int(params.get("sweep_lookback", 20))
            confirm_momentum = int(params.get("confirm_momentum", 10))
            mapped["lookback"] = max(10, sweep_lookback)
            mapped["sma_short"] = max(5, confirm_momentum)
            mapped["sma_long"] = max(mapped["sma_short"] + 10, sweep_lookback * 2)
        elif strategy_name == "Adaptive Fractal Regime":
            trend_lookback = int(params.get("trend_lookback", 55))
            mean_window = int(params.get("mean_window", 20))
            mapped["lookback"] = max(10, trend_lookback)
            mapped["sma_short"] = max(5, mean_window // 2)
            mapped["sma_long"] = max(mapped["sma_short"] + 10, trend_lookback)
            mapped["target_vol"] = float(params.get("target_daily_vol", 0.012)) * 16.0

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
    app._append_output("1) 전략 테스트 또는 1-A/1-B) 전체 전략 테스트 후 2) 즉시 적용을 실행하세요.")
    app._append_output("심볼 목록은 symbols.json 파일을 우선 사용합니다.")
    root.mainloop()


if __name__ == "__main__":
    main()
