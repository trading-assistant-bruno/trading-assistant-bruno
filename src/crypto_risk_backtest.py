from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import crypto_backtest as base
import crypto_backtest_yahoo as yahoo

ROOT = Path(__file__).resolve().parents[1]
RISK_CONFIG = yaml.safe_load((ROOT / "crypto_risk_config.yml").read_text(encoding="utf-8"))
OUT_DIR = ROOT / "data" / "crypto_risk"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Position:
    symbol: str
    qty: float
    entry_date: pd.Timestamp
    entry_price: float
    entry_fee: float
    initial_stop: float
    current_stop: float
    risk_per_unit: float
    risk_cash: float
    target: float | None
    highest_high: float
    exit_next_open: bool = False
    scheduled_exit_reason: str | None = None


@dataclass
class VariantResult:
    variant: str
    stop_rule: str
    exit_rule: str
    cagr_pct: float
    max_drawdown_pct: float
    sharpe: float
    sortino: float
    calmar: float
    total_return_pct: float
    final_equity: float
    trades: int
    win_rate_pct: float
    profit_factor: float
    avg_r: float
    median_r: float
    exposure_pct: float
    max_simultaneous_positions: int
    total_commissions: float
    total_slippage_estimate: float
    avg_initial_risk_pct_equity: float


def commission(notional: float) -> float:
    if notional <= 0:
        return 0.0
    pct = float(RISK_CONFIG["execution"]["commission_pct_one_way"]) / 100.0
    minimum = float(RISK_CONFIG["execution"]["minimum_commission_usd"])
    fee = max(notional * pct, minimum)
    return min(fee, notional * 0.01)


def slip_buy(price: float) -> float:
    slip = float(RISK_CONFIG["execution"]["slippage_pct_one_way"]) / 100.0
    return price * (1.0 + slip)


def slip_sell(price: float) -> float:
    slip = float(RISK_CONFIG["execution"]["slippage_pct_one_way"]) / 100.0
    return price * (1.0 - slip)


def atr(df: pd.DataFrame, window: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean()


def metrics_from_equity(
    variant: str,
    stop_rule: str,
    exit_rule: str,
    equity: pd.Series,
    trades_df: pd.DataFrame,
    exposure_days: int,
    max_positions_seen: int,
) -> VariantResult:
    equity = equity.dropna()
    daily = equity.pct_change().fillna(0.0)
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25)
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0 if equity.iloc[-1] > 0 else -1.0
    dd = equity / equity.cummax() - 1.0
    max_dd = float(dd.min())

    ann_vol = float(daily.std(ddof=0) * math.sqrt(365))
    ann_ret = float(daily.mean() * 365)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    downside = float(daily[daily < 0].std(ddof=0) * math.sqrt(365))
    sortino = ann_ret / downside if downside > 0 else float("nan")
    calmar = cagr / abs(max_dd) if max_dd < 0 else float("nan")

    if trades_df.empty:
        wins = pd.Series(dtype=float)
        losses = pd.Series(dtype=float)
        win_rate = 0.0
        pf = float("nan")
        avg_r = float("nan")
        median_r = float("nan")
        total_commissions = 0.0
        total_slippage = 0.0
        avg_risk_pct = float("nan")
    else:
        wins = trades_df.loc[trades_df["pnl"] > 0, "pnl"]
        losses = trades_df.loc[trades_df["pnl"] < 0, "pnl"]
        win_rate = float((trades_df["pnl"] > 0).mean() * 100)
        pf = float(wins.sum() / abs(losses.sum())) if len(losses) and abs(losses.sum()) > 0 else float("inf")
        avg_r = float(trades_df["r_multiple"].mean())
        median_r = float(trades_df["r_multiple"].median())
        total_commissions = float(trades_df["entry_fee"].sum() + trades_df["exit_fee"].sum())
        total_slippage = float(trades_df["slippage_cost_estimate"].sum())
        avg_risk_pct = float(trades_df["initial_risk_pct_equity"].mean())

    exposure = 100.0 * exposure_days / len(equity) if len(equity) else 0.0

    return VariantResult(
        variant=variant,
        stop_rule=stop_rule,
        exit_rule=exit_rule,
        cagr_pct=100 * cagr,
        max_drawdown_pct=100 * max_dd,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        total_return_pct=100 * total_return,
        final_equity=float(equity.iloc[-1]),
        trades=int(len(trades_df)),
        win_rate_pct=win_rate,
        profit_factor=pf,
        avg_r=avg_r,
        median_r=median_r,
        exposure_pct=exposure,
        max_simultaneous_positions=max_positions_seen,
        total_commissions=total_commissions,
        total_slippage_estimate=total_slippage,
        avg_initial_risk_pct_equity=avg_risk_pct,
    )


def build_indicators(prices: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, pd.Series], pd.DataFrame, pd.Series, pd.DataFrame]:
    close = base.align_close(prices)
    atr_window = int(RISK_CONFIG["risk"]["atr_days"])
    atrs = {symbol: atr(df, atr_window) for symbol, df in prices.items()}

    hybrid = base.build_signals(prices)[RISK_CONFIG["strategy"]["signal"]].reindex(close.index).reindex(columns=close.columns).fillna(0.0)
    rank_days = int(RISK_CONFIG["strategy"]["ranking_momentum_days"])
    ranking_momentum = close.pct_change(rank_days)

    s = base.CONFIG["strategies"]["hybrid"]
    benchmark = base.CONFIG["universe"]["benchmark"]
    market_days = int(s["market_momentum_days"])
    regime = (close[benchmark].pct_change(market_days) > 0).fillna(False)
    sma100 = close.rolling(100).mean()
    return close, atrs, hybrid, regime, ranking_momentum, sma100


def equity_at_price(cash: float, positions: dict[str, Position], close_ffill: pd.DataFrame, dt: pd.Timestamp) -> float:
    value = cash
    for symbol, pos in positions.items():
        try:
            px = float(close_ffill.at[dt, symbol])
        except Exception:
            px = float("nan")
        if math.isfinite(px):
            value += pos.qty * px
    return value


def exit_position(
    pos: Position,
    raw_exit_price: float,
    dt: pd.Timestamp,
    reason: str,
    cash: float,
    equity_reference: float,
    trades: list[dict],
) -> float:
    fill = slip_sell(raw_exit_price)
    notional = pos.qty * fill
    fee = commission(notional)
    proceeds = notional - fee
    cash += proceeds

    pnl = pos.qty * (fill - pos.entry_price) - pos.entry_fee - fee
    r_multiple = pnl / pos.risk_cash if pos.risk_cash > 0 else float("nan")
    slip_pct = float(RISK_CONFIG["execution"]["slippage_pct_one_way"]) / 100.0
    slippage_cost = pos.qty * (pos.entry_price / (1.0 + slip_pct)) * slip_pct + pos.qty * raw_exit_price * slip_pct

    trades.append(
        {
            "symbol": pos.symbol,
            "entry_date": pos.entry_date,
            "exit_date": dt,
            "entry_price": pos.entry_price,
            "exit_price": fill,
            "initial_stop": pos.initial_stop,
            "risk_cash": pos.risk_cash,
            "pnl": pnl,
            "r_multiple": r_multiple,
            "exit_reason": reason,
            "entry_fee": pos.entry_fee,
            "exit_fee": fee,
            "slippage_cost_estimate": slippage_cost,
            "initial_risk_pct_equity": 100.0 * pos.risk_cash / equity_reference if equity_reference > 0 else float("nan"),
            "holding_days": int((dt - pos.entry_date).days),
        }
    )
    return cash


def run_variant(
    prices: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    close_ffill: pd.DataFrame,
    atrs: dict[str, pd.Series],
    hybrid: pd.DataFrame,
    regime: pd.Series,
    ranking_momentum: pd.DataFrame,
    sma100: pd.DataFrame,
    stop_cfg: dict,
    exit_cfg: dict,
) -> tuple[VariantResult, pd.Series, pd.DataFrame]:
    initial_capital = float(RISK_CONFIG["capital"])
    risk_pct = float(RISK_CONFIG["risk_per_trade_pct"]) / 100.0
    max_positions = int(RISK_CONFIG["max_positions"])
    max_position_pct = float(RISK_CONFIG["max_position_equity_pct"]) / 100.0
    common_regime_exit = bool(RISK_CONFIG["strategy"].get("common_regime_exit", True))

    dates = close.index.sort_values()
    cash = initial_capital
    positions: dict[str, Position] = {}
    trades: list[dict] = []
    equity_curve: dict[pd.Timestamp, float] = {}
    exposure_days = 0
    max_positions_seen = 0

    for i, dt in enumerate(dates):
        if i == 0:
            equity_curve[dt] = initial_capital
            continue
        prev_dt = dates[i - 1]
        prev_equity = equity_curve.get(prev_dt, initial_capital)
        if not math.isfinite(prev_equity) or prev_equity <= 0:
            prev_equity = equity_at_price(cash, positions, close_ffill, prev_dt)

        # 1) Scheduled exits execute at today's open using information from yesterday's close.
        for symbol in list(positions.keys()):
            pos = positions[symbol]
            if not pos.exit_next_open:
                continue
            row = prices[symbol].loc[dt] if dt in prices[symbol].index else None
            if row is None or not math.isfinite(float(row["open"])):
                continue
            cash = exit_position(
                pos,
                float(row["open"]),
                dt,
                pos.scheduled_exit_reason or "scheduled_exit",
                cash,
                prev_equity,
                trades,
            )
            del positions[symbol]

        # 2) New entries. Eligibility and ranking are based on yesterday's close only.
        slots = max_positions - len(positions)
        if slots > 0 and prev_dt in hybrid.index:
            eligible = hybrid.loc[prev_dt]
            candidates = [s for s in eligible.index if eligible.get(s, 0.0) > 0 and s not in positions]
            if candidates:
                ranks = ranking_momentum.loc[prev_dt, candidates].dropna().sort_values(ascending=False)
                candidates = ranks.index.tolist()[:slots]

                for symbol in candidates:
                    if symbol not in prices or dt not in prices[symbol].index or prev_dt not in prices[symbol].index:
                        continue
                    row = prices[symbol].loc[dt]
                    raw_open = float(row["open"])
                    if not math.isfinite(raw_open) or raw_open <= 0:
                        continue

                    atr_prev = float(atrs[symbol].get(prev_dt, np.nan))
                    if not math.isfinite(atr_prev) or atr_prev <= 0:
                        continue

                    entry_fill = slip_buy(raw_open)
                    if stop_cfg["type"] == "atr":
                        stop = entry_fill - float(stop_cfg["multiplier"]) * atr_prev
                    elif stop_cfg["type"] == "swing":
                        lookback = int(stop_cfg.get("lookback_days", 20))
                        hist = prices[symbol].loc[:prev_dt].tail(lookback)
                        if hist.empty:
                            continue
                        stop = float(hist["low"].min())
                    else:
                        raise ValueError(f"Unknown stop type: {stop_cfg['type']}")

                    if not math.isfinite(stop) or stop <= 0 or stop >= entry_fill:
                        continue

                    risk_per_unit = entry_fill - stop
                    desired_risk_cash = prev_equity * risk_pct
                    qty_by_risk = desired_risk_cash / risk_per_unit
                    max_notional = prev_equity * max_position_pct
                    affordable_notional = max(0.0, (cash - float(RISK_CONFIG["execution"]["minimum_commission_usd"])) / (1.0 + float(RISK_CONFIG["execution"]["commission_pct_one_way"]) / 100.0))
                    desired_notional = min(qty_by_risk * entry_fill, max_notional, affordable_notional)
                    if desired_notional <= 0:
                        continue

                    qty = desired_notional / entry_fill
                    entry_fee = commission(desired_notional)
                    if desired_notional + entry_fee > cash:
                        desired_notional = max(0.0, cash - entry_fee)
                        qty = desired_notional / entry_fill if entry_fill > 0 else 0.0
                    if qty <= 0:
                        continue

                    actual_risk_cash = qty * risk_per_unit
                    cash -= qty * entry_fill + entry_fee

                    target = None
                    if exit_cfg["type"] == "target_r":
                        target = entry_fill + float(exit_cfg["multiple"]) * risk_per_unit

                    positions[symbol] = Position(
                        symbol=symbol,
                        qty=qty,
                        entry_date=dt,
                        entry_price=entry_fill,
                        entry_fee=entry_fee,
                        initial_stop=stop,
                        current_stop=stop,
                        risk_per_unit=risk_per_unit,
                        risk_cash=actual_risk_cash,
                        target=target,
                        highest_high=raw_open,
                    )

        max_positions_seen = max(max_positions_seen, len(positions))

        # 3) Intraday protective stop / target logic. Gap behavior approximates stop-market execution.
        for symbol in list(positions.keys()):
            pos = positions[symbol]
            if dt not in prices[symbol].index:
                continue
            row = prices[symbol].loc[dt]
            raw_open = float(row["open"])
            high = float(row["high"])
            low = float(row["low"])
            if not all(math.isfinite(x) for x in [raw_open, high, low]):
                continue

            stop = pos.current_stop
            target = pos.target
            exit_price = None
            exit_reason = None

            if raw_open <= stop:
                exit_price = raw_open
                exit_reason = "gap_stop"
            elif target is not None and raw_open >= target:
                exit_price = raw_open
                exit_reason = f"target_{exit_cfg['multiple']}R_gap"
            else:
                stop_hit = low <= stop
                target_hit = target is not None and high >= target
                if stop_hit and target_hit:
                    # Conservative daily-bar assumption: adverse stop happens first.
                    exit_price = stop
                    exit_reason = "stop_and_target_same_bar_assume_stop"
                elif stop_hit:
                    exit_price = stop
                    exit_reason = "protective_stop"
                elif target_hit:
                    exit_price = target
                    exit_reason = f"target_{exit_cfg['multiple']}R"

            if exit_price is not None:
                cash = exit_position(pos, float(exit_price), dt, exit_reason or "intraday_exit", cash, prev_equity, trades)
                del positions[symbol]

        # 4) End-of-day trailing-stop update and scheduling for next open.
        for symbol, pos in positions.items():
            if dt not in prices[symbol].index:
                continue
            row = prices[symbol].loc[dt]
            high = float(row["high"])
            pos.highest_high = max(pos.highest_high, high) if math.isfinite(high) else pos.highest_high

            if exit_cfg["type"] == "atr_trail":
                atr_today = float(atrs[symbol].get(dt, np.nan))
                if math.isfinite(atr_today) and atr_today > 0:
                    trail = pos.highest_high - float(exit_cfg["multiplier"]) * atr_today
                    if math.isfinite(trail):
                        pos.current_stop = max(pos.current_stop, trail)

            reason = None
            if common_regime_exit and dt in regime.index and not bool(regime.loc[dt]):
                reason = "market_regime_off"
            elif exit_cfg["type"] == "sma":
                sma_window = int(exit_cfg.get("window_days", 100))
                if sma_window != 100:
                    local_sma = close[symbol].rolling(sma_window).mean()
                    sma_value = float(local_sma.get(dt, np.nan))
                else:
                    sma_value = float(sma100.at[dt, symbol]) if dt in sma100.index and symbol in sma100.columns else float("nan")
                close_value = float(close.at[dt, symbol]) if dt in close.index and symbol in close.columns else float("nan")
                if math.isfinite(sma_value) and math.isfinite(close_value) and close_value <= sma_value:
                    reason = f"close_below_sma{sma_window}"

            if reason:
                pos.exit_next_open = True
                pos.scheduled_exit_reason = reason

        if positions:
            exposure_days += 1
        equity_curve[dt] = equity_at_price(cash, positions, close_ffill, dt)

    # Liquidate any remaining positions at the final close.
    final_dt = dates[-1]
    final_equity_before = equity_curve[final_dt]
    for symbol in list(positions.keys()):
        pos = positions[symbol]
        raw_close = float(close_ffill.at[final_dt, symbol])
        cash = exit_position(pos, raw_close, final_dt, "end_of_backtest", cash, final_equity_before, trades)
        del positions[symbol]
    equity_curve[final_dt] = cash

    equity = pd.Series(equity_curve).sort_index().astype(float)
    trades_df = pd.DataFrame(trades)
    variant_name = f"{stop_cfg['name']}__{exit_cfg['name']}"
    result = metrics_from_equity(
        variant_name,
        stop_cfg["name"],
        exit_cfg["name"],
        equity,
        trades_df,
        exposure_days,
        max_positions_seen,
    )
    return result, equity, trades_df


def main() -> None:
    # Reuse the corrected Yahoo historical loader and the existing D_hybrid signal engine.
    prices = yahoo.load_prices_yahoo()
    benchmark = base.CONFIG["universe"]["benchmark"]
    if benchmark not in prices:
        raise RuntimeError("BTC benchmark unavailable")

    close, atrs, hybrid, regime, ranking_momentum, sma100 = build_indicators(prices)
    close_ffill = close.ffill()

    results: list[VariantResult] = []
    annual_frames: dict[str, pd.Series] = {}
    all_trades: list[pd.DataFrame] = []

    for stop_cfg in RISK_CONFIG["risk"]["stops"]:
        for exit_cfg in RISK_CONFIG["exits"]:
            print(f"Running {stop_cfg['name']} + {exit_cfg['name']}")
            result, equity, trades_df = run_variant(
                prices,
                close,
                close_ffill,
                atrs,
                hybrid,
                regime,
                ranking_momentum,
                sma100,
                stop_cfg,
                exit_cfg,
            )
            results.append(result)
            annual_frames[result.variant] = equity.pct_change().fillna(0.0)
            if not trades_df.empty:
                trades_df = trades_df.copy()
                trades_df.insert(0, "variant", result.variant)
                all_trades.append(trades_df)

    result_df = pd.DataFrame([r.__dict__ for r in results]).sort_values(["calmar", "cagr_pct"], ascending=[False, False])
    annual = pd.DataFrame({name: (1.0 + ret).groupby(ret.index.year).prod() - 1.0 for name, ret in annual_frames.items()}) * 100.0
    trades_out = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    result_df.to_csv(OUT_DIR / "risk_backtest_results.csv", index=False)
    annual.to_csv(OUT_DIR / "risk_backtest_annual_returns.csv")
    trades_out.to_csv(OUT_DIR / "risk_backtest_trades.csv", index=False)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_source": "Yahoo Finance daily OHLC via yfinance",
        "base_signal": "D_hybrid from crypto-backtest branch",
        "config": RISK_CONFIG,
        "symbols_loaded": sorted(prices.keys()),
        "results": result_df.replace({np.nan: None, np.inf: None, -np.inf: None}).to_dict(orient="records"),
        "annual_returns_pct": annual.replace({np.nan: None, np.inf: None, -np.inf: None}).to_dict(),
    }
    (OUT_DIR / "risk_backtest_results.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print("\n=== RISK-NORMALIZED CRYPTO BACKTEST ===")
    print(result_df.to_string(index=False))
    print("\n=== ANNUAL RETURNS (%) ===")
    print(annual.to_string())
    if not trades_out.empty:
        print("\n=== EXIT REASONS ===")
        print(trades_out.groupby(["variant", "exit_reason"]).size().to_string())


if __name__ == "__main__":
    main()
