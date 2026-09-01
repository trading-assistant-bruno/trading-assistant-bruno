from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import active_momentum_rotation as rot
import meta_regime_backtest as meta
import meta_regime_backtest_v2 as robust

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "momentum_burst_backtest"
OUT.mkdir(parents=True, exist_ok=True)

INITIAL = 100_000.0
DELIST_HAIRCUT = 0.02
STOP_ATR = 2.5
ADD1_FRACTION = 0.50
ADD2_FRACTION = 0.50
EXIT_RANK = 50


@dataclass(frozen=True)
class Variant:
    name: str
    max_positions: int
    pool_rank: int
    risk_pct: float
    gate_kind: str
    breakout_n: int
    trail_atr: float
    pyramid: bool
    max_position_pct: float


VARIANTS = [
    Variant("Burst3_R075_Select_B20_T4_Pyr", 3, 10, 0.0075, "selective", 20, 4.0, True, 0.35),
    Variant("Burst5_R075_Select_B20_T4_Pyr", 5, 15, 0.0075, "selective", 20, 4.0, True, 0.25),
    Variant("Burst5_R100_Select_B20_T4_Pyr", 5, 15, 0.0100, "selective", 20, 4.0, True, 0.25),
    Variant("Burst5_R075_Strong_B20_T4_Pyr", 5, 15, 0.0075, "strong", 20, 4.0, True, 0.25),
    Variant("Burst5_R075_Select_B20_T3_Pyr", 5, 15, 0.0075, "selective", 20, 3.0, True, 0.25),
    Variant("Burst5_R075_Select_B20_T4_NoPyr", 5, 15, 0.0075, "selective", 20, 4.0, False, 0.25),
    Variant("Burst5_R075_Select_B55_T4_Pyr", 5, 15, 0.0075, "selective", 55, 4.0, True, 0.25),
]


def weekly_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index=index, data=index)
    return pd.DatetimeIndex(s.groupby(index.to_period("W-FRI")).max().values)


def true_range(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    prev = close.shift(1)
    a = high - low
    b = (high - prev).abs()
    c = (low - prev).abs()
    return a.where((a >= b) & (a >= c), b.where(b >= c, c))


def gate_features(date, members, close, ind, qqq_sma200, spy_mom63, qqq_mom63, spy_vol20):
    priced = [t for t in members if t in close.columns and t in ind["sma200"].columns
              and pd.notna(close.at[date, t]) and pd.notna(ind["sma200"].at[date, t])]
    breadth = float(np.mean([close.at[date, t] > ind["sma200"].at[date, t] for t in priced])) if priced else np.nan
    spy = float(close.at[date, "SPY"])
    spy200 = float(ind["sma200"].at[date, "SPY"]) if pd.notna(ind["sma200"].at[date, "SPY"]) else np.nan
    q = float(close.at[date, "QQQ"]) if "QQQ" in close.columns and pd.notna(close.at[date, "QQQ"]) else np.nan
    q200 = float(qqq_sma200.at[date]) if pd.notna(qqq_sma200.at[date]) else np.nan
    sm3 = float(spy_mom63.at[date]) if pd.notna(spy_mom63.at[date]) else np.nan
    qm3 = float(qqq_mom63.at[date]) if pd.notna(qqq_mom63.at[date]) else np.nan
    vol = float(spy_vol20.at[date]) if pd.notna(spy_vol20.at[date]) else np.nan
    broad = bool(np.isfinite(spy200) and spy > spy200 and np.isfinite(sm3) and sm3 > 0 and np.isfinite(breadth) and breadth >= 0.50)
    strong = bool(broad and np.isfinite(q) and np.isfinite(q200) and q > q200 and np.isfinite(qm3) and qm3 > 0 and breadth >= 0.60)
    selective = bool(strong and breadth >= 0.65 and sm3 >= 0.03 and np.isfinite(vol) and vol < 0.25)
    return {
        "date": date,
        "breadth": breadth,
        "spy_mom63": sm3,
        "qqq_mom63": qm3,
        "spy_vol20_ann": vol,
        "broad": broad,
        "strong": strong,
        "selective": selective,
        "priced_members": len(priced),
    }


def buy_fill(cash, raw_open, qty, max_cash=None):
    if not np.isfinite(raw_open) or raw_open <= 0 or qty <= 0:
        return cash, 0.0, 0.0, 0.0
    px = raw_open * (1 + rot.SLIPPAGE)
    desired = qty * px
    if max_cash is not None:
        desired = min(desired, max(0.0, max_cash))
    desired = min(desired, max(0.0, cash - rot.MIN_COMMISSION))
    if desired <= 100:
        return cash, 0.0, 0.0, 0.0
    fee = rot.commission(desired)
    if desired + fee > cash:
        desired = max(0.0, cash - fee)
    if desired <= 100:
        return cash, 0.0, 0.0, 0.0
    qty2 = desired / px
    cash -= desired + fee
    return cash, qty2, desired, fee


def sell_fill(cash, raw_px, qty, haircut=0.0):
    if not np.isfinite(raw_px) or raw_px <= 0 or qty <= 0:
        return cash, 0.0, 0.0, 0.0
    px = raw_px * (1 - haircut) * (1 - rot.SLIPPAGE)
    notion = qty * px
    fee = rot.commission(notion)
    cash += notion - fee
    return cash, notion, fee, px


def simulate(v: Variant, open_, high, low, close, close_ffill, ind, atr20, sma50, breakout_map,
             week_dates, memberships, sections, gates):
    dates = close.index[(close.index >= rot.TEST_START) & (close.index < rot.END_EXCLUSIVE)]
    weekset = set(week_dates)
    cash = INITIAL
    positions = {}
    current_gate = False
    current_cs = pd.DataFrame()
    pending_entries = []
    pending_exits = {}
    pending_adds = {}
    equity_rows = []
    fills = []
    closed_trades = []
    exposure_sum = 0.0
    gate_days = 0

    def mark(date, t):
        if t not in close_ffill.columns:
            return np.nan
        z = close_ffill.at[date, t]
        return float(z) if pd.notna(z) else np.nan

    def portfolio_equity(date):
        total = cash
        invested = 0.0
        for t, p in positions.items():
            px = mark(date, t)
            if np.isfinite(px):
                val = p["shares"] * px
                invested += val
                total += val
        return total, invested

    def close_position(date, t, reason, raw_px=None, haircut=0.0):
        nonlocal cash
        if t not in positions:
            return
        p = positions[t]
        if raw_px is None or not np.isfinite(raw_px) or raw_px <= 0:
            raw_px = mark(date, t)
            haircut = max(haircut, DELIST_HAIRCUT)
        cash, notion, fee, px = sell_fill(cash, float(raw_px), p["shares"], haircut)
        if notion <= 0:
            return
        proceeds = notion - fee
        pnl = proceeds - p["buy_cash"]
        r_mult = pnl / p["initial_risk_cash"] if p["initial_risk_cash"] > 0 else np.nan
        closed_trades.append({
            "variant": v.name, "ticker": t, "entry_date": p["entry_date"], "exit_date": date,
            "reason": reason, "pnl": pnl, "r_multiple": r_mult, "days_held": (date-p["entry_date"]).days,
            "adds": p["adds"], "initial_risk_cash": p["initial_risk_cash"],
            "buy_cash": p["buy_cash"], "sell_proceeds": proceeds,
        })
        fills.append({"variant":v.name,"date":date,"ticker":t,"side":"SELL","reason":reason,"notional":notion,"fee":fee,"price":px})
        positions.pop(t, None)
        pending_adds.pop(t, None)

    for date in dates:
        # 1) Execute exits signaled at prior close.
        for t, reason in list(pending_exits.items()):
            if t not in positions:
                continue
            op = open_.at[date, t] if t in open_.columns else np.nan
            close_position(date, t, reason, float(op) if pd.notna(op) else None)
        pending_exits = {}

        # 2) Execute pyramids signaled at prior close.
        eq_open, _ = portfolio_equity(date)
        for t, stage in list(pending_adds.items()):
            if t not in positions:
                continue
            p = positions[t]
            op = open_.at[date, t] if t in open_.columns else np.nan
            if pd.isna(op) or float(op) <= 0:
                continue
            pos_cap = eq_open * v.max_position_pct
            cur_val = p["shares"] * float(op)
            room = max(0.0, pos_cap - cur_val)
            add_qty = p["initial_units"] * (ADD1_FRACTION if stage == 1 else ADD2_FRACTION)
            cash, qty, notion, fee = buy_fill(cash, float(op), add_qty, room)
            if qty > 0:
                p["shares"] += qty
                p["buy_cash"] += notion + fee
                p["adds"] += 1
                if stage == 1:
                    p["add1_done"] = True
                    p["stop"] = max(p["stop"], p["entry_price"])
                else:
                    p["add2_done"] = True
                    p["stop"] = max(p["stop"], p["entry_price"] + p["stop_dist"])
                fills.append({"variant":v.name,"date":date,"ticker":t,"side":"BUY_ADD","reason":f"ADD{stage}","notional":notion,"fee":fee,"price":float(op)*(1+rot.SLIPPAGE)})
        pending_adds = {}

        # 3) Execute new entries signaled at prior close.
        eq_open, _ = portfolio_equity(date)
        for ent in pending_entries:
            t = ent["ticker"]
            if not current_gate or t in positions or len(positions) >= v.max_positions:
                continue
            op = open_.at[date, t] if t in open_.columns else np.nan
            if pd.isna(op) or float(op) <= 0:
                continue
            stop_dist = STOP_ATR * ent["atr"]
            if not np.isfinite(stop_dist) or stop_dist <= 0:
                continue
            risk_budget = eq_open * v.risk_pct
            wanted_qty = risk_budget / stop_dist
            max_cash = eq_open * v.max_position_pct
            cash, qty, notion, fee = buy_fill(cash, float(op), wanted_qty, max_cash)
            if qty <= 0:
                continue
            fill_px = float(op) * (1 + rot.SLIPPAGE)
            actual_risk = qty * stop_dist
            positions[t] = {
                "shares": qty, "initial_units": qty, "entry_date": date, "entry_price": fill_px,
                "stop_dist": stop_dist, "stop": fill_px - stop_dist, "highest_close": fill_px,
                "initial_risk_cash": actual_risk, "buy_cash": notion + fee, "adds": 0,
                "add1_done": False, "add2_done": False,
            }
            fills.append({"variant":v.name,"date":date,"ticker":t,"side":"BUY","reason":"BREAKOUT","notional":notion,"fee":fee,"price":fill_px})
        pending_entries = []

        # 4) Intraday stops, using stops known before this bar.
        for t in list(positions):
            p = positions[t]
            if t not in low.columns or pd.isna(low.at[date,t]):
                continue
            lo = float(low.at[date,t]); op = open_.at[date,t] if t in open_.columns else np.nan
            if lo <= p["stop"]:
                raw = float(op) if pd.notna(op) and float(op) < p["stop"] else p["stop"]
                close_position(date, t, "TRAIL_STOP", raw)

        # 5) Close marks and trailing stop update.
        total, invested = portfolio_equity(date)
        exposure = invested / total if total > 0 else 0.0
        exposure_sum += exposure
        gate_days += int(current_gate)
        equity_rows.append((date,total,cash,len(positions),exposure,current_gate))

        for t,p in list(positions.items()):
            if t not in close.columns or pd.isna(close.at[date,t]):
                continue
            cl = float(close.at[date,t])
            p["highest_close"] = max(p["highest_close"], cl)
            a = atr20.at[date,t] if t in atr20.columns else np.nan
            if pd.notna(a) and float(a)>0:
                p["stop"] = max(p["stop"], p["highest_close"] - v.trail_atr*float(a))

        # 6) Weekly information becomes actionable for the next session.
        if date in weekset:
            current_cs = sections.get(date, pd.DataFrame())
            current_gate = bool(gates.get(date, {}).get(v.gate_kind, False)) if v.gate_kind != "none" else True
            ranks = current_cs["rank"].to_dict() if not current_cs.empty else {}
            for t in list(positions):
                if not current_gate:
                    pending_exits[t] = "GATE_OFF"
                elif ranks.get(t, np.inf) > EXIT_RANK:
                    pending_exits[t] = "RANK_EXIT"

        # 7) Pyramid signals at close. Cancel if exit is already pending.
        if v.pyramid and current_gate:
            for t,p in positions.items():
                if t in pending_exits or t not in close.columns or pd.isna(close.at[date,t]):
                    continue
                cl = float(close.at[date,t])
                if (not p["add1_done"]) and cl >= p["entry_price"] + p["stop_dist"]:
                    pending_adds[t] = 1
                elif p["add1_done"] and (not p["add2_done"]) and cl >= p["entry_price"] + 2*p["stop_dist"]:
                    pending_adds[t] = 2

        # 8) New breakout signals using current weekly ranking; execute next open.
        if current_gate and not current_cs.empty:
            future_count = len(positions) - len([t for t in pending_exits if t in positions])
            slots = max(0, v.max_positions - future_count)
            if slots > 0:
                elig = current_cs[current_cs.base_eligible].copy()
                elig = elig[(elig["rank"] <= v.pool_rank)]
                candidates=[]
                for t,row in elig.iterrows():
                    if t in positions or t in pending_exits:
                        continue
                    if t not in sma50.columns or t not in ind["sma200"].columns or t not in close.columns:
                        continue
                    vals=[close.at[date,t],sma50.at[date,t],ind["sma200"].at[date,t],ind["high52"].at[date,t],ind["mom6"].at[date,t],atr20.at[date,t]]
                    if any(pd.isna(z) for z in vals):
                        continue
                    cl,s50,s200,h52,m6,a=map(float,vals)
                    if not (cl > s50 > s200 and h52 >= 0.90 and m6 > 0 and a > 0):
                        continue
                    br = breakout_map[v.breakout_n].at[date,t] if t in breakout_map[v.breakout_n].columns else False
                    if bool(br):
                        candidates.append((float(row["rank"]),t,a))
                candidates.sort()
                pending_entries=[{"ticker":t,"atr":a,"rank":r} for r,t,a in candidates[:slots]]

    # Terminal liquidation at last known prices.
    if len(dates):
        last=dates[-1]
        for t in list(positions):
            close_position(last,t,"TERMINAL",mark(last,t),0.0)
        if equity_rows:
            equity_rows[-1]=(last,cash,cash,0,0.0,current_gate)

    eq=pd.DataFrame(equity_rows,columns=["date","equity","cash","positions","exposure","gate_on"]).set_index("date")
    trades=pd.DataFrame(closed_trades)
    fills_df=pd.DataFrame(fills)
    return eq,trades,fills_df


def trade_stats(trades):
    if trades.empty:
        return {}
    wins=trades[trades.pnl>0]; losses=trades[trades.pnl<0]
    gp=float(wins.pnl.sum()); gl=float(-losses.pnl.sum())
    return {
        "trades":int(len(trades)),
        "win_rate_pct":100*float((trades.pnl>0).mean()),
        "profit_factor":gp/gl if gl>0 else np.nan,
        "avg_r":float(trades.r_multiple.mean()),
        "median_r":float(trades.r_multiple.median()),
        "avg_win_r":float(wins.r_multiple.mean()) if len(wins) else np.nan,
        "avg_loss_r":float(losses.r_multiple.mean()) if len(losses) else np.nan,
        "max_trade_r":float(trades.r_multiple.max()),
        "trades_ge_5r":int((trades.r_multiple>=5).sum()),
        "trades_ge_10r":int((trades.r_multiple>=10).sum()),
        "median_hold_days":float(trades.days_held.median()),
        "mean_hold_days":float(trades.days_held.mean()),
    }


def blend_annual(world, burst, weight_burst):
    common=world.index.intersection(burst.index)
    wr=world.reindex(common).pct_change().fillna(0)
    br=burst.reindex(common).pct_change().fillna(0)
    eq=INITIAL
    world_val=eq*(1-weight_burst); burst_val=eq*weight_burst
    vals=[]; current_year=None
    for d in common:
        if current_year is None:
            current_year=d.year
        elif d.year!=current_year:
            total=world_val+burst_val
            world_val=total*(1-weight_burst); burst_val=total*weight_burst
            current_year=d.year
        world_val*=1+float(wr.at[d]); burst_val*=1+float(br.at[d])
        vals.append((d,world_val+burst_val))
    return pd.Series(dict(vals)).sort_index()


def main():
    hist,symbols=rot.load_history_symbols()
    symbols=sorted(set(symbols)|{"QQQ"})
    print(f"Downloading {len(symbols)} point-in-time symbols + QQQ")
    frames=rot.base.download_prices(symbols,rot.DATA_START.strftime("%Y-%m-%d"),(rot.END_EXCLUSIVE-pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    open_,high,low,close,volume=rot.base.matrices(frames)
    close=close.loc[(close.index>=rot.DATA_START)&(close.index<rot.END_EXCLUSIVE)]
    open_=open_.reindex(close.index); high=high.reindex(close.index); low=low.reindex(close.index); volume=volume.reindex(close.index)
    if "SPY" not in close.columns or "QQQ" not in close.columns:
        raise RuntimeError("SPY/QQQ unavailable")
    close_ffill=close.ffill()
    ind=rot.indicators(close,volume)
    tr=true_range(high,low,close)
    atr20=tr.rolling(20).mean()
    sma50=close.rolling(50).mean()
    qqq_sma200=close.QQQ.rolling(200).mean()
    spy_mom63=close.SPY/close.SPY.shift(63)-1
    qqq_mom63=close.QQQ/close.QQQ.shift(63)-1
    spy_vol20=close.SPY.pct_change().rolling(20).std(ddof=0)*math.sqrt(252)
    breakout_map={n: close > close.shift(1).rolling(n).max() for n in sorted({v.breakout_n for v in VARIANTS})}

    test_index=close.index[(close.index>=rot.TEST_START)&(close.index<rot.END_EXCLUSIVE)]
    weeks=weekly_dates(test_index)
    mem=rot.memberships(hist,weeks)
    sections={}; gates={}; gate_rows=[]; coverage=[]
    for d in weeks:
        m=mem.get(d,set())
        cs=rot.section(d,m,close,ind,"composite")
        sections[d]=cs
        gf=gate_features(d,m,close,ind,qqq_sma200,spy_mom63,qqq_mom63,spy_vol20)
        gates[d]=gf; gate_rows.append(gf)
        coverage.append({"date":d,"members":len(m),"priced_members":gf["priced_members"],"coverage_pct":100*gf["priced_members"]/max(len(m),1)})

    last=test_index[-1]
    world_raw=rot.world_netr(rot.TEST_START,last)
    common=min(last,world_raw.index.max())
    calendar=test_index[test_index<=common]
    world=rot.benchmark(world_raw,rot.TEST_START,common).reindex(calendar).ffill().bfill()
    spy=rot.benchmark(close.SPY,rot.TEST_START,common).reindex(calendar).ffill().bfill()

    # Robust Top20 benchmark from the prior engine on the same downloaded universe.
    msigs=rot.monthly_dates(test_index)
    mmem=rot.memberships(hist,msigs)
    msecs={("composite",d):rot.section(d,mmem.get(d,set()),close,ind,"composite") for d in msigs}
    top20df,_,_=robust.robust_simulate(meta.TOP20,open_,close,msigs,msecs,ind)
    top20=top20df.equity.reindex(calendar).ffill().bfill()

    rows=[]; annual={}; curves={"MSCI_World_NETR":world,"SPY_BuyHold":spy,"Active_Top20":top20};
    all_trades=[]; all_fills=[]; subs=[]; blends=[]
    for v in VARIANTS:
        eqdf,trades,fills=simulate(v,open_,high,low,close,close_ffill,ind,atr20,sma50,breakout_map,weeks,mem,sections,gates)
        eq=eqdf.equity.loc[:common]
        z=rot.stats(eq); ts=trade_stats(trades)
        ann=((1+eq.pct_change().fillna(0)).groupby(eq.index.year).prod()-1)*100
        z.update(strategy=v.name,start=str(eq.index[0].date()),end=str(eq.index[-1].date()),
                 avg_exposure_pct=100*float(eqdf.loc[:common,"exposure"].mean()),
                 gate_on_pct=100*float(eqdf.loc[:common,"gate_on"].mean()),
                 best_year_pct=float(ann.max()),worst_year_pct=float(ann.min()),
                 years_ge_40=int((ann>=40).sum()),years_ge_50=int((ann>=50).sum()),**ts)
        rows.append(z); annual[v.name]=ann; curves[v.name]=eq
        if not trades.empty: all_trades.append(trades)
        if not fills.empty: all_fills.append(fills)
        for lab,a,b in [("2011-2017","2011-01-03","2017-12-31"),("2018-2022","2018-01-01","2022-12-31"),("2023-2026","2023-01-01","2026-08-31")]:
            s=eq.loc[pd.Timestamp(a):pd.Timestamp(b)]
            if len(s)>30:
                zz=rot.stats(s); zz.update(strategy=v.name,period=lab); subs.append(zz)
        for w in [0.05,0.10]:
            b=blend_annual(world,eq,w)
            bz=rot.stats(b); bz.update(strategy=f"World{int((1-w)*100)}_Burst{int(w*100)}__{v.name}",burst_weight_pct=100*w,burst_variant=v.name)
            blends.append(bz)

    for name,eq in [("MSCI_World_NETR",world),("SPY_BuyHold",spy),("Active_Top20",top20)]:
        z=rot.stats(eq)
        ann=((1+eq.pct_change().fillna(0)).groupby(eq.index.year).prod()-1)*100
        z.update(strategy=name,start=str(eq.index[0].date()),end=str(eq.index[-1].date()),best_year_pct=float(ann.max()),worst_year_pct=float(ann.min()),years_ge_40=int((ann>=40).sum()),years_ge_50=int((ann>=50).sum()))
        rows.append(z); annual[name]=ann

    res=pd.DataFrame(rows).sort_values("cagr_pct",ascending=False)
    ann_df=pd.DataFrame(annual)
    curve_df=pd.DataFrame(curves)
    trades_df=pd.concat(all_trades,ignore_index=True) if all_trades else pd.DataFrame()
    fills_df=pd.concat(all_fills,ignore_index=True) if all_fills else pd.DataFrame()
    blend_df=pd.DataFrame(blends).sort_values("cagr_pct",ascending=False)
    gate_df=pd.DataFrame(gate_rows); cov_df=pd.DataFrame(coverage); sub_df=pd.DataFrame(subs)

    res.to_csv(OUT/"results.csv",index=False); ann_df.to_csv(OUT/"annual_returns_pct.csv"); curve_df.to_csv(OUT/"equity_curves.csv")
    trades_df.to_csv(OUT/"closed_trades.csv",index=False); fills_df.to_csv(OUT/"fills.csv",index=False); blend_df.to_csv(OUT/"world_burst_blends.csv",index=False)
    gate_df.to_csv(OUT/"weekly_gate_features.csv",index=False); cov_df.to_csv(OUT/"universe_coverage.csv",index=False); sub_df.to_csv(OUT/"subperiod_results.csv",index=False)

    # Latest actionable research snapshot (not individualized advice).
    last_week=weeks[weeks<=common][-1]
    latest_cs=sections[last_week]
    latest_elig=latest_cs[latest_cs.base_eligible].copy()
    latest_elig=latest_elig[(latest_elig.close>latest_elig.sma200)&(latest_elig.high52>=0.90)&(latest_elig.mom6>0)].head(20)
    latest_elig.reset_index(names="ticker").to_csv(OUT/"latest_leader_snapshot.csv",index=False)

    payload={
        "generated_at":datetime.now(timezone.utc).isoformat(),
        "method":"Aggressive long-only Momentum Burst pocket. Point-in-time S&P500 universe; weekly composite momentum ranking; daily 20/55-day breakout; 2.5 ATR initial stop; 3/4 ATR Chandelier trail; optional 0.5x+0.5x pyramids at +1R/+2R; next-open execution; 0.08% commission + 0.05% one-way slippage; no leverage; gate can force cash.",
        "goal":"Test a small, aggressive, convex satellite capable of large gains in favorable regimes rather than a full-capital replacement for MSCI World.",
        "gate_definitions":{"broad":"SPY>SMA200, SPY 3m momentum>0, breadth>=50%","strong":"broad + QQQ>SMA200 + QQQ 3m momentum>0 + breadth>=60%","selective":"strong + breadth>=65% + SPY 3m momentum>=3% + SPY 20d annualized vol<25%"},
        "limitations":["Yahoo missing delisted/renamed historical names leaves residual data-availability bias.","2% haircut is only an approximation for stale/delisted forced exits.","No taxes modeled.","Pyramiding/stop execution uses daily OHLC, so exact intraday path is unknown; gaps are handled conservatively at the open.","Variants were specified ex ante for this test, but selecting the best variant after seeing results is still subject to data-mining risk."],
        "coverage":{"median_pct":float(cov_df.coverage_pct.median()),"min_pct":float(cov_df.coverage_pct.min()),"last_pct":float(cov_df.coverage_pct.iloc[-1])},
        "latest_gate":gates[last_week],
        "results":res.replace({np.nan:None,np.inf:None,-np.inf:None}).to_dict("records"),
        "blends_top":blend_df.head(10).replace({np.nan:None,np.inf:None,-np.inf:None}).to_dict("records"),
    }
    (OUT/"results.json").write_text(json.dumps(payload,indent=2,default=str),encoding="utf-8")

    cols=["strategy","cagr_pct","max_drawdown_pct","sharpe_0rf","final_value","volatility_pct","avg_exposure_pct","best_year_pct","worst_year_pct","years_ge_40","trades","win_rate_pct","profit_factor","avg_r","avg_win_r","avg_loss_r","max_trade_r","trades_ge_5r","trades_ge_10r"]
    print("COMMON END",common.date())
    print("\nGATE FREQUENCY\n",gate_df[["broad","strong","selective"]].mean().mul(100).to_string())
    print("\nRESULTS\n",res.reindex(columns=cols).to_string(index=False))
    print("\nTOP WORLD+BURST BLENDS\n",blend_df[["strategy","cagr_pct","max_drawdown_pct","sharpe_0rf","final_value"]].head(14).to_string(index=False))
    print("\nSUBPERIODS\n",sub_df[["strategy","period","cagr_pct","max_drawdown_pct","sharpe_0rf"]].to_string(index=False))
    print("\nLATEST GATE\n",pd.Series(gates[last_week]).to_string())
    print("\nLATEST LEADERS\n",latest_elig[["rank","score","mom12","mom6","high52"]].head(15).to_string())


if __name__=="__main__":
    main()
