\
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import yaml
import yfinance as yf
from jinja2 import Template

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config.yml").read_text(encoding="utf-8"))
DOCS = ROOT / "docs"
DATA = ROOT / "data"
DOCS.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)


@dataclass
class Candidate:
    ticker: str
    close: float
    pivot: float
    entry: float
    stop: float
    stop_pct: float
    shares: int
    position_value: float
    risk_eur: float
    score: float
    distance_52w_high_pct: float
    avg_dollar_volume: float
    reasons: list[str]


def download_universe() -> list[str]:
    """Download US-listed symbols from Nasdaq Trader symbol directories."""
    urls = [
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
    ]
    symbols: list[str] = []
    for url in urls:
        try:
            df = pd.read_csv(url, sep="|")
            col = "Symbol" if "Symbol" in df.columns else "ACT Symbol"
            if col not in df.columns:
                continue
            for raw in df[col].dropna().astype(str):
                s = raw.strip().replace(".", "-")
                if (
                    s
                    and "File Creation Time" not in s
                    and "$" not in s
                    and len(s) <= 6
                ):
                    symbols.append(s)
        except Exception as exc:
            print(f"Universe source failed: {url}: {exc}")
    # Remove obvious warrants/rights/units where suffix conventions are common.
    filtered = sorted(
        {
            s
            for s in symbols
            if not s.endswith(("-W", "-WS", "-U", "-R"))
            and "^" not in s
        }
    )
    max_symbols = int(CONFIG["universe"]["max_symbols"])
    return filtered[:max_symbols]


def batched(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def fetch_prices(symbols: list[str], period: str = "2y") -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for n, batch in enumerate(batched(symbols, 150), start=1):
        print(f"Downloading batch {n}: {len(batch)} symbols")
        try:
            data = yf.download(
                tickers=batch,
                period=period,
                interval="1d",
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False,
                timeout=30,
            )
            if len(batch) == 1:
                ticker = batch[0]
                if not data.empty:
                    out[ticker] = data.dropna(how="all")
            else:
                for ticker in batch:
                    try:
                        frame = data[ticker].dropna(how="all")
                        if not frame.empty:
                            out[ticker] = frame
                    except Exception:
                        continue
        except Exception as exc:
            print(f"Batch failed: {exc}")
        time.sleep(0.25)
    return out


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.columns = [str(c).title() for c in x.columns]
    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(set(x.columns)):
        raise ValueError("Missing OHLCV columns")
    for n in (21, 50, 150, 200):
        x[f"SMA{n}"] = x["Close"].rolling(n).mean()
    x["High52"] = x["High"].rolling(252).max()
    x["AvgDollarVol20"] = (x["Close"] * x["Volume"]).rolling(20).mean()
    return x


def market_regime() -> dict:
    tickers = ["QQQ", "SPY", "^IXIC", "^VIX"]
    frames = fetch_prices(tickers, period="1y")
    details = {}
    positives = 0

    for ticker in ["QQQ", "SPY", "^IXIC"]:
        df = add_indicators(frames[ticker])
        last = df.iloc[-1]
        slope_days = int(CONFIG["strategy"]["sma50_slope_days"])
        sma50_rising = bool(last["SMA50"] > df["SMA50"].iloc[-1 - slope_days])
        above_50 = bool(last["Close"] > last["SMA50"])
        above_200 = bool(last["Close"] > last["SMA200"])
        score = sum([above_50, above_200, sma50_rising])
        positives += score
        details[ticker] = {
            "close": round(float(last["Close"]), 2),
            "above_sma50": above_50,
            "above_sma200": above_200,
            "sma50_rising": sma50_rising,
        }

    vix = add_indicators(frames["^VIX"])
    vix_close = float(vix.iloc[-1]["Close"])
    vix_ok = vix_close < 25
    positives += int(vix_ok)
    details["VIX"] = {"close": round(vix_close, 2), "below_25": vix_ok}

    # 10 binary points: 3 indices x 3 conditions + VIX.
    if positives >= 9:
        color = "VERT"
    elif positives >= 6:
        color = "ORANGE"
    else:
        color = "ROUGE"

    return {
        "color": color,
        "score": positives * 10,
        "details": details,
        "new_positions_allowed": {"VERT": 4, "ORANGE": 1, "ROUGE": 0}[color],
    }


def analyze_symbol(ticker: str, raw: pd.DataFrame) -> Candidate | None:
    try:
        df = add_indicators(raw).dropna()
        if len(df) < 252:
            return None
        last = df.iloc[-1]
        close = float(last["Close"])
        cfg_u = CONFIG["universe"]
        cfg_s = CONFIG["strategy"]

        if close < float(cfg_u["min_price"]):
            return None
        avg_dollar_vol = float(last["AvgDollarVol20"])
        if avg_dollar_vol < float(cfg_u["min_avg_dollar_volume"]):
            return None

        # Objective Minervini-style trend template.
        if not (close > last["SMA50"] > last["SMA150"] > last["SMA200"]):
            return None
        slope_days = int(cfg_s["sma200_slope_days"])
        if float(last["SMA200"]) <= float(df["SMA200"].iloc[-1 - slope_days]):
            return None

        high52 = float(last["High52"])
        dist_high = (high52 - close) / high52 * 100
        if dist_high > float(cfg_s["max_distance_from_52w_high_pct"]):
            return None

        lookback = int(cfg_s["pivot_lookback_days"])
        exclude = int(cfg_s["pivot_exclude_recent_days"])
        pivot_window = df["High"].iloc[-lookback:-exclude]
        pivot = float(pivot_window.max())
        entry = pivot * 1.001
        distance_above_pivot = (close - pivot) / pivot * 100
        if distance_above_pivot > float(cfg_s["max_distance_above_pivot_pct"]):
            return None

        recent_swing_low = float(df["Low"].iloc[-10:].min()) * 0.995
        min_stop = entry * (1 - float(cfg_s["min_stop_distance_pct"]) / 100)
        max_stop = entry * (1 - float(cfg_s["max_stop_distance_pct"]) / 100)
        # Stop below swing low but constrained to the permitted 3–8% band.
        stop = min(min_stop, recent_swing_low)
        stop = max(stop, max_stop)
        stop_pct = (entry - stop) / entry * 100
        if stop_pct <= 0 or stop_pct > float(cfg_s["max_stop_distance_pct"]):
            return None

        capital = float(CONFIG["capital_eur"])
        risk_eur = capital * float(CONFIG["risk_per_trade_pct"]) / 100
        shares = max(0, math.floor(risk_eur / (entry - stop)))
        if shares < 1:
            return None

        position_value = shares * entry
        volume_ratio = float(last["Volume"] / max(df["Volume"].iloc[-21:-1].mean(), 1))
        volatility_now = float((df["High"].iloc[-10:] - df["Low"].iloc[-10:]).mean() / close)
        volatility_prev = float((df["High"].iloc[-30:-10] - df["Low"].iloc[-30:-10]).mean() / close)
        contraction = volatility_now < volatility_prev

        score = 55.0
        reasons = [
            "Cours > MM50 > MM150 > MM200",
            "MM200 montante",
            f"À {dist_high:.1f}% du plus haut 52 semaines",
        ]
        score += max(0, 15 - dist_high * 0.6)
        if contraction:
            score += 12
            reasons.append("Volatilité récente en contraction")
        if volume_ratio > 1.2:
            score += 8
            reasons.append("Volume supérieur à la moyenne")
        proximity = abs((close - pivot) / pivot * 100)
        score += max(0, 10 - proximity * 2)
        score = min(100, round(score, 1))

        return Candidate(
            ticker=ticker,
            close=round(close, 2),
            pivot=round(pivot, 2),
            entry=round(entry, 2),
            stop=round(stop, 2),
            stop_pct=round(stop_pct, 2),
            shares=shares,
            position_value=round(position_value, 2),
            risk_eur=round(shares * (entry - stop), 2),
            score=score,
            distance_52w_high_pct=round(dist_high, 2),
            avg_dollar_volume=round(avg_dollar_vol, 0),
            reasons=reasons,
        )
    except Exception:
        return None


HTML_TEMPLATE = """<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trading Assistant Bruno</title>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#f5f7fa;margin:0;color:#18212f}
main{max-width:820px;margin:auto;padding:18px}.card{background:white;border-radius:16px;padding:18px;margin:14px 0;box-shadow:0 3px 18px #00000012}
h1{font-size:24px}.regime{font-size:28px;font-weight:800}.VERT{color:#138a42}.ORANGE{color:#c46b00}.ROUGE{color:#c62828}
.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}.metric{background:#f4f6f8;border-radius:10px;padding:10px}
.ticker{font-size:22px;font-weight:800}.score{float:right}.buy{font-size:18px;font-weight:700}
small{color:#667085} ul{padding-left:20px}
@media(max-width:600px){.grid{grid-template-columns:1fr}}
</style>
</head>
<body><main>
<h1>Trading Assistant Bruno</h1>
<div class="card">
<div class="regime {{ regime.color }}">● Marché {{ regime.color }}</div>
<p>Score de régime : {{ regime.score }}/100 — nouvelles positions autorisées : {{ regime.new_positions_allowed }}</p>
<small>Calculé le {{ generated_at }}</small>
</div>
{% if regime.color == "ROUGE" %}
<div class="card"><h2>Aucun nouvel achat</h2><p>Le régime rouge bloque automatiquement les nouvelles positions.</p></div>
{% endif %}
{% for c in candidates %}
<div class="card">
<div class="ticker">{{ loop.index }}. {{ c.ticker }} <span class="score">{{ c.score }}/100</span></div>
<p class="buy">Ordre conditionnel : achat stop-limit autour de {{ c.entry }} $</p>
<div class="grid">
<div class="metric"><small>Pivot</small><br><b>{{ c.pivot }} $</b></div>
<div class="metric"><small>Stop initial</small><br><b>{{ c.stop }} $</b> ({{ c.stop_pct }}%)</div>
<div class="metric"><small>Taille</small><br><b>{{ c.shares }} actions</b></div>
<div class="metric"><small>Risque estimé</small><br><b>{{ c.risk_eur }} €</b></div>
</div>
<ul>{% for r in c.reasons %}<li>{{ r }}</li>{% endfor %}</ul>
<p><a href="https://www.tradingview.com/chart/?symbol={{ c.ticker }}" target="_blank">Ouvrir le graphique</a></p>
</div>
{% endfor %}
<div class="card"><small>Prototype d'aide à la décision. Les niveaux sont calculés mécaniquement sur données quotidiennes et doivent être vérifiés avant tout ordre.</small></div>
</main></body></html>"""


def send_telegram(regime: dict, candidates: list[Candidate]) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Telegram secrets missing; notification skipped.")
        return
    lines = [
        f"📈 Trading Assistant",
        f"Marché : {regime['color']} ({regime['score']}/100)",
        f"Nouvelles positions : {regime['new_positions_allowed']}",
        "",
    ]
    if regime["color"] == "ROUGE":
        lines.append("⛔ Aucun nouvel achat.")
    elif not candidates:
        lines.append("Aucune configuration conforme aujourd'hui.")
    else:
        for i, c in enumerate(candidates, 1):
            lines += [
                f"{i}. {c.ticker} — score {c.score}/100",
                f"Entrée {c.entry}$ | Stop {c.stop}$ | {c.shares} actions",
                "",
            ]
    requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": "\n".join(lines)},
        timeout=20,
    ).raise_for_status()


def main() -> None:
    regime = market_regime()
    symbols = download_universe()
    print(f"Universe: {len(symbols)} symbols")
    frames = fetch_prices(symbols, period="2y")

    candidates: list[Candidate] = []
    if regime["color"] != "ROUGE":
        for ticker, frame in frames.items():
            c = analyze_symbol(ticker, frame)
            if c:
                candidates.append(c)
        candidates.sort(key=lambda c: c.score, reverse=True)
        candidates = candidates[: int(CONFIG["max_new_candidates"])]

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "regime": regime,
        "candidates": [asdict(c) for c in candidates],
    }
    (DATA / "latest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    html = Template(HTML_TEMPLATE).render(
        regime=regime,
        candidates=candidates,
        generated_at=datetime.now().astimezone().strftime("%d/%m/%Y %H:%M"),
    )
    (DOCS / "index.html").write_text(html, encoding="utf-8")
    send_telegram(regime, candidates)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
