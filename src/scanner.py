from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

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


def cfg(section: str, key: str, default):
    return CONFIG.get(section, {}).get(key, default)


@dataclass
class Candidate:
    ticker: str
    close: float
    status: str
    score: float
    pivot: float
    entry_trigger: float
    buy_zone_max: float
    extension_vs_pivot_pct: float
    stop: float
    stop_pct: float
    shares: int
    position_value_usd: float
    risk_usd: float
    risk_eur: float
    eurusd: float
    perf_3m_pct: float
    perf_6m_pct: float
    perf_12m_pct: float
    rs_6m_vs_spy_pct: float
    rs_12m_vs_spy_pct: float
    distance_52w_high_pct: float
    avg_dollar_volume: float
    volume_ratio: float
    contraction: bool
    reasons: list[str]


def download_universe() -> list[str]:
    urls = [
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
    ]

    symbols: list[str] = []
    banned_terms = (
        "warrant", "warrants", "rights", "unit", "units", "preferred",
        "bond", "bonds", "debenture", "notes", "closed-end fund",
        "depositary shares",
    )

    for url in urls:
        try:
            df = pd.read_csv(url, sep="|")

            if "Symbol" in df.columns:
                symbol_col = "Symbol"
            elif "ACT Symbol" in df.columns:
                symbol_col = "ACT Symbol"
            else:
                continue

            if "ETF" in df.columns:
                df = df[df["ETF"].astype(str).str.upper().eq("N")]

            if "Test Issue" in df.columns:
                df = df[df["Test Issue"].astype(str).str.upper().eq("N")]

            if "Security Name" in df.columns:
                names = df["Security Name"].astype(str).str.lower()
                mask = ~names.apply(lambda name: any(term in name for term in banned_terms))
                df = df[mask]

            for raw in df[symbol_col].dropna().astype(str):
                ticker = raw.strip().replace(".", "-")
                if not ticker:
                    continue
                if "File Creation Time" in ticker:
                    continue
                if any(char in ticker for char in ("$", "^")):
                    continue
                if len(ticker) > 6:
                    continue
                symbols.append(ticker)

        except Exception as exc:
            print(f"Universe source failed: {url}: {exc}")

    symbols = sorted(set(symbols))
    max_symbols = int(CONFIG.get("universe", {}).get("max_symbols", 5000))
    if max_symbols > 0:
        symbols = symbols[:max_symbols]

    print(f"Universe after filtering: {len(symbols)}")
    return symbols


def batched(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _extract_downloaded_frames(
    data: pd.DataFrame,
    batch: list[str],
) -> dict[str, pd.DataFrame]:
    output: dict[str, pd.DataFrame] = {}

    if data is None or data.empty:
        return output

    if len(batch) == 1:
        ticker = batch[0]
        frame = data.dropna(how="all")
        if not frame.empty:
            output[ticker] = frame
        return output

    for ticker in batch:
        try:
            frame = data[ticker].dropna(how="all")
            if not frame.empty:
                output[ticker] = frame
        except Exception:
            continue

    return output


def fetch_prices(
    symbols: list[str],
    period: str = "2y",
    parallel: bool = True,
    retries: int = 3,
) -> dict[str, pd.DataFrame]:
    """
    Télécharge les prix avec reprise automatique.

    - Pour les indices/benchmarks : utiliser parallel=False afin
      d'éviter les conflits de cache SQLite de yfinance.
    - Pour le gros univers d'actions : parallel=True pour conserver
      un temps de scan raisonnable.
    """
    output: dict[str, pd.DataFrame] = {}

    batch_size = 150 if parallel else 1

    for batch_no, batch in enumerate(batched(symbols, batch_size), start=1):
        print(f"Downloading batch {batch_no}: {len(batch)} symbols")

        batch_output: dict[str, pd.DataFrame] = {}

        for attempt in range(1, retries + 1):
            try:
                data = yf.download(
                    tickers=batch,
                    period=period,
                    interval="1d",
                    auto_adjust=True,
                    group_by="ticker",
                    threads=parallel,
                    progress=False,
                    timeout=30,
                )

                batch_output = _extract_downloaded_frames(data, batch)

                if len(batch_output) == len(batch):
                    break

            except Exception as exc:
                print(
                    f"Batch attempt {attempt}/{retries} failed "
                    f"for {batch}: {exc}"
                )

            time.sleep(1.5 * attempt)

        output.update(batch_output)

        # Reprise individuelle des tickers manquants.
        missing = [ticker for ticker in batch if ticker not in output]

        for ticker in missing:
            for attempt in range(1, retries + 1):
                try:
                    print(
                        f"Retrying {ticker} individually "
                        f"({attempt}/{retries})"
                    )

                    data = yf.download(
                        ticker,
                        period=period,
                        interval="1d",
                        auto_adjust=True,
                        progress=False,
                        threads=False,
                        timeout=30,
                    )

                    frames = _extract_downloaded_frames(data, [ticker])

                    if ticker in frames:
                        output[ticker] = frames[ticker]
                        break

                except Exception as exc:
                    print(
                        f"{ticker} retry {attempt}/{retries} failed: {exc}"
                    )

                time.sleep(2 * attempt)

        time.sleep(0.20)

    return output

def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise les colonnes yfinance, qu'elles soient simples
    ou en MultiIndex (Price/Ticker ou Ticker/Price).
    """
    x = df.copy()

    required = {"Open", "High", "Low", "Close", "Volume"}

    if isinstance(x.columns, pd.MultiIndex):
        best_level = None
        best_score = -1

        for level in range(x.columns.nlevels):
            values = [
                str(value).title()
                for value in x.columns.get_level_values(level)
            ]
            score = sum(value in required for value in values)

            if score > best_score:
                best_score = score
                best_level = level

        if best_level is None or best_score == 0:
            raise ValueError(
                f"Impossible d'identifier les colonnes OHLCV dans MultiIndex: "
                f"{list(x.columns)}"
            )

        x.columns = [
            str(value).title()
            for value in x.columns.get_level_values(best_level)
        ]

    else:
        x.columns = [
            str(column).title()
            for column in x.columns
        ]

    x = x.loc[:, ~x.columns.duplicated()]

    missing = required.difference(set(x.columns))
    if missing:
        raise ValueError(
            f"Missing OHLCV columns: {sorted(missing)}. "
            f"Columns received: {list(x.columns)}"
        )

    return x


def close_series(df: pd.DataFrame) -> pd.Series:
    """
    Extrait uniquement la série Close de manière robuste.
    Utilisé pour indices, VIX, FX et benchmark afin de ne pas
    dépendre de colonnes OHLCV inutiles.
    """
    x = df.copy()

    if x is None or x.empty:
        raise ValueError("Empty market data")

    if isinstance(x.columns, pd.MultiIndex):
        # Cherche le niveau qui contient 'Close'
        for level in range(x.columns.nlevels):
            values = [
                str(value).title()
                for value in x.columns.get_level_values(level)
            ]

            if "Close" in values:
                close_cols = [
                    col
                    for col, value in zip(x.columns, values)
                    if value == "Close"
                ]

                if not close_cols:
                    continue

                series = x[close_cols[0]]

                # Selon pandas/yfinance, l'indexation peut encore
                # retourner un DataFrame à une seule colonne.
                if isinstance(series, pd.DataFrame):
                    series = series.iloc[:, 0]

                return pd.to_numeric(series, errors="coerce").dropna()

        raise ValueError(
            f"Close introuvable dans MultiIndex: {list(x.columns)}"
        )

    # Colonnes simples
    mapping = {
        str(column).title(): column
        for column in x.columns
    }

    if "Close" not in mapping:
        raise ValueError(
            f"Close introuvable. Colonnes reçues: {list(x.columns)}"
        )

    series = x[mapping["Close"]]

    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]

    return pd.to_numeric(series, errors="coerce").dropna()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = normalize_ohlcv(df)

    for n in (21, 50, 150, 200):
        x[f"SMA{n}"] = x["Close"].rolling(n).mean()

    x["High52"] = x["High"].rolling(252).max()
    x["AvgDollarVol20"] = (x["Close"] * x["Volume"]).rolling(20).mean()
    return x


def pct_return(close: pd.Series, sessions: int) -> float:
    if len(close) <= sessions:
        return float("nan")
    old = float(close.iloc[-1 - sessions])
    new = float(close.iloc[-1])
    if old <= 0:
        return float("nan")
    return (new / old - 1) * 100


def get_eurusd() -> float:
    fallback = float(
        CONFIG.get("fx", {}).get("eurusd_fallback", 1.10)
    )

    try:
        frames = fetch_prices(
            ["EURUSD=X"],
            period="10d",
            parallel=False,
            retries=3,
        )

        if "EURUSD=X" not in frames:
            return fallback

        close = close_series(frames["EURUSD=X"])

        if close.empty:
            return fallback

        value = float(close.iloc[-1])

        return value if value > 0 else fallback

    except Exception as exc:
        print(
            f"EURUSD failed, using fallback {fallback}: {exc}"
        )
        return fallback

def market_regime() -> dict:
    tickers = ["QQQ", "SPY", "^IXIC", "^VIX"]

    frames = fetch_prices(
        tickers,
        period="1y",
        parallel=False,
        retries=4,
    )

    missing = [ticker for ticker in tickers if ticker not in frames]
    if missing:
        raise RuntimeError(
            "Données de marché indisponibles après plusieurs tentatives : "
            + ", ".join(missing)
        )

    details = {}
    positives = 0
    slope_days = int(cfg("strategy", "sma50_slope_days", 20))

    for ticker in ["QQQ", "SPY", "^IXIC"]:
        close = close_series(frames[ticker])

        if len(close) < 220:
            raise RuntimeError(
                f"Historique insuffisant pour {ticker}: {len(close)} séances"
            )

        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()

        last_close = float(close.iloc[-1])
        last_sma50 = float(sma50.iloc[-1])
        last_sma200 = float(sma200.iloc[-1])

        previous_sma50 = float(sma50.iloc[-1 - slope_days])

        sma50_rising = bool(last_sma50 > previous_sma50)
        above_50 = bool(last_close > last_sma50)
        above_200 = bool(last_close > last_sma200)

        positives += sum([above_50, above_200, sma50_rising])

        details[ticker] = {
            "close": round(last_close, 2),
            "above_sma50": above_50,
            "above_sma200": above_200,
            "sma50_rising": sma50_rising,
        }

    vix_close_series = close_series(frames["^VIX"])
    if vix_close_series.empty:
        raise RuntimeError("Historique VIX vide")

    vix_close = float(vix_close_series.iloc[-1])
    vix_limit = float(cfg("market", "vix_green_max", 25))
    vix_ok = bool(vix_close < vix_limit)
    positives += int(vix_ok)

    details["VIX"] = {
        "close": round(vix_close, 2),
        "below_limit": vix_ok,
        "limit": vix_limit,
    }

    if positives >= 9:
        color = "VERT"
    elif positives >= 6:
        color = "ORANGE"
    else:
        color = "ROUGE"

    allowed = {
        "VERT": int(CONFIG.get("max_positions_green", 4)),
        "ORANGE": int(CONFIG.get("max_positions_orange", 1)),
        "ROUGE": 0,
    }[color]

    return {
        "color": color,
        "score": positives * 10,
        "details": details,
        "new_positions_allowed": allowed,
    }

def benchmark_returns() -> dict:
    frames = fetch_prices(
        ["SPY"],
        period="2y",
        parallel=False,
        retries=4,
    )

    if "SPY" not in frames:
        raise RuntimeError(
            "Impossible de télécharger SPY pour la force relative"
        )

    close = close_series(frames["SPY"])

    return {
        "perf_6m": pct_return(close, 126),
        "perf_12m": pct_return(close, 252),
    }

def relative_return(stock_pct: float, benchmark_pct: float) -> float:
    if pd.isna(stock_pct) or pd.isna(benchmark_pct):
        return float("nan")
    stock_mult = 1 + stock_pct / 100
    bench_mult = 1 + benchmark_pct / 100
    if bench_mult <= 0:
        return float("nan")
    return (stock_mult / bench_mult - 1) * 100


def determine_status(close: float, pivot: float, entry: float, buy_zone_max: float) -> str:
    watch_below_pct = float(cfg("strategy", "watchlist_below_pivot_pct", 3.0))

    if entry <= close <= buy_zone_max:
        return "ACHAT POSSIBLE"
    if close > buy_zone_max:
        return "TROP ETENDU"

    distance_below = (pivot - close) / pivot * 100
    if 0 <= distance_below <= watch_below_pct:
        return "ATTENDRE CASSURE"

    return "WATCHLIST"


def analyze_symbol(
    ticker: str,
    raw: pd.DataFrame,
    spy_returns: dict,
    eurusd: float,
) -> tuple[Candidate | None, str]:

    try:
        df = add_indicators(raw)

        if len(df) < 260:
            return None, "historique"

        df = df.dropna()
        if df.empty:
            return None, "historique"

        last = df.iloc[-1]
        close = float(last["Close"])

        if close < float(cfg("universe", "min_price", 5)):
            return None, "prix"

        avg_dollar_vol = float(last["AvgDollarVol20"])
        if avg_dollar_vol < float(cfg("universe", "min_avg_dollar_volume", 5_000_000)):
            return None, "liquidite"

        if not (close > last["SMA50"] > last["SMA150"] > last["SMA200"]):
            return None, "trend_template"

        slope_days = int(cfg("strategy", "sma200_slope_days", 20))
        if float(last["SMA200"]) <= float(df["SMA200"].iloc[-1 - slope_days]):
            return None, "sma200"

        high52 = float(last["High52"])
        distance_high = (high52 - close) / high52 * 100
        if distance_high > float(cfg("strategy", "max_distance_from_52w_high_pct", 25)):
            return None, "52w_high"

        perf_3m = pct_return(df["Close"], 63)
        perf_6m = pct_return(df["Close"], 126)
        perf_12m = pct_return(df["Close"], 252)

        if pd.isna(perf_6m) or perf_6m < float(cfg("strategy", "min_perf_6m_pct", 0)):
            return None, "momentum"

        rs_6m = relative_return(perf_6m, spy_returns["perf_6m"])
        rs_12m = relative_return(perf_12m, spy_returns["perf_12m"])

        if pd.isna(rs_6m) or rs_6m < float(cfg("strategy", "min_rs_6m_vs_spy_pct", 0)):
            return None, "relative_strength"

        lookback = int(cfg("strategy", "pivot_lookback_days", 60))
        exclude_recent = int(cfg("strategy", "pivot_exclude_recent_days", 5))

        pivot_window = df["High"].iloc[-lookback:-exclude_recent]
        if pivot_window.empty:
            return None, "pivot"

        pivot = float(pivot_window.max())
        entry_buffer_pct = float(cfg("strategy", "entry_buffer_pct", 0.10))
        entry = pivot * (1 + entry_buffer_pct / 100)

        buy_zone_pct = float(cfg("strategy", "max_distance_above_pivot_pct", 1.5))
        buy_zone_max = pivot * (1 + buy_zone_pct / 100)

        extension_vs_pivot = (close - pivot) / pivot * 100
        status = determine_status(close, pivot, entry, buy_zone_max)

        recent_swing_low = float(df["Low"].iloc[-10:].min()) * 0.995
        min_stop_pct = float(cfg("strategy", "min_stop_distance_pct", 3))
        max_stop_pct = float(cfg("strategy", "max_stop_distance_pct", 8))

        min_stop = entry * (1 - min_stop_pct / 100)
        max_stop = entry * (1 - max_stop_pct / 100)

        stop = min(min_stop, recent_swing_low)
        stop = max(stop, max_stop)

        stop_pct = (entry - stop) / entry * 100
        if stop_pct <= 0 or stop_pct > max_stop_pct + 1e-6:
            return None, "stop"

        capital_eur = float(CONFIG.get("capital_eur", 10_000))
        risk_pct = float(CONFIG.get("risk_per_trade_pct", 0.5))
        max_risk_eur = capital_eur * risk_pct / 100
        max_risk_usd = max_risk_eur * eurusd

        risk_per_share_usd = entry - stop
        if risk_per_share_usd <= 0:
            return None, "stop"

        shares = math.floor(max_risk_usd / risk_per_share_usd)
        if shares < 1:
            return None, "taille"

        risk_usd = shares * risk_per_share_usd
        actual_risk_eur = risk_usd / eurusd
        position_value_usd = shares * entry

        avg_vol = float(df["Volume"].iloc[-21:-1].mean())
        volume_ratio = float(last["Volume"] / max(avg_vol, 1))

        volatility_now = float(((df["High"].iloc[-10:] - df["Low"].iloc[-10:]).mean()) / close)
        volatility_prev = float(((df["High"].iloc[-30:-10] - df["Low"].iloc[-30:-10]).mean()) / close)
        contraction = volatility_now < volatility_prev

        score = 40.0
        score += max(0, min(15, perf_6m / 4))
        score += max(0, min(10, perf_12m / 10))
        score += max(0, min(15, rs_6m / 2))
        score += max(0, 10 - distance_high * 0.4)

        if contraction:
            score += 5
        if volume_ratio > 1.2:
            score += 5

        score = min(100, round(score, 1))

        reasons = [
            "Cours > MM50 > MM150 > MM200",
            "MM200 montante",
            f"Performance 6 mois : {perf_6m:.1f}%",
            f"RS 6 mois vs SPY : {rs_6m:+.1f}%",
            f"À {distance_high:.1f}% du plus haut 52 semaines",
        ]

        if contraction:
            reasons.append("Volatilité récente en contraction")
        if volume_ratio > 1.2:
            reasons.append(f"Volume {volume_ratio:.1f}× la moyenne 20j")

        return Candidate(
            ticker=ticker,
            close=round(close, 2),
            status=status,
            score=score,
            pivot=round(pivot, 2),
            entry_trigger=round(entry, 2),
            buy_zone_max=round(buy_zone_max, 2),
            extension_vs_pivot_pct=round(extension_vs_pivot, 2),
            stop=round(stop, 2),
            stop_pct=round(stop_pct, 2),
            shares=shares,
            position_value_usd=round(position_value_usd, 2),
            risk_usd=round(risk_usd, 2),
            risk_eur=round(actual_risk_eur, 2),
            eurusd=round(eurusd, 4),
            perf_3m_pct=round(perf_3m, 2),
            perf_6m_pct=round(perf_6m, 2),
            perf_12m_pct=round(perf_12m, 2),
            rs_6m_vs_spy_pct=round(rs_6m, 2),
            rs_12m_vs_spy_pct=round(rs_12m, 2),
            distance_52w_high_pct=round(distance_high, 2),
            avg_dollar_volume=round(avg_dollar_vol, 0),
            volume_ratio=round(volume_ratio, 2),
            contraction=contraction,
            reasons=reasons,
        ), "retenu"

    except Exception as exc:
        print(f"{ticker}: analysis failed: {exc}")
        return None, "erreur"


HTML_TEMPLATE = """
<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trading Assistant Bruno</title>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#f5f7fa;margin:0;color:#18212f}
main{max-width:860px;margin:auto;padding:18px}
.card{background:white;border-radius:16px;padding:18px;margin:14px 0;box-shadow:0 3px 18px #00000012}
h1{font-size:24px}.regime{font-size:28px;font-weight:800}
.VERT{color:#138a42}.ORANGE{color:#c46b00}.ROUGE{color:#c62828}
.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}
.metric{background:#f4f6f8;border-radius:10px;padding:10px}
.ticker{font-size:22px;font-weight:800}.score{float:right}
.status{font-size:18px;font-weight:800;margin-top:8px}
small{color:#667085}ul{padding-left:20px}
table{width:100%;border-collapse:collapse}td{padding:6px;border-bottom:1px solid #eee}
@media(max-width:600px){.grid{grid-template-columns:1fr}}
</style>
</head>
<body><main>
<h1>Trading Assistant Bruno</h1>

<div class="card">
<div class="regime {{ regime.color }}">● Marché {{ regime.color }}</div>
<p>Score de régime : {{ regime.score }}/100 — nouvelles positions autorisées : {{ regime.new_positions_allowed }}</p>
<p>EUR/USD utilisé : {{ eurusd }}</p>
<small>Calculé le {{ generated_at }}</small>
</div>

<div class="card">
<h2>Entonnoir du scan</h2>
<table>
{% for key, value in funnel.items() %}
<tr><td>{{ key }}</td><td><b>{{ value }}</b></td></tr>
{% endfor %}
</table>
</div>

{% if regime.color == "ROUGE" %}
<div class="card"><h2>Aucun nouvel achat</h2><p>Le régime rouge bloque automatiquement les nouvelles positions.</p></div>
{% endif %}

{% for c in candidates %}
<div class="card">
<div class="ticker">{{ loop.index }}. {{ c.ticker }} <span class="score">{{ c.score }}/100</span></div>
<div class="status">{{ c.status }}</div>
<p>Cours : <b>{{ c.close }} $</b> — pivot : <b>{{ c.pivot }} $</b></p>

<div class="grid">
<div class="metric"><small>Déclenchement</small><br><b>{{ c.entry_trigger }} $</b></div>
<div class="metric"><small>Zone d'achat max</small><br><b>{{ c.buy_zone_max }} $</b></div>
<div class="metric"><small>Stop initial</small><br><b>{{ c.stop }} $</b> ({{ c.stop_pct }}%)</div>
<div class="metric"><small>Taille</small><br><b>{{ c.shares }} actions</b></div>
<div class="metric"><small>Risque réel</small><br><b>{{ c.risk_eur }} €</b></div>
<div class="metric"><small>Position</small><br><b>{{ c.position_value_usd }} $</b></div>
<div class="metric"><small>Perf. 6 mois</small><br><b>{{ c.perf_6m_pct }}%</b></div>
<div class="metric"><small>RS 6m vs SPY</small><br><b>{{ c.rs_6m_vs_spy_pct }}%</b></div>
</div>

<ul>{% for r in c.reasons %}<li>{{ r }}</li>{% endfor %}</ul>
<p><a href="https://www.tradingview.com/chart/?symbol={{ c.ticker }}" target="_blank">Ouvrir dans TradingView</a></p>
</div>
{% endfor %}

<div class="card">
<small>
Le score est un classement technique, pas une probabilité de gain.
Cette version ne vérifie pas encore automatiquement les résultats d'entreprise ni la qualité visuelle VCP/flat base.
Aucun ordre n'est envoyé au broker.
</small>
</div>
</main></body></html>
"""


def send_telegram(regime: dict, candidates: list[Candidate]) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("Telegram secrets missing; notification skipped.")
        return

    lines = [
        "📈 Trading Assistant",
        f"Marché : {regime['color']} ({regime['score']}/100)",
        f"Nouvelles positions autorisées : {regime['new_positions_allowed']}",
        "",
    ]

    if regime["color"] == "ROUGE":
        lines.append("⛔ Aucun nouvel achat.")
    elif not candidates:
        lines.append("Aucune configuration conforme aujourd'hui.")
    else:
        for i, c in enumerate(candidates, 1):
            lines += [
                f"{i}. {c.ticker} — {c.status} — score {c.score}/100",
                f"Cours {c.close}$ | Pivot {c.pivot}$",
                f"Déclenchement {c.entry_trigger}$ | Zone max {c.buy_zone_max}$",
                f"Stop {c.stop}$ | {c.shares} actions | risque {c.risk_eur}€",
                "",
            ]

    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": "\n".join(lines)},
            timeout=20,
        ).raise_for_status()
    except Exception as exc:
        print(f"Telegram error: {exc}")


def main() -> None:
    print("Starting Trading Assistant v1.1.2")

    regime = market_regime()
    eurusd = get_eurusd()
    spy_returns = benchmark_returns()

    print(f"Market regime: {regime['color']} ({regime['score']}/100)")
    print(f"EUR/USD: {eurusd:.4f}")
    print(f"SPY 6m: {spy_returns['perf_6m']:.2f}% | SPY 12m: {spy_returns['perf_12m']:.2f}%")

    symbols = download_universe()
    frames = fetch_prices(symbols, period="2y", parallel=True, retries=2)

    funnel = {
        "Univers filtré": len(symbols),
        "Données téléchargées": len(frames),
        "Historique insuffisant": 0,
        "Prix minimum": 0,
        "Liquidité": 0,
        "Trend template": 0,
        "MM200 montante": 0,
        "Proximité 52 semaines": 0,
        "Momentum 6 mois": 0,
        "Force relative vs SPY": 0,
        "Pivot / stop / taille": 0,
        "Retenus": 0,
        "Erreurs": 0,
    }

    stage_map = {
        "historique": "Historique insuffisant",
        "prix": "Prix minimum",
        "liquidite": "Liquidité",
        "trend_template": "Trend template",
        "sma200": "MM200 montante",
        "52w_high": "Proximité 52 semaines",
        "momentum": "Momentum 6 mois",
        "relative_strength": "Force relative vs SPY",
        "pivot": "Pivot / stop / taille",
        "stop": "Pivot / stop / taille",
        "taille": "Pivot / stop / taille",
        "erreur": "Erreurs",
    }

    all_candidates: list[Candidate] = []

    if regime["color"] != "ROUGE":
        for ticker, frame in frames.items():
            candidate, stage = analyze_symbol(ticker, frame, spy_returns, eurusd)
            if candidate:
                all_candidates.append(candidate)
                funnel["Retenus"] += 1
            elif stage in stage_map:
                funnel[stage_map[stage]] += 1

    status_priority = {
        "ACHAT POSSIBLE": 0,
        "ATTENDRE CASSURE": 1,
        "WATCHLIST": 2,
        "TROP ETENDU": 3,
    }

    all_candidates.sort(
        key=lambda c: (status_priority.get(c.status, 9), -c.score)
    )

    max_new = int(CONFIG.get("max_new_candidates", 8))
    candidates = all_candidates[:max_new]

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "version": "1.1.2",
        "regime": regime,
        "eurusd": round(eurusd, 4),
        "spy_returns": {
            "perf_6m_pct": round(spy_returns["perf_6m"], 2),
            "perf_12m_pct": round(spy_returns["perf_12m"], 2),
        },
        "funnel": funnel,
        "candidates": [asdict(c) for c in candidates],
    }

    (DATA / "latest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    html = Template(HTML_TEMPLATE).render(
        regime=regime,
        eurusd=round(eurusd, 4),
        funnel=funnel,
        candidates=candidates,
        generated_at=datetime.now().astimezone().strftime("%d/%m/%Y %H:%M"),
    )
    (DOCS / "index.html").write_text(html, encoding="utf-8")

    send_telegram(regime, candidates)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
