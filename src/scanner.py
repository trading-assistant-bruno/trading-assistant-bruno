from __future__ import annotations

import base64
import io
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    technical_stop_pct: float
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
    rs_rank: int
    distance_52w_high_pct: float
    distance_above_52w_low_pct: float
    avg_dollar_volume: float
    volume_ratio: float
    base_depth_pct: float
    base_tightness_pct: float
    volatility_contraction_ratio: float
    volume_dryup_ratio: float
    contraction_steps: int
    base_quality_score: float
    reasons: list[str]

    sector: str = "INCONNU"
    industry: str = "INCONNU"
    next_earnings_date: str | None = None
    earnings_days: int | None = None
    earnings_status: str = "INCONNU"
    chart_validation_score: float = 0.0
    final_decision: str = "À ANALYSER"
    final_reason: str = ""


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


def compute_universe_momentum(
    frames: dict[str, pd.DataFrame]
) -> dict[str, dict]:
    """RS Rank interne 1-99, percentile de momentum sur l'univers."""
    rows = {}

    for ticker, raw in frames.items():
        try:
            close = close_series(raw)
            p3 = pct_return(close, 63)
            p6 = pct_return(close, 126)
            p12 = pct_return(close, 252)

            if pd.isna(p6):
                continue

            components = []
            weights = []

            if not pd.isna(p3):
                components.append(p3)
                weights.append(0.40)

            if not pd.isna(p6):
                components.append(p6)
                weights.append(0.30)

            if not pd.isna(p12):
                components.append(p12)
                weights.append(0.30)

            if not components:
                continue

            total_weight = sum(weights)

            momentum_score = sum(
                value * weight
                for value, weight in zip(components, weights)
            ) / total_weight

            rows[ticker] = {
                "perf_3m_pct": p3,
                "perf_6m_pct": p6,
                "perf_12m_pct": p12,
                "momentum_score": momentum_score,
            }

        except Exception:
            continue

    if not rows:
        return {}

    scores = pd.Series(
        {
            ticker: metrics["momentum_score"]
            for ticker, metrics in rows.items()
        },
        dtype=float,
    )

    percentiles = scores.rank(pct=True, method="average")

    for ticker, metrics in rows.items():
        pct = float(percentiles.loc[ticker])
        metrics["rs_rank"] = int(
            max(1, min(99, round(1 + pct * 98)))
        )

    return rows


def determine_status(
    close: float,
    pivot: float,
    entry: float,
    buy_zone_max: float,
) -> str:
    ready_below_pct = float(
        cfg("strategy", "ready_below_pivot_pct", 0.7)
    )
    watch_below_pct = float(
        cfg("strategy", "watchlist_below_pivot_pct", 3.0)
    )
    max_extension_retain_pct = float(
        cfg("strategy", "max_extension_retain_pct", 4.0)
    )

    if entry <= close <= buy_zone_max:
        return "DANS ZONE D'ACHAT"

    if close > buy_zone_max:
        extension = (close - pivot) / pivot * 100
        if extension <= max_extension_retain_pct:
            return "TROP ÉTENDU"
        return "REJETÉ"

    distance_below = (pivot - close) / pivot * 100

    if 0 <= distance_below <= ready_below_pct:
        return "PRÊT À DÉCLENCHER"

    if ready_below_pct < distance_below <= watch_below_pct:
        return "ATTENDRE"

    return "REJETÉ"


def compute_base_quality(
    df: pd.DataFrame,
    close: float,
) -> dict:
    lookback = int(
        cfg("strategy", "base_lookback_days", 50)
    )

    base = df.tail(lookback)

    if len(base) < 30:
        raise ValueError("Base history too short")

    base_high = float(base["High"].max())
    base_low = float(base["Low"].min())

    base_depth_pct = (
        (base_high - base_low) / base_high * 100
    )

    recent_5 = base.tail(5)

    base_tightness_pct = (
        (
            float(recent_5["High"].max())
            - float(recent_5["Low"].min())
        )
        / close
        * 100
    )

    recent_range = float(
        (
            df["High"].iloc[-10:]
            - df["Low"].iloc[-10:]
        ).mean()
        / close
    )

    previous_range = float(
        (
            df["High"].iloc[-30:-10]
            - df["Low"].iloc[-30:-10]
        ).mean()
        / close
    )

    volatility_contraction_ratio = (
        recent_range / previous_range
        if previous_range > 0
        else 1.0
    )

    recent_vol = float(
        df["Volume"].iloc[-11:-1].mean()
    )

    previous_vol = float(
        df["Volume"].iloc[-41:-11].mean()
    )

    volume_dryup_ratio = (
        recent_vol / previous_vol
        if previous_vol > 0
        else 1.0
    )

    blocks = [
        df.iloc[-30:-20],
        df.iloc[-20:-10],
        df.iloc[-10:],
    ]

    block_ranges = []

    for block in blocks:
        mean_close = float(block["Close"].mean())

        if mean_close <= 0:
            block_ranges.append(float("nan"))
            continue

        range_pct = (
            (
                float(block["High"].max())
                - float(block["Low"].min())
            )
            / mean_close
            * 100
        )

        block_ranges.append(range_pct)

    contraction_steps = 0

    if not any(pd.isna(value) for value in block_ranges):
        if block_ranges[1] < block_ranges[0]:
            contraction_steps += 1

        if block_ranges[2] < block_ranges[1]:
            contraction_steps += 1

    quality = 0.0

    if base_depth_pct <= 10:
        quality += 25
    elif base_depth_pct <= 15:
        quality += 22
    elif base_depth_pct <= 20:
        quality += 16
    elif base_depth_pct <= 25:
        quality += 8

    if volatility_contraction_ratio <= 0.60:
        quality += 25
    elif volatility_contraction_ratio <= 0.80:
        quality += 20
    elif volatility_contraction_ratio <= 1.00:
        quality += 10

    if volume_dryup_ratio <= 0.70:
        quality += 20
    elif volume_dryup_ratio <= 0.85:
        quality += 15
    elif volume_dryup_ratio <= 1.00:
        quality += 8

    if contraction_steps == 2:
        quality += 20
    elif contraction_steps == 1:
        quality += 10

    if base_tightness_pct <= 3:
        quality += 10
    elif base_tightness_pct <= 5:
        quality += 7
    elif base_tightness_pct <= 7:
        quality += 4

    return {
        "base_depth_pct": base_depth_pct,
        "base_tightness_pct": base_tightness_pct,
        "volatility_contraction_ratio": volatility_contraction_ratio,
        "volume_dryup_ratio": volume_dryup_ratio,
        "contraction_steps": contraction_steps,
        "base_quality_score": min(100, quality),
        "block_ranges_pct": block_ranges,
    }


def analyze_symbol(
    ticker: str,
    raw: pd.DataFrame,
    universe_momentum: dict[str, dict],
    spy_returns: dict,
    eurusd: float,
) -> tuple[Candidate | None, str]:

    try:
        ohlcv = normalize_ohlcv(raw).dropna(
            subset=["High", "Low", "Close", "Volume"]
        )

        if len(ohlcv) < 260:
            return None, "historique"

        momentum = universe_momentum.get(ticker)

        if not momentum:
            return None, "momentum"

        perf_3m = momentum["perf_3m_pct"]
        perf_6m = momentum["perf_6m_pct"]
        perf_12m = momentum["perf_12m_pct"]
        rs_rank = int(momentum["rs_rank"])

        if rs_rank < int(
            cfg("strategy", "min_rs_rank", 80)
        ):
            return None, "rs_rank"

        df = add_indicators(raw)

        indicator_rows = df.dropna(
            subset=[
                "SMA50",
                "SMA150",
                "SMA200",
                "High52",
                "AvgDollarVol20",
            ]
        )

        if indicator_rows.empty:
            return None, "historique"

        last = indicator_rows.iloc[-1]
        close = float(last["Close"])

        if close < float(
            cfg("universe", "min_price", 5)
        ):
            return None, "prix"

        avg_dollar_vol = float(
            last["AvgDollarVol20"]
        )

        if avg_dollar_vol < float(
            cfg(
                "universe",
                "min_avg_dollar_volume",
                5_000_000,
            )
        ):
            return None, "liquidite"

        if not (
            close
            > float(last["SMA50"])
            > float(last["SMA150"])
            > float(last["SMA200"])
        ):
            return None, "trend_template"

        slope_days = int(
            cfg("strategy", "sma200_slope_days", 20)
        )

        sma200 = df["SMA200"].dropna()

        if len(sma200) <= slope_days:
            return None, "historique"

        if float(sma200.iloc[-1]) <= float(
            sma200.iloc[-1 - slope_days]
        ):
            return None, "sma200"

        tail_52w = ohlcv.tail(252)
        high52 = float(tail_52w["High"].max())
        low52 = float(tail_52w["Low"].min())

        distance_high = (
            (high52 - close) / high52 * 100
        )

        distance_above_low = (
            (close / low52 - 1) * 100
        )

        if distance_high > float(
            cfg(
                "strategy",
                "max_distance_from_52w_high_pct",
                25,
            )
        ):
            return None, "52w_high"

        if distance_above_low < float(
            cfg(
                "strategy",
                "min_distance_above_52w_low_pct",
                30,
            )
        ):
            return None, "52w_low"

        if (
            pd.isna(perf_6m)
            or perf_6m
            < float(
                cfg(
                    "strategy",
                    "min_perf_6m_pct",
                    10,
                )
            )
        ):
            return None, "momentum"

        rs_6m = relative_return(
            perf_6m,
            spy_returns["perf_6m"],
        )

        rs_12m = relative_return(
            perf_12m,
            spy_returns["perf_12m"],
        )

        if (
            pd.isna(rs_6m)
            or rs_6m
            < float(
                cfg(
                    "strategy",
                    "min_rs_6m_vs_spy_pct",
                    0,
                )
            )
        ):
            return None, "relative_strength"

        base_metrics = compute_base_quality(
            ohlcv,
            close,
        )

        if (
            base_metrics["base_depth_pct"]
            > float(
                cfg(
                    "strategy",
                    "max_base_depth_pct",
                    25,
                )
            )
        ):
            return None, "base_depth"

        if (
            base_metrics["base_quality_score"]
            < float(
                cfg(
                    "strategy",
                    "min_base_quality_score",
                    45,
                )
            )
        ):
            return None, "base_quality"

        lookback = int(
            cfg("strategy", "pivot_lookback_days", 60)
        )

        exclude_recent = int(
            cfg(
                "strategy",
                "pivot_exclude_recent_days",
                5,
            )
        )

        pivot_window = ohlcv["High"].iloc[
            -lookback:-exclude_recent
        ]

        if pivot_window.empty:
            return None, "pivot"

        pivot = float(pivot_window.max())

        entry_buffer_pct = float(
            cfg("strategy", "entry_buffer_pct", 0.10)
        )

        entry = pivot * (
            1 + entry_buffer_pct / 100
        )

        buy_zone_pct = float(
            cfg(
                "strategy",
                "max_distance_above_pivot_pct",
                1.5,
            )
        )

        buy_zone_max = pivot * (
            1 + buy_zone_pct / 100
        )

        extension_vs_pivot = (
            (close - pivot) / pivot * 100
        )

        status = determine_status(
            close,
            pivot,
            entry,
            buy_zone_max,
        )

        if status == "REJETÉ":
            return None, "distance_pivot"

        recent_swing_low = float(
            ohlcv["Low"].iloc[-10:].min()
        ) * 0.995

        technical_stop_pct = (
            (entry - recent_swing_low)
            / entry
            * 100
        )

        max_stop_pct = float(
            cfg(
                "strategy",
                "max_stop_distance_pct",
                8,
            )
        )

        min_stop_pct = float(
            cfg(
                "strategy",
                "min_stop_distance_pct",
                3,
            )
        )

        if technical_stop_pct <= 0:
            return None, "stop"

        if technical_stop_pct > max_stop_pct:
            return None, "stop_trop_large"

        if technical_stop_pct < min_stop_pct:
            stop = entry * (
                1 - min_stop_pct / 100
            )
        else:
            stop = recent_swing_low

        stop_pct = (
            (entry - stop)
            / entry
            * 100
        )

        capital_eur = float(
            CONFIG.get("capital_eur", 10_000)
        )

        risk_pct = float(
            CONFIG.get("risk_per_trade_pct", 0.5)
        )

        max_risk_eur = (
            capital_eur
            * risk_pct
            / 100
        )

        max_risk_usd = (
            max_risk_eur
            * eurusd
        )

        risk_per_share_usd = (
            entry - stop
        )

        if risk_per_share_usd <= 0:
            return None, "stop"

        shares = math.floor(
            max_risk_usd
            / risk_per_share_usd
        )

        if shares < 1:
            return None, "taille"

        risk_usd = (
            shares
            * risk_per_share_usd
        )

        actual_risk_eur = (
            risk_usd
            / eurusd
        )

        position_value_usd = (
            shares
            * entry
        )

        avg_vol = float(
            ohlcv["Volume"].iloc[-21:-1].mean()
        )

        volume_ratio = float(
            ohlcv["Volume"].iloc[-1]
            / max(avg_vol, 1)
        )

        score = 0.0
        score += rs_rank * 0.35
        score += (
            base_metrics["base_quality_score"]
            * 0.35
        )
        score += max(
            0,
            min(15, perf_6m / 6)
        )
        score += max(
            0,
            10 - distance_high * 0.4
        )

        if volume_ratio >= 1.5:
            score += 5
        elif volume_ratio >= 1.2:
            score += 3

        score = min(
            100,
            round(score, 1),
        )

        reasons = [
            f"RS Rank interne : {rs_rank}/99",
            "Cours > MM50 > MM150 > MM200",
            "MM200 montante",
            f"Performance 3 mois : {perf_3m:.1f}%",
            f"Performance 6 mois : {perf_6m:.1f}%",
            (
                f"Performance 12 mois : {perf_12m:.1f}%"
                if not pd.isna(perf_12m)
                else "Historique 12 mois incomplet"
            ),
            f"RS 6 mois vs SPY : {rs_6m:+.1f}%",
            f"À {distance_high:.1f}% du plus haut 52 semaines",
            f"{distance_above_low:.1f}% au-dessus du plus bas 52 semaines",
            f"Qualité de base : {base_metrics['base_quality_score']:.0f}/100",
            f"Profondeur base : {base_metrics['base_depth_pct']:.1f}%",
        ]

        if base_metrics["contraction_steps"] == 2:
            reasons.append(
                "Deux contractions successives détectées"
            )
        elif base_metrics["contraction_steps"] == 1:
            reasons.append(
                "Une contraction successive détectée"
            )

        if base_metrics["volume_dryup_ratio"] <= 0.85:
            reasons.append(
                "Assèchement récent du volume"
            )

        if base_metrics["volatility_contraction_ratio"] < 1:
            reasons.append(
                "Volatilité récente en contraction"
            )

        if volume_ratio >= 1.2:
            reasons.append(
                f"Volume séance : {volume_ratio:.1f}× moyenne 20j"
            )

        return Candidate(
            ticker=ticker,
            close=round(close, 2),
            status=status,
            score=score,
            pivot=round(pivot, 2),
            entry_trigger=round(entry, 2),
            buy_zone_max=round(buy_zone_max, 2),
            extension_vs_pivot_pct=round(
                extension_vs_pivot,
                2,
            ),
            stop=round(stop, 2),
            stop_pct=round(stop_pct, 2),
            technical_stop_pct=round(
                technical_stop_pct,
                2,
            ),
            shares=shares,
            position_value_usd=round(
                position_value_usd,
                2,
            ),
            risk_usd=round(risk_usd, 2),
            risk_eur=round(
                actual_risk_eur,
                2,
            ),
            eurusd=round(eurusd, 4),
            perf_3m_pct=round(
                perf_3m,
                2,
            ),
            perf_6m_pct=round(
                perf_6m,
                2,
            ),
            perf_12m_pct=(
                round(perf_12m, 2)
                if not pd.isna(perf_12m)
                else float("nan")
            ),
            rs_6m_vs_spy_pct=round(
                rs_6m,
                2,
            ),
            rs_12m_vs_spy_pct=(
                round(rs_12m, 2)
                if not pd.isna(rs_12m)
                else float("nan")
            ),
            rs_rank=rs_rank,
            distance_52w_high_pct=round(
                distance_high,
                2,
            ),
            distance_above_52w_low_pct=round(
                distance_above_low,
                2,
            ),
            avg_dollar_volume=round(
                avg_dollar_vol,
                0,
            ),
            volume_ratio=round(
                volume_ratio,
                2,
            ),
            base_depth_pct=round(
                base_metrics["base_depth_pct"],
                2,
            ),
            base_tightness_pct=round(
                base_metrics["base_tightness_pct"],
                2,
            ),
            volatility_contraction_ratio=round(
                base_metrics[
                    "volatility_contraction_ratio"
                ],
                2,
            ),
            volume_dryup_ratio=round(
                base_metrics["volume_dryup_ratio"],
                2,
            ),
            contraction_steps=int(
                base_metrics["contraction_steps"]
            ),
            base_quality_score=round(
                base_metrics["base_quality_score"],
                1,
            ),
            reasons=reasons,
        ), "retenu"

    except Exception as exc:
        print(
            f"{ticker}: analysis failed: {exc}"
        )
        return None, "erreur"


def _to_utc_timestamp(value) -> pd.Timestamp | None:
    try:
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return None
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts
    except Exception:
        return None


def fetch_company_metadata(ticker: str) -> dict:
    """Best-effort metadata. Never fails the workflow."""
    result = {
        "sector": "INCONNU",
        "industry": "INCONNU",
    }

    try:
        info = yf.Ticker(ticker).get_info()
        if isinstance(info, dict):
            result["sector"] = str(info.get("sector") or "INCONNU")
            result["industry"] = str(info.get("industry") or "INCONNU")
    except Exception as exc:
        print(f"{ticker}: metadata unavailable: {exc}")

    return result


def fetch_next_earnings(ticker: str) -> dict:
    """
    Best-effort next earnings date from Yahoo/yfinance.
    Unknown earnings are deliberately treated conservatively downstream.
    """
    now = pd.Timestamp.now(tz="UTC")
    future_dates: list[pd.Timestamp] = []
    tk = yf.Ticker(ticker)

    # 1) Calendar endpoint
    try:
        calendar = tk.calendar

        values = []
        if isinstance(calendar, dict):
            for key, value in calendar.items():
                if "earning" in str(key).lower():
                    if isinstance(value, (list, tuple)):
                        values.extend(value)
                    else:
                        values.append(value)
        elif isinstance(calendar, pd.DataFrame):
            for label in list(calendar.index) + list(calendar.columns):
                if "earning" in str(label).lower():
                    try:
                        if label in calendar.index:
                            values.extend(calendar.loc[label].tolist())
                        if label in calendar.columns:
                            values.extend(calendar[label].tolist())
                    except Exception:
                        pass

        for value in values:
            ts = _to_utc_timestamp(value)
            if ts is not None and ts >= now - pd.Timedelta(days=1):
                future_dates.append(ts)

    except Exception as exc:
        print(f"{ticker}: calendar earnings unavailable: {exc}")

    # 2) Earnings history/forecast endpoint fallback
    try:
        earnings_dates = tk.get_earnings_dates(limit=8)
        if earnings_dates is not None and not earnings_dates.empty:
            for value in earnings_dates.index:
                ts = _to_utc_timestamp(value)
                if ts is not None and ts >= now - pd.Timedelta(days=1):
                    future_dates.append(ts)
    except Exception as exc:
        print(f"{ticker}: earnings dates fallback unavailable: {exc}")

    if not future_dates:
        return {
            "next_earnings_date": None,
            "earnings_days": None,
            "earnings_status": "INCONNU",
        }

    next_date = min(future_dates)
    days = int((next_date.normalize() - now.normalize()).days)

    block_days = int(
        CONFIG.get("validation", {}).get("earnings_block_days", 7)
    )

    if days <= block_days:
        status = "PROCHE"
    else:
        status = "OK"

    return {
        "next_earnings_date": next_date.date().isoformat(),
        "earnings_days": days,
        "earnings_status": status,
    }


def compute_chart_validation_score(candidate: Candidate) -> float:
    """
    Quantitative chart validation. This is NOT image-AI vision.
    It scores the geometry already extracted from price/volume history.
    """
    score = candidate.base_quality_score * 0.60

    # Recent tightness: 15 pts
    if candidate.base_tightness_pct <= 3:
        score += 15
    elif candidate.base_tightness_pct <= 5:
        score += 11
    elif candidate.base_tightness_pct <= 7:
        score += 6

    # Pivot proximity: 15 pts
    distance = abs(candidate.extension_vs_pivot_pct)
    if distance <= 0.7:
        score += 15
    elif distance <= 1.5:
        score += 11
    elif distance <= 3:
        score += 5

    # Stop efficiency: 10 pts
    if candidate.stop_pct <= 4:
        score += 10
    elif candidate.stop_pct <= 6:
        score += 7
    elif candidate.stop_pct <= 8:
        score += 3

    return round(min(100, score), 1)


def decide_final_action(candidate: Candidate) -> tuple[str, str]:
    min_chart_score = float(
        CONFIG.get("validation", {}).get("min_chart_validation_score", 60)
    )

    if candidate.status == "TROP ÉTENDU":
        return "NE PAS ACHETER", "Cours déjà au-delà de la zone d'achat autorisée."

    if candidate.earnings_status == "INCONNU":
        return (
            "VÉRIFIER RÉSULTATS",
            "Date des résultats non confirmée automatiquement : achat non autorisé sans contrôle manuel.",
        )

    if candidate.earnings_status == "PROCHE":
        return (
            "BLOQUÉ RÉSULTATS",
            f"Résultats prévus dans {candidate.earnings_days} jour(s), fenêtre de sécurité active.",
        )

    if candidate.chart_validation_score < min_chart_score:
        return (
            "REJETER STRUCTURE",
            f"Validation graphique quantitative {candidate.chart_validation_score}/100 < {min_chart_score:.0f}/100.",
        )

    if candidate.status == "DANS ZONE D'ACHAT":
        return (
            "ACHAT CONDITIONNEL",
            "Setup validé : ordre autorisé uniquement tant que le cours reste sous la borne haute de la zone d'achat.",
        )

    if candidate.status == "PRÊT À DÉCLENCHER":
        return (
            "PRÉPARER ORDRE",
            "Setup validé mais cassure non encore confirmée : préparer un ordre conditionnel au déclenchement.",
        )

    if candidate.status == "ATTENDRE":
        return (
            "ATTENDRE",
            "Setup intéressant mais encore trop éloigné du pivot pour déclencher un ordre.",
        )

    return "NE RIEN FAIRE", "Aucune action requise."


def _figure_to_data_uri(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def make_price_chart_uri(raw: pd.DataFrame, candidate: Candidate) -> str:
    try:
        df = add_indicators(raw).tail(140)
        fig = plt.figure(figsize=(9, 4.8))
        ax = fig.add_subplot(111)

        ax.plot(df.index, df["Close"], label="Cours")
        ax.plot(df.index, df["SMA21"], label="MM21")
        ax.plot(df.index, df["SMA50"], label="MM50")
        ax.plot(df.index, df["SMA150"], label="MM150")
        ax.plot(df.index, df["SMA200"], label="MM200")
        ax.axhline(candidate.pivot, linestyle="--", label="Pivot")
        ax.axhline(candidate.buy_zone_max, linestyle=":", label="Zone achat max")
        ax.axhline(candidate.stop, linestyle="--", label="Stop")
        ax.set_title(f"{candidate.ticker} — Prix / moyennes / niveaux")
        ax.set_ylabel("USD")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, ncol=4)
        fig.autofmt_xdate()

        return _figure_to_data_uri(fig)
    except Exception as exc:
        print(f"{candidate.ticker}: price chart failed: {exc}")
        return ""


def make_volume_chart_uri(raw: pd.DataFrame, candidate: Candidate) -> str:
    try:
        df = normalize_ohlcv(raw).tail(90).copy()
        df["Volume20"] = df["Volume"].rolling(20).mean()

        fig = plt.figure(figsize=(9, 2.8))
        ax = fig.add_subplot(111)
        ax.bar(df.index, df["Volume"])
        ax.plot(df.index, df["Volume20"], label="Volume moyen 20j")
        ax.set_title(f"{candidate.ticker} — Volume")
        ax.set_ylabel("Volume")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        fig.autofmt_xdate()

        return _figure_to_data_uri(fig)
    except Exception as exc:
        print(f"{candidate.ticker}: volume chart failed: {exc}")
        return ""


def enrich_finalist(candidate: Candidate) -> Candidate:
    metadata = fetch_company_metadata(candidate.ticker)
    earnings = fetch_next_earnings(candidate.ticker)

    candidate.sector = metadata["sector"]
    candidate.industry = metadata["industry"]
    candidate.next_earnings_date = earnings["next_earnings_date"]
    candidate.earnings_days = earnings["earnings_days"]
    candidate.earnings_status = earnings["earnings_status"]
    candidate.chart_validation_score = compute_chart_validation_score(candidate)
    candidate.final_decision, candidate.final_reason = decide_final_action(candidate)

    pause = float(
        CONFIG.get("validation", {}).get("metadata_pause_seconds", 0.3)
    )
    time.sleep(max(0, pause))

    return candidate


HTML_TEMPLATE = """
<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trading Assistant Bruno</title>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#f5f7fa;margin:0;color:#18212f}
main{max-width:960px;margin:auto;padding:18px}
.card{background:white;border-radius:16px;padding:18px;margin:14px 0;box-shadow:0 3px 18px #00000012}
h1{font-size:24px}.regime{font-size:28px;font-weight:800}.VERT{color:#138a42}.ORANGE{color:#c46b00}.ROUGE{color:#c62828}
.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}.metric{background:#f4f6f8;border-radius:10px;padding:10px}
.ticker{font-size:22px;font-weight:800}.score{float:right}.status{font-size:17px;font-weight:700;margin-top:8px}.decision{font-size:21px;font-weight:900;margin:12px 0}
small{color:#667085}ul{padding-left:20px}table{width:100%;border-collapse:collapse}td{padding:6px;border-bottom:1px solid #eee}
.chart{width:100%;border-radius:10px;margin-top:12px}.muted{color:#667085}
@media(max-width:600px){.grid{grid-template-columns:1fr}}
</style>
</head>
<body><main>
<h1>Trading Assistant Bruno — v1.3</h1>

<div class="card">
<div class="regime {{ regime.color }}">● Marché {{ regime.color }}</div>
<p>Score régime : {{ regime.score }}/100 — nouvelles positions autorisées : {{ regime.new_positions_allowed }}</p>
<p>EUR/USD : {{ eurusd }}</p>
<small>Calculé le {{ generated_at }}</small>
</div>

<div class="card">
<h2>Entonnoir du scan</h2>
<table>{% for key, value in funnel.items() %}<tr><td>{{ key }}</td><td><b>{{ value }}</b></td></tr>{% endfor %}</table>
</div>

{% if regime.color == "ROUGE" %}
<div class="card"><h2>Aucun nouvel achat</h2><p>Le régime rouge bloque automatiquement les nouvelles positions.</p></div>
{% endif %}

{% for c in display_candidates %}
<div class="card">
<div class="ticker">{{ loop.index }}. {{ c.ticker }} <span class="score">{{ c.score }}/100</span></div>
<div class="status">Scanner : {{ c.status }}</div>
<div class="decision">Décision : {{ c.final_decision }}</div>
<p>{{ c.final_reason }}</p>
<p><b>{{ c.sector }}</b> — {{ c.industry }}</p>

<div class="grid">
<div class="metric"><small>Cours / pivot</small><br><b>{{ c.close }} $ / {{ c.pivot }} $</b></div>
<div class="metric"><small>Déclenchement</small><br><b>{{ c.entry_trigger }} $</b></div>
<div class="metric"><small>Zone achat max</small><br><b>{{ c.buy_zone_max }} $</b></div>
<div class="metric"><small>Stop</small><br><b>{{ c.stop }} $ ({{ c.stop_pct }}%)</b></div>
<div class="metric"><small>Taille</small><br><b>{{ c.shares }} actions</b></div>
<div class="metric"><small>Risque réel</small><br><b>{{ c.risk_eur }} €</b></div>
<div class="metric"><small>RS Rank</small><br><b>{{ c.rs_rank }}/99</b></div>
<div class="metric"><small>Qualité base</small><br><b>{{ c.base_quality_score }}/100</b></div>
<div class="metric"><small>Validation graphique quantitative</small><br><b>{{ c.chart_validation_score }}/100</b></div>
<div class="metric"><small>Prochains résultats</small><br><b>{{ c.next_earnings_date or 'INCONNU' }}</b> {% if c.earnings_days is not none %}({{ c.earnings_days }} j){% endif %}</div>
</div>

{% if c.price_chart_uri %}<img class="chart" src="{{ c.price_chart_uri }}" alt="Graphique prix {{ c.ticker }}">{% endif %}
{% if c.volume_chart_uri %}<img class="chart" src="{{ c.volume_chart_uri }}" alt="Graphique volume {{ c.ticker }}">{% endif %}

<ul>{% for r in c.reasons %}<li>{{ r }}</li>{% endfor %}</ul>
<p><a href="https://www.tradingview.com/chart/?symbol={{ c.ticker }}" target="_blank">Ouvrir dans TradingView</a></p>
</div>
{% endfor %}

<div class="card"><small>
La validation graphique v1.3 est quantitative (prix/volume), pas une IA visuelle. Si la date de résultats est inconnue, l'achat reste bloqué jusqu'à contrôle manuel. Le secteur est informatif dans cette version. Aucun ordre n'est envoyé au broker.
</small></div>
</main></body></html>
"""



def send_telegram(
    regime: dict,
    candidates: list[Candidate],
) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("Telegram secrets missing; notification skipped.")
        return

    lines = [
        "📈 Trading Assistant v1.3",
        f"Marché : {regime['color']} ({regime['score']}/100)",
        f"Nouvelles positions autorisées : {regime['new_positions_allowed']}",
        "",
    ]

    if regime["color"] == "ROUGE":
        lines.append("⛔ Aucun nouvel achat.")
    elif not candidates:
        lines.append("Aucun finaliste conforme aujourd'hui.")
    else:
        for i, c in enumerate(candidates, 1):
            lines += [
                f"{i}. {c.ticker} — {c.final_decision}",
                f"Scanner {c.status} | Score {c.score}/100 | Graph {c.chart_validation_score}/100",
                f"Entrée {c.entry_trigger}$ | Max {c.buy_zone_max}$ | Stop {c.stop}$",
                f"{c.shares} actions | risque {c.risk_eur}€",
                f"Résultats : {c.next_earnings_date or 'INCONNU'}",
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
    print("Starting Trading Assistant v1.3")

    regime = market_regime()
    eurusd = get_eurusd()
    spy_returns = benchmark_returns()

    print(f"Market regime: {regime['color']} ({regime['score']}/100)")
    print(f"EUR/USD: {eurusd:.4f}")
    print(
        f"SPY 6m: {spy_returns['perf_6m']:.2f}% | "
        f"SPY 12m: {spy_returns['perf_12m']:.2f}%"
    )

    symbols = download_universe()
    frames = {}

    if regime["color"] != "ROUGE":
        frames = fetch_prices(
            symbols,
            period="2y",
            parallel=True,
            retries=2,
        )

    funnel = {
        "Univers filtré": len(symbols),
        "Données téléchargées": len(frames),
        "Rejet - historique": 0,
        "Rejet - prix": 0,
        "Rejet - liquidité": 0,
        "Rejet - RS Rank": 0,
        "Rejet - Trend Template": 0,
        "Rejet - MM200": 0,
        "Rejet - range 52 semaines": 0,
        "Rejet - momentum": 0,
        "Rejet - RS vs SPY": 0,
        "Rejet - profondeur base": 0,
        "Rejet - qualité base": 0,
        "Rejet - distance pivot": 0,
        "Rejet - stop > 8% / invalide": 0,
        "Rejet - taille": 0,
        "Erreurs": 0,
        "Finalistes": 0,
        "Affichés": 0,
    }

    stage_map = {
        "historique": "Rejet - historique",
        "prix": "Rejet - prix",
        "liquidite": "Rejet - liquidité",
        "rs_rank": "Rejet - RS Rank",
        "trend_template": "Rejet - Trend Template",
        "sma200": "Rejet - MM200",
        "52w_high": "Rejet - range 52 semaines",
        "52w_low": "Rejet - range 52 semaines",
        "momentum": "Rejet - momentum",
        "relative_strength": "Rejet - RS vs SPY",
        "base_depth": "Rejet - profondeur base",
        "base_quality": "Rejet - qualité base",
        "pivot": "Rejet - distance pivot",
        "distance_pivot": "Rejet - distance pivot",
        "stop": "Rejet - stop > 8% / invalide",
        "stop_trop_large": "Rejet - stop > 8% / invalide",
        "taille": "Rejet - taille",
        "erreur": "Erreurs",
    }

    all_candidates: list[Candidate] = []

    if regime["color"] != "ROUGE":
        universe_momentum = compute_universe_momentum(frames)

        print(
            f"Momentum ranks computed: {len(universe_momentum)}"
        )

        for ticker, frame in frames.items():
            candidate, stage = analyze_symbol(
                ticker,
                frame,
                universe_momentum,
                spy_returns,
                eurusd,
            )

            if candidate:
                all_candidates.append(candidate)
            elif stage in stage_map:
                funnel[stage_map[stage]] += 1

    funnel["Finalistes"] = len(all_candidates)

    status_priority = {
        "DANS ZONE D'ACHAT": 0,
        "PRÊT À DÉCLENCHER": 1,
        "ATTENDRE": 2,
        "TROP ÉTENDU": 3,
    }

    all_candidates.sort(
        key=lambda c: (
            status_priority.get(c.status, 9),
            -c.score,
            -c.rs_rank,
        )
    )

    max_new = int(
        CONFIG.get("max_new_candidates", 8)
    )

    candidates = all_candidates[:max_new]
    funnel["Affichés"] = len(candidates)

    # Validation finale uniquement sur les quelques titres affichés.
    enriched_candidates: list[Candidate] = []
    display_candidates: list[dict] = []

    for candidate in candidates:
        candidate = enrich_finalist(candidate)
        enriched_candidates.append(candidate)

        display = asdict(candidate)
        frame = frames.get(candidate.ticker)

        if frame is not None:
            display["price_chart_uri"] = make_price_chart_uri(frame, candidate)
            display["volume_chart_uri"] = make_volume_chart_uri(frame, candidate)
        else:
            display["price_chart_uri"] = ""
            display["volume_chart_uri"] = ""

        display_candidates.append(display)

    candidates = enriched_candidates

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "version": "1.3",
        "regime": regime,
        "eurusd": round(eurusd, 4),
        "spy_returns": {
            "perf_6m_pct": round(spy_returns["perf_6m"], 2),
            "perf_12m_pct": round(spy_returns["perf_12m"], 2),
        },
        "rs_rank_definition": (
            "Percentile momentum interne 1-99, "
            "non équivalent au rating IBD propriétaire"
        ),
        "funnel": funnel,
        "candidates": [asdict(c) for c in candidates],
    }

    (DATA / "latest.json").write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=True,
        ),
        encoding="utf-8",
    )

    html = Template(HTML_TEMPLATE).render(
        regime=regime,
        eurusd=round(eurusd, 4),
        funnel=funnel,
        candidates=candidates,
        display_candidates=display_candidates,
        generated_at=datetime.now().astimezone().strftime("%d/%m/%Y %H:%M"),
    )

    (DOCS / "index.html").write_text(
        html,
        encoding="utf-8",
    )

    send_telegram(regime, candidates)

    print(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=True,
        )
    )


if __name__ == "__main__":
    main()

