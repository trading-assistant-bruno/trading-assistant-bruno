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


# ─────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────

ROOT = Path(__file__).resolve().parents[1]

CONFIG = yaml.safe_load(
    (ROOT / "config.yml").read_text(encoding="utf-8")
)

DOCS = ROOT / "docs"
DATA = ROOT / "data"

DOCS.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)


# ─────────────────────────────────────
# STRUCTURE D'UNE ACTION CANDIDATE
# ─────────────────────────────────────

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


# ─────────────────────────────────────
# UNIVERS D'ACTIONS
# ─────────────────────────────────────

def download_universe() -> list[str]:
    """
    Télécharge les titres cotés aux États-Unis.

    Exclut :
    - ETF
    - warrants
    - rights
    - units
    - preferred shares
    - obligations
    - closed-end funds
    - titres de test Nasdaq
    """

    urls = [
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
    ]

    symbols: list[str] = []

    banned_terms = (
        "warrant",
        "warrants",
        "rights",
        "unit",
        "units",
        "preferred",
        "bond",
        "bonds",
        "debenture",
        "notes",
        "closed-end fund",
    )

    for url in urls:

        try:

            df = pd.read_csv(
                url,
                sep="|"
            )

            if "Symbol" in df.columns:
                symbol_col = "Symbol"
            elif "ACT Symbol" in df.columns:
                symbol_col = "ACT Symbol"
            else:
                continue

            # ──────────────────────────
            # SUPPRESSION DES ETF
            # ──────────────────────────

            if "ETF" in df.columns:

                df = df[
                    df["ETF"]
                    .astype(str)
                    .str.upper()
                    .eq("N")
                ]

            # ──────────────────────────
            # SUPPRESSION DES TITRES TEST
            # ──────────────────────────

            if "Test Issue" in df.columns:

                df = df[
                    df["Test Issue"]
                    .astype(str)
                    .str.upper()
                    .eq("N")
                ]

            # ──────────────────────────
            # SUPPRESSION AUTRES PRODUITS
            # ──────────────────────────

            if "Security Name" in df.columns:

                security_names = (
                    df["Security Name"]
                    .astype(str)
                    .str.lower()
                )

                mask = ~security_names.apply(
                    lambda name: any(
                        term in name
                        for term in banned_terms
                    )
                )

                df = df[mask]

            # ──────────────────────────
            # TICKERS
            # ──────────────────────────

            for raw in df[symbol_col].dropna().astype(str):

                ticker = (
                    raw
                    .strip()
                    .replace(".", "-")
                )

                if not ticker:
                    continue

                if "File Creation Time" in ticker:
                    continue

                if "$" in ticker:
                    continue

                if "^" in ticker:
                    continue

                if len(ticker) > 6:
                    continue

                symbols.append(ticker)

        except Exception as exc:

            print(
                f"Universe source failed: "
                f"{url}: {exc}"
            )

    # Supprime les doublons
    symbols = sorted(set(symbols))

    max_symbols = int(
        CONFIG["universe"]["max_symbols"]
    )

    if max_symbols > 0:
        symbols = symbols[:max_symbols]

    print(
        f"Universe after filtering: "
        f"{len(symbols)} securities"
    )

    return symbols


# ─────────────────────────────────────
# CRÉATION DE LOTS
# ─────────────────────────────────────

def batched(
    items: list[str],
    size: int
) -> Iterable[list[str]]:

    for i in range(
        0,
        len(items),
        size
    ):
        yield items[i:i + size]


# ─────────────────────────────────────
# TÉLÉCHARGEMENT DES PRIX
# ─────────────────────────────────────

def fetch_prices(
    symbols: list[str],
    period: str = "2y"
) -> dict[str, pd.DataFrame]:

    output: dict[str, pd.DataFrame] = {}

    for batch_number, batch in enumerate(
        batched(symbols, 150),
        start=1
    ):

        print(
            f"Downloading batch "
            f"{batch_number}: "
            f"{len(batch)} symbols"
        )

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
                    output[ticker] = (
                        data.dropna(how="all")
                    )

            else:

                for ticker in batch:

                    try:

                        frame = (
                            data[ticker]
                            .dropna(how="all")
                        )

                        if not frame.empty:
                            output[ticker] = frame

                    except Exception:
                        continue

        except Exception as exc:

            print(
                f"Batch failed: {exc}"
            )

        # Petite pause pour limiter les problèmes Yahoo
        time.sleep(0.25)

    return output


# ─────────────────────────────────────
# INDICATEURS TECHNIQUES
# ─────────────────────────────────────

def add_indicators(
    df: pd.DataFrame
) -> pd.DataFrame:

    x = df.copy()

    x.columns = [
        str(column).title()
        for column in x.columns
    ]

    required = {
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    }

    if not required.issubset(
        set(x.columns)
    ):
        raise ValueError(
            "Missing OHLCV columns"
        )

    # Moyennes mobiles
    for period in (
        21,
        50,
        150,
        200
    ):

        x[f"SMA{period}"] = (
            x["Close"]
            .rolling(period)
            .mean()
        )

    # Plus haut 52 semaines
    x["High52"] = (
        x["High"]
        .rolling(252)
        .max()
    )

    # Liquidité moyenne
    x["AvgDollarVol20"] = (
        (
            x["Close"]
            * x["Volume"]
        )
        .rolling(20)
        .mean()
    )

    return x


# ─────────────────────────────────────
# RÉGIME DE MARCHÉ
# ─────────────────────────────────────

def market_regime() -> dict:

    tickers = [
        "QQQ",
        "SPY",
        "^IXIC",
        "^VIX",
    ]

    frames = fetch_prices(
        tickers,
        period="1y"
    )

    details = {}

    positives = 0

    for ticker in [
        "QQQ",
        "SPY",
        "^IXIC",
    ]:

        df = add_indicators(
            frames[ticker]
        )

        last = df.iloc[-1]

        slope_days = int(
            CONFIG["strategy"][
                "sma50_slope_days"
            ]
        )

        sma50_rising = bool(
            last["SMA50"]
            >
            df["SMA50"].iloc[
                -1 - slope_days
            ]
        )

        above_50 = bool(
            last["Close"]
            >
            last["SMA50"]
        )

        above_200 = bool(
            last["Close"]
            >
            last["SMA200"]
        )

        score = sum(
            [
                above_50,
                above_200,
                sma50_rising,
            ]
        )

        positives += score

        details[ticker] = {

            "close": round(
                float(last["Close"]),
                2
            ),

            "above_sma50":
                above_50,

            "above_sma200":
                above_200,

            "sma50_rising":
                sma50_rising,
        }

    # ──────────────────────────────
    # VIX
    # ──────────────────────────────

    vix = add_indicators(
        frames["^VIX"]
    )

    vix_close = float(
        vix.iloc[-1]["Close"]
    )

    vix_ok = (
        vix_close < 25
    )

    positives += int(
        vix_ok
    )

    details["VIX"] = {

        "close": round(
            vix_close,
            2
        ),

        "below_25":
            vix_ok,
    }

    # 10 critères binaires
    # 3 indices x 3 conditions
    # + VIX

    if positives >= 9:

        color = "VERT"

    elif positives >= 6:

        color = "ORANGE"

    else:

        color = "ROUGE"

    return {

        "color":
            color,

        "score":
            positives * 10,

        "details":
            details,

        "new_positions_allowed": {
            "VERT": 4,
            "ORANGE": 1,
            "ROUGE": 0,
        }[color],
    }


# ─────────────────────────────────────
# ANALYSE D'UNE ACTION
# ─────────────────────────────────────

def analyze_symbol(
    ticker: str,
    raw: pd.DataFrame
) -> Candidate | None:

    try:

        df = add_indicators(raw)

        # Il faut assez d'historique
        # pour calculer le plus haut 52 semaines

        if len(df) < 260:
            return None

        df = df.dropna()

        if df.empty:
            return None

        last = df.iloc[-1]

        close = float(
            last["Close"]
        )

        cfg_u = CONFIG["universe"]
        cfg_s = CONFIG["strategy"]

        # ──────────────────────────
        # PRIX MINIMUM
        # ──────────────────────────

        if close < float(
            cfg_u["min_price"]
        ):
            return None

        # ──────────────────────────
        # LIQUIDITÉ
        # ──────────────────────────

        avg_dollar_vol = float(
            last["AvgDollarVol20"]
        )

        if avg_dollar_vol < float(
            cfg_u["min_avg_dollar_volume"]
        ):
            return None

        # ──────────────────────────
        # TREND TEMPLATE
        # ──────────────────────────

        if not (
            close
            >
            last["SMA50"]
            >
            last["SMA150"]
            >
            last["SMA200"]
        ):
            return None

        # ──────────────────────────
        # MM200 MONTANTE
        # ──────────────────────────

        slope_days = int(
            cfg_s[
                "sma200_slope_days"
            ]
        )

        if float(
            last["SMA200"]
        ) <= float(
            df["SMA200"].iloc[
                -1 - slope_days
            ]
        ):
            return None

        # ──────────────────────────
        # DISTANCE PLUS HAUT 52 SEMAINES
        # ──────────────────────────

        high52 = float(
            last["High52"]
        )

        distance_high = (
            (
                high52
                - close
            )
            / high52
            * 100
        )

        if distance_high > float(
            cfg_s[
                "max_distance_from_52w_high_pct"
            ]
        ):
            return None

        # ──────────────────────────
        # PIVOT MÉCANIQUE
        # ──────────────────────────

        lookback = int(
            cfg_s[
                "pivot_lookback_days"
            ]
        )

        exclude_recent = int(
            cfg_s[
                "pivot_exclude_recent_days"
            ]
        )

        pivot_window = (
            df["High"]
            .iloc[
                -lookback:
                -exclude_recent
            ]
        )

        if pivot_window.empty:
            return None

        pivot = float(
            pivot_window.max()
        )

        # Entrée légèrement au-dessus du pivot
        entry = (
            pivot * 1.001
        )

        distance_above_pivot = (
            (
                close
                - pivot
            )
            / pivot
            * 100
        )

        if distance_above_pivot > float(
            cfg_s[
                "max_distance_above_pivot_pct"
            ]
        ):
            return None

        # ──────────────────────────
        # STOP
        # ──────────────────────────

        recent_swing_low = float(
            df["Low"]
            .iloc[-10:]
            .min()
        ) * 0.995

        min_stop = (
            entry
            *
            (
                1
                -
                float(
                    cfg_s[
                        "min_stop_distance_pct"
                    ]
                )
                / 100
            )
        )

        max_stop = (
            entry
            *
            (
                1
                -
                float(
                    cfg_s[
                        "max_stop_distance_pct"
                    ]
                )
                / 100
            )
        )

        # Stop sous le récent plus bas,
        # mais limité à la bande 3-8 %

        stop = min(
            min_stop,
            recent_swing_low
        )

        stop = max(
            stop,
            max_stop
        )

        stop_pct = (
            (
                entry
                - stop
            )
            / entry
            * 100
        )

        if stop_pct <= 0:
            return None

        if stop_pct > float(
            cfg_s[
                "max_stop_distance_pct"
            ]
        ):
            return None

        # ──────────────────────────
        # TAILLE DE POSITION
        # ──────────────────────────

        capital = float(
            CONFIG["capital_eur"]
        )

        risk_eur = (
            capital
            *
            float(
                CONFIG[
                    "risk_per_trade_pct"
                ]
            )
            / 100
        )

        risk_per_share = (
            entry
            - stop
        )

        if risk_per_share <= 0:
            return None

        shares = max(
            0,
            math.floor(
                risk_eur
                /
                risk_per_share
            )
        )

        if shares < 1:
            return None

        position_value = (
            shares
            * entry
        )

        # ──────────────────────────
        # VOLUME
        # ──────────────────────────

        average_volume = (
            df["Volume"]
            .iloc[-21:-1]
            .mean()
        )

        volume_ratio = float(
            last["Volume"]
            /
            max(
                average_volume,
                1
            )
        )

        # ──────────────────────────
        # CONTRACTION VOLATILITÉ
        # ──────────────────────────

        volatility_now = float(

            (
                df["High"]
                .iloc[-10:]
                -
                df["Low"]
                .iloc[-10:]
            )
            .mean()
            /
            close
        )

        volatility_previous = float(

            (
                df["High"]
                .iloc[-30:-10]
                -
                df["Low"]
                .iloc[-30:-10]
            )
            .mean()
            /
            close
        )

        contraction = (
            volatility_now
            <
            volatility_previous
        )

        # ──────────────────────────
        # SCORE
        # ──────────────────────────

        score = 55.0

        reasons = [

            "Cours > MM50 > MM150 > MM200",

            "MM200 montante",

            (
                f"À {distance_high:.1f}% "
                f"du plus haut 52 semaines"
            ),
        ]

        # Proximité du plus haut annuel
        score += max(
            0,
            15
            -
            distance_high
            * 0.6
        )

        # Contraction
        if contraction:

            score += 12

            reasons.append(
                "Volatilité récente en contraction"
            )

        # Volume
        if volume_ratio > 1.2:

            score += 8

            reasons.append(
                "Volume supérieur à la moyenne"
            )

        # Proximité pivot
        proximity = abs(
            (
                close
                - pivot
            )
            /
            pivot
            * 100
        )

        score += max(
            0,
            10
            -
            proximity
            * 2
        )

        score = min(
            100,
            round(
                score,
                1
            )
        )

        return Candidate(

            ticker=
                ticker,

            close=
                round(
                    close,
                    2
                ),

            pivot=
                round(
                    pivot,
                    2
                ),

            entry=
                round(
                    entry,
                    2
                ),

            stop=
                round(
                    stop,
                    2
                ),

            stop_pct=
                round(
                    stop_pct,
                    2
                ),

            shares=
                shares,

            position_value=
                round(
                    position_value,
                    2
                ),

            risk_eur=
                round(
                    shares
                    *
                    (
                        entry
                        - stop
                    ),
                    2
                ),

            score=
                score,

            distance_52w_high_pct=
                round(
                    distance_high,
                    2
                ),

            avg_dollar_volume=
                round(
                    avg_dollar_vol,
                    0
                ),

            reasons=
                reasons,
        )

    except Exception as exc:

        print(
            f"{ticker}: analysis failed: {exc}"
        )

        return None


# ─────────────────────────────────────
# INTERFACE WEB
# ─────────────────────────────────────

HTML_TEMPLATE = """
<!doctype html>

<html lang="fr">

<head>

<meta charset="utf-8">

<meta
    name="viewport"
    content="width=device-width,initial-scale=1"
>

<title>
Trading Assistant Bruno
</title>

<style>

body {
    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;

    background:#f5f7fa;
    margin:0;
    color:#18212f;
}

main {
    max-width:820px;
    margin:auto;
    padding:18px;
}

.card {
    background:white;
    border-radius:16px;
    padding:18px;
    margin:14px 0;
    box-shadow:
        0 3px 18px
        #00000012;
}

h1 {
    font-size:24px;
}

.regime {
    font-size:28px;
    font-weight:800;
}

.VERT {
    color:#138a42;
}

.ORANGE {
    color:#c46b00;
}

.ROUGE {
    color:#c62828;
}

.grid {
    display:grid;
    grid-template-columns:
        repeat(2,1fr);
    gap:10px;
}

.metric {
    background:#f4f6f8;
    border-radius:10px;
    padding:10px;
}

.ticker {
    font-size:22px;
    font-weight:800;
}

.score {
    float:right;
}

.buy {
    font-size:18px;
    font-weight:700;
}

small {
    color:#667085;
}

ul {
    padding-left:20px;
}

@media(max-width:600px) {

    .grid {
        grid-template-columns:1fr;
    }

}

</style>

</head>

<body>

<main>

<h1>
Trading Assistant Bruno
</h1>

<div class="card">

<div class="regime {{ regime.color }}">

● Marché {{ regime.color }}

</div>

<p>

Score de régime :
{{ regime.score }}/100

—

Nouvelles positions autorisées :
{{ regime.new_positions_allowed }}

</p>

<small>

Calculé le
{{ generated_at }}

</small>

</div>


{% if regime.color == "ROUGE" %}

<div class="card">

<h2>
Aucun nouvel achat
</h2>

<p>
Le régime rouge bloque automatiquement
les nouvelles positions.
</p>

</div>

{% endif %}


{% if not candidates and regime.color != "ROUGE" %}

<div class="card">

<h2>
Aucune configuration retenue
</h2>

<p>
Aucune action ne respecte actuellement
tous les critères du scanner.
</p>

</div>

{% endif %}


{% for c in candidates %}

<div class="card">

<div class="ticker">

{{ loop.index }}.
{{ c.ticker }}

<span class="score">
{{ c.score }}/100
</span>

</div>

<p class="buy">

Ordre conditionnel :
achat stop-limit autour de
{{ c.entry }} $

</p>

<div class="grid">

<div class="metric">

<small>
Pivot
</small>

<br>

<b>
{{ c.pivot }} $
</b>

</div>


<div class="metric">

<small>
Stop initial
</small>

<br>

<b>
{{ c.stop }} $
</b>

({{ c.stop_pct }}%)

</div>


<div class="metric">

<small>
Taille
</small>

<br>

<b>
{{ c.shares }} actions
</b>

</div>


<div class="metric">

<small>
Risque estimé
</small>

<br>

<b>
{{ c.risk_eur }} €
</b>

</div>

</div>


<ul>

{% for reason in c.reasons %}

<li>
{{ reason }}
</li>

{% endfor %}

</ul>


<p>

<a
href="https://www.tradingview.com/chart/?symbol={{ c.ticker }}"
target="_blank"
>

Ouvrir le graphique

</a>

</p>

</div>

{% endfor %}


<div class="card">

<small>

Prototype d'aide à la décision.

Les niveaux sont calculés mécaniquement
sur données quotidiennes et doivent être
vérifiés avant tout ordre réel.

</small>

</div>

</main>

</body>

</html>
"""


# ─────────────────────────────────────
# TELEGRAM
# ─────────────────────────────────────

def send_telegram(
    regime: dict,
    candidates: list[Candidate]
) -> None:

    token = os.getenv(
        "TELEGRAM_BOT_TOKEN"
    )

    chat_id = os.getenv(
        "TELEGRAM_CHAT_ID"
    )

    if not token or not chat_id:

        print(
            "Telegram secrets missing; "
            "notification skipped."
        )

        return

    lines = [

        "📈 Trading Assistant",

        (
            f"Marché : "
            f"{regime['color']} "
            f"({regime['score']}/100)"
        ),

        (
            f"Nouvelles positions : "
            f"{regime['new_positions_allowed']}"
        ),

        "",
    ]

    if regime["color"] == "ROUGE":

        lines.append(
            "⛔ Aucun nouvel achat."
        )

    elif not candidates:

        lines.append(
            "Aucune configuration conforme aujourd'hui."
        )

    else:

        for index, candidate in enumerate(
            candidates,
            start=1
        ):

            lines.extend(
                [
                    (
                        f"{index}. "
                        f"{candidate.ticker} "
                        f"— score "
                        f"{candidate.score}/100"
                    ),

                    (
                        f"Entrée "
                        f"{candidate.entry}$ "
                        f"| Stop "
                        f"{candidate.stop}$ "
                        f"| "
                        f"{candidate.shares} actions"
                    ),

                    "",
                ]
            )

    try:

        requests.post(

            (
                f"https://api.telegram.org/"
                f"bot{token}/sendMessage"
            ),

            json={
                "chat_id":
                    chat_id,

                "text":
                    "\n".join(lines),
            },

            timeout=20,

        ).raise_for_status()

    except Exception as exc:

        print(
            f"Telegram error: {exc}"
        )


# ─────────────────────────────────────
# PROGRAMME PRINCIPAL
# ─────────────────────────────────────

def main() -> None:

    print(
        "Starting Trading Assistant..."
    )

    # ──────────────────────────────
    # RÉGIME
    # ──────────────────────────────

    regime = market_regime()

    print(
        f"Market regime: "
        f"{regime['color']} "
        f"({regime['score']}/100)"
    )

    # ──────────────────────────────
    # UNIVERS
    # ──────────────────────────────

    symbols = download_universe()

    print(
        f"Universe: "
        f"{len(symbols)} symbols"
    )

    # ──────────────────────────────
    # DONNÉES ACTIONS
    # ──────────────────────────────

    frames = fetch_prices(
        symbols,
        period="2y"
    )

    print(
        f"Price data downloaded: "
        f"{len(frames)} symbols"
    )

    # ──────────────────────────────
    # SCREENING
    # ──────────────────────────────

    candidates: list[Candidate] = []

    if regime["color"] != "ROUGE":

        for ticker, frame in frames.items():

            candidate = analyze_symbol(
                ticker,
                frame
            )

            if candidate:

                candidates.append(
                    candidate
                )

        candidates.sort(
            key=lambda candidate:
                candidate.score,
            reverse=True
        )

        candidates = candidates[
            :
            int(
                CONFIG[
                    "max_new_candidates"
                ]
            )
        ]

    print(
        f"Candidates retained: "
        f"{len(candidates)}"
    )

    for candidate in candidates:

        print(
            candidate.ticker,
            candidate.score,
            candidate.entry,
            candidate.stop,
        )

    # ──────────────────────────────
    # JSON
    # ──────────────────────────────

    payload = {

        "generated_at":
            datetime
            .now()
            .astimezone()
            .isoformat(),

        "regime":
            regime,

        "candidates":
            [
                asdict(candidate)
                for candidate
                in candidates
            ],
    }

    (
        DATA
        /
        "latest.json"
    ).write_text(

        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False
        ),

        encoding="utf-8"
    )

    # ──────────────────────────────
    # PAGE WEB
    # ──────────────────────────────

    html = Template(
        HTML_TEMPLATE
    ).render(

        regime=
            regime,

        candidates=
            candidates,

        generated_at=
            datetime
            .now()
            .astimezone()
            .strftime(
                "%d/%m/%Y %H:%M"
            ),
    )

    (
        DOCS
        /
        "index.html"
    ).write_text(

        html,

        encoding="utf-8"
    )

    # ──────────────────────────────
    # TELEGRAM
    # ──────────────────────────────

    send_telegram(
        regime,
        candidates
    )

    # ──────────────────────────────
    # LOG FINAL
    # ──────────────────────────────

    print(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False
        )
    )


# ─────────────────────────────────────
# LANCEMENT
# ─────────────────────────────────────

if __name__ == "__main__":
    main()
