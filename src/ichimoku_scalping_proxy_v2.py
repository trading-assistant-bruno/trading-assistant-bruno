from __future__ import annotations

import pandas as pd
import ichimoku_scalping_proxy as m


def fixed_add_daily_pivots(x: pd.DataFrame) -> pd.DataFrame:
    d = x[["High", "Low", "Close"]].resample("1D").agg({"High": "max", "Low": "min", "Close": "last"}).dropna()
    p = (d["High"] + d["Low"] + d["Close"]) / 3.0
    piv = pd.DataFrame(index=d.index)
    piv["pivot"] = p.shift(1)
    piv["r1"] = (2 * p - d["Low"]).shift(1)
    piv["s1"] = (2 * p - d["High"]).shift(1)
    key = x.index.floor("1D")
    out = x.copy()
    out["pivot"] = pd.Series(key.map(piv["pivot"]), index=x.index).values
    out["r1"] = pd.Series(key.map(piv["r1"]), index=x.index).values
    out["s1"] = pd.Series(key.map(piv["s1"]), index=x.index).values
    return out


m.add_daily_pivots = fixed_add_daily_pivots

if __name__ == "__main__":
    m.main()
