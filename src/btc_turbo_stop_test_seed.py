from pathlib import Path
import pandas as pd
import btc_turbo_stop_test as bt


def seeded_collect_snapshots():
    candidates = [
        bt.ROOT / "seed" / "snapshots.csv",
        bt.ROOT / "data" / "btc_turbo_pit_monthly" / "snapshots.csv",
    ]
    for path in candidates:
        if path.exists():
            s = pd.read_csv(path)
            s["date"] = pd.to_datetime(s["date"])
            s["Rank"] = pd.to_numeric(s["Rank"], errors="coerce")
            s["Price"] = pd.to_numeric(s["Price"], errors="coerce")
            s["Symbol"] = s["Symbol"].astype(str).str.upper().str.strip()
            s = s.dropna(subset=["date", "Rank", "Price", "Symbol"])
            s = s[(s["Rank"] >= 1) & (s["Rank"] <= bt.UNIVERSE_N)]
            s = s[(s["date"] >= bt.START) & (s["date"] <= bt.END)]
            s = s.drop_duplicates(["date", "Symbol"]).sort_values(["date", "Rank"])
            if s["date"].nunique() < 100:
                raise RuntimeError(f"Seed snapshot file incomplete: {s['date'].nunique()} dates")
            s.to_csv(bt.OUT / "snapshots.csv", index=False)
            pd.DataFrame(columns=["date", "error"]).to_csv(bt.OUT / "snapshot_failures.csv", index=False)
            print(f"Using validated PIT seed: {path} with {s['date'].nunique()} snapshots")
            return s
    raise FileNotFoundError("Validated PIT snapshot seed not found")


bt.collect_snapshots = seeded_collect_snapshots
bt.main()
