"""P3: event-grouped, market-disjoint, chronological splits for Polymarket decision points.

market_splits.parquet : one row per market: event_id, event_t_first, split
    test_ood : event first traded >= 2026-01-01 UTC
    val      : event first traded in [2025-10-01)
    train / test_id : remaining events hashed (sha1 of event_id) 90 / 10
splits.parquet : one row per decision point (point_id, market_id, kind, split); for train / val / test_id the
    target horizon must end before 2026-01-01 (t + 24h < cutoff), points violating this are dropped.
"""
import sys, os, time, hashlib, datetime as dt
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eventworld.pm_state import ROOT, DAY

CUT_VAL = int(dt.datetime(2025, 10, 1, tzinfo=dt.timezone.utc).timestamp())
CUT_OOD = int(dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc).timestamp())


def hash_bucket(s, mod=100):
    return int(hashlib.sha1(s.encode()).hexdigest(), 16) % mod


def main():
    t0 = time.time()
    m = pl.read_parquet(f"{ROOT}/markets.parquet").select("market_id", "event_id", "t_first", "t_last", "topic")
    ev = m.group_by("event_id").agg(pl.col("t_first").min().alias("event_t_first"), pl.len().alias("n_markets_event"))
    ev = ev.with_columns(pl.col("event_id").map_elements(hash_bucket, return_dtype=pl.Int64).alias("bucket"))
    ev = ev.with_columns(
        pl.when(pl.col("event_t_first") >= CUT_OOD).then(pl.lit("test_ood"))
        .when(pl.col("event_t_first") >= CUT_VAL).then(pl.lit("val"))
        .when(pl.col("bucket") < 90).then(pl.lit("train")).otherwise(pl.lit("test_id")).alias("split"))
    ms = m.join(ev, on="event_id", how="left").select("market_id", "event_id", "topic", "t_first", "t_last", "event_t_first", "n_markets_event", "split")
    ms.write_parquet(f"{ROOT}/market_splits.parquet")
    pts = pl.read_parquet(f"{ROOT}/points.parquet", columns=["point_id", "market_id", "kind", "t"])
    sp = pts.join(ms.select("market_id", "split"), on="market_id", how="left")
    n_before = sp.height
    sp = sp.filter((pl.col("split") == "test_ood") | (pl.col("t") + DAY < CUT_OOD))
    sp.select("point_id", "market_id", "kind", "split").write_parquet(f"{ROOT}/splits.parquet")
    el = time.time() - t0
    tab_m = ms.group_by("split").agg(pl.len().alias("markets"), pl.col("event_id").n_unique().alias("events")).sort("split")
    tab_p = sp.group_by("split", "kind").len().sort("split", "kind").pivot(on="kind", index="split", values="len").sort("split")
    lines = ["# P3 Polymarket splits", "",
             f"- cutoffs (UTC): val >= 2025-10-01, test_ood >= 2026-01-01 on the event's first trade; train/test_id = sha1(event_id) mod 100 < 90 / >= 90",
             f"- points before truncation {n_before:,}; after dropping non-OOD points with t+24h >= 2026-01-01: {sp.height:,}",
             "", "markets / events per split:", tab_m.to_pandas().to_markdown(index=False),
             "", "points per split:", tab_p.to_pandas().to_markdown(index=False),
             "", f"- runtime {el:.1f}s; outputs `{ROOT}/market_splits.parquet`, `{ROOT}/splits.parquet`"]
    open("out/build/pm_splits.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
