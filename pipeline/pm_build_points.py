"""P2: decision points = anchors (burst-start large orders) + uniform control windows, with block-strict state summary.

anchor: order k of wallet w with q/D >= 0.1 (q = order usd, D = 24h volume in blocks < b_k, +1 smoothing),
        burst start = w has no order in this market with ts >= t-1h in blocks < b_k,
        t >= first trade + 24h, t + 24h <= last trade, p_buy and p_sell both defined.
control: 100 times per market uniform in [first+24h, last-24h] (seeded per market); no designated actor;
        history = orders with ts < t.
Output: data/processed/polymarket/points.parquet
"""
import sys, time, os
import multiprocessing as mp
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eventworld.pm_state import MarketOrders, market_meta, ROOT, DAY, HOUR

N_CONTROL = 100
QD_MIN = 0.1
BURST_GAP = HOUR


def band_of(mid):
    return np.where((mid >= 0.2) & (mid <= 0.8), "mid", "extreme")


def rho_bin(qd):
    return np.select([qd < 0.2, qd < 0.5, qd < 1.0], ["0.1-0.2", "0.2-0.5", "0.5-1"], ">=1")


def vec_state(m, t, n_hist):
    """Vectorized phi(H) for arrays t, n_hist (no actor part)."""
    t = np.asarray(t, dtype=np.int64); n_hist = np.asarray(n_hist, dtype=np.int64)
    has = n_hist > 0
    j = np.clip(n_hist - 1, 0, max(m.n - 1, 0))
    pb = np.where(has, m.lb[j], np.nan); ps = np.where(has, m.ls[j], np.nan)
    both = ~np.isnan(pb) & ~np.isnan(ps)
    mid = np.where(both, (pb + ps) / 2, np.where(np.isnan(pb), ps, pb))
    a24 = np.minimum(np.searchsorted(m.ts, t - DAY, side="left"), n_hist)
    a1 = np.minimum(np.searchsorted(m.ts, t - HOUR, side="left"), n_hist)
    since = np.where(has, t - m.ts[j], -1)
    ttl = np.full(len(t), np.nan)
    if m.end_ts is not None:
        d = (m.end_ts - t) / DAY
        ttl = np.where((d >= 0) & (d <= 5 * 365), d, np.nan)
    return {"p_buy": pb, "p_sell": ps, "mid": mid, "spread": pb - ps,
            "D": m.cum_usd[n_hist] - m.cum_usd[a24],
            "N1h": (n_hist - a1).astype(np.int64), "F1h": m.cum_flow[n_hist] - m.cum_flow[a1], "V1h": m.cum_usd[n_hist] - m.cum_usd[a1],
            "age_d": (t - m.t_first) / DAY, "ttl_d": ttl, "since_last_s": since}


def build_market(mid):
    m = MarketOrders(mid, meta=market_meta(mid))
    if m.n < 2:
        return None, {"market_id": mid, "n_large": 0, "n_burst": 0, "n_anchor": 0, "n_control": 0}
    n = m.n
    ts = m.ts
    nh = m.block_start
    # block-strict D and q/D for all orders
    a24 = np.minimum(np.searchsorted(ts, ts - DAY, side="left"), nh)
    D = m.cum_usd[nh] - m.cum_usd[a24]
    qd = m.usd / (D + 1.0)
    # previous order of the same wallet (by idx) -> its ts; wallet rank (count of w's earlier orders)
    order = m.w_order
    ws = m.wallet[order]
    prev_same = np.concatenate([[False], ws[1:] == ws[:-1]])
    prev_ts_sorted = np.where(prev_same, np.concatenate([[-1], ts[order][:-1]]), -10**12)
    inv = np.empty(n, dtype=np.int64); inv[order] = np.arange(n)
    prev_ts = prev_ts_sorted[inv]
    seg_start = np.zeros(n, dtype=np.int64)
    starts = np.flatnonzero(~prev_same)
    seg_id = np.cumsum(~prev_same) - 1
    seg_start = starts[seg_id]  # in sorted arrangement
    rank_sorted = np.arange(n) - seg_start
    rank = rank_sorted[inv]
    pos_before = (m.w_pos_pref[np.arange(n)] - m.w_pos_pref[seg_start])[inv]
    flow_before = (m.w_flow_pref[np.arange(n)] - m.w_flow_pref[seg_start])[inv]
    large = qd >= QD_MIN
    burst = prev_ts < ts - BURST_GAP
    ok_time = (ts >= m.t_first + DAY) & (ts + DAY <= m.t_last)
    st_all = None
    sel = np.flatnonzero(large & burst & ok_time)
    frames = []
    n_anchor = 0
    if len(sel):
        st = vec_state(m, ts[sel], nh[sel])
        defined = ~np.isnan(st["p_buy"]) & ~np.isnan(st["p_sell"])
        sel = sel[defined]; st = {k: v[defined] for k, v in st.items()}
        n_anchor = len(sel)
        # prev order of w must be block-strict: same-block earlier order of w has ts >= t - gap, so burst already excludes it
        w_since = np.where(prev_ts[sel] > 0, ts[sel] - prev_ts[sel], -1)
        frames.append(pl.DataFrame({
            "point_id": [f"{mid}:{i}" for i in sel], "market_id": mid, "kind": "anchor", "t": ts[sel], "n_hist": nh[sel],
            "anchor_idx": sel, "block": m.block[sel], "wallet": m.wallet[sel], "d": m.sign[sel], "q": m.usd[sel], "qD": qd[sel], "p_exec": m.p[sel],
            **st, "w_pos": pos_before[sel], "w_n_prev": rank[sel], "w_flow": flow_before[sel], "w_since_last_s": w_since}))
    # controls
    lo, hi = m.t_first + DAY, m.t_last - DAY
    n_control = 0
    if hi > lo:
        rng = np.random.default_rng(int(mid) % (2**32))
        tc = np.sort(rng.integers(lo, hi + 1, size=N_CONTROL)).astype(np.int64)
        nhc = np.searchsorted(ts, tc, side="left")
        st = vec_state(m, tc, nhc)
        defined = ~np.isnan(st["p_buy"]) & ~np.isnan(st["p_sell"])
        tc = tc[defined]; nhc = nhc[defined]; st = {k: v[defined] for k, v in st.items()}
        n_control = len(tc)
        frames.append(pl.DataFrame({
            "point_id": [f"{mid}:c{j}" for j in range(len(tc))], "market_id": mid, "kind": "control", "t": tc, "n_hist": nhc,
            "anchor_idx": np.full(len(tc), -1, dtype=np.int64), "block": np.full(len(tc), -1, dtype=np.int64), "wallet": np.zeros(len(tc), dtype=np.int64),
            "d": np.zeros(len(tc), dtype=np.int64), "q": np.zeros(len(tc)), "qD": np.zeros(len(tc)), "p_exec": np.full(len(tc), np.nan),
            **st, "w_pos": np.full(len(tc), np.nan), "w_n_prev": np.full(len(tc), -1, dtype=np.int64), "w_flow": np.full(len(tc), np.nan), "w_since_last_s": np.full(len(tc), -1, dtype=np.int64)}))
    stats = {"market_id": mid, "n_large": int(large.sum()), "n_burst": int((large & burst).sum()), "n_anchor": n_anchor, "n_control": n_control}
    if not frames:
        return None, stats
    df = pl.concat([f.select(frames[0].columns) for f in frames]).with_columns([
        pl.col("anchor_idx").cast(pl.Int64), pl.col("block").cast(pl.Int64), pl.col("wallet").cast(pl.UInt32), pl.col("d").cast(pl.Int8),
        pl.col("n_hist").cast(pl.UInt32), pl.col("t").cast(pl.Int64), pl.col("N1h").cast(pl.Int64), pl.col("w_n_prev").cast(pl.Int64), pl.col("since_last_s").cast(pl.Int64), pl.col("w_since_last_s").cast(pl.Int64)])
    return df, stats


def main():
    t0 = time.time()
    meta = market_meta()
    mids = sorted(meta.keys())
    print("markets:", len(mids))
    with mp.get_context("spawn").Pool(48) as pool:
        res = pool.map(build_market, mids, chunksize=8)
    frames = [r[0] for r in res if r[0] is not None]
    stats = pl.DataFrame([r[1] for r in res])
    pts = pl.concat(frames)
    mk = pl.read_parquet(f"{ROOT}/markets.parquet").select("market_id", "event_id", "topic", "t_first", "t_last")
    pts = pts.join(mk, on="market_id", how="left").with_columns([
        pl.Series("band", band_of(pts["mid"].to_numpy())), pl.Series("rho_bin", np.where(pts["kind"].to_numpy() == "anchor", rho_bin(pts["qD"].to_numpy()), "control")),
    ]).sort(["market_id", "t", "point_id"])
    pts.write_parquet(f"{ROOT}/points.parquet")
    na = (pts["kind"] == "anchor").sum(); nc = (pts["kind"] == "control").sum()
    el = time.time() - t0
    print(f"points: {pts.height:,} (anchors {na:,}, controls {nc:,}); large orders {stats['n_large'].sum():,}, burst-starts {stats['n_burst'].sum():,}; elapsed {el:.0f}s")
    lines = ["# P2 Polymarket decision points", "",
             f"- large orders (q/D >= 0.1, block-strict D): {stats['n_large'].sum():,}",
             f"- of which burst-starts (no order by the same wallet in the previous 1h): {stats['n_burst'].sum():,}",
             f"- anchors after time constraints (t >= open+24h, t+24h <= last trade) and defined p_buy/p_sell: {na:,}",
             f"- controls: {nc:,} ({N_CONTROL} uniform times per market in [open+24h, last-24h]); markets with anchors: {(stats['n_anchor']>0).sum()} / {len(mids)}",
             f"- anchors per market: median {stats['n_anchor'].median():.0f}, p10 {stats['n_anchor'].quantile(.1):.0f}, p90 {stats['n_anchor'].quantile(.9):.0f}",
             f"- band (mid in [0.2,0.8]) share among anchors: {(pts.filter(pl.col('kind')=='anchor')['band']=='mid').mean():.3f}",
             "", "rho bins (anchors):", pts.filter(pl.col("kind") == "anchor").group_by("rho_bin").len().sort("rho_bin").to_pandas().to_markdown(index=False),
             "", f"- runtime {el:.0f}s; output `{ROOT}/points.parquet` ({pts.height:,} rows)"]
    open("out/build/pm_points.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
