"""P4: horizon targets for every decision point, for Delta in {5m, 1h, 24h}.

For point (market, t, n_hist, wallet w): window = orders[n_hist:n_end], n_end = #orders with ts < t+Delta.
  end state    : p_buy/p_sell/mid/spread at t+Delta from orders[:n_end] (block-strict history + whole window), D_end
  main target  : dmid = mid_end - mid_t, is_zero, dbin (21-bin signed grid)
  joint action : N_all / F_all / V_all over the window
  own action   : list of (dt, d, q, p) of w's orders in the window (anchors), own_N/F/V/dpos, own vwap
  others       : N_o / F_o / V_o = joint - own
  bookkeeping  : w_pos_end = w_pos + own_dpos, pnl_mtm = own_dpos * mid_end - own_F (mark-to-market PnL of the window's own trades)
Vectorized per market; verified against eventworld.pm_state.MarketOrders.window in verify_pm.py.
Output: data/processed/polymarket/targets_{5m,1h,24h}.parquet
"""
import sys, os, time
import multiprocessing as mp
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eventworld.pm_state import MarketOrders, market_meta, ROOT, DELTAS, delta_bin

PTS = None


def _init():
    global PTS
    PTS = pl.read_parquet(f"{ROOT}/points.parquet", columns=["point_id", "market_id", "kind", "t", "n_hist", "wallet", "mid", "w_pos", "D"])


def build_market(mid):
    pts = PTS.filter(pl.col("market_id") == mid)
    if pts.height == 0:
        return {}
    m = MarketOrders(mid, meta=market_meta(mid))
    n = m.n
    t = pts["t"].to_numpy().astype(np.int64)
    nh = pts["n_hist"].to_numpy().astype(np.int64)
    w = pts["wallet"].to_numpy().astype(np.int64)
    mid0 = pts["mid"].to_numpy()
    is_anchor = (pts["kind"] == "anchor").to_numpy()
    # wallet-sorted keys for own-order range queries
    keys = m.wallet[m.w_order].astype(np.int64) * (n + 1) + m.w_order.astype(np.int64)
    out = {}
    for name, delta in DELTAS.items():
        n_end = np.maximum(np.searchsorted(m.ts, t + delta, side="left"), nh)
        j = np.clip(n_end - 1, 0, max(n - 1, 0))
        has = n_end > 0
        pb = np.where(has, m.lb[j], np.nan); ps = np.where(has, m.ls[j], np.nan)
        both = ~np.isnan(pb) & ~np.isnan(ps)
        mid1 = np.where(both, (pb + ps) / 2, np.where(np.isnan(pb), ps, pb))
        te = t + delta
        a24 = np.minimum(np.searchsorted(m.ts, te - 86400, side="left"), n_end)
        D_end = m.cum_usd[n_end] - m.cum_usd[a24]
        N_all = n_end - nh
        F_all = m.cum_flow[n_end] - m.cum_flow[nh]
        V_all = m.cum_usd[n_end] - m.cum_usd[nh]
        # own orders in [nh, n_end)
        a = np.searchsorted(keys, w * (n + 1) + nh, side="left")
        b = np.searchsorted(keys, w * (n + 1) + n_end, side="left")
        a = np.where(is_anchor, a, 0); b = np.where(is_anchor, b, 0)
        own_N = b - a
        own_V = m.w_usd_pref[b] - m.w_usd_pref[a]
        own_F = m.w_flow_pref[b] - m.w_flow_pref[a]
        own_dpos = m.w_pos_pref[b] - m.w_pos_pref[a]
        # flat own order lists
        tot = int(own_N.sum())
        if tot:
            rep = np.repeat(np.arange(len(t)), own_N)
            offs = np.repeat(a, own_N) + (np.arange(tot) - np.repeat(np.cumsum(own_N) - own_N, own_N))
            rows = m.w_order[offs]
            own_dt = (m.ts[rows] - t[rep]).astype(np.int64)
            own_d = m.sign[rows].astype(np.int8); own_q = m.usd[rows]; own_p = m.p[rows]
        else:
            own_dt = np.empty(0, np.int64); own_d = np.empty(0, np.int8); own_q = np.empty(0); own_p = np.empty(0)
        offsets = np.concatenate([[0], np.cumsum(own_N)]).astype(np.int64)
        own_list = pl.Series("own", [
            {"dt": own_dt[offsets[i]:offsets[i + 1]].tolist(), "d": own_d[offsets[i]:offsets[i + 1]].tolist(),
             "q": own_q[offsets[i]:offsets[i + 1]].tolist(), "p": own_p[offsets[i]:offsets[i + 1]].tolist()}
            for i in range(len(t))],
            dtype=pl.Struct({"dt": pl.List(pl.Int64), "d": pl.List(pl.Int8), "q": pl.List(pl.Float64), "p": pl.List(pl.Float64)}))
        dmid = mid1 - mid0
        with np.errstate(invalid="ignore", divide="ignore"):
            own_vwap = np.where(own_V > 0, (m.w_usd_pref[b] - m.w_usd_pref[a]) / np.maximum(np.abs(own_dpos), 1e-12), np.nan)
        df = pl.DataFrame({
            "point_id": pts["point_id"], "delta": name, "n_end": n_end.astype(np.int64),
            "p_buy_end": pb, "p_sell_end": ps, "mid_end": mid1, "spread_end": pb - ps, "D_end": D_end,
            "dmid": dmid, "is_zero": dmid == 0, "dbin": delta_bin(np.nan_to_num(dmid)).astype(np.int8),
            "N_all": N_all.astype(np.int64), "F_all": F_all, "V_all": V_all,
            "own_N": own_N.astype(np.int64), "own_F": own_F, "own_V": own_V, "own_dpos": own_dpos, "own_vwap": own_vwap,
            "N_o": (N_all - own_N).astype(np.int64), "F_o": F_all - own_F, "V_o": V_all - own_V,
            "w_pos_end": np.where(is_anchor, pts["w_pos"].to_numpy() + own_dpos, np.nan),
            "pnl_mtm": np.where(is_anchor, own_dpos * mid1 - own_F, np.nan),
            "rho": np.where(is_anchor, own_V / (pts["D"].to_numpy() + 1.0), np.nan),
        }).with_columns(own_list)
        out[name] = df
    return out


def main():
    t0 = time.time()
    _init()
    mids = PTS["market_id"].unique().sort().to_list()
    print("markets with points:", len(mids), "points:", PTS.height)
    with mp.get_context("spawn").Pool(48, initializer=_init) as pool:
        res = pool.map(build_market, mids, chunksize=4)
    lines = ["# P4 Polymarket horizon targets", ""]
    for name in DELTAS:
        df = pl.concat([r[name] for r in res if name in r]).sort("point_id")
        df.write_parquet(f"{ROOT}/targets_{name}.parquet")
        j = df.join(PTS.select("point_id", "kind"), on="point_id")
        anc = j.filter(pl.col("kind") == "anchor"); ctl = j.filter(pl.col("kind") == "control")
        lines += [f"## Delta = {name}: {df.height:,} rows",
                  f"- anchors: p_zero {anc['is_zero'].mean():.3f}, mean |dmid| {anc['dmid'].abs().mean():.4f}, mean N_all {anc['N_all'].mean():.1f}, mean own_N {anc['own_N'].mean():.2f}, own_N>1 share {(anc['own_N']>1).mean():.3f}, N_o==0 share {(anc['N_o']==0).mean():.3f}",
                  f"- controls: p_zero {ctl['is_zero'].mean():.3f}, mean |dmid| {ctl['dmid'].abs().mean():.4f}, mean N_all {ctl['N_all'].mean():.1f}",
                  f"- mid_end null: {df['mid_end'].is_null().sum() + int(np.isnan(df['mid_end'].to_numpy()).sum())}", ""]
        print(lines[-5])
    el = time.time() - t0
    lines.append(f"- runtime {el:.0f}s; outputs `{ROOT}/targets_{{5m,1h,24h}}.parquet`")
    open("out/build/pm_targets.md", "w").write("\n".join(lines) + "\n")
    print(f"done in {el:.0f}s")


if __name__ == "__main__":
    main()
