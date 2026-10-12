"""Polymarket scorer metrics (used by scripts/eval_pm.py).

Prediction contract per sample: {"sample_id", "task", "variant", "pred": {p_buy, p_sell, p_zero, q10, q50, q90, others?}}; q's are Delta-mid quantiles.
Targets come from eval_index.parquet (y_* columns) / interventions.parquet.
"""
import numpy as np
import polars as pl
from scipy.stats import spearmanr

BIN_EDGES = np.array([0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.25])
X_GRID = np.linspace(-1, 1, 4001)
EPS = 1e-6


def bin_index(x):
    x = np.asarray(x, float)
    mag = np.searchsorted(BIN_EDGES, np.abs(x), side="left") + 1
    return np.where(x == 0, 10, np.where(x > 0, 10 + mag, 10 - mag))


def bin_center(b):
    """Representative value of bin b (0..20); 10 -> 0; outer bins -> 0.35 * sign."""
    e = np.concatenate([[0.0], BIN_EDGES, [0.45]])
    mag = np.abs(b - 10)
    c = np.where(mag == 0, 0.0, (e[np.clip(mag - 1, 0, 9)] + e[np.clip(mag, 0, 10)]) / 2)
    return np.sign(b - 10) * c


def quantile_cdf(q10, q50, q90, x=X_GRID):
    """Piecewise-linear CDF through (q10,.1),(q50,.5),(q90,.9), extended linearly to (q10 - 2(q50-q10), 0) and (q90 + 2(q90-q50), 1)."""
    lo = q10 - 2 * np.maximum(q50 - q10, 1e-4); hi = q90 + 2 * np.maximum(q90 - q50, 1e-4)
    xs = np.stack([lo, q10, q50, q90, hi], -1); ps = np.array([0, .1, .5, .9, 1.0])
    xs = np.maximum.accumulate(xs + np.arange(5) * 1e-9, axis=-1)
    F = np.empty((len(q10), len(x)))
    for i in range(len(q10)):
        F[i] = np.interp(x, xs[i], ps, left=0.0, right=1.0)
    return F


def crps_from_quantiles(q10, q50, q90, y):
    """CRPS of the quantile-implied CDF vs observation y, integrated on the [-1,1] grid."""
    F = quantile_cdf(np.asarray(q10, float), np.asarray(q50, float), np.asarray(q90, float))
    H = (X_GRID[None, :] >= np.asarray(y, float)[:, None]).astype(float)
    return np.trapezoid((F - H) ** 2, X_GRID, axis=1)


def brier(p, y):
    p = np.clip(np.asarray(p, float), 0, 1); y = np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def logloss(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS); y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def r2(pred, y):
    y = np.asarray(y, float); pred = np.asarray(pred, float)
    return float(1 - np.sum((y - pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12))


def clip_physical(d):
    """Project predictions onto their physical ranges before scoring: prices and probabilities to [0, 1], Delta-mid quantiles to [-1, 1], quantiles sorted.
    In-range predictions (all floor baselines) are unchanged; this only bounds the penalty for models that emit nonsense values."""
    if d.height == 0:
        return d
    q = np.sort(np.clip(np.stack([d["q10"].to_numpy(), d["q50"].to_numpy(), d["q90"].to_numpy()], 1).astype(float), -1, 1), axis=1)
    return d.with_columns([pl.Series("q10", q[:, 0]), pl.Series("q50", q[:, 1]), pl.Series("q90", q[:, 2])] +
                          [pl.col(c).cast(pl.Float64).clip(0, 1) for c in ("p_buy", "p_sell", "p_zero")])


def score_block(df):
    """df: joined predictions+targets with columns p_zero, q10, q50, q90, p_buy, p_sell, y_dmid, y_is_zero, y_mid_t, kind, (others_N/F/V, y_N_o...).
    Returns dict of metrics for one (task, delta, split) block."""
    out = {"n": df.height, "parsed": float(df["parsed"].mean())}
    d = df.filter(pl.col("parsed"))  # eval_pm.load_preds already applies clip_physical-equivalent bounds
    if d.height == 0:
        return out
    yz = d["y_is_zero"].to_numpy().astype(float); pz = d["p_zero"].to_numpy()
    out["p_zero_brier"] = brier(pz, yz); out["p_zero_logloss"] = logloss(pz, yz); out["p_zero_base_rate"] = float(yz.mean()); out["p_zero_mean_pred"] = float(pz.mean())
    out["is_zero_acc"] = float(np.mean((pz >= 0.5) == (yz == 1)))
    nz = d.filter(~pl.col("y_is_zero"))
    out["n_nonzero"] = nz.height
    if nz.height:
        y = nz["y_dmid"].to_numpy(); q50 = nz["q50"].to_numpy(); q10 = nz["q10"].to_numpy(); q90 = nz["q90"].to_numpy()
        out["mae"] = float(np.mean(np.abs(q50 - y))); out["mae_nochange"] = float(np.mean(np.abs(y))); out["skill_vs_nochange"] = 1 - out["mae"] / max(out["mae_nochange"], 1e-12)
        out["mae_bin"] = float(np.mean(np.abs(bin_center(bin_index(q50)) - bin_center(bin_index(y)))))
        out["bin_acc"] = float(np.mean(bin_index(q50) == bin_index(y)))
        out["crps"] = float(np.mean(crps_from_quantiles(q10, q50, q90, y)))
        pm = (nz["p_buy"].to_numpy() + nz["p_sell"].to_numpy()) / 2 - nz["y_mid_t"].to_numpy()
        pred_dir = np.where(q50 != 0, np.sign(q50), np.sign(pm))
        out["direction_acc"] = float(np.mean(pred_dir == np.sign(y))); out["direction_coverage"] = float(np.mean(pred_dir != 0))
        out["direction_acc_covered"] = float(np.mean(pred_dir[pred_dir != 0] == np.sign(y)[pred_dir != 0])) if (pred_dir != 0).any() else None
        out["coverage_80"] = float(np.mean((y >= q10) & (y <= q90))); out["width_80"] = float(np.mean(q90 - q10))
        out["r2_q50"] = r2(q50, y)
    # all-window R^2 / MAE of the point prediction E[dmid] = (1 - p_zero) * q50
    y_all = d["y_dmid"].to_numpy(); e_all = (1 - d["p_zero"].to_numpy()) * d["q50"].to_numpy()
    out["r2_all"] = r2(e_all, y_all); out["mae_all"] = float(np.mean(np.abs(e_all - y_all))); out["mae_all_nochange"] = float(np.mean(np.abs(y_all)))
    # price-quantile self-consistency: sign of implied mid change from (p_buy, p_sell) agrees with sign(q50) when both non-zero
    pm_all = (d["p_buy"].to_numpy() + d["p_sell"].to_numpy()) / 2 - d["y_mid_t"].to_numpy(); q = d["q50"].to_numpy(); both = (pm_all != 0) & (q != 0)
    out["price_quantile_consistency"] = float(np.mean(np.sign(pm_all[both]) == np.sign(q[both]))) if both.any() else None
    # controls: p_zero vs base rate and spurious response
    c = d.filter(pl.col("kind") == "control")
    if c.height:
        out["control_n"] = c.height; out["control_p_zero_pred"] = float(c["p_zero"].mean()); out["control_p_zero_true"] = float(c["y_is_zero"].mean())
        out["control_p_zero_brier"] = brier(c["p_zero"].to_numpy(), c["y_is_zero"].to_numpy().astype(float)); out["control_mean_abs_q50"] = float(c["q50"].abs().mean())
    # actor PnL sign (anchors): predicted pnl = own_dpos * (mid_t + q50) - own_F vs true pnl_mtm
    a = d.filter((pl.col("kind") == "anchor") & pl.col("y_pnl_mtm").is_not_null() & (pl.col("y_pnl_mtm") != 0))
    if a.height:
        pp = a["y_own_dpos"].to_numpy() * (a["y_mid_t"].to_numpy() + a["q50"].to_numpy()) - a["y_own_F"].to_numpy()
        out["pnl_sign_acc"] = float(np.mean(np.sign(pp) == np.sign(a["y_pnl_mtm"].to_numpy())))
    # Task S others
    if "others_N" in d.columns and d["others_N"].null_count() < d.height:
        o = d.filter(pl.col("others_N").is_not_null())
        for k, yk in [("N", "y_N_o"), ("F", "y_F_o"), ("V", "y_V_o")]:
            p = o[f"others_{k}"].to_numpy(); y = o[yk].to_numpy()
            out[f"others_{k}_mae"] = float(np.mean(np.abs(p - y))); out[f"others_{k}_mae_mean_baseline"] = float(np.mean(np.abs(y - y.mean())))
        out["others_F_sign_acc"] = float(np.mean(np.sign(o["others_F"].to_numpy()) == np.sign(o["y_F_o"].to_numpy())))
    return out


def horizon_monotonicity(df):
    """Per point: p_zero non-increasing and width (q90-q10) non-decreasing over 5m -> 1h -> 24h. Returns pass rates."""
    piv = df.filter(pl.col("parsed")).pivot(on="delta", index="point_id", values=["p_zero", "width"], aggregate_function="first").drop_nulls()
    if piv.height == 0 or any(f"p_zero_{d}" not in piv.columns for d in ["5m", "1h", "24h"]):  # needs all three horizons; 1h-only runs skip it
        return {}
    pz = np.stack([piv[f"p_zero_{d}"].to_numpy() for d in ["5m", "1h", "24h"]], 1); wd = np.stack([piv[f"width_{d}"].to_numpy() for d in ["5m", "1h", "24h"]], 1)
    return {"n_points": piv.height, "p_zero_monotone_rate": float(np.mean(np.all(np.diff(pz, axis=1) <= 1e-9, axis=1))),
            "width_monotone_rate": float(np.mean(np.all(np.diff(wd, axis=1) >= -1e-9, axis=1))),
            "p_zero_strict_rate": float(np.mean(np.all(np.diff(pz, axis=1) < 0, axis=1)))}


def intervention_metrics(df, thr=0.0005):
    """df: predictions on interventions joined with interventions.parquet (variant, ref_response, r0, rho, d, q50, parsed).
    response(variant) = q50(variant) - q50(factual). Sign agreement / magnitude ratio vs ref on remove/reverse/scale, Spearman(rho, d*(q50_factual - q50_remove)),
    null spurious rate = P(|q50(null) - q50(factual)| >= thr)."""
    d = df.filter(pl.col("parsed"))
    fac = d.filter(pl.col("variant") == "factual").select("point_id", pl.col("q50").alias("q50_f"))
    j = d.filter(pl.col("variant") != "factual").join(fac, on="point_id", how="inner").with_columns((pl.col("q50") - pl.col("q50_f")).alias("resp"))
    out = {"n_points": fac.height}
    for v in ["remove", "reverse", "scale"]:
        s = j.filter((pl.col("variant") == v) & (pl.col("ref_response") != 0))
        if s.height == 0:
            continue
        r = s["resp"].to_numpy(); ref = s["ref_response"].to_numpy()
        out[f"{v}_n"] = s.height; out[f"{v}_sign_agreement"] = float(np.mean(np.sign(r) == np.sign(ref))); out[f"{v}_insensitive_rate"] = float(np.mean(np.abs(r) < thr))
        out[f"{v}_magnitude_ratio"] = float(np.mean(np.abs(r)) / max(np.mean(np.abs(ref)), 1e-12)); out[f"{v}_mean_resp"] = float(np.mean(r)); out[f"{v}_mean_ref"] = float(np.mean(ref))
    rm = j.filter(pl.col("variant") == "remove")
    if rm.height > 2:
        eff = rm["d"].to_numpy() * (rm["q50_f"].to_numpy() - rm["q50"].to_numpy())  # model's own action effect
        rho = rm["rho"].to_numpy()
        out["monotonicity_spearman"] = float(spearmanr(rho, eff).statistic) if np.std(eff) > 0 else 0.0
        bins = np.digitize(rho, [0.2, 0.5, 1.0])
        out["effect_by_rho_bin"] = [float(eff[bins == b].mean()) if (bins == b).any() else None for b in range(4)]
        out["ref_effect_by_rho_bin"] = [float((rm["d"].to_numpy() * -rm["ref_response"].to_numpy())[bins == b].mean()) if (bins == b).any() else None for b in range(4)]
    nl = j.filter(pl.col("variant") == "null")
    if nl.height:
        out["null_n"] = nl.height; out["null_spurious_rate"] = float(np.mean(np.abs(nl["resp"].to_numpy()) >= thr)); out["null_mean_abs_resp"] = float(nl["resp"].abs().mean())
    return out


def rollout_metrics(pred_steps, truth):
    """pred_steps: DataFrame chain_id, step, q50, q10, q90 (per-step predicted dmid, model fed its own state);
    truth: chain_id, step, y_mid_t (true mid at step start), y_mid_end. MAE of the accumulated path vs true mid per step + 80% coverage of the per-step dmid."""
    j = pred_steps.join(truth, on=["chain_id", "step"]).sort("chain_id", "step")
    j = j.with_columns(pl.col("q50").cum_sum().over("chain_id").alias("path"), (pl.col("y_mid_end") - pl.col("y_mid_t").first().over("chain_id")).alias("true_path"))
    per = j.group_by("step").agg((pl.col("path") - pl.col("true_path")).abs().mean().alias("mae_path"), pl.col("true_path").abs().mean().alias("mae_path_nochange"),
                                 ((pl.col("y_mid_end") - pl.col("y_mid_t") >= pl.col("q10")) & (pl.col("y_mid_end") - pl.col("y_mid_t") <= pl.col("q90"))).mean().alias("coverage_80")).sort("step")
    return per
