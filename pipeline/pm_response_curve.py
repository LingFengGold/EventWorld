"""P5: train-only empirical response curve r(rho, band, Delta) = E[d * dmid | rho, band, Delta] (isotonic in log rho),
with market-bootstrap 90% CI, plus zero-move base rates and residual quantiles for the curve-lookup baseline.

Inputs : points.parquet, splits.parquet (train anchors / controls), targets_{Delta}.parquet
Outputs: data/processed/polymarket/response_curve.parquet        (band, delta, rho grid, r, lo, hi, n)
         data/processed/polymarket/response_curve_stats.parquet  (kind, band, delta, rho_bin: p_zero, n, quantiles of signed move on non-zero windows)
         out/build/pm_response_curve.md
"""
import sys, os, time
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eventworld.pm_state import ROOT, DELTAS

GRID = np.exp(np.linspace(np.log(0.05), np.log(50), 60))
N_BOOT = 200
BANDS = ["mid", "extreme"]


def fit_curve(x, y):
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(np.log(x), y)
    return iso.predict(np.log(GRID))


def curve_lookup(curve, band, delta, rho):
    """Interpolate r at rho (array) for a (band, delta) from the curve table."""
    c = curve.filter((pl.col("band") == band) & (pl.col("delta") == delta)).sort("rho")
    return np.interp(np.log(np.clip(rho, GRID[0], GRID[-1])), np.log(c["rho"].to_numpy()), c["r"].to_numpy())


def main():
    t0 = time.time()
    pts = pl.read_parquet(f"{ROOT}/points.parquet", columns=["point_id", "market_id", "kind", "d", "band", "rho_bin", "qD"])
    sp = pl.read_parquet(f"{ROOT}/splits.parquet").filter(pl.col("split") == "train").select("point_id")
    pts = pts.join(sp, on="point_id", how="inner")
    rows, stats, lines = [], [], ["# P5 Polymarket response curve (train only)", ""]
    rng = np.random.default_rng(0)
    for delta in DELTAS:
        tg = pl.read_parquet(f"{ROOT}/targets_{delta}.parquet", columns=["point_id", "dmid", "is_zero", "rho", "own_V"])
        df = pts.join(tg, on="point_id", how="inner")
        anc = df.filter((pl.col("kind") == "anchor") & pl.col("dmid").is_not_nan() & (pl.col("rho") > 0))
        anc = anc.with_columns((pl.col("d").cast(pl.Float64) * pl.col("dmid")).alias("y"))
        for band in BANDS:
            a = anc.filter(pl.col("band") == band)
            x = a["rho"].to_numpy(); y = a["y"].to_numpy(); mk = a["market_id"].to_numpy()
            r = fit_curve(x, y)
            umk = np.unique(mk); inv = np.searchsorted(umk, mk)
            boots = np.empty((N_BOOT, len(GRID)))
            for b in range(N_BOOT):
                w = np.bincount(rng.integers(0, len(umk), len(umk)), minlength=len(umk))[inv].astype(float)
                iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(np.log(x), y, sample_weight=w)
                boots[b] = iso.predict(np.log(GRID))
            lo, hi = np.percentile(boots, [5, 95], axis=0)
            cnt = np.array([(x <= g).sum() for g in GRID])
            for g, rr, l, h, c in zip(GRID, r, lo, hi, cnt):
                rows.append({"band": band, "delta": delta, "rho": float(g), "r": float(rr), "lo": float(l), "hi": float(h), "n_le": int(c), "n": len(x)})
            lines.append(f"- {delta} {band}: n={len(x):,}; r(0.1)={np.interp(np.log(0.1), np.log(GRID), r):.4f} r(0.3)={np.interp(np.log(0.3), np.log(GRID), r):.4f} r(1)={np.interp(0, np.log(GRID), r):.4f} r(3)={np.interp(np.log(3), np.log(GRID), r):.4f} r(10)={np.interp(np.log(10), np.log(GRID), r):.4f}; monotone by construction, CI width at rho=1: {np.interp(0, np.log(GRID), hi - lo):.4f}")
        # stats: base rates and residual quantiles per (kind, band, rho_bin)
        st = df.with_columns((pl.col("d").cast(pl.Float64) * pl.col("dmid")).alias("y")).group_by("kind", "band", "rho_bin").agg(
            pl.len().alias("n"), pl.col("is_zero").mean().alias("p_zero"),
            pl.col("y").filter(~pl.col("is_zero")).quantile(0.1).alias("q10"), pl.col("y").filter(~pl.col("is_zero")).quantile(0.5).alias("q50"),
            pl.col("y").filter(~pl.col("is_zero")).quantile(0.9).alias("q90"),
            pl.col("dmid").filter(~pl.col("is_zero")).quantile(0.1).alias("dq10"), pl.col("dmid").filter(~pl.col("is_zero")).quantile(0.5).alias("dq50"),
            pl.col("dmid").filter(~pl.col("is_zero")).quantile(0.9).alias("dq90"),
            pl.col("dmid").filter(~pl.col("is_zero")).abs().mean().alias("mean_abs_nz"),
            (pl.col("y").filter(~pl.col("is_zero")) > 0).mean().alias("p_same_dir_nz")).with_columns(pl.lit(delta).alias("delta"))
        stats.append(st)
        lines.append(f"- {delta} zero-move base rate: anchors {df.filter(pl.col('kind')=='anchor')['is_zero'].mean():.3f}, controls {df.filter(pl.col('kind')=='control')['is_zero'].mean():.3f}")
    curve = pl.DataFrame(rows); curve.write_parquet(f"{ROOT}/response_curve.parquet")
    stats = pl.concat(stats).sort("delta", "kind", "band", "rho_bin"); stats.write_parquet(f"{ROOT}/response_curve_stats.parquet")
    el = time.time() - t0
    lines += ["", "stats (train):", stats.filter(pl.col("delta") == "1h").to_pandas().to_markdown(index=False, floatfmt=".4f"), "",
              f"- runtime {el:.0f}s; outputs `{ROOT}/response_curve.parquet` ({curve.height} rows), `{ROOT}/response_curve_stats.parquet` ({stats.height} rows)"]
    open("out/build/pm_response_curve.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
