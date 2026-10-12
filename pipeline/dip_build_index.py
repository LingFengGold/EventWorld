"""D1: game index for Diplomacy. Filters engine-reproduced no-press games, assigns populations and splits."""
import hashlib, json, re, sys
CUTOFFS = None
import pandas as pd

REPRO = "out/dip_deep/repro_all.csv"
GAMES = "out/dip_deep/standard_no_press.games.csv"
OUT = "data/processed/diplomacy/games_index.parquet"
REPORT = "out/build/dip_splits.md"


def bucket(game_id, salt=""):
    return int(hashlib.sha1((salt + game_id).encode()).hexdigest()[:8], 16)


def main():
    r = pd.read_csv(REPRO)
    r = r[(r.file == "standard_no_press") & (r.status == "ok")]
    g = pd.read_csv(GAMES)
    df = r.merge(g[["id", "n_phases", "n_M", "n_R", "n_A", "last_year", "n_powers", "winner", "max_centers"]], on="id", how="left")
    df["pop"] = df["map"].map(lambda m: "7p" if m == "standard" else "2p")
    df["no_check"] = df["rules"].fillna("").str.contains("NO_CHECK")
    # Split v2. DipNet (Paquette et al. 2019) trained on all games except the last 5% of game ids in sorted order, which was its
    # validation set. To make DipNet a genuinely held-out baseline, our test set is the last 4% of the sorted ids of the WHOLE dump (all files,
    # 156k games); the 4-5% tail is a buffer assigned to val so that the approximation of DipNet's cutoff (computed on its own filtered game list,
    # which we do not have) cannot leak DipNet-trained games into test. The remaining games are hash-split 95/5 into train/val.
    all_ids = sorted(pd.read_csv(REPRO, usecols=["id"])["id"].astype(str).unique())
    rank = {gid: i / len(all_ids) for i, gid in enumerate(all_ids)}
    rk = df["id"].map(rank)
    b = df["id"].map(bucket)
    hash_split = pd.Series(pd.cut(b % 100, [-1, 94, 99], labels=["train", "val"]).astype(str), index=df.index)
    df["split"] = hash_split.where(rk < 0.95, "val").where(rk < 0.96, "test")
    df["id_rank"] = rk
    global CUTOFFS; CUTOFFS = (all_ids[int(0.95 * len(all_ids))], all_ids[int(0.96 * len(all_ids))], len(all_ids))
    b2 = df["id"].map(lambda s: bucket(s, "scale")) % 1000
    df["scale_bucket"] = pd.cut(b2, [-1, 9, 99, 999], labels=["1pct", "10pct", "100pct"]).astype(str)
    df = df.rename(columns={"id": "game_id"})[["game_id", "file", "map", "pop", "rules", "no_check", "n_transitions",
                                                  "n_phases", "n_M", "n_R", "n_A", "last_year", "n_powers", "winner", "max_centers", "split", "scale_bucket", "id_rank"]]
    df.to_parquet(OUT, index=False)
    lines = ["# Diplomacy game index and splits (v2: test = last 4% of sorted game ids over the whole dump, aligned with DipNet's held-out tail)", "",
             f"games kept: {len(df)} (engine-reproduced no-press)", f"whole-dump games: {CUTOFFS[2]}; DipNet-approx validation cutoff id (95%): {CUTOFFS[0]}; our test cutoff id (96%): {CUTOFFS[1]}", ""]
    for pop in ["7p", "2p"]:
        d = df[df["pop"] == pop]
        lines.append(f"## population {pop}: {len(d)} games, {int(d.n_transitions.sum()):,} transitions")
        t = d.groupby("split").agg(games=("game_id", "size"), transitions=("n_transitions", "sum"),
                                   M=("n_M", "sum"), R=("n_R", "sum"), A=("n_A", "sum"),
                                   phases_p50=("n_phases", "median"), no_check=("no_check", "mean"),
                                   solo=("max_centers", lambda x: (x >= 18).mean())).round(3)
        lines.append(t.to_markdown()); lines.append("")
        tr = d[d.split == "train"]
        rows = []
        for name, keep in [("1pct", ["1pct"]), ("10pct", ["1pct", "10pct"]), ("100pct", ["1pct", "10pct", "100pct"])]:
            sub = tr[tr.scale_bucket.isin(keep)]
            rows.append({"subset": name, "games": len(sub), "transitions": int(sub.n_transitions.sum())})
        lines.append("nested train subsets (1pct ⊂ 10pct ⊂ 100pct):"); lines.append(pd.DataFrame(rows).to_markdown(index=False)); lines.append("")
    open(REPORT, "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
