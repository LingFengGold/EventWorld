"""D5: common evaluation sets, Task S designations, rollout chains, SFT id lists, and materialized eval text."""
import json, hashlib, random, os, sys, time
import polars as pl, pandas as pd
from eventworld.dip_serialize import render_task_j, render_task_j_target, render_task_s, render_task_s_direct_target

random.seed(0)
D = "data/processed/diplomacy"
os.makedirs(f"{D}/sft_lists", exist_ok=True); os.makedirs(f"{D}/eval_text", exist_ok=True)
t0 = time.time()
cols = ["sample_id", "game_id", "pop", "split", "scale_bucket", "map", "rules_no_check", "phase_idx", "phase_name", "phase_type",
        "state_json", "orders_json", "results_json", "next_state_json", "n_powers_alive", "difficulty_tier"]
T = pl.scan_parquet(f"{D}/transitions.parquet").select(cols)

def alive_powers(state_json):
    st = json.loads(state_json)
    return [p for p, us in st["units"].items() if us]

# ---------- common evaluation sets ----------
test = T.filter(pl.col("split") == "test").collect().to_pandas()
by_game = {g: d.sort_values("phase_idx") for g, d in test.groupby("game_id")}
common = {}
for pop, n in [("7p", 3000), ("2p", 1000)]:
    d = test[test["pop"] == pop]
    pick = d.sample(n=n, random_state=0).copy()
    pick["designated"] = [json.dumps(random.sample(alive_powers(s), min(2, len(alive_powers(s))))) for s in pick["state_json"]]
    pick[["sample_id", "game_id", "phase_idx", "phase_name", "phase_type", "difficulty_tier", "designated"]].to_parquet(f"{D}/eval_common_{pop}.parquet", index=False)
    common[pop] = pick

# ---------- rollout chains (7p test, start at a movement phase, 5 consecutive transitions) ----------
chains = []
games7 = [g for g, d in by_game.items() if d["pop"].iloc[0] == "7p" and len(d) >= 8]
random.shuffle(games7)
for g in games7:
    d = by_game[g]; idx = list(d.index)
    starts = [i for i in range(len(idx) - 5) if d.loc[idx[i], "phase_type"] == "M" and d.loc[idx[i], "phase_idx"] >= 2]
    if not starts:
        continue
    i = random.choice(starts)
    chains.append(dict(chain_id=f"{g}:{d.loc[idx[i], 'phase_idx']}", game_id=g, start_sample=d.loc[idx[i], "sample_id"],
                       sample_ids=json.dumps([d.loc[idx[j], "sample_id"] for j in range(i, i + 5)])))
    if len(chains) >= 500:
        break
pd.DataFrame(chains).to_parquet(f"{D}/rollout_chains.parquet", index=False)

# ---------- SFT id lists (train, 7p and 2p): J-pool / S-pool disjoint by transition hash ----------
train = T.filter(pl.col("split") == "train").select("sample_id", "pop", "state_json", "phase_type").collect().to_pandas()
def h(s, salt=""):
    return int(hashlib.sha1((salt + s).encode()).hexdigest()[:8], 16)
train["pool"] = train["sample_id"].map(lambda s: "J" if h(s, "pool") % 2 == 0 else "S")
train["order"] = train["sample_id"].map(lambda s: h(s, "order"))
sizes = [5000, 20000, 80000, 320000]
summary = []
for pop in ["7p", "2p"]:
    for pool in ["J", "S"]:
        d = train[(train["pop"] == pop) & (train["pool"] == pool)].sort_values("order")
        if pool == "S":
            rng = random.Random(1)
            d = d.assign(designated=[rng.choice(alive_powers(s)) for s in d["state_json"]])
            ids = [f"{sid}:{pw}" for sid, pw in zip(d["sample_id"], d["designated"])]
        else:
            ids = list(d["sample_id"])
        for n in sizes + [len(ids)]:
            name = f"{D}/sft_lists/{pop}_{pool}_{'all' if n == len(ids) else n}.txt"
            if n <= len(ids):
                open(name, "w").write("\n".join(ids[:n]))
        summary.append((pop, pool, len(ids)))

# ---------- materialize eval text for the common sets ----------
def hist_rows(game_id, phase_idx, K):
    d = by_game[game_id]; d = d[d["phase_idx"] < phase_idx].tail(K)
    return [(json.loads(r.state_json), json.loads(r.orders_json), json.loads(r.results_json)) for r in d.itertuples()]

lens = {}
KS = [0, 3]  # protocol v3: history length K in {0, 3} for both tasks; J and S targets are both "NEXT <phase>" + STATE
for pop, pick in common.items():
    npow = 7 if pop == "7p" else 2
    for K in KS:
        with open(f"{D}/eval_text/common_{pop}_J_K{K}.jsonl", "w") as fj, open(f"{D}/eval_text/common_{pop}_S_K{K}.jsonl", "w") as fs:
            for r in pick.itertuples():
                st, o, res, ns = json.loads(r.state_json), json.loads(r.orders_json), json.loads(r.results_json), json.loads(r.next_state_json)
                rules = ["NO_CHECK"] if r.rules_no_check else []
                hist = hist_rows(r.game_id, r.phase_idx, K) if K else []
                prompt = render_task_j(r.map, rules, npow, st, o, history=hist); target = render_task_j_target(res, st, ns)
                fj.write(json.dumps(dict(sample_id=r.sample_id, task="J", K=K, pop=pop, phase_type=r.phase_type, tier=r.difficulty_tier, prompt=prompt, target=target)) + "\n")
                lens.setdefault(f"{pop}_J_K{K}", []).append(len(prompt) + len(target))
                for pw in json.loads(r.designated):
                    prompt = render_task_s(r.map, rules, npow, hist, st, pw, o.get(pw)); target = render_task_s_direct_target(ns)
                    fs.write(json.dumps(dict(sample_id=f"{r.sample_id}:{pw}", task="S", K=K, pop=pop, power=pw, phase_type=r.phase_type, tier=r.difficulty_tier, prompt=prompt, target=target)) + "\n")
                    lens.setdefault(f"{pop}_S_K{K}", []).append(len(prompt) + len(target))

# ---------- fixed 3-shot demonstrations (train split): one movement, one retreat, one adjustment phase, each with >= 3 prior phases ----------
DEMO_ACTOR = "FRANCE"
tr7 = T.filter((pl.col("split") == "train") & (pl.col("pop") == "7p") & (pl.col("phase_idx") >= 6)).select(cols).collect().to_pandas()
tr_games = {g: d.sort_values("phase_idx") for g, d in tr7.groupby("game_id")}
def tr_hist(game_id, phase_idx, K):
    d = tr_games[game_id]; d = d[d["phase_idx"] < phase_idx].tail(K)
    return [(json.loads(r.state_json), json.loads(r.orders_json), json.loads(r.results_json)) for r in d.itertuples()]
demo_rows = []
rng = random.Random(11)
for pt in ["M", "R", "A"]:
    cand = tr7[tr7["phase_type"] == pt].sample(frac=1, random_state=11)
    for r in cand.itertuples():
        o = json.loads(r.orders_json)
        if pt == "M" and not o.get(DEMO_ACTOR):
            continue
        if len(tr_games[r.game_id][tr_games[r.game_id]["phase_idx"] < r.phase_idx]) < 3:
            continue
        demo_rows.append(r); break
for K in KS:
    with open(f"{D}/eval_text/demos_dip_J_K{K}.jsonl", "w") as fj, open(f"{D}/eval_text/demos_dip_S_K{K}.jsonl", "w") as fs:
        for r in demo_rows:
            st, o, res, ns = json.loads(r.state_json), json.loads(r.orders_json), json.loads(r.results_json), json.loads(r.next_state_json)
            rules = ["NO_CHECK"] if r.rules_no_check else []
            hist = tr_hist(r.game_id, r.phase_idx, K) if K else []
            fj.write(json.dumps(dict(sample_id=r.sample_id, task="J", K=K, phase_type=r.phase_type, prompt=render_task_j(r.map, rules, 7, st, o, history=hist), target=render_task_j_target(res, st, ns))) + "\n")
            fs.write(json.dumps(dict(sample_id=f"{r.sample_id}:{DEMO_ACTOR}", task="S", K=K, phase_type=r.phase_type, power=DEMO_ACTOR, prompt=render_task_s(r.map, rules, 7, hist, st, DEMO_ACTOR, o.get(DEMO_ACTOR)), target=render_task_s_direct_target(ns))) + "\n")
demo_desc = ", ".join(f"{r.phase_name} ({r.phase_type})" for r in demo_rows)

lines = ["# Diplomacy samples", "", f"common eval sets: 7p {len(common['7p'])} transitions ({len(common['7p'])*2} Task S samples), 2p {len(common['2p'])}",
         f"phase mix 7p: {common['7p']['phase_type'].value_counts().to_dict()}; tiers (M only): {common['7p'][common['7p'].phase_type=='M']['difficulty_tier'].value_counts().sort_index().to_dict()}",
         f"rollout chains: {len(chains)} (5 transitions each, start at a movement phase)", f"3-shot demonstrations (train, fixed, actor {DEMO_ACTOR} for S): {demo_desc}", "", "SFT pools (train):"]
lines += [f"- {pop} pool {pool}: {n:,} samples; lists at 5k/20k/80k/320k/all" for pop, pool, n in summary]
lines += ["", "text length in characters (prompt+target), p50 / p90:"]
import statistics
for k, v in lens.items():
    q = statistics.quantiles(v, n=10)
    lines.append(f"- {k}: {q[4]:.0f} / {q[8]:.0f} (max {max(v)})")
open("out/build/dip_samples.md", "w").write("\n".join(lines)); print("\n".join(lines)); print(f"{time.time()-t0:.0f}s")
