"""Diplomacy PRESS pipeline: Task S with the acting power's negotiation messages (SM / SFM) vs without (S / SF), all within press games.
Mirrors the no-press pipeline (dip_build_index -> dip_extract_transitions -> dip_build_samples -> dip_build_sft); outputs under
data/processed/diplomacy_press/. Stages (run in this order, from the repo root):

  python pipeline/press_build.py index         # games_index.parquet (engine-reproduced press games with per-phase message storage, split v2 rule over
                                                           # the whole dump) + games_excluded.parquet (games whose messages sit only in the final container)
  python pipeline/press_build.py transitions   # transitions.parquet (+ msgs_json per phase, engine check), 24 worker processes
  python pipeline/press_build.py eval          # eval_text/press_test_{J,S,SM,SF,SFM}_K0.jsonl (1,000 test transitions x 2 acting powers)
  python pipeline/press_build.py sft           # sft/press_20k_{plain,msg,sf,sfmsg}{,_val}.json + dataset_info.json + token lengths + cutoff

(From `eval` on, the GPU environment is needed for the Qwen3 tokenizer that measures the MESSAGES budget; the SFT runs use cutoff_len 2560.)
Message view of power p at phase t (K = 0, current phase only): see pipeline/press_msgs.py.
"""
import argparse, hashlib, json, os, random, sys, time
import numpy as np, pandas as pd
RAW = "data/diplomacy/standard_press_with_msgs.jsonl"
FILE = "standard_press_with_msgs"
REPRO = "out/dip_deep/repro_all.csv"
GAMES = f"out/dip_deep/{FILE}.games.csv"
D = "data/processed/diplomacy_press"
WORKERS = 24
CONDS = {"plain": "S", "msg": "SM", "sf": "SF", "sfmsg": "SFM"}   # SFT condition -> task of the S-type half (the J half is shared)
N_EVAL, N_J, N_S, N_VAL = 1000, 10000, 10000, 1000
# Message-storage artifact: in ~18% of press games the dump stores the whole negotiation in the final container
# (labelled COMPLETED or with the last phase's name) and every transition container is empty. Those games are excluded (games_excluded.parquet).
# In the remaining games, the transitions after the last message-bearing container ("message tail") are excluded from eval/SFT sampling when the
# final container's messages span more than FINAL_SPAN_AMBIG_H hours, because part of that final container may then belong to the tail phases.
FINAL_SPAN_AMBIG_H = 48


def bucket(game_id, salt=""):
    return int(hashlib.sha1((salt + game_id).encode()).hexdigest()[:8], 16)


def dump(obj, name):
    os.makedirs(D, exist_ok=True)
    json.dump(obj, open(f"{D}/{name}", "w"), indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))


# ---------------------------------------------------------------- index
def _storage(line):
    """Per-game message storage in the raw dump. Transition containers are phases[:-1] (each has a next state); the final container is phases[-1]
    (COMPLETED, or the last phase of a game that has no next state)."""
    g = json.loads(line); ph = g["phases"]
    cnt = [len(p.get("messages") or []) for p in ph]
    tr_ts = [m["time_sent"] for p in ph[:-1] for m in (p.get("messages") or [])]
    fts = [m["time_sent"] for m in (ph[-1].get("messages") or [])]
    tr_max = max(tr_ts) if tr_ts else None
    sent, sent_tr = {}, {}
    for k, p in enumerate(ph):
        for m in p.get("messages") or []:
            sent[m["sender"]] = sent.get(m["sender"], 0) + 1
            if k < len(ph) - 1:
                sent_tr[m["sender"]] = sent_tr.get(m["sender"], 0) + 1
    return dict(game_id=g["id"], n_containers=len(ph), final_name=ph[-1]["name"], n_msgs_raw=sum(cnt), n_msgs_trans_containers=sum(cnt[:-1]),
                n_msgs_final_container=cnt[-1], n_trans_containers_with_msgs=sum(1 for c in cnt[:-1] if c),
                last_msg_container=max((k for k, c in enumerate(cnt[:-1]) if c), default=-1),
                final_span_h=(max(fts) - min(fts)) / 3600 if fts else 0.0,
                final_first_after_last_trans_h=(min(fts) - tr_max) / 3600 if fts and tr_max is not None else None,
                n_final_before_last_trans=sum(1 for t in fts if tr_max is not None and t < tr_max),
                sent_by_power_json=json.dumps(sent, sort_keys=True), sent_trans_by_power_json=json.dumps(sent_tr, sort_keys=True))


def stage_index():
    from multiprocessing import Pool
    r = pd.read_csv(REPRO)
    all_ids = sorted(r["id"].astype(str).unique())          # same id universe as dip_build_index (whole dump, all five files)
    rank = {gid: i / len(all_ids) for i, gid in enumerate(all_ids)}
    p = r[r.file == FILE].copy()
    st = {"press_games_in_dump": int(len(p)), "repro_status": p.status.value_counts().to_dict(), "whole_dump_games": len(all_ids)}
    p = p[p.status == "ok"]
    build_any = p.rules.fillna("").str.contains("BUILD_ANY")
    st["dropped_build_any"] = int(build_any.sum())           # eval_dip / SF scoring rebuild the engine with NO_CHECK only, so BUILD_ANY games would be mis-adjudicated
    p = p[~build_any]
    g = pd.read_csv(GAMES)
    df = p.merge(g[["id", "n_phases", "n_M", "n_R", "n_A", "last_year", "n_powers", "winner", "max_centers", "n_msgs", "n_msgs_global",
                    "n_phases_with_msgs"]], on="id", how="left")
    df["pop"] = "7p"
    df["no_check"] = df["rules"].fillna("").str.contains("NO_CHECK")
    rk = df["id"].astype(str).map(rank)
    b = df["id"].map(bucket)
    hash_split = pd.Series(pd.cut(b % 100, [-1, 94, 99], labels=["train", "val"]).astype(str), index=df.index)
    df["split"] = hash_split.where(rk < 0.95, "val").where(rk < 0.96, "test")       # split v2 rule of dip_build_index, unchanged
    df["id_rank"] = rk
    b2 = df["id"].map(lambda s: bucket(s, "scale")) % 1000
    df["scale_bucket"] = pd.cut(b2, [-1, 9, 99, 999], labels=["1pct", "10pct", "100pct"]).astype(str)
    df = df.rename(columns={"id": "game_id"})[["game_id", "file", "map", "pop", "rules", "no_check", "n_transitions", "n_phases", "n_M", "n_R", "n_A",
                                                  "last_year", "n_powers", "winner", "max_centers", "n_msgs", "n_msgs_global", "n_phases_with_msgs",
                                                  "split", "scale_bucket", "id_rank"]]
    st["engine_kept_games"] = int(len(df))
    # message storage per game, from the raw dump (all 30,423 press games)
    with open(RAW) as f, Pool(WORKERS) as pool:
        S = pd.DataFrame(list(pool.imap_unordered(_storage, f, chunksize=4)))
    S = S.sort_values("game_id").reset_index(drop=True)
    df = df.merge(S, on="game_id", how="left")
    assert df.n_msgs_raw.notna().all()
    final_only = (df.n_msgs_trans_containers == 0) & (df.n_msgs_final_container > 0)
    df["msg_storage"] = np.where(final_only, "final_only", "per_phase")
    os.makedirs(D, exist_ok=True)
    ex = df[final_only].assign(exclude_reason="msgs_final_only")
    ex.to_parquet(f"{D}/games_excluded.parquet", index=False)
    df = df[~final_only].reset_index(drop=True)
    df.to_parquet(f"{D}/games_index.parquet", index=False)
    comp = df.final_name == "COMPLETED"
    st["message_storage"] = {
        "engine_kept_games": int(final_only.size), "final_only_games_excluded": int(final_only.sum()), "final_only_share": float(final_only.mean()),
        "final_only_by_split": ex.split.value_counts().to_dict(), "final_only_final_container_COMPLETED": int((ex.final_name == "COMPLETED").sum()),
        "final_only_msgs_per_game_p50": float(ex.n_msgs_raw.median()), "per_phase_msgs_per_game_p50": float(df.n_msgs_raw.median()),
        "final_only_final_span_h_p50": float(ex.final_span_h.median()),
        "per_phase_final_span_h_quantiles": df[df.n_msgs_final_container > 0].final_span_h.quantile([.5, .75, .9, .95, .99]).round(2).to_dict(),
        "per_phase_games_with_nonempty_final": int((df.n_msgs_final_container > 0).sum()),
        "per_phase_final_msgs": int(df.n_msgs_final_container.sum()), "per_phase_final_msgs_before_last_transition_msg": int(df.n_final_before_last_trans.sum()),
        "per_phase_games_final_span_gt_ambig": int(((df.n_msgs_final_container > 0) & (df.final_span_h > FINAL_SPAN_AMBIG_H)).sum()),
        "final_span_ambig_h": FINAL_SPAN_AMBIG_H}
    # where every message of the dump ends up (the transitions stage splits the per-phase games' transition-container messages further)
    dropped = S[~S.game_id.isin(set(df.game_id) | set(ex.game_id))]
    st["message_accounting"] = {
        "dump_total": int(S.n_msgs_raw.sum()), "in_engine_dropped_games": int(dropped.n_msgs_raw.sum()), "engine_dropped_games": int(len(dropped)),
        "in_final_only_games": int(ex.n_msgs_raw.sum()),
        "in_final_only_games_COMPLETED_container": int(ex[ex.final_name == "COMPLETED"].n_msgs_final_container.sum()),
        "in_final_only_games_last_phase_container": int(ex[ex.final_name != "COMPLETED"].n_msgs_final_container.sum()),
        "per_phase_games_transition_containers": int(df.n_msgs_trans_containers.sum()),
        "per_phase_games_COMPLETED_container": int(df[comp].n_msgs_final_container.sum()),
        "per_phase_games_last_phase_container": int(df[~comp].n_msgs_final_container.sum())}
    st["kept_games"] = int(len(df))
    st["test_cutoff_id_96pct"] = all_ids[int(0.96 * len(all_ids))]
    st["by_split"] = df.groupby("split").agg(games=("game_id", "size"), transitions=("n_transitions", "sum"), msgs=("n_msgs", "sum"),
                                             msgs_p50=("n_msgs", "median"), phases_p50=("n_phases", "median"), no_check=("no_check", "mean"),
                                             rank_min=("id_rank", "min"), rank_max=("id_rank", "max")).round(4).to_dict(orient="index")
    st["n_msgs_per_game_quantiles"] = df.n_msgs.quantile([0, .1, .25, .5, .75, .9, 1]).to_dict()
    dump(st, "stats_index.json"); print(json.dumps(st, indent=1, default=str))


# ---------------------------------------------------------------- transitions
IDX = None


def process_game(line):
    from eventworld.dip_engine import step
    from dip_extract_transitions import difficulty
    g = json.loads(line)
    gid = g["id"]
    if gid not in IDX.index:
        return [], {}
    meta = IDX.loc[gid]
    rules = g.get("rules", [])
    ph = g["phases"]
    names = [p["name"] for p in ph]
    # min time_sent over all messages stored in containers after k (strict upper bound for messages of phase k)
    later_min = [None] * len(ph); run = None
    for k in range(len(ph) - 1, -1, -1):
        later_min[k] = run
        ts = [m["time_sent"] for m in ph[k].get("messages") or []]
        if ts:
            run = min(ts) if run is None else min(run, min(ts))
    label_elsewhere = {}
    for k, p in enumerate(ph):
        for m in p.get("messages") or []:
            if m["phase"] != p["name"]:
                label_elsewhere[m["phase"]] = label_elsewhere.get(m["phase"], 0) + 1
    gs = {"recipients": {}, "senders": {}, "completed_msgs": 0}
    for p in ph:
        for m in p.get("messages") or []:
            gs["recipients"][m["recipient"]] = gs["recipients"].get(m["recipient"], 0) + 1
            gs["senders"][m["sender"]] = gs["senders"].get(m["sender"], 0) + 1
        if p["name"] == "COMPLETED":
            gs["completed_msgs"] += len(p.get("messages") or [])
    rows = []
    for k in range(len(ph) - 1):
        p, nxt = ph[k], ph[k + 1]
        name = p["name"]
        if name == "COMPLETED" or name[-1] not in "MRA":
            continue
        st = p["state"]; ns = nxt["state"]
        st_min = {kk: st.get(kk) for kk in ("name", "units", "centers", "retreats", "builds", "homes")}
        ns_min = {kk: ns.get(kk) for kk in ("name", "units", "centers", "retreats", "builds")}
        try:
            eng_ns, eng_res = step(st_min | {"homes": st.get("homes")}, p["orders"], g["map"], rules)
            eng_ok = all(sorted(eng_ns["units"].get(pw, [])) == sorted(ns["units"].get(pw, [])) for pw in ns["units"]) and \
                     all(sorted(eng_ns["centers"].get(pw, [])) == sorted(ns["centers"].get(pw, [])) for pw in ns["centers"])
            res_match = None
            if name.endswith("M"):
                rec = {u: sorted(r) for u, r in (p["results"] or {}).items() if r}
                eng = {u: sorted(r) for u, r in eng_res.items() if r}
                res_match = rec == eng
        except Exception:
            eng_ok, res_match = False, None
        # messages of this phase: stored in this container, labelled with this phase, sent strictly before any message of a later container
        own = [m for m in (p.get("messages") or []) if m["phase"] == name]
        lm = later_min[k]
        keep = [m for m in own if lm is None or m["time_sent"] < lm]
        keep_sorted = [m for _, m in sorted(enumerate(keep), key=lambda x: (x[1]["time_sent"], x[0]))]
        msgs = [[m["sender"], m["recipient"], m["time_sent"], m["message"]] for m in keep_sorted]
        alive = [pw for pw, us in st["units"].items() if us]
        n_noorders = sum(1 for pw in alive if not (p["orders"] or {}).get(pw))
        row = dict(sample_id=f"{gid}:{k}", game_id=gid, pop=meta["pop"], split=meta["split"], scale_bucket=meta["scale_bucket"],
                   map=g["map"], rules_no_check=bool(meta["no_check"]), phase_idx=k, phase_name=name, phase_type=name[-1],
                   year=int(name[1:5]), season=name[0], is_primary=name.endswith("M"),
                   state_json=json.dumps(st_min, separators=(",", ":")), orders_json=json.dumps(p["orders"] or {}, separators=(",", ":")),
                   results_json=json.dumps(p["results"] or {}, separators=(",", ":")), next_state_json=json.dumps(ns_min, separators=(",", ":")),
                   engine_ok=bool(eng_ok), results_match_M=res_match,
                   n_units_total=sum(len(us) for us in st["units"].values()), n_powers_alive=len(alive), n_powers_no_orders=n_noorders,
                   msgs_json=json.dumps(msgs, ensure_ascii=False, separators=(",", ":")), n_msgs_phase=len(msgs),
                   n_msgs_container=len(p.get("messages") or []), n_msgs_dropped_label=len(p.get("messages") or []) - len(own),
                   n_msgs_dropped_late=len(own) - len(keep), n_msgs_dropped_tie=sum(1 for m in own if lm is not None and m["time_sent"] == lm),
                   n_msgs_labelled_elsewhere=label_elsewhere.get(name, 0), later_min_time=lm,
                   msgs_max_time=max((m[2] for m in msgs), default=None),
                   # after the game's last message-bearing transition container; ambiguous if the final container then spans > FINAL_SPAN_AMBIG_H
                   msg_tail=bool(k > meta["last_msg_container"]),
                   msg_tail_ambiguous=bool(k > meta["last_msg_container"] and meta["n_msgs_final_container"] > 0
                                           and meta["final_span_h"] > FINAL_SPAN_AMBIG_H))
        if name.endswith("M"):
            row.update(difficulty(st, p["orders"], p["results"], ns))
        rows.append(row)
    return rows, gs


DIFF_COLS = ["n_move", "n_support", "n_convoy_orders", "n_via", "n_hold", "n_contested", "n_units_in_conflict", "n_bounce", "n_cut", "n_dislodged",
             "n_void", "n_noconvoy", "n_changed_units", "difficulty_tier"]


def _table(batch, schema):
    import pyarrow as pa
    df = pd.DataFrame(batch)
    for c in DIFF_COLS:
        df[c] = pd.to_numeric(df.get(c), errors="coerce").astype("float64")
    for c in ("later_min_time", "msgs_max_time"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df["results_match_M"] = df["results_match_M"].astype("object")
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    return tbl if schema is None else tbl.select(schema.names).cast(schema)


def stage_transitions():
    import pyarrow.parquet as pq
    from multiprocessing import Pool
    global IDX
    IDX = pd.read_parquet(f"{D}/games_index.parquet").set_index("game_id")
    out = f"{D}/transitions.parquet"
    t0 = time.time(); writer = None; schema = None; n = 0; batch = []
    agg = {"recipients": {}, "senders": {}, "completed_msgs": 0, "games_seen": 0}
    with open(RAW) as f, Pool(WORKERS) as pool:
        for rows, gs in pool.imap_unordered(process_game, f, chunksize=4):
            if rows:
                agg["games_seen"] += 1
            for kk in ("recipients", "senders"):
                for a, c in gs.get(kk, {}).items():
                    agg[kk][a] = agg[kk].get(a, 0) + c
            agg["completed_msgs"] += gs.get("completed_msgs", 0)
            batch.extend(rows)
            if len(batch) >= 20000:
                tbl = _table(batch, schema)
                if writer is None:
                    schema = tbl.schema; writer = pq.ParquetWriter(out, schema, compression="zstd")
                writer.write_table(tbl); n += len(batch); batch = []
                print(f"  {n:,} rows  {time.time()-t0:.0f}s", flush=True)
        if batch:
            tbl = _table(batch, schema)
            if writer is None:
                schema = tbl.schema; writer = pq.ParquetWriter(out, schema, compression="zstd")
            writer.write_table(tbl); n += len(batch)
    writer.close()
    import polars as pl
    T = pl.scan_parquet(out)
    s = T.group_by("split").agg(pl.len().alias("transitions"), pl.col("game_id").n_unique().alias("games"), pl.col("engine_ok").mean(),
                                pl.col("results_match_M").mean(), pl.col("n_msgs_phase").sum().alias("msgs_kept"),
                                pl.col("n_msgs_container").sum().alias("msgs_in_containers"), pl.col("n_msgs_dropped_label").sum(),
                                pl.col("n_msgs_dropped_late").sum(), pl.col("n_msgs_dropped_tie").sum(), pl.col("n_msgs_labelled_elsewhere").sum(),
                                (pl.col("n_msgs_phase") > 0).mean().alias("share_phases_with_msgs"), pl.col("msg_tail").sum().alias("msg_tail_transitions"),
                                pl.col("msg_tail_ambiguous").sum().alias("msg_tail_ambiguous_transitions")).collect().to_pandas().set_index("split")
    by_pt = T.group_by("phase_type").agg(pl.len(), pl.col("n_msgs_phase").mean().alias("msgs_mean"), (pl.col("n_msgs_phase") > 0).mean().alias("share_with_msgs")
                                         ).collect().to_pandas().set_index("phase_type")
    st = {"rows": n, "seconds": round(time.time() - t0), "by_split": s.round(5).to_dict(orient="index"), "by_phase_type": by_pt.round(4).to_dict(orient="index"),
          "message_recipients": agg["recipients"], "message_senders": agg["senders"], "msgs_in_COMPLETED_phase_excluded": agg["completed_msgs"],
          "games_with_rows": agg["games_seen"]}
    dump(st, "stats_transitions.json"); print(json.dumps(st, indent=1, default=str)[:4000])


# ---------------------------------------------------------------- shared rendering
COLS = ["sample_id", "game_id", "pop", "split", "map", "rules_no_check", "phase_idx", "phase_name", "phase_type", "state_json", "orders_json",
        "results_json", "next_state_json", "difficulty_tier", "msgs_json", "n_msgs_phase", "msg_tail"]


def alive_powers(state_json):
    return [p for p, us in json.loads(state_json)["units"].items() if us]


def interaction(phase_type):
    """Interaction split of the paper: high-interaction = movement phases, low = retreat and adjustment phases."""
    return "high" if phase_type == "M" else "low"


class Builder:
    def __init__(self):
        from press_msgs import Renderer
        self.R = Renderer()

    def render_all(self, r, power):
        """-> dict task -> (prompt, target), info about the MESSAGES block, provenance of the kept messages."""
        from eventworld.dip_serialize import render_task_j, render_task_j_target, render_task_s, render_task_s_direct_target, render_task_s_target
        from press_msgs import insert_block, view
        st, o, res, ns = (json.loads(r[c]) for c in ("state_json", "orders_json", "results_json", "next_state_json"))
        rules = ["NO_CHECK"] if r["rules_no_check"] else []
        out = {"J": (render_task_j(r["map"], rules, 7, st, o, history=[]), render_task_j_target(res, st, ns))}
        if power is None:
            return out, None, None
        msgs = json.loads(r["msgs_json"])
        s_prompt = render_task_s(r["map"], rules, 7, [], st, power, o.get(power))
        block, info = self.R.render(msgs, power)
        sm_prompt = insert_block(s_prompt, block)
        t_s = render_task_s_direct_target(ns); t_sf = render_task_s_target(o, power, ns)
        out.update(S=(s_prompt, t_s), SM=(sm_prompt, t_s), SF=(s_prompt, t_sf), SFM=(sm_prompt, t_sf))
        v = [m for m in view(msgs, power)]
        from press_msgs import clean
        v = [m for m in v if clean(m[3])]
        prov = [[m[0], m[1], m[2]] for m in v[len(v) - info["n_kept"]:]] if info["n_kept"] else []
        return out, info, prov


def load_T(filter_expr):
    import polars as pl
    return pl.scan_parquet(f"{D}/transitions.parquet").filter(filter_expr).select(COLS).collect().to_pandas()


# ---------------------------------------------------------------- eval
def stage_eval():
    import polars as pl
    B = Builder()
    # transitions.parquet row order depends on worker completion order (imap_unordered), so sort before any seeded sampling
    # pool: test transitions of per-phase-storage games (final-only games are not in transitions.parquet), minus the ambiguous message-tail ones
    test = load_T((pl.col("split") == "test") & ~pl.col("msg_tail_ambiguous")).sort_values(["game_id", "phase_idx"]).reset_index(drop=True)
    pick = test.sample(n=N_EVAL, random_state=0).copy()                       # as dip_build_samples does for the common set
    rng = random.Random(0)
    pick["designated"] = [json.dumps(rng.sample(alive_powers(s), min(2, len(alive_powers(s))))) for s in pick["state_json"]]
    pick["interaction"] = pick["phase_type"].map(interaction)
    pick[["sample_id", "game_id", "phase_idx", "phase_name", "phase_type", "difficulty_tier", "interaction", "msg_tail", "designated"]].to_parquet(f"{D}/eval_test.parquet", index=False)
    os.makedirs(f"{D}/eval_text", exist_ok=True)
    fh = {t: open(f"{D}/eval_text/press_test_{t}_K0.jsonl", "w") for t in ("J", "S", "SM", "SF", "SFM")}
    # optional inference-time control for the message-trained checkpoints: the same SM / SFM prompts with every MESSAGES block replaced by
    # "MESSAGES\nnone" (task stays SM / SFM, so routing and the system prompt are unchanged); n_msgs keeps the original count
    from press_msgs import strip_block, insert_block
    fh_null = {t: open(f"{D}/eval_text/press_test_{t}_K0_nomsg.jsonl", "w") for t in ("SM", "SFM")}
    prov_f = open(f"{D}/eval_text/press_test_msg_provenance.jsonl", "w")
    for _, r in pick.iterrows():
        base = dict(K=0, pop="7p", phase_type=r.phase_type, tier=r.difficulty_tier, interaction=r.interaction)
        out, _, _ = B.render_all(r, None)
        fh["J"].write(json.dumps(dict(sample_id=r.sample_id, task="J", **base, prompt=out["J"][0], target=out["J"][1])) + "\n")
        for pw in json.loads(r.designated):
            out, info, prov = B.render_all(r, pw)
            sid = f"{r.sample_id}:{pw}"
            minfo = dict(n_msgs=info["n_msgs"], n_msgs_kept=info["n_kept"], msgs_truncated=info["truncated"], msg_block_tokens=info["tokens"], n_msgs_cut=info["n_cut_msgs"])
            for t in ("S", "SM", "SF", "SFM"):
                fh[t].write(json.dumps(dict(sample_id=sid, task=t, **base, power=pw, **minfo, prompt=out[t][0], target=out[t][1])) + "\n")
            for t in ("SM", "SFM"):
                fh_null[t].write(json.dumps(dict(sample_id=sid, task=t, **base, power=pw, **minfo, msgs_ablated=True,
                                                 prompt=insert_block(strip_block(out[t][0]), "MESSAGES\nnone"), target=out[t][1])) + "\n")
            prov_f.write(json.dumps(dict(sample_id=sid, split="test", phase_name=r.phase_name, msgs=prov)) + "\n")
    for f in list(fh.values()) + list(fh_null.values()) + [prov_f]:
        f.close()
    st = {"transitions": len(pick), "games": int(pick.game_id.nunique()), "phase_type": pick.phase_type.value_counts().to_dict(),
          "interaction": pick.interaction.value_counts().to_dict(), "tier_M": pick[pick.phase_type == "M"].difficulty_tier.value_counts().sort_index().to_dict(),
          "test_pool_transitions": len(test), "test_pool_games": int(test.game_id.nunique()),
          "test_transitions_excluded_msg_tail_ambiguous": int(pl.scan_parquet(f"{D}/transitions.parquet").filter((pl.col("split") == "test") & pl.col("msg_tail_ambiguous")).select(pl.len()).collect().item()),
          "eval_transitions_in_msg_tail": int(pick.msg_tail.sum())}
    dump(st, "stats_eval.json"); print(json.dumps(st, indent=1, default=str))


# ---------------------------------------------------------------- sft
def h(s, salt=""):
    return int(hashlib.sha1((salt + s).encode()).hexdigest()[:8], 16)


def stage_sft():
    import polars as pl
    from eventworld.prompts import SYS
    B = Builder()
    # id lists exactly as dip_build_samples: train transitions split into a J pool and an S pool by hash, each ordered by hash; the S pool's
    # acting power drawn with random.Random(1) in pool order. The first 10k of each pool form the 20k mixture shared by all four conditions.
    tr = pl.scan_parquet(f"{D}/transitions.parquet").filter((pl.col("split") == "train") & ~pl.col("msg_tail_ambiguous")).select("sample_id", "state_json").collect().to_pandas()
    tr = tr.sort_values("sample_id").reset_index(drop=True)          # deterministic regardless of the parquet row order
    tr["pool"] = tr["sample_id"].map(lambda s: "J" if h(s, "pool") % 2 == 0 else "S")
    tr["order"] = tr["sample_id"].map(lambda s: h(s, "order"))
    dj = tr[tr.pool == "J"].sort_values("order", kind="stable")
    ds = tr[tr.pool == "S"].sort_values("order", kind="stable").head(N_S)
    rng = random.Random(1)
    ids_j = list(dj.sample_id[:N_J])
    ids_s = [f"{sid}:{rng.choice(alive_powers(s))}" for sid, s in zip(ds.sample_id, ds.state_json)]
    os.makedirs(f"{D}/sft_lists", exist_ok=True); os.makedirs(f"{D}/sft", exist_ok=True)
    open(f"{D}/sft_lists/train_J_10k.txt", "w").write("\n".join(ids_j)); open(f"{D}/sft_lists/train_S_10k.txt", "w").write("\n".join(ids_s))
    # validation: as dip_build_sft (1,000 val transitions, random_state 0; first 500 -> J, last 500 -> S with random.Random(sid).choice(alive))
    V = pl.scan_parquet(f"{D}/transitions.parquet").filter((pl.col("split") == "val") & ~pl.col("msg_tail_ambiguous")).select("sample_id", "state_json").collect().to_pandas()
    V = V.sort_values("sample_id").reset_index(drop=True).sample(n=N_VAL, random_state=0)
    vj = list(V.sample_id[:N_VAL // 2])
    vs = [f"{sid}:{random.Random(sid).choice(alive_powers(sj))}" for sid, sj in zip(V.sample_id[N_VAL // 2:], V.state_json[N_VAL // 2:])]
    open(f"{D}/sft_lists/val_J_500.txt", "w").write("\n".join(vj)); open(f"{D}/sft_lists/val_S_500.txt", "w").write("\n".join(vs))

    def build(jids, sids, split):
        need = set(jids) | {s.rsplit(":", 1)[0] for s in sids}
        T = load_T(pl.col("sample_id").is_in(list(need))).set_index("sample_id")
        recs = {c: [] for c in CONDS}; prov = []; minfo = []
        jrecs = []
        for sid in jids:
            out, _, _ = B.render_all(T.loc[sid], None)
            jrecs.append({"system": SYS["dip_J"], "instruction": out["J"][0], "output": out["J"][1], "_id": sid, "_task": "J"})
        for s in sids:
            tid, pw = s.rsplit(":", 1)
            r = T.loc[tid]
            out, info, pv = B.render_all(r, pw)
            for c, t in CONDS.items():
                recs[c].append({"system": SYS[f"dip_{t}"], "instruction": out[t][0], "output": out[t][1], "_id": s, "_task": t})
            prov.append(dict(sample_id=s, split=split, phase_name=r.phase_name, msgs=pv))
            minfo.append(dict(sample_id=s, phase_type=r.phase_type, **info))
        data = {}
        for c in CONDS:
            d = jrecs + recs[c]
            random.Random(0).shuffle(d)          # same permutation for every condition (same length, same seed) -> same sample order
            data[c] = d
        return data, prov, minfo

    info_path = f"{D}/sft/dataset_info.json"
    info = json.load(open(info_path)) if os.path.exists(info_path) else {}
    lens = {}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    allprov = []; allminfo = {}
    for split, (jids, sids) in (("train", (ids_j, ids_s)), ("val", (vj, vs))):
        data, prov, minfo = build(jids, sids, split)
        allprov += prov; allminfo[split] = minfo
        for c, d in data.items():
            name = f"press_20k_{c}" + ("_val" if split == "val" else "")
            json.dump([{k: v for k, v in x.items() if not k.startswith("_")} for x in d], open(f"{D}/sft/{name}.json", "w"))
            info[name] = {"file_name": f"{name}.json", "columns": {"prompt": "instruction", "response": "output", "system": "system"}}
            # token lengths in the training render: chat template (system + user, generation prompt, empty think block) + output + <|im_end|>\n
            pre = [tok.apply_chat_template([{"role": "system", "content": x["system"]}, {"role": "user", "content": x["instruction"]}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False) for x in d]
            pt = [len(a) for a in tok(pre, add_special_tokens=False)["input_ids"]]
            ot = [len(a) for a in tok([x["output"] + "<|im_end|>\n" for x in d], add_special_tokens=False)["input_ids"]]
            lens[name] = pd.DataFrame(dict(task=[x["_task"] for x in d], prompt_tok=pt, target_tok=ot, total=np.array(pt) + np.array(ot)))
    json.dump(info, open(info_path, "w"), indent=1)
    with open(f"{D}/sft/press_20k_msg_provenance.jsonl", "w") as f:
        for p in allprov:
            f.write(json.dumps(p) + "\n")
    pd.DataFrame(allminfo["train"]).to_parquet(f"{D}/sft/press_20k_train_msginfo.parquet", index=False)
    pd.DataFrame(allminfo["val"]).to_parquet(f"{D}/sft/press_20k_val_msginfo.parquet", index=False)
    cands = [2048, 2560, 3072, 3584, 4096, 5120, 6144]
    trunc = {L: {n: float((v.total > L).mean()) for n, v in lens.items()} for L in cands}
    # smallest candidate that truncates nothing in any of the eight files (requirement: <= 0.5%; 2048 would meet it with 0.375% of press_20k_sfmsg,
    # i.e. 0.75% of its SFM rows, whose targets would be cut; padding is per batch, so a larger cutoff costs nothing on the other rows)
    cutoff = next((L for L in cands if all(x == 0 for x in trunc[L].values())), None) or next(L for L in cands if all(x <= 0.005 for x in trunc[L].values()))
    min_ok = next(L for L in cands if all(x <= 0.005 for x in trunc[L].values()))
    st = {"n_train_J": len(ids_j), "n_train_S": len(ids_s), "n_val_J": len(vj), "n_val_S": len(vs),
          "train_games": int(len({i.split(':')[0] for i in ids_j} | {i.split(':')[0] for i in ids_s})), "cutoff_len": cutoff,
          "smallest_cutoff_meeting_0.5pct": min_ok,
          "truncation_rate_at_cutoff": trunc[cutoff], "truncation_by_candidate": trunc,
          "token_lengths": {n: {t: {k: v[v.task == t][k].describe(percentiles=[.5, .9, .99, .995]).round(1).to_dict() for k in ("prompt_tok", "target_tok", "total")}
                                for t in sorted(v.task.unique())} for n, v in lens.items()}}
    dump(st, "stats_sft.json"); print(json.dumps({k: v for k, v in st.items() if k != "token_lengths"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["index", "transitions", "eval", "sft"]); a = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    globals()[f"stage_{a.stage}"]()
