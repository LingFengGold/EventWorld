"""Leakage check, direct probe: does a closed model know how a Polymarket market resolved? For every closed Yes/No market whose final
outcome_prices are decided (Yes price >= 0.99 or <= 0.01) that appears in the 1,333-point test_id or test_ood subsets (docs/sub1k_ids.json, docs/sub1k_ood_ids.json), the model sees only
the question, the event title and the scheduled end date (no prices, no orders) and returns {"known": bool, "p_yes": prob}. If the model's
p_yes separates YES from NO markets (AUC) on test_ood about as well as on test_id, it carries knowledge of 2026 outcomes. Questions alone carry
some prior signal (e.g. long-shot range questions resolve NO), so the reading is relative across end dates, not against AUC 0.5.
Same deployment settings as the Task J/S runs (reasoning_effort low, max_completion_tokens 2000, temperature 0); responses cached in out/api_raw.
Usage: (EW_API_BASE / EW_API_KEY set as for scripts/infer_api.py)
       python scripts/leakage_recall_probe.py run <deployment> <concurrency>    -> out/eval/api/leakage_recall/<deployment>.jsonl
       python scripts/leakage_recall_probe.py score <deployment> [<deployment> ...]"""
import sys, os, json, re, concurrent.futures as cf, numpy as np, pandas as pd, polars as pl
O = "out/eval/api/leakage_recall"
SYSTEM = "You answer questions about past events from memory. Reply with a single JSON object and nothing else."
USER = ("A Polymarket prediction market:\nQuestion: {q}\nEvent: {e}\nScheduled end: {end}\n\nFrom your own knowledge (no browsing), do you know how this market "
        "resolved? Reply with JSON only: {{\"known\": <true if you actually know the outcome, false if you would be guessing>, \"p_yes\": <your probability that it resolved Yes>}}")


def markets():
    ids = {"test_id": json.load(open("docs/sub1k_ids.json"))["polymarket_points"], "test_ood": json.load(open("docs/sub1k_ood_ids.json"))["polymarket_points_test_ood"]}
    sp = {p.split(":")[0]: s for s, v in ids.items() for p in v}
    m = pl.read_parquet("data/processed/polymarket/markets.parquet").filter(pl.col("market_id").is_in(list(sp)) & (pl.col("closed") == 1)
        & (pl.col("answer1") == "Yes") & (pl.col("answer2") == "No")).to_pandas()
    op1 = m.outcome_prices.str.extract(r"'([\d.]+)'", expand=False).astype(float); m = m[(op1 >= 0.99) | (op1 <= 0.01)].copy()
    m["split"] = m.market_id.map(sp); m["yes"] = (op1[m.index] >= 0.99).astype(int)
    return m.sort_values("market_id").reset_index(drop=True)


def run(model, conc):
    from infer_api import call, extract
    m = markets(); os.makedirs(O, exist_ok=True)

    def work(r):
        d, hit = call(model, SYSTEM, USER.format(q=r.question, e=r.event_title, end=str(r.end_date)[:10]), 2000, "low", 0.0)
        text, finish, u = extract(model, d); known, p = None, None
        mm = re.search(r"\{.*\}", text or "", re.S)
        if mm:
            try:
                j = json.loads(mm.group(0)); known = bool(j.get("known")); p = float(j.get("p_yes"))
            except Exception:
                pass
        return dict(market_id=r.market_id, split=r.split, end_date=str(r.end_date)[:10], yes=int(r.yes), known=known, p_yes=p, text=text, finish=finish, usage=u, cached=hit, error=d.get("error"))
    with cf.ThreadPoolExecutor(conc) as ex:
        recs = list(ex.map(work, [r for r in m.itertuples()]))
    with open(f"{O}/{model}.jsonl", "w") as f:
        for r in recs: f.write(json.dumps(r) + "\n")
    tin = sum(r["usage"].get("in", 0) for r in recs); tout = sum(r["usage"].get("out", 0) for r in recs)
    print(f"USAGE model={model} n={len(recs)} errors={sum(r['error'] is not None for r in recs)} parsed={sum(r['p_yes'] is not None for r in recs)} in={tin} out={tout}")


def auc(p, y):
    p, y = np.asarray(p, float), np.asarray(y, int)
    if y.min() == y.max(): return np.nan
    r = pd.Series(p).rank().values; n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def score(models):
    rows = []
    for model in models:
        d = pd.read_json(f"{O}/{model}.jsonl", lines=True); d["half"] = d.end_date.str[:4] + "Q" + ((d.end_date.str[5:7].astype(int) - 1) // 3 + 1).astype(str)  # quarter of the scheduled end date
        d = d[d.p_yes.notna()].copy(); d["p_yes"] = d.p_yes.clip(0, 1)
        rng = np.random.default_rng(0)
        for grp, g in [("split", s) for s in ("test_id", "test_ood")] + [("half", h) for h in sorted(d.half.unique())]:
            x = d[d[grp] == g]
            if len(x) < 10: continue
            bs = [auc(x.p_yes.values[i], x.yes.values[i]) for i in (rng.integers(0, len(x), len(x)) for _ in range(500))]
            kn = x[x.known == True]
            rows.append(dict(model=model, group=g, n=len(x), yes=int(x.yes.sum()), auc=auc(x.p_yes, x.yes), auc_se=np.nanstd(bs), brier=((x.p_yes - x.yes) ** 2).mean(),
                             brier_base=((x.yes.mean() - x.yes) ** 2).mean(), known=float((x.known == True).mean()),
                             acc_known=float(((kn.p_yes >= 0.5).astype(int) == kn.yes).mean()) if len(kn) else np.nan, n_known=len(kn)))
    r = pd.DataFrame(rows); r.to_csv(f"{O}/summary.csv", index=False)
    print("| Model | Group | n (YES) | AUC (± SE) | Brier | Brier of base rate | self-reported known | Acc when known (n) |\n|---|---|---|---|---|---|---|---|")
    for x in r.itertuples():
        print(f"| {x.model} | {x.group} | {x.n} ({x.yes}) | {x.auc:.3f} ± {x.auc_se:.3f} | {x.brier:.3f} | {x.brier_base:.3f} | {x.known:.3f} | "
              + ("--" if np.isnan(x.acc_known) else f"{x.acc_known:.3f} ({x.n_known})") + " |")


if __name__ == "__main__":
    if sys.argv[1] == "run": run(sys.argv[2], int(sys.argv[3]))
    else: score(sys.argv[2:])
