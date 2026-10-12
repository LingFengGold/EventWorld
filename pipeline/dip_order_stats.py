"""D3: train-only order-template frequencies per (pop, phase_type, unit_type, location)."""
import json, collections, time
import polars as pl, pandas as pd
t0 = time.time()
tr = pl.scan_parquet("data/processed/diplomacy/transitions.parquet").filter(pl.col("split") == "train").select("pop", "phase_type", "orders_json").collect()
cnt = collections.Counter(); marg = collections.Counter()
for pop, pt, oj in zip(tr["pop"], tr["phase_type"], tr["orders_json"]):
    for p, ol in json.loads(oj).items():
        for o in ol or []:
            tok = o.split()
            if len(tok) < 2 or tok[0] not in ("A", "F"):
                continue
            typ, loc = tok[0], tok[1]
            template = "* " + " ".join(tok[2:]) if len(tok) > 2 else "* H"
            cnt[(pop, pt, typ, loc, template)] += 1
            kind = tok[2] if len(tok) > 2 else "H"
            marg[(pop, pt, typ, kind)] += 1
df = pd.DataFrame([dict(pop=k[0], phase_type=k[1], unit_type=k[2], loc=k[3], template=k[4], count=v) for k, v in cnt.items()])
df["prob"] = df["count"] / df.groupby(["pop", "phase_type", "unit_type", "loc"])["count"].transform("sum")
df.to_parquet("data/processed/diplomacy/order_freq_train.parquet", index=False)
mg = pd.DataFrame([dict(pop=k[0], phase_type=k[1], unit_type=k[2], order_kind=k[3], count=v) for k, v in marg.items()])
mg.to_parquet("data/processed/diplomacy/order_kind_marginal_train.parquet", index=False)
print(f"{len(df):,} (pop, phase, unit, loc, template) rows; {int(df['count'].sum()):,} orders; {time.time()-t0:.0f}s")
top = df[(df.pop == "7p") & (df.phase_type == "M") & (df.loc == "PAR") & (df.unit_type == "A")].sort_values("prob", ascending=False).head(5)
print(top[["template", "count", "prob"]].to_string(index=False))
