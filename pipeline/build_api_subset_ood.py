"""Leakage check: fixed 1,333-point subset of the Polymarket later-markets split test_ood (events first traded >= 2026-01-01),
1,000 anchors + 333 controls at Delta = 1h, drawn at random (seed 0) from the frozen eval_test_ood_K0.jsonl, i.e. the same construction as the
test_id sub1k subset of pipeline/build_api_subset.py (that script's rng had first drawn the Diplomacy transitions; here a fresh Random(0) is used,
since the pool is different anyway). Rows (J and S, 1h) are copied verbatim, so every fine-tuned model's full test_ood run contains them.
Writes data/processed/polymarket/eval/sub1k_test_ood_K0.jsonl and docs/sub1k_ood_ids.json."""
import json, random
P = "data/processed/polymarket/eval"
rng = random.Random(0)
pm = [json.loads(l) for l in open(f"{P}/eval_test_ood_K0.jsonl")]
anch = sorted({r["point_id"] for r in pm if r["delta"] == "1h" and r["kind"] == "anchor"})
ctrl = sorted({r["point_id"] for r in pm if r["delta"] == "1h" and r["kind"] == "control"})
sub_p = set(rng.sample(anch, 1000)) | set(rng.sample(ctrl, 333))
rows = [l for l in open(f"{P}/eval_test_ood_K0.jsonl") if (lambda r: r["delta"] == "1h" and r["point_id"] in sub_p)(json.loads(l))]
open(f"{P}/sub1k_test_ood_K0.jsonl", "w").writelines(rows); print(f"sub1k_test_ood_K0: {len(rows)} (1h, J+S) from {len(anch)} anchors / {len(ctrl)} controls")
json.dump({"polymarket_points_test_ood": sorted(sub_p)}, open("docs/sub1k_ood_ids.json", "w"))
