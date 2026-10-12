"""Shared evaluation subsets: 1,000 of the 3,000 Diplomacy seven-player test transitions (Task S: both designated powers -> 2,000 samples) and
1,333 of the Polymarket test_id points (1,000 anchors + 333 controls), drawn at random (seed 0) from the frozen evaluation sets. All models
are compared on these subsets (closed models are costly), so comparisons stay paired.
Writes diplomacy/eval_text/sub1k_7p_{J,S}_K{0,3}.jsonl and sub1k_7p_CF_{S,J}_K0.jsonl (counterfactuals of the subset transitions, Task S and
Task J form), polymarket/eval/sub1k_test_id_K{0,3}.jsonl (1 h) and sub1k_test_id_K0_h5m24h.jsonl (5 min and 24 h), and docs/sub1k_ids.json."""
import json, random
D, P = "data/processed/diplomacy/eval_text", "data/processed/polymarket/eval"
rng = random.Random(0)
tids = sorted({json.loads(l)["sample_id"] for l in open(f"{D}/common_7p_J_K0.jsonl")})
sub_t = set(rng.sample(tids, 1000))
pm = [json.loads(l) for l in open(f"{P}/eval_test_id_K0.jsonl")]
anch = sorted({r["point_id"] for r in pm if r["delta"] == "1h" and r["kind"] == "anchor"}); ctrl = sorted({r["point_id"] for r in pm if r["delta"] == "1h" and r["kind"] == "control"})
sub_p = set(rng.sample(anch, 1000)) | set(rng.sample(ctrl, 333))


def write(path, rows):
    open(path, "w").writelines(rows); print(f"{path}: {len(rows)}")


for K in (0, 3):
    write(f"{D}/sub1k_7p_J_K{K}.jsonl", [l for l in open(f"{D}/common_7p_J_K{K}.jsonl") if json.loads(l)["sample_id"] in sub_t])
    write(f"{D}/sub1k_7p_S_K{K}.jsonl", [l for l in open(f"{D}/common_7p_S_K{K}.jsonl") if json.loads(l)["sample_id"].rsplit(":", 1)[0] in sub_t])
    write(f"{P}/sub1k_test_id_K{K}.jsonl", [l for l in open(f"{P}/eval_test_id_K{K}.jsonl") if (lambda r: r["delta"] == "1h" and r["point_id"] in sub_p)(json.loads(l))])
for T, src in (("S", "common_7p_CF_S_K0"), ("J", "common_7p_CF_J")):   # cf_id = <transition id>:<POWER>:<scheme>
    write(f"{D}/sub1k_7p_CF_{T}_K0.jsonl", [l for l in open(f"{D}/{src}.jsonl") if json.loads(l)["cf_id"].rsplit(":", 2)[0] in sub_t])
write(f"{P}/sub1k_test_id_K0_h5m24h.jsonl", [l for l in open(f"{P}/eval_test_id_K0.jsonl") if (lambda r: r["delta"] in ("5m", "24h") and r["point_id"] in sub_p)(json.loads(l))])
json.dump({"diplomacy_transitions": sorted(sub_t), "polymarket_points": sorted(sub_p)}, open("docs/sub1k_ids.json", "w"))
