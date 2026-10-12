"""Prompted-model inference through an OpenAI-compatible chat-completions endpoint. Same input/output contract as infer_vllm.py.

Usage:
  export EW_API_BASE=<endpoint, e.g. https://<resource>.openai.azure.com/openai/v1>  EW_API_KEY=<key>
  python scripts/infer_api.py --model gpt-6-sol --input data/processed/diplomacy/eval_text/sub1k_7p_J_K0.jsonl --out out/eval/api/gpt-6-sol_sub1k/dip_J_K0_zs.jsonl
--model is the deployment (model) name. Every raw response is cached under out/api_raw/<model>/<sha1 of request>.json, so re-running never
re-sends a request. Token usage is summed and printed as one USAGE line.
"""
import sys, os, json, argparse, hashlib, time, threading, concurrent.futures as cf
import requests
from eventworld.prompts import SYS

KEY = os.environ.get("EW_API_KEY", "")
BASE = os.environ.get("EW_API_BASE", "https://api.openai.com/v1").rstrip("/")
RAW = "out/api_raw"
_lock = threading.Lock()


def sys_prompt(domain, task, version):
    k = f"{domain}_{task if task in ('J', 'S', 'SF', 'P', 'SM', 'SFM') else 'J'}"
    k2 = f"{k}_{version}"
    return SYS[k2] if version != "v1" and k2 in SYS else SYS[k]


def call(model, system, user, max_tokens, reasoning, temperature, demos=(), shots_mode="turns"):
    """demos: fixed few-shot records. shots_mode "turns": prior user/assistant turns (deterministic targets); "outcomes": demo inputs and realised
    outcomes composed into the user prompt (probabilistic targets)."""
    turns = []
    if demos and shots_mode == "turns":
        for d in demos:
            turns += [{"role": "user", "content": d["prompt"]}, {"role": "assistant", "content": d["target"]}]
    elif demos:
        from eventworld.pm_serialize import fewshot_outcomes_user
        user = fewshot_outcomes_user(user, demos)
    turns.append({"role": "user", "content": user})
    body = {"model": model, "messages": [{"role": "system", "content": system}] + turns, "max_completion_tokens": max_tokens}
    if reasoning:
        body["reasoning_effort"] = reasoning
    url, headers = BASE + "/chat/completions", {"api-key": KEY, "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
    h = hashlib.sha1(json.dumps(body, sort_keys=True).encode()).hexdigest()
    path = f"{RAW}/{model}/{h}.json"
    if os.path.exists(path):
        return json.load(open(path)), True
    delay = 2.0
    for attempt in range(8):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=600)
            if r.status_code == 200:
                d = r.json()
                os.makedirs(os.path.dirname(path), exist_ok=True)
                json.dump(d, open(path, "w"))
                return d, False
            if r.status_code in (429, 500, 502, 503, 504, 408):
                ra = r.headers.get("retry-after")
                time.sleep(float(ra) if ra else delay); delay = min(delay * 2, 60); continue
            return {"error": {"status": r.status_code, "body": r.text[:2000]}}, False
        except requests.RequestException as e:
            time.sleep(delay); delay = min(delay * 2, 60); last = str(e)
    return {"error": {"status": "exhausted", "body": last if 'last' in dir() else ""}}, False


def extract(model, d):
    """-> (text, finish, usage dict with in/out/reasoning/cached)"""
    if "error" in d:
        return "", "error", {}
    c = d["choices"][0]; u = d.get("usage", {})
    return c["message"].get("content") or "", c.get("finish_reason"), {"in": u.get("prompt_tokens", 0), "out": u.get("completion_tokens", 0), "reasoning": (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0), "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--input", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0); ap.add_argument("--where", action="append", default=[])
    ap.add_argument("--shots", type=int, default=0); ap.add_argument("--shots_from", default=None, help="demos jsonl (prompt/target); the first --shots rows are used for every sample"); ap.add_argument("--shots_mode", default="auto", help="turns | outcomes | auto (turns for Diplomacy, outcomes for Polymarket)")
    ap.add_argument("--max_tokens", type=int, default=2500); ap.add_argument("--reasoning", default="auto", help="reasoning_effort: auto (low for gpt-6, none otherwise) | none | low | medium | high")
    ap.add_argument("--temperature", type=float, default=0.0); ap.add_argument("--prompt_version", default="v1"); ap.add_argument("--concurrency", type=int, default=8)
    a = ap.parse_args()
    domain = "pm" if "polymarket" in a.input else "dip"
    rows = [json.loads(l) for l in open(a.input)]
    for w in a.where:
        k, v = w.split("=", 1); rows = [r for r in rows if str(r.get(k)) == v]
    if a.n:
        rows = rows[:a.n]
    key = "cf_id" if rows and "cf_id" in rows[0] else "sample_id"
    if a.reasoning == "auto":   # gpt-6 deployments reject "none"; "low" is their cheapest setting
        a.reasoning = "low" if a.model.startswith("gpt-6") else "none"
    reasoning = a.reasoning
    demos = []
    if a.shots:
        demos = [json.loads(l) for l in open(a.shots_from)][:a.shots]
    shots_mode = a.shots_mode if a.shots_mode != "auto" else ("outcomes" if domain == "pm" else "turns")
    parse_pm = None
    if domain == "pm":
        from eventworld.pm_serialize import parse_output as parse_pm
    tot = {"in": 0, "out": 0, "reasoning": 0, "cached": 0, "calls": 0, "cached_calls": 0, "errors": 0}
    t0 = time.time()

    def work(r):
        task = r.get("task", "J")
        d, hit = call(a.model, sys_prompt(domain, task, a.prompt_version), r["prompt"], a.max_tokens, reasoning, a.temperature, demos, shots_mode)
        text, finish, u = extract(a.model, d)
        pred = {"text": text}
        if parse_pm and text:
            try:
                pred.update(parse_pm(text, task) or {})
            except Exception:
                pass
        rec = {key: r[key], "task": task, "variant": f"{a.model}|shots{a.shots}|{a.prompt_version}|r={reasoning}", "pred": pred, "samples": [{"text": text}], "logprob": None,
               "finish": finish, "usage": u, "error": d.get("error")}
        with _lock:
            for k in ("in", "out", "reasoning", "cached"): tot[k] += u.get(k, 0)
            tot["calls"] += 1; tot["cached_calls"] += int(hit); tot["errors"] += int("error" in d)
        return rec

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with cf.ThreadPoolExecutor(a.concurrency) as ex:
        recs = list(ex.map(work, rows))
    with open(a.out, "w") as f:
        for rec in recs:
            f.write(json.dumps(rec) + "\n")
    n = max(1, len(recs))
    print(f"USAGE model={a.model} input={os.path.basename(a.input)} n={len(recs)} errors={tot['errors']} cached_calls={tot['cached_calls']} "
          f"in={tot['in']} out={tot['out']} reasoning={tot['reasoning']} per_call_in={tot['in']/n:.0f} per_call_out={tot['out']/n:.0f} per_call_reasoning={tot['reasoning']/n:.0f} "
          f"finish={json.dumps(dict(__import__('collections').Counter(r['finish'] for r in recs)))} secs={time.time()-t0:.0f}")
    if tot["errors"]:
        errs = [r["error"] for r in recs if r["error"]][:3]; print("ERRORS", json.dumps(errs)[:800])


if __name__ == "__main__":
    main()
