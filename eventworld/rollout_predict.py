"""Shared predictor for the L1 rollout experiments (Diplomacy chains, Diplomacy CR, Polymarket 24-step rollouts).

Each call writes one step's prompts to a JSONL file and runs the existing inference CLI on it as a subprocess, so prompts, chat templates,
demonstrations and output parsing are exactly those of the single-step L1 evaluation:
  model "api:<deployment>"  -> scripts/infer_api.py (responses cached in out/api_raw, so a killed run resumes for free)
  model "<local dir or HF id>" -> scripts/infer_vllm.py (Polymarket SFT checkpoints need binned=True)
A step whose output file already exists with one line per input row is not recomputed (resume after a crash).
The file path must contain "diplomacy" or "polymarket": both CLIs infer the domain from the input path.
"""
import json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
# Interpreters for the two CLIs: the API client runs anywhere; vLLM needs the GPU environment (see README). Both default to this interpreter.
PY_CPU = os.environ.get("EW_PYTHON", sys.executable)
PY_GPU = os.environ.get("EW_PYTHON_GPU", sys.executable)


# vLLM GPU memory share (EW_GPU_MEM overrides it). 0.3 fits the 1.7B model on a 96 GB card; use 0.8 for 8B models on 40 GB cards.
GPU_MEM = float(os.environ.get("EW_GPU_MEM", "0.3"))


def predict(model, rows, path, binned=False, max_tokens=None, concurrency=8, gpu_mem=GPU_MEM, log=None):
    """rows: list of {sample_id, task, prompt}. Returns {sample_id: pred dict} (pred has "text" and, for Polymarket, parsed fields)."""
    assert ("diplomacy" in path) or ("polymarket" in path), path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    inp, out = path + ".in.jsonl", path + ".out.jsonl"
    if not rows:
        return {}
    body = "".join(json.dumps({"sample_id": r["sample_id"], "task": r.get("task", "J"), "prompt": r["prompt"]}) + "\n" for r in rows)
    if os.path.exists(out) and os.path.exists(inp) and open(inp).read() == body:   # identical inputs: reuse the finished step
        return {json.loads(l)["sample_id"]: json.loads(l)["pred"] for l in open(out)}
    with open(inp, "w") as f:
        f.write(body)
    if model.startswith("api:"):
        cmd = [PY_CPU, "scripts/infer_api.py", "--model", model[4:], "--input", inp, "--out", out + ".part",
               "--concurrency", str(concurrency), "--max_tokens", str(max_tokens or 8000)]
        env = dict(os.environ)
    else:
        cmd = [PY_GPU, "scripts/infer_vllm.py", "--model", model, "--input", inp, "--out", out + ".part",
               "--gpu_mem", str(gpu_mem), "--max_model_len", "16384", "--max_tokens", str(max_tokens or (60 if binned else 700)), "--cuda_graphs"]
        if binned:
            cmd.append("--pm_binned")
        env = dict(os.environ, VLLM_BATCH_INVARIANT="0")
    for attempt in range(3):   # API errors are not cached, so a rerun re-sends only the failed requests
        r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
        tail = "\n".join(l for l in (r.stdout + r.stderr).splitlines() if any(k in l for k in ("USAGE", "prompts x", "Traceback", "Error", "ERRORS")))[-1500:]
        if log:
            with open(log, "a") as f:
                f.write(f"[{os.path.basename(path)}] {tail}\n")
        if r.returncode != 0 or not os.path.exists(out + ".part"):
            raise RuntimeError(f"inference failed for {path}:\n{(r.stdout + r.stderr)[-3000:]}")
        n_err = sum(1 for l in open(out + ".part") if json.loads(l).get("error"))
        if n_err == 0:
            break
    if n_err > 0.01 * len(rows):   # more than 1% failed requests (quota, outage): stop rather than score empty answers as wrong
        raise RuntimeError(f"{n_err}/{len(rows)} API errors for {path}; last output:\n{tail}")
    os.replace(out + ".part", out)
    return {json.loads(l)["sample_id"]: json.loads(l)["pred"] for l in open(out)}
