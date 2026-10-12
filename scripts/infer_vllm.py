"""Zero-shot / few-shot inference with a local model via vLLM. Reads an eval_text JSONL (fields: sample_id or cf_id, task, prompt, target),
writes prediction JSONL in the shared contract. Thinking is disabled for Qwen3 unless --thinking is given (then the <think> block is
stripped before parsing and its length is recorded; an unterminated think block counts as an empty answer).
Usage: python scripts/infer_vllm.py --model Qwen/Qwen3-8B --input data/processed/diplomacy/eval_text/sub1k_7p_J_K0.jsonl --out out/eval/local/qwen3-8b/dip_J_K0_zs.jsonl [--n 200] [--shots 0] [--max_tokens 700]
"""
import argparse, json, os, random, sys, time
import os as _os
# Deterministic greedy decoding regardless of batch composition. Measured on 600 Task S prompts (Qwen3-1.7B SFT): default kernels
# reproduced 75.5% of outputs across two runs, VLLM_BATCH_INVARIANT=1 reproduced 100%, at ~20% lower throughput. Needed for null-counterfactual
# scoring (identical prompts must give identical predictions). Set VLLM_BATCH_INVARIANT=0 in the environment to opt out.
_os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
from vllm import LLM, SamplingParams

from eventworld.prompts import SYS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--input", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0); ap.add_argument("--samples", type=int, default=1); ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--prompt_version", default="v1", help="system prompt variant suffix in eventworld.prompts.SYS; set to sft by --pm_binned")
    ap.add_argument("--response_prefix", default="", help="text appended after the generation prompt and prepended to the recorded output, e.g. 'OTHERS\\n' for Task S (SFT checkpoints at 20k are ~50/50 on the first header token)")
    ap.add_argument("--shots_from", default=None, help="jsonl with prompt/target pairs used as demonstrations (train split)"); ap.add_argument("--shots_mode", default="auto", help="turns | outcomes | auto (turns for Diplomacy, outcomes for Polymarket)")
    ap.add_argument("--max_tokens", type=int, default=700); ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--gpu_mem", type=float, default=0.35); ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--where", action="append", default=[], help="key=value filters on the input rows, e.g. delta=1h")
    ap.add_argument("--pm_binned", action="store_true", help="Polymarket SFT checkpoints: binned output format; the distribution is read from the probabilities of the 21 bin letters")
    ap.add_argument("--cuda_graphs", action="store_true", help="enable CUDA graphs (faster decoding; default off for historical Blackwell compatibility)")
    ap.add_argument("--thinking", action="store_true", help="Qwen3 thinking mode (enable_thinking=True); raise --max_tokens accordingly")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.input)]
    for w in a.where:
        k, v = w.split("=", 1); rows = [r for r in rows if str(r.get(k)) == v]
    if a.n:
        rows = rows[:a.n]
    domain = "dip" if "diplomacy" in a.input else "pm"
    demos = []
    if a.shots and a.shots_from:
        demos = [json.loads(l) for l in open(a.shots_from)][:a.shots]  # fixed demonstrations, in file order (Diplomacy: M, R, A)
    shots_mode = a.shots_mode if a.shots_mode != "auto" else ("outcomes" if domain == "pm" else "turns")
    if a.pm_binned:
        from eventworld import pm_serialize as _ser
        a.response_prefix = "BIN"; a.prompt_version = "sft"
    llm = LLM(model=a.model, gpu_memory_utilization=a.gpu_mem, max_model_len=a.max_model_len, enforce_eager=not a.cuda_graphs, **({"max_logprobs": 30} if a.pm_binned else {}))
    tok = llm.get_tokenizer()
    def chat(prompt, task):
        k = f"{domain}_{task if task in ('J', 'S', 'SF', 'P', 'SM', 'SFM') else 'J'}"  # P: Diplomacy orders policy (agent usefulness, synthetic data)
        k = f"{k}_{a.prompt_version}" if a.prompt_version != "v1" and f"{k}_{a.prompt_version}" in SYS else k
        msgs = [{"role": "system", "content": SYS[k]}]
        if demos and shots_mode == "turns":  # demonstrations as prior user/assistant turns (deterministic targets, e.g. Diplomacy)
            for d in demos:
                msgs += [{"role": "user", "content": d["prompt"]}, {"role": "assistant", "content": d["target"]}]
        elif demos:  # "outcomes": demo inputs + realised outcomes inside the user prompt (probabilistic targets, e.g. Polymarket)
            from eventworld.pm_serialize import fewshot_outcomes_user
            prompt = fewshot_outcomes_user(prompt, demos)
        if a.pm_binned:
            prompt = _ser.sft_prompt(prompt)
        msgs.append({"role": "user", "content": prompt})
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=a.thinking)
        except TypeError:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return text + prefix_for(task)  # format-constrained decoding: the target's fixed header is teacher-forced and re-attached to the output below
    def prefix_for(task):  # binned Task SF starts with the other wallets' line; its bin letter is read further down (see below)
        return "OTHERS" if (a.pm_binned and task == "SF") else a.response_prefix
    prompts = [chat(r["prompt"], r.get("task", "J")) for r in rows]
    parse_pm = None
    if domain == "pm":
        from eventworld.pm_serialize import parse_output as parse_pm
    sp = SamplingParams(max_tokens=a.max_tokens, temperature=a.temperature if a.samples == 1 else max(a.temperature, 0.7), n=a.samples, stop=["<|im_end|>"], logprobs=(30 if a.pm_binned else None))
    letter_ids = None
    if a.pm_binned:
        letter_ids = [tok.encode(" " + c, add_special_tokens=False)[0] for c in _ser.SFT_LETTERS]
        assert all(len(tok.encode(" " + c, add_special_tokens=False)) == 1 for c in _ser.SFT_LETTERS)
    t0 = time.time(); outs = llm.generate(prompts, sp); dt = time.time() - t0
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for r, o in zip(rows, outs):
            raw = [c.text for c in o.outputs]
            think_len = None
            if a.thinking:  # keep only the answer after </think>; unterminated thinking (hit max_tokens) -> empty answer
                think_len = [len(t.split("</think>", 1)[0]) if "</think>" in t else len(t) for t in raw]
                raw = [t.split("</think>", 1)[1] if "</think>" in t else "" for t in raw]
            pre = prefix_for(r.get("task", "J"))
            texts = [(pre + t).strip() for t in raw]
            key = "cf_id" if "cf_id" in r else "sample_id"
            pred = {"text": texts[0]}
            if a.pm_binned:
                import math
                if pre == "OTHERS":  # Task SF: the bin letter follows the generated "BIN"; find that step and read its distribution
                    ids, lps, pos, acc = o.outputs[0].token_ids, o.outputs[0].logprobs or [], None, pre
                    for i, tid in enumerate(ids):
                        if acc.rstrip().endswith("BIN"):
                            pos = i; break
                        acc += tok.decode([tid])
                    lp = lps[pos] if (pos is not None and pos < len(lps)) else {}
                    probs = [math.exp(lp[t].logprob) if t in lp else 0.0 for t in letter_ids]
                    k = texts[0].find("BIN")
                    parsed = _ser.sft_decode(texts[0][k + 3:], probs if sum(probs) > 0 else None) if k >= 0 else None
                    oth = _ser.sft_decode_others(texts[0])
                    if parsed and oth:
                        parsed["others"] = oth
                else:
                    lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
                    probs = [math.exp(lp[t].logprob) if t in lp else 0.0 for t in letter_ids]
                    parsed = _ser.sft_decode(texts[0][len("BIN"):], probs)
                if parsed:
                    pred.update(parsed)
            elif parse_pm is not None:
                parsed = parse_pm(texts[0], r.get("task", "J"))
                if parsed:
                    pred.update(parsed)
            rec = {key: r[key], "task": r["task"], "variant": f"{os.path.basename(a.model)}|shots{a.shots}|{a.prompt_version}|think{int(a.thinking)}", "pred": pred, "samples": [{"text": t} for t in texts], "logprob": None, "think_chars": think_len, "finish": [c.finish_reason for c in o.outputs]}
            f.write(json.dumps(rec) + "\n")
    ntok = sum(len(c.token_ids) for o in outs for c in o.outputs)
    print(f"{len(rows)} prompts x {a.samples} samples in {dt:.0f}s ({ntok/dt:.0f} output tok/s) -> {a.out}")

if __name__ == "__main__":
    main()
