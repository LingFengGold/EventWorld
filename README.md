# EventWorld

Code for the submission *From Single-Actor Environments to Multi-Actor Event Worlds: Can LLMs Be Event World Models?*
It builds the two event-world benchmarks (Diplomacy and Polymarket), fine-tunes and prompts LLMs as event world models, and runs every
experiment of the paper.

```
eventworld/   library: Diplomacy engine wrapper and serializer, Polymarket state and serializer, prompts, metrics, DipNet client
pipeline/     builds the benchmarks from the raw records
scripts/      inference (vLLM, OpenAI-compatible API), scoring, baselines, rollouts, agent experiments
configs/      LLaMA-Factory fine-tuning recipes
run/          one script per group of experiments
docs/         ids of the shared evaluation subsets used for the prompted models
tools/        assembles the released data folder
```

## Setup

Python 3.11. Two environments keep the pinned versions apart (one environment also works if the versions agree):

```
pip install -r requirements.txt && pip install -e .     # CPU: data, scoring, baselines, API calls       -> PY
pip install -r requirements-gpu.txt && pip install -e . # GPU: vLLM inference, LLaMA-Factory fine-tuning -> PY_GPU
```

The run scripts use `$PY` and `$PY_GPU` (default `python`). Prompted models (GPT-6-Luna, GPT-6-Sol) are called through an OpenAI-compatible
endpoint: set `EW_API_BASE`, `EW_API_KEY`, and the deployment names `LUNA` and `SOL`. DipNet (Diplomacy baselines, opponents and agents):
see `third_party/dipnet/README.md`.

## Data

The processed benchmarks and all model outputs are on Hugging Face (anonymous link in the submission). Download them into this directory:

```
hf download <dataset> --repo-type dataset --local-dir .     # creates data/processed/ and out/
```

To rebuild the benchmarks from the raw records instead, place the webDiplomacy games of Paquette et al. (2019) in `data/diplomacy/`
(`standard_no_press.jsonl`, `standard_press_with_msgs.jsonl`) and `trades.parquet` / `markets.parquet` of the Hugging Face dataset
`SII-WANGZJ/Polymarket_data` in `data/polymarket/`, then run `bash run/build_data.sh`.

## Run

```
bash run/train.sh             # fine-tune all world models (LLaMA-Factory; 2 GPUs, 4 for Qwen3-8B)
bash run/q1_q2.sh             # Sections 6.1-6.2 and their appendix experiments
bash run/q3.sh                # Section 6.3: rollouts, actions never taken
bash run/q4.sh                # Section 6.4: agent experiments
```

Each script also takes stage names, e.g. `bash run/q1_q2.sh baselines prompted`; finished models and prediction files are skipped.

| Paper | Stage |
|---|---|
| Table 2, Figure 3, single-step results, seeds, robustness, from-scratch control (App.) | `run/q1_q2.sh baselines prompted finetuned` |
| Figure 4 (horizons, idle windows) | `run/q1_q2.sh horizons` |
| Figure 5, history sweep (App.) | `run/q1_q2.sh history` |
| Factored Task S (App.) | `run/q1_q2.sh factored` |
| Negotiation messages, outcome knowledge (App.) | `run/q1_q2.sh press leakage` |
| Table 3, Figure 6, consistency with an RL agent (App.) | `run/q3.sh rollouts` |
| Table 4, counterfactuals and interventions (App.) | `run/q3.sh counterfactuals interventions` |
| Figure 7, Table 5 | `run/q4.sh` |

Predictions go to `out/eval/{local,api,baselines}/<run>/<set>_<task>_K<history>_<zs|3s>.jsonl`. Diplomacy predictions are scored into a
directory of the same name (`metrics.json`, `per_sample.parquet`); Polymarket predictions are scored with
`python scripts/eval_pm.py --pred <predictions.jsonl> --name <run>`. All inference is greedy; responses of the API models are cached under
`out/api_raw/`.
