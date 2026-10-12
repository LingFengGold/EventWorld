---
license: other
pretty_name: EventWorld
---

# EventWorld data

Data for the anonymous submission *From Single-Actor Environments to Multi-Actor Event Worlds: Can LLMs Be Event World Models?*
Download it into the root of the code repository; the code reads these paths unchanged.

```
hf download <this dataset> --repo-type dataset --local-dir <code repository>
```

| Path | Content |
|---|---|
| `data/processed/diplomacy/` | Engine-verified seven- and two-player Diplomacy transitions (`transitions.parquet`), the test decision points, evaluation prompts (`eval_text/`: Task J / S at K = 0 and 3, factored Task S, counterfactuals, history sweep, shared 1,000-transition subset), fine-tuning sets (`sft/`, LLaMA-Factory format with `dataset_info.json`) and behavior-cloning sets for the agent experiments (`sft_agent/`) |
| `data/processed/diplomacy_press/` | Press games with negotiation messages: transitions, evaluation prompts, fine-tuning sets |
| `data/processed/polymarket/` | Decision points (large orders and control windows), splits (training period, later markets), targets at 5 min / 1 h / 24 h, the response curve, evaluation prompts (`eval/`), fine-tuning sets (`sft/`), and the no-change and LightGBM predictions (`preds/`) |
| `out/eval/` | Predictions and scores of every model in the paper (`local/`: Qwen3 models, `api/`: GPT-6-Luna and GPT-6-Sol, `baselines/`) |
| `out/rollout/`, `out/agent/` | Rollouts, consistency runs, and the agent experiments (verifier, behavior cloning) |

Sources: the Diplomacy games are the webDiplomacy records released by Paquette et al. (2019, *No-Press Diplomacy: Modeling Multi-Agent
Gameplay*); the Polymarket records are the public on-chain trades of the Hugging Face dataset `SII-WANGZJ/Polymarket_data`. Their terms apply
to the derived files. Polymarket wallet addresses are replaced by per-sample pseudonyms in all prompts.
