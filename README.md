# GRASPrune

Official implementation of **GRASPrune: Global Gating for Budgeted Structured Pruning of Large Language Models**, accepted to **ACL 2026 Main Conference**.

[[Paper]](https://arxiv.org/abs/2604.19398)

GRASPrune is a post-training structured pruning toolkit for decoder-only LLMs. It learns global gates over FFN intermediate channels and attention KV groups, projects them into budget-feasible hard masks, applies lightweight rescale compensation, and exports a dense pruned checkpoint.

The code currently supports LLaMA-style and Qwen-style models, including `meta-llama/Llama-2-7b-hf` and `Qwen/Qwen3-8B`.

## Installation

Python 3.10 is recommended.

```bash
pip install -r requirements.txt
```

## Pruning

Run commands from the repository root.

### LLaMA

```bash
python train.py \
  --model-id meta-llama/Llama-2-7b-hf \
  --model-family llama \
  --device cuda:0 \
  --keep-ratio 0.6 \
  --epochs 4 \
  --num-samples 512
```

### Qwen

```bash
python train.py \
  --model-id Qwen/Qwen3-8B \
  --model-family qwen \
  --trust-remote-code \
  --device cuda:0 \
  --keep-ratio 0.8 \
  --epochs 4 \
  --num-samples 512
```

If the model is not cached locally, add:

```bash
--allow-remote-model
```

Main options:

- `--keep-ratio`: retained structural budget.
- `--epochs`: number of gate-training epochs.
- `--num-samples`: number of calibration sequences.
- `--max-len`: calibration sequence length.
- `--output-dir`: custom output directory.

## Outputs

Checkpoints are saved under `outputs/` by default. Each output directory contains:

```text
pruned_state_dict.safetensors
meta.json
layer_mask_report.csv
```

where `pruned_state_dict.safetensors` stores the exported dense pruned weights, `meta.json` stores the pruning metadata required for loading, and `layer_mask_report.csv` records layer-wise retention statistics.

## Evaluation

### Perplexity

```bash
python eval_pruned_ppl.py \
  --model-id meta-llama/Llama-2-7b-hf \
  --sd outputs/Llama-2-7b-hf_pruned_60/pruned_state_dict.safetensors \
  --meta outputs/Llama-2-7b-hf_pruned_60/meta.json \
  --device cuda:0
```

For Qwen:

```bash
python eval_pruned_ppl.py \
  --model-id Qwen/Qwen3-8B \
  --sd outputs/Qwen3-8B_pruned_80/pruned_state_dict.safetensors \
  --meta outputs/Qwen3-8B_pruned_80/meta.json \
  --trust-remote-code \
  --use-fast-tokenizer \
  --device cuda:0
```

### Accuracy

```bash
python eval_pruned_acc.py \
  --model-id meta-llama/Llama-2-7b-hf \
  --sd outputs/Llama-2-7b-hf_pruned_60/pruned_state_dict.safetensors \
  --meta outputs/Llama-2-7b-hf_pruned_60/meta.json \
  --device cuda:0
```

Accuracy evaluation uses `lm-eval`.

## C4 Evaluation

C4 evaluation is optional. To enable it, prepare:

```text
data/c4-validation.json
```

WikiText-2 and PTB are loaded through Hugging Face `datasets`.

## Citation

If you find GRASPrune useful, please cite:

```bibtex
@article{wang2026grasprune,
  title={GRASPrune: Global Gating for Budgeted Structured Pruning of Large Language Models},
  author={Wang, Ziyang and Xiao, Jiangfeng and Xiao, Chuan and Li, Ruoxiang and Mao, Rui and Qin, Jianbin},
  journal={arXiv preprint arXiv:2604.19398},
  year={2026}
}
```