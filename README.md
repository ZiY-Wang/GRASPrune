# GRASPrune

GRASPrune is a structured pruning toolkit for decoder-only language models. It learns global gates over FFN intermediate channels and attention KV groups, hardens the learned gates into deterministic masks, applies a lightweight rescale compensation stage, and exports a loadable pruned checkpoint.

The current implementation supports:

- LLaMA-style models, such as `meta-llama/Llama-2-7b-hf`
- Qwen-style models, such as `Qwen/Qwen3-8B`
- Perplexity evaluation on WikiText-2, PTB, and optional C4
- Accuracy evaluation through `lm-eval`

## Installation

Python 3.10 is recommended.

```bash
pip install -r requirements.txt
```

If you use `uv`:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Train and Prune

Run commands from the `GRASPrune` directory.

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

If the model is not already cached locally, add:

```bash
--allow-remote-model
```

## Outputs

By default, checkpoints are saved under `outputs/`:

```bash
outputs/Llama-2-7b-hf_pruned_60/
outputs/Qwen3-8B_pruned_80/
```

Each output directory contains:

- `pruned_state_dict.safetensors`
- `meta.json`
- `layer_mask_report.csv`

The exported checkpoint can be reloaded with the provided evaluation scripts.

## Evaluate a Pruned Checkpoint

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

## Common Options

- `--keep-ratio`: target retained structural budget.
- `--epochs`: number of gate-training epochs.
- `--num-samples`: number of training blocks used for gate optimization.
- `--max-len`: sequence length for pruning data.
- `--run-ppl-eval`: run perplexity evaluation after pruning.
- `--skip-acc-eval`: skip accuracy evaluation after pruning.
- `--output-dir`: custom output directory.
- `--report-csv`: custom mask report path.

## C4 Evaluation Data

C4 support is optional. To evaluate on C4, prepare a local file:

```bash
data/c4-validation.json
```

Commands that include `c4` will fail if this file is missing. WikiText-2 and PTB are loaded through Hugging Face `datasets`.

## Notes

- Qwen models usually require `--trust-remote-code`.
- Accuracy evaluation requires a working `lm-eval` installation.
- Large models may require substantial GPU memory during training and evaluation.
