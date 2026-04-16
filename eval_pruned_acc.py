from __future__ import annotations

import argparse

import torch

from evals import evaluate_accuracy
from rebuild import load_pruned_model


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a saved pruned model with lm-eval tasks.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--sd", required=True, help="Path to pruned_state_dict.safetensors")
    parser.add_argument("--meta", required=True, help="Path to meta.json")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=sorted(DTYPE_MAP.keys()), default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--allow-remote-model", action="store_true")
    parser.add_argument("--use-fast-tokenizer", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model, tokenizer, _ = load_pruned_model(
        model_id=args.model_id,
        state_dict_path=args.sd,
        meta_path=args.meta,
        torch_dtype=DTYPE_MAP[args.dtype],
        device=args.device,
        use_fast_tokenizer=args.use_fast_tokenizer,
        trust_remote_code=args.trust_remote_code,
        hf_token=args.hf_token,
        local_only=not args.allow_remote_model,
    )
    metrics = evaluate_accuracy(model, tokenizer, device=args.device)
    if metrics:
        print(metrics)


if __name__ == "__main__":
    main()
