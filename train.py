from __future__ import annotations

import argparse

import torch

from config import PruningConfig
from pipeline import run_pruning_pipeline


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run GRASPrune gate pruning on a LLaMA-style model.")
    parser.add_argument("--model-id", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--model-family", default="auto", choices=["auto", "llama", "qwen"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=sorted(DTYPE_MAP.keys()), default="bfloat16")
    parser.add_argument("--keep-ratio", type=float, default=0.6)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--tau", type=float, default=1.5)
    parser.add_argument("--gate-init", type=float, default=0.0)
    parser.add_argument("--train-dataset", default="wikitext2")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--report-csv", default=None)
    parser.add_argument("--run-ppl-eval", action="store_true")
    parser.add_argument("--skip-acc-eval", action="store_true")
    parser.add_argument("--rescale-num-samples", type=int, default=512)
    parser.add_argument("--rescale-epochs", type=int, default=1)
    parser.add_argument("--rescale-lr", type=float, default=1e-2)
    parser.add_argument("--rescale-alpha", type=float, default=0.5)
    parser.add_argument("--allow-remote-model", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--hf-token", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = PruningConfig(
        model_id=args.model_id,
        model_family=args.model_family,
        dtype=DTYPE_MAP[args.dtype],
        device=args.device or PruningConfig().device,
        local_only=not args.allow_remote_model,
        trust_remote_code=args.trust_remote_code,
        hf_token=args.hf_token,
        max_len=args.max_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        tau=args.tau,
        gate_init=args.gate_init,
        keep_ratio=args.keep_ratio,
        train_split=args.train_split,
        num_samples=args.num_samples,
        train_dataset_name=args.train_dataset,
        run_ppl_eval=args.run_ppl_eval,
        run_acc_eval=not args.skip_acc_eval,
        rescale_num_samples=args.rescale_num_samples,
        rescale_epochs=args.rescale_epochs,
        rescale_lr=args.rescale_lr,
        rescale_alpha=args.rescale_alpha,
        report_csv=args.report_csv,
        output_dir=args.output_dir,
    )
    run_pruning_pipeline(config)


if __name__ == "__main__":
    main()
