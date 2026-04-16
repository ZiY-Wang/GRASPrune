from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

os.environ.setdefault("HF_DATASETS_DISABLE_CACHING", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

DEFAULT_LM_EVAL_TASKS = ("piqa", "winogrande", "hellaswag", "arc_easy", "arc_challenge")


class EvalSequenceDataset(Dataset):
    def __init__(self, tensors: torch.Tensor):
        self.tensors = tensors

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.tensors[index]

    def __len__(self) -> int:
        return len(self.tensors)


def load_local_c4_dataset():
    c4_path = Path(__file__).resolve().parent / "data" / "c4-validation.json"
    try:
        disable_progress_bar()
        return load_dataset("json", data_files=str(c4_path))["train"]
    except Exception as exc:
        raise FileNotFoundError(
            "C4 evaluation expects a local file named `data/c4-validation.json` "
            "in the GRASPrune root directory. Please download or prepare it manually."
        ) from exc


def build_eval_sequence_dataset(
    samples: Any,
    tokenizer: Any,
    seq_len: int,
    field_name: str,
) -> EvalSequenceDataset:
    test_ids = tokenizer("\n\n".join(samples[field_name]), return_tensors="pt").input_ids[0]
    num_samples = test_ids.numel() // seq_len
    batches = [test_ids[i * seq_len : (i + 1) * seq_len] for i in range(num_samples)]
    return EvalSequenceDataset(torch.stack(batches))


def build_eval_dataloader(name: str, tokenizer: Any, seq_len: int = 2048, batch_size: int = 8) -> DataLoader:
    if name == "wikitext2":
        test_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        dataset = build_eval_sequence_dataset(test_data, tokenizer, seq_len, "text")
    elif name == "ptb":
        test_data = load_dataset("ptb_text_only", "penn_treebank", split="test")
        dataset = build_eval_sequence_dataset(test_data, tokenizer, seq_len, "sentence")
    elif name == "c4":
        test_data = load_local_c4_dataset()
        dataset = build_eval_sequence_dataset(test_data[:2000], tokenizer, seq_len, "text")
    else:
        raise ValueError(f"Unsupported evaluation dataset: {name}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


@torch.no_grad()
def evaluate_perplexity(
    model: Any,
    tokenizer: Any,
    datasets: Iterable[str] = ("wikitext2", "ptb", "c4"),
    model_seq_len: int = 2048,
    batch_size: int = 8,
    device: str = "cuda",
) -> Dict[str, float]:
    model.to(device)
    model.eval()

    ppls: Dict[str, float] = {}
    for dataset in datasets:
        print(f"\n=== Evaluating {dataset} ===")
        test_loader = build_eval_dataloader(dataset, tokenizer, seq_len=model_seq_len, batch_size=batch_size)
        nlls = []
        for batch in tqdm(test_loader):
            batch = batch.to(device)
            output = model(batch, use_cache=False)
            logits = output.logits
            if not torch.isfinite(logits).all():
                continue

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
            nlls.append(loss)

        ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
        ppls[dataset] = float(ppl)

    print(f"\nPPL results: {ppls}")
    if torch.cuda.is_available():
        print(f"Weight Memory: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MiB\n")
    return ppls


def evaluate_accuracy(model: Any, tokenizer: Any, device: str, batch_size: int = 8) -> dict[str, float]:
    try:
        import lm_eval
        from lm_eval import utils as lm_eval_utils
        from lm_eval.api.registry import ALL_TASKS
        from lm_eval.models.huggingface import HFLM
        from lm_eval.tasks import initialize_tasks
    except ImportError as exc:
        raise ImportError(
            "Accuracy evaluation requires `lm-eval` to be installed in the current environment. "
            "Please install dependencies from requirements.txt before running accuracy evaluation."
        ) from exc

    model.to(device)
    initialize_tasks()

    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    task_names = lm_eval_utils.pattern_match(list(DEFAULT_LM_EVAL_TASKS), ALL_TASKS)
    if not task_names:
        raise ValueError("No lm-eval tasks matched the default accuracy task list.")

    logging.info("Selected lm-eval tasks: %s", task_names)
    results = lm_eval.simple_evaluate(
        model=hflm,
        tasks=task_names,
        num_fewshot=0,
        batch_size=batch_size,
    )["results"]

    metric_vals = {
        task: round(float(result.get("acc_norm,none", result["acc,none"])), 4)
        for task, result in results.items()
    }
    metric_vals["avg_accuracy"] = round(sum(metric_vals.values()) / len(metric_vals), 4)
    logging.info("Accuracy metrics:\n%s", json.dumps(metric_vals, indent=2, ensure_ascii=False))
    return metric_vals


IndexDataset = EvalSequenceDataset


def _load_local_c4_json():
    return load_local_c4_dataset()


def _process_eval_data(samples: Any, tokenizer: Any, seq_len: int, field_name: str) -> EvalSequenceDataset:
    return build_eval_sequence_dataset(samples, tokenizer, seq_len, field_name)


def get_test_loader(name: str, tokenizer: Any, seq_len: int = 2048, batch_size: int = 8) -> DataLoader:
    return build_eval_dataloader(name, tokenizer, seq_len=seq_len, batch_size=batch_size)


def ppl_eval(
    model: Any,
    tokenizer: Any,
    datasets: Iterable[str] = ("wikitext2", "ptb", "c4"),
    model_seq_len: int = 2048,
    batch_size: int = 8,
    device: str = "cuda",
) -> Dict[str, float]:
    return evaluate_perplexity(
        model,
        tokenizer,
        datasets=datasets,
        model_seq_len=model_seq_len,
        batch_size=batch_size,
        device=device,
    )


def acc_eval(model: Any, tokenizer: Any, device: str, batch_size: int = 8) -> dict[str, float]:
    return evaluate_accuracy(model, tokenizer, device=device, batch_size=batch_size)
