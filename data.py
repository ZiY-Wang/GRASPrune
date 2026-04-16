from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import Dataset


class TokenBlockDataset(Dataset):
    """Concatenate text samples and slice them into fixed-length token blocks."""

    def __init__(self, ds: Any, tokenizer: Any, block_size: int, text_key: str | None = None):
        self.block_size = int(block_size)

        if text_key is None:
            for candidate in ("text", "sentence", "content", "document"):
                if candidate in ds.column_names:
                    text_key = candidate
                    break
        if text_key is None:
            raise KeyError(
                f"No text column found in dataset columns={ds.column_names}. "
                "Pass text_key explicitly."
            )

        texts = ds[text_key]
        tokenized = tokenizer("\n\n".join(map(str, texts)), return_tensors="pt", add_special_tokens=False)
        self.tokens = tokenized["input_ids"][0]

        if self.tokens.numel() < self.block_size + 1:
            raise ValueError(
                f"Too few tokens: {self.tokens.numel()} < block_size + 1 = {self.block_size + 1}"
            )

    def __len__(self) -> int:
        return (self.tokens.numel() - 1) // self.block_size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = idx * self.block_size
        end = start + self.block_size
        input_ids = self.tokens[start:end].clone()
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def load_local_c4_dataset():
    c4_path = Path(__file__).resolve().parent / "data" / "c4-validation.json"
    try:
        disable_progress_bar()
        return load_dataset("json", data_files=str(c4_path))["train"]
    except Exception as exc:
        raise FileNotFoundError(
            "C4 evaluation/training support expects a local file named "
            "`data/c4-validation.json` in the GRASPrune root directory. "
            "Please download or prepare it manually before using dataset='c4'."
        ) from exc


def load_text_dataset(name: str, split: str):
    if name == "wikitext2":
        return load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    if name == "ptb":
        return load_dataset("ptb_text_only", "penn_treebank", split=split)
    if name == "c4":
        ds = load_local_c4_dataset()
        return ds.select(range(min(2000, len(ds)))) if split != "train" else ds
    raise ValueError(f"Unsupported dataset: {name}")


def _load_local_c4_json():
    return load_local_c4_dataset()


def load_raw_dataset(name: str, split: str):
    return load_text_dataset(name, split)
