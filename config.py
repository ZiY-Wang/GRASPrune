from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import torch


def default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@dataclass
class PruningConfig:
    model_id: str = "meta-llama/Llama-2-7b-hf"
    model_family: str = "auto"
    dtype: torch.dtype = torch.bfloat16
    device: str = field(default_factory=default_device)
    local_only: bool = True
    trust_remote_code: bool = False
    hf_token: str | None = None
    seed: int = 42

    max_len: int = 512
    batch_size: int = 1
    epochs: int = 4
    lr: float = 1e-2
    tau: float = 1.5
    gate_init: float = 0.0
    keep_ratio: float = 0.6
    train_split: str = "train"
    num_samples: int = 512

    train_dataset_name: str = "wikitext2"
    ppl_datasets: Tuple[str, ...] = ("wikitext2", "ptb", "c4")
    run_ppl_eval: bool = False
    run_acc_eval: bool = True

    rescale_num_samples: int = 512
    rescale_epochs: int = 1
    rescale_lr: float = 1e-2
    rescale_alpha: float = 0.5

    report_csv: str | None = None
    output_dir: str | None = None
    qwen_eval_in_mask_state: bool = False

    def __post_init__(self) -> None:
        root_dir = Path(__file__).resolve().parent
        if self.output_dir is None:
            model_slug = self.model_id.split("/")[-1]
            self.output_dir = str(root_dir / "outputs" / f"{model_slug}_pruned_{int(self.keep_ratio * 100)}")
        if self.report_csv is None:
            self.report_csv = str(Path(self.output_dir) / "layer_mask_report.csv")

    @property
    def pruned_state_dict_path(self) -> str:
        return f"{self.output_dir}/pruned_state_dict.safetensors"

    @property
    def pruned_meta_path(self) -> str:
        return f"{self.output_dir}/meta.json"
