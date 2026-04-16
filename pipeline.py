from __future__ import annotations

import csv
import json
import os
import random
from typing import Any

import torch
from safetensors.torch import save_file as st_save_file
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import PruningConfig
from data import TokenBlockDataset, load_text_dataset
from evals import evaluate_accuracy, evaluate_perplexity
from gates import (
    LlamaAttnGate,
    LlamaMLPGate,
    QwenAttnGate,
    QwenMLPGate,
    attach_pruning_gates,
    get_transformer_layers,
)
from rebuild import load_pruned_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def compute_ffn_unit_cost(device: str) -> torch.Tensor:
    return torch.tensor(1.0, device=device)


@torch.no_grad()
def compute_attention_group_cost(cfg: Any, device: str) -> torch.Tensor:
    num_heads = cfg.num_attention_heads
    num_kv_heads = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // max(1, num_heads))
    group_size = max(1, num_heads // max(1, num_kv_heads))
    alpha_param = ((2 * group_size + 2) * head_dim) / 3.0
    return torch.tensor(alpha_param, dtype=torch.float32, device=device)


@torch.no_grad()
def build_batch_budget_masks(
    model: Any,
    f_cost: torch.Tensor,
    a_cost: torch.Tensor,
    keep_ratio: float,
    tau: float,
):
    scores, costs, holders = [], [], []

    for layer_idx, block in enumerate(get_transformer_layers(model)):
        if hasattr(block.mlp, "gate_score"):
            values = torch.sigmoid(block.mlp.gate_score.detach().float() / tau)
            scores.append(values)
            costs.append(torch.full_like(values, f_cost))
            holders.append(("ffn", block.mlp, values.numel(), layer_idx))
        if hasattr(block.self_attn, "gate_score_kv"):
            values = torch.sigmoid(block.self_attn.gate_score_kv.detach().float() / tau)
            scores.append(values)
            costs.append(torch.full_like(values, a_cost))
            holders.append(("kv", block.self_attn, values.numel(), layer_idx))

    if not scores:
        return []

    score_tensor = torch.cat(scores)
    cost_tensor = torch.cat(costs)
    budget = max(1.0, cost_tensor.sum().item() * keep_ratio)

    order = torch.argsort(score_tensor, descending=True)
    cumsum = torch.cumsum(cost_tensor[order], dim=0)
    chosen = order[cumsum <= budget]
    if chosen.numel() == 0:
        chosen = order[:1]

    masks, base = [], 0
    for kind, module, length, layer_idx in holders:
        take = torch.zeros(length, dtype=torch.bool, device=score_tensor.device)
        selected = chosen[(chosen >= base) & (chosen < base + length)] - base
        if selected.numel() > 0:
            take[selected.long()] = True
        masks.append((kind, module, take, layer_idx))
        base += length

    per_layer: dict[int, dict[str, tuple[Any, torch.Tensor]]] = {}
    for kind, module, take, layer_idx in masks:
        per_layer.setdefault(layer_idx, {})[kind] = (module, take)

    for layer_idx, grouped in per_layer.items():
        if "ffn" in grouped:
            module, take = grouped["ffn"]
            if take.sum() == 0:
                score = torch.sigmoid(module.gate_score.detach().float() / tau)
                take[torch.argmax(score).item()] = True
        if "kv" in grouped:
            module, take = grouped["kv"]
            if take.sum() == 0:
                score = torch.sigmoid(module.gate_score_kv.detach().float() / tau)
                take[torch.argmax(score).item()] = True

    return [(kind, module, take) for kind, module, take, _ in masks]


def build_train_dataloader(tokenizer: Any, config: PruningConfig) -> DataLoader:
    raw_dataset = load_text_dataset(config.train_dataset_name, config.train_split)
    dataset = TokenBlockDataset(raw_dataset, tokenizer, config.max_len)
    dataset = torch.utils.data.Subset(dataset, list(range(min(config.num_samples, len(dataset)))))
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True)


def train_gates(model: Any, tokenizer: Any, config: PruningConfig) -> None:
    dataloader = build_train_dataloader(tokenizer, config)
    f_cost = compute_ffn_unit_cost(config.device)
    a_cost = compute_attention_group_cost(model.config, config.device)

    f_params, a_params = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith("mlp.gate_score"):
            f_params.append(parameter)
        elif name.endswith("self_attn.gate_score_kv"):
            a_params.append(parameter)

    optimizer = torch.optim.AdamW(
        [{"params": f_params, "lr": config.lr}, {"params": a_params, "lr": config.lr}],
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0.0,
    )

    model.train()
    for epoch in range(config.epochs):
        progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        for batch in progress:
            batch = {key: value.to(config.device) for key, value in batch.items()}
            masks = build_batch_budget_masks(model, f_cost, a_cost, config.keep_ratio, config.tau)
            for kind, module, take in masks:
                module._batch_mask = take.view(1, 1, -1) if kind == "ffn" else take

            output = model(**batch, labels=batch["input_ids"])
            loss = output.loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            for _, module, _ in masks:
                if hasattr(module, "_batch_mask"):
                    delattr(module, "_batch_mask")

            with torch.no_grad():
                mean_ffn = []
                mean_kv = []
                for block in get_transformer_layers(model):
                    if hasattr(block.mlp, "gate_score"):
                        mean_ffn.append(torch.sigmoid(block.mlp.gate_score / config.tau).mean().item())
                    if hasattr(block.self_attn, "gate_score_kv"):
                        mean_kv.append(torch.sigmoid(block.self_attn.gate_score_kv / config.tau).mean().item())
            progress.set_postfix(
                loss=f"{loss.item():.3f}",
                p_ffn=f"{sum(mean_ffn) / max(1, len(mean_ffn)):.3f}",
                p_kv=f"{sum(mean_kv) / max(1, len(mean_kv)):.3f}",
            )


@torch.no_grad()
def apply_final_hard_masks(model: Any, config: PruningConfig) -> None:
    f_cost = compute_ffn_unit_cost(config.device)
    a_cost = compute_attention_group_cost(model.config, config.device)
    masks = build_batch_budget_masks(model, f_cost, a_cost, keep_ratio=config.keep_ratio, tau=config.tau)

    for kind, module, take in masks:
        if kind == "ffn":
            module.register_hard_mask(take)
        else:
            module.register_hard_mask_kv(take)

    for block in get_transformer_layers(model):
        if hasattr(block.mlp, "eval_keep_mask"):
            take = block.mlp.eval_keep_mask.to(torch.bool)
            if take.sum() == 0:
                score = torch.sigmoid(block.mlp.gate_score.detach().float() / config.tau)
                take[torch.argmax(score).item()] = True
                block.mlp.register_hard_mask(take)
        if hasattr(block.self_attn, "eval_keep_mask_kv"):
            take = block.self_attn.eval_keep_mask_kv.to(torch.bool)
            if take.sum() == 0:
                score = torch.sigmoid(block.self_attn.gate_score_kv.detach().float() / config.tau)
                take[torch.argmax(score).item()] = True
                block.self_attn.register_hard_mask_kv(take)


@torch.no_grad()
def summarize_pruning_masks(model: Any, config: PruningConfig) -> list[dict[str, float]]:
    device = next(model.parameters()).device
    cfg = model.config
    f_unit = compute_ffn_unit_cost(str(device)).to(device)
    a_unit = compute_attention_group_cost(cfg, str(device)).to(device)
    num_heads = cfg.num_attention_heads
    num_kv_heads = cfg.num_key_value_heads
    group_size = max(1, num_heads // max(1, num_kv_heads))

    rows = []
    total_ffn_all = total_ffn_keep = 0
    total_kv_all = total_kv_keep = 0
    total_params_kept = 0.0

    for layer_idx, block in enumerate(get_transformer_layers(model)):
        if hasattr(block.mlp, "eval_keep_mask"):
            f_keep = int(block.mlp.eval_keep_mask.to(torch.bool).sum().item())
            f_total = int(block.mlp.eval_keep_mask.numel())
        else:
            score = getattr(block.mlp, "gate_score", None)
            f_total = int(score.numel()) if score is not None else getattr(block.mlp.up_proj, "out_features", 0)
            f_keep = f_total

        if hasattr(block.self_attn, "eval_keep_mask_kv"):
            a_keep = int(block.self_attn.eval_keep_mask_kv.to(torch.bool).sum().item())
            a_total = int(block.self_attn.eval_keep_mask_kv.numel())
        else:
            score = getattr(block.self_attn, "gate_score_kv", None)
            a_total = int(score.numel()) if score is not None else cfg.num_key_value_heads
            a_keep = a_total

        total_ffn_all += f_total
        total_ffn_keep += f_keep
        total_kv_all += a_total
        total_kv_keep += a_keep
        kept_cost = float(f_unit) * f_keep + float(a_unit) * a_keep
        total_params_kept += kept_cost

        rows.append(
            {
                "layer": layer_idx,
                "ffn_keep": f_keep,
                "ffn_total": f_total,
                "ffn_ratio": f_keep / max(1, f_total),
                "kv_keep": a_keep,
                "kv_total": a_total,
                "kv_ratio": a_keep / max(1, a_total),
                "head_keep_eq": int(a_keep * group_size),
                "head_total": int(num_heads),
                "head_ratio_eq": a_keep * group_size / max(1, num_heads),
                "~params_kept": int(kept_cost),
            }
        )

    total_prunable = float(f_unit) * total_ffn_all + float(a_unit) * total_kv_all
    target_cost = total_prunable * config.keep_ratio
    diff = total_params_kept - target_cost
    pct = (diff / max(1.0, target_cost)) * 100.0

    print("\n[Final HARD Masks Report] (KV-based)")
    print("-" * 112)
    print(
        f"{'Layer':>5} | {'FFN keep/total (ratio)':>26} | {'KV keep/total (ratio)':>26} | "
        f"{'Head≈keep/total':>18} | {'~Params kept':>14}"
    )
    print("-" * 112)
    for row in rows:
        print(
            f"{row['layer']:5d} | {row['ffn_keep']:5d}/{row['ffn_total']:5d} ({row['ffn_ratio'] * 100:6.2f}%) | "
            f"{row['kv_keep']:4d}/{row['kv_total']:4d} ({row['kv_ratio'] * 100:6.2f}%) | "
            f"{row['head_keep_eq']:4d}/{row['head_total']:4d} ({row['head_ratio_eq'] * 100:6.2f}%) | "
            f"{row['~params_kept']:14,d}"
        )
    print("-" * 112)
    print(
        f"TOTAL: FFN {total_ffn_keep}/{total_ffn_all} ({(total_ffn_keep / max(1, total_ffn_all)) * 100:.2f}%) | "
        f"KV {total_kv_keep}/{total_kv_all} ({(total_kv_keep / max(1, total_kv_all)) * 100:.2f}%) | "
        f"~Params kept ≈ {int(total_params_kept):,}"
    )
    print(f"TARGET keep_ratio={config.keep_ratio:.3f}: ~Params ≈ {int(target_cost):,}")
    print(f"DIFF: {int(diff):,} ({pct:+.3f}%)\n")

    if config.report_csv:
        os.makedirs(os.path.dirname(config.report_csv), exist_ok=True)
        with open(config.report_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[Saved] layer-wise mask report -> {config.report_csv}")

    return rows


@torch.no_grad()
def materialize_all_pruning(model: Any) -> None:
    for block in get_transformer_layers(model):
        if hasattr(block.mlp, "materialize_pruning"):
            block.mlp.materialize_pruning()
        if hasattr(block.self_attn, "materialize_pruning"):
            new_attn = block.self_attn.materialize_pruning()
            if new_attn is not None:
                block.self_attn = new_attn


@torch.no_grad()
def export_pruned_state_dict_safetensors(model: Any, path: str) -> None:
    state_dict = model.state_dict()
    cpu_state_dict = {key: value.detach().cpu() for key, value in state_dict.items() if torch.is_tensor(value)}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    st_save_file(cpu_state_dict, path)
    print(f"[Saved] pruned state_dict (safetensors) -> {path}")


@torch.no_grad()
def export_pruning_metadata(model: Any, config: PruningConfig) -> dict[str, Any]:
    cfg = model.config
    base_head_dim = getattr(cfg, "head_dim", None)
    if not isinstance(base_head_dim, int) or base_head_dim <= 0:
        base_head_dim = cfg.hidden_size // max(1, getattr(cfg, "num_attention_heads", 1))

    num_heads_orig = getattr(cfg, "num_attention_heads", None)
    num_kv_heads_orig = getattr(cfg, "num_key_value_heads", None)
    if not isinstance(num_heads_orig, int) or num_heads_orig <= 0:
        num_heads_orig = getattr(get_transformer_layers(model)[0].self_attn, "num_heads", 1)
    if not isinstance(num_kv_heads_orig, int) or num_kv_heads_orig <= 0:
        num_kv_heads_orig = getattr(get_transformer_layers(model)[0].self_attn, "num_key_value_heads", 1)
    group_size = max(1, num_heads_orig // max(1, num_kv_heads_orig))

    export_format = "materialized"
    meta = {
        "model_id": config.model_id,
        "model_family": config.model_family,
        "dtype": str(config.dtype),
        "export_format": export_format,
        "rescale_alpha": float(config.rescale_alpha),
        "layers": [],
    }
    for layer_idx, block in enumerate(get_transformer_layers(model)):
        assert hasattr(block.mlp, "eval_keep_mask"), (
            f"Layer {layer_idx} mlp missing eval_keep_mask. Run apply_final_hard_masks first."
        )
        assert hasattr(block.self_attn, "eval_keep_mask_kv"), (
            f"Layer {layer_idx} attn missing eval_keep_mask_kv. Run apply_final_hard_masks first."
        )

        ffn_keep = int(block.mlp.eval_keep_mask.to(torch.bool).sum().item())
        kv_mask = block.self_attn.eval_keep_mask_kv.to(torch.bool)
        num_kv_heads = int(kv_mask.sum().item())
        num_heads = max(1, min(num_heads_orig, num_kv_heads * group_size))
        layer_info = {
            "ffn_intermediate_size": max(1, ffn_keep),
            "num_heads": int(num_heads),
            "num_key_value_heads": max(1, int(num_kv_heads)),
            "head_dim": int(base_head_dim),
        }
        meta["layers"].append(layer_info)

    os.makedirs(os.path.dirname(config.pruned_meta_path), exist_ok=True)
    with open(config.pruned_meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    print(f"[Saved] meta(from masks) -> {config.pruned_meta_path}")
    return meta


def train_rescale(model: Any, tokenizer: Any, config: PruningConfig) -> None:
    for block in get_transformer_layers(model):
        if isinstance(block.mlp, (LlamaMLPGate, QwenMLPGate)):
            block.mlp.rescale_alpha = config.rescale_alpha
        if isinstance(block.self_attn, (LlamaAttnGate, QwenAttnGate)):
            block.self_attn.rescale_alpha = config.rescale_alpha

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    rescale_params = []
    for block in get_transformer_layers(model):
        if isinstance(block.mlp, (LlamaMLPGate, QwenMLPGate)) and hasattr(block.mlp, "eval_keep_mask"):
            with torch.no_grad():
                block.mlp.gate_score.zero_()
            block.mlp.gate_score.requires_grad_(True)
            block.mlp._rescale_mode = True
            rescale_params.append(block.mlp.gate_score)
        if isinstance(block.self_attn, (LlamaAttnGate, QwenAttnGate)) and hasattr(block.self_attn, "eval_keep_mask_kv"):
            with torch.no_grad():
                block.self_attn.gate_score_kv.zero_()
            block.self_attn.gate_score_kv.requires_grad_(True)
            block.self_attn._rescale_mode = True
            rescale_params.append(block.self_attn.gate_score_kv)

    if not rescale_params:
        print("[Stage B] No rescale params found, skip.")
        return

    optimizer = torch.optim.AdamW(
        rescale_params,
        lr=config.rescale_lr,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0.0,
    )

    raw_dataset = load_text_dataset("wikitext2", "validation")
    dataset = TokenBlockDataset(raw_dataset, tokenizer, config.max_len)
    dataset = torch.utils.data.Subset(
        dataset,
        list(range(min(config.rescale_num_samples, len(dataset)))),
    )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model.train()
    for epoch in range(config.rescale_epochs):
        progress = tqdm(dataloader, desc=f"[Stage B] Epoch {epoch + 1}")
        for batch in progress:
            batch = {key: value.to(config.device) for key, value in batch.items()}
            output = model(**batch, labels=batch["input_ids"])
            loss = output.loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            progress.set_postfix(loss=f"{loss.item():.3f}")


def load_model_and_tokenizer(config: PruningConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        use_fast=False if config.model_family != "qwen" else True,
        local_files_only=config.local_only,
        trust_remote_code=config.trust_remote_code,
        token=config.hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=config.dtype,
        local_files_only=config.local_only,
        trust_remote_code=config.trust_remote_code,
        token=config.hf_token,
    ).to(config.device)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return model, tokenizer


def run_pruning_pipeline(config: PruningConfig) -> dict[str, Any]:
    set_seed(config.seed)
    model, tokenizer = load_model_and_tokenizer(config)

    attach_pruning_gates(
        model,
        gate_init=config.gate_init,
        tau=config.tau,
        device=config.device,
        model_family=config.model_family,
    )
    train_gates(model, tokenizer, config)
    apply_final_hard_masks(model, config)
    summarize_pruning_masks(model, config)
    meta = export_pruning_metadata(model, config)
    train_rescale(model, tokenizer, config)

    materialize_all_pruning(model)
    export_pruned_state_dict_safetensors(model, config.pruned_state_dict_path)

    metrics: dict[str, Any] = {"meta": meta}
    if config.run_ppl_eval:
        eval_model, eval_tokenizer, _ = load_pruned_model(
            model_id=config.model_id,
            state_dict_path=config.pruned_state_dict_path,
            meta_path=config.pruned_meta_path,
            torch_dtype=config.dtype,
            device=config.device,
            use_fast_tokenizer=(config.model_family == "qwen"),
            trust_remote_code=config.trust_remote_code,
            hf_token=config.hf_token,
            local_only=config.local_only,
        )
        metrics["ppl"] = evaluate_perplexity(
            eval_model,
            eval_tokenizer,
            datasets=config.ppl_datasets,
            device=config.device,
        )
    if config.run_acc_eval:
        if not config.run_ppl_eval:
            eval_model, eval_tokenizer, _ = load_pruned_model(
                model_id=config.model_id,
                state_dict_path=config.pruned_state_dict_path,
                meta_path=config.pruned_meta_path,
                torch_dtype=config.dtype,
                device=config.device,
                use_fast_tokenizer=(config.model_family == "qwen"),
                trust_remote_code=config.trust_remote_code,
                hf_token=config.hf_token,
                local_only=config.local_only,
            )
        metrics["acc"] = evaluate_accuracy(eval_model, eval_tokenizer, device=config.device)
    print("[Done] Pruning pipeline finished.")
    return metrics


def ffn_unit_cost(device: str) -> torch.Tensor:
    return compute_ffn_unit_cost(device)


def attn_kv_group_cost(cfg: Any, device: str) -> torch.Tensor:
    return compute_attention_group_cost(cfg, device)


def build_batch_budget_mask(
    model: Any,
    f_cost: torch.Tensor,
    a_cost: torch.Tensor,
    keep_ratio: float,
    tau: float,
):
    return build_batch_budget_masks(model, f_cost, a_cost, keep_ratio, tau)


def final_hardening_apply(model: Any, config: PruningConfig) -> None:
    apply_final_hard_masks(model, config)


def report_masks(model: Any, config: PruningConfig) -> list[dict[str, float]]:
    return summarize_pruning_masks(model, config)


def export_meta_from_masks(model: Any, config: PruningConfig) -> dict[str, Any]:
    return export_pruning_metadata(model, config)


def create_model_and_tokenizer(config: PruningConfig):
    return load_model_and_tokenizer(config)
