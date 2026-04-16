from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import load_file as st_load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from gates import get_transformer_layers


def restore_dense_ffn(block: nn.Module, intermediate_size: int) -> None:
    hidden_size = block.mlp.down_proj.out_features
    input_hidden = block.mlp.gate_proj.in_features

    new_gate = nn.Linear(input_hidden, intermediate_size, bias=(block.mlp.gate_proj.bias is not None))
    new_up = nn.Linear(input_hidden, intermediate_size, bias=(block.mlp.up_proj.bias is not None))
    new_down = nn.Linear(intermediate_size, hidden_size, bias=(block.mlp.down_proj.bias is not None))

    dtype = block.mlp.gate_proj.weight.dtype
    device = block.mlp.gate_proj.weight.device
    block.mlp.gate_proj = new_gate.to(device, dtype=dtype)
    block.mlp.up_proj = new_up.to(device, dtype=dtype)
    block.mlp.down_proj = new_down.to(device, dtype=dtype)

    if hasattr(block.mlp, "intermediate_size"):
        block.mlp.intermediate_size = int(intermediate_size)


def restore_dense_attention_projections(
    block: nn.Module,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> None:
    hidden_size = block.self_attn.q_proj.in_features
    q_out = int(num_heads) * int(head_dim)
    kv_out = int(num_kv_heads) * int(head_dim)
    o_out = block.self_attn.o_proj.out_features

    dtype = block.self_attn.q_proj.weight.dtype
    device = block.self_attn.q_proj.weight.device

    block.self_attn.q_proj = nn.Linear(hidden_size, q_out, bias=(block.self_attn.q_proj.bias is not None)).to(
        device, dtype=dtype
    )
    block.self_attn.k_proj = nn.Linear(hidden_size, kv_out, bias=(block.self_attn.k_proj.bias is not None)).to(
        device, dtype=dtype
    )
    block.self_attn.v_proj = nn.Linear(hidden_size, kv_out, bias=(block.self_attn.v_proj.bias is not None)).to(
        device, dtype=dtype
    )
    block.self_attn.o_proj = nn.Linear(q_out, o_out, bias=(block.self_attn.o_proj.bias is not None)).to(
        device, dtype=dtype
    )

    block.self_attn.num_heads = int(num_heads)
    block.self_attn.num_key_value_heads = int(num_kv_heads)
    block.self_attn.num_key_value_groups = max(
        1,
        block.self_attn.num_heads // max(1, block.self_attn.num_key_value_heads),
    )
    block.self_attn.head_dim = int(head_dim)
    block.self_attn.hidden_size = int(num_heads * head_dim)


def rebuild_from_meta(model: Any, meta: dict[str, Any]) -> None:
    layers = get_transformer_layers(model)
    assert "layers" in meta, "meta.json must contain key: layers"
    assert len(meta["layers"]) == len(layers), f"meta layers={len(meta['layers'])} != model layers={len(layers)}"

    for idx, block in enumerate(layers):
        info = meta["layers"][idx]
        restore_dense_ffn(block, intermediate_size=int(info["ffn_intermediate_size"]))
        restore_dense_attention_projections(
            block,
            num_heads=int(info["num_heads"]),
            num_kv_heads=int(info["num_key_value_heads"]),
            head_dim=int(info["head_dim"]),
        )


def load_pruned_model(
    model_id: str,
    state_dict_path: str,
    meta_path: str,
    torch_dtype: torch.dtype,
    device: str,
    *,
    use_fast_tokenizer: bool = False,
    trust_remote_code: bool = False,
    hf_token: str | None = None,
    local_only: bool = True,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        local_files_only=local_only,
        trust_remote_code=trust_remote_code,
        token=hf_token,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    rebuild_from_meta(model, meta)

    state_dict = OrderedDict(st_load_file(state_dict_path, device="cpu"))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("[WARN] missing keys:", len(missing))
    if unexpected:
        print("[WARN] unexpected keys:", len(unexpected))

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=use_fast_tokenizer,
        local_files_only=local_only,
        trust_remote_code=trust_remote_code,
        token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, meta


def replace_ffn_with_plain(block: nn.Module, inter_size: int) -> None:
    restore_dense_ffn(block, intermediate_size=inter_size)


def replace_attn_proj_with_plain(block: nn.Module, num_heads: int, num_kv_heads: int, head_dim: int) -> None:
    restore_dense_attention_projections(
        block,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
