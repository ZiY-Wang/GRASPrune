from __future__ import annotations

import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.models.llama.modeling_llama import (
    ALL_ATTENTION_FUNCTIONS,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


def get_transformer_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    raise RuntimeError("Unable to locate decoder layers at model.model.layers.")


def infer_model_family(model_or_config) -> str:
    config = getattr(model_or_config, "config", model_or_config)
    model_type = str(getattr(config, "model_type", "")).lower()
    architectures = " ".join(getattr(config, "architectures", []) or []).lower()
    if "qwen" in model_type or "qwen" in architectures:
        return "qwen"
    return "llama"


class LlamaMLPGate(nn.Module):
    def __init__(self, cfg, src: nn.Module, gate_init: float, tau: float, device: str):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.intermediate_size = cfg.intermediate_size
        self.act_fn = ACT2FN[cfg.hidden_act]
        self.tau = tau

        self.gate_proj = src.gate_proj
        self.up_proj = src.up_proj
        self.down_proj = src.down_proj
        for parameter in (
            list(self.gate_proj.parameters())
            + list(self.up_proj.parameters())
            + list(self.down_proj.parameters())
        ):
            parameter.requires_grad_(False)

        self.gate_score = nn.Parameter(torch.full((self.intermediate_size,), gate_init, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = self.up_proj(x)
        gate = self.act_fn(self.gate_proj(x))
        inter = up * gate
        act_dtype = inter.dtype

        if getattr(self, "_rescale_mode", False) and hasattr(self, "eval_keep_mask"):
            mask = self.eval_keep_mask.to(act_dtype).view(1, 1, -1)
            alpha = getattr(self, "rescale_alpha", 0.5)
            scale = 1.0 + alpha * torch.tanh(self.gate_score.to(act_dtype)).view(1, 1, -1)
            z = mask * scale
            return self.down_proj(inter * z)

        if self.training and hasattr(self, "_batch_mask"):
            hard_mask = self._batch_mask.to(act_dtype)
            soft_prob = torch.sigmoid(self.gate_score / self.tau).to(act_dtype).view(1, 1, -1)
            z = hard_mask + (soft_prob - soft_prob.detach())
        elif hasattr(self, "eval_keep_mask"):
            z = self.eval_keep_mask.to(act_dtype).view(1, 1, -1)
        else:
            z = torch.sigmoid(self.gate_score / self.tau).to(act_dtype).view(1, 1, -1)

        return self.down_proj(inter * z)

    @torch.no_grad()
    def register_hard_mask(self, keep_mask_1d: torch.Tensor) -> None:
        if hasattr(self, "eval_keep_mask"):
            self.eval_keep_mask.data = keep_mask_1d.float()
        else:
            self.register_buffer("eval_keep_mask", keep_mask_1d.float())

    @torch.no_grad()
    def materialize_pruning(self) -> None:
        if not hasattr(self, "eval_keep_mask"):
            return

        keep = self.eval_keep_mask.to(torch.bool)
        idx = keep.nonzero(as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            idx = torch.tensor([0], device=keep.device)

        scale_keep = None
        if getattr(self, "_rescale_mode", False) and hasattr(self, "gate_score"):
            alpha = getattr(self, "rescale_alpha", 0.5)
            full_scale = 1.0 + alpha * torch.tanh(self.gate_score.detach().to(self.down_proj.weight.dtype))
            scale_keep = full_scale[keep]

        def prune_out(linear: nn.Linear, indices: torch.Tensor) -> nn.Linear:
            weight = linear.weight.data[indices, :].clone()
            bias = linear.bias.data[indices].clone() if linear.bias is not None else None
            new_linear = nn.Linear(linear.in_features, indices.numel(), bias=(linear.bias is not None)).to(
                weight.device, dtype=weight.dtype
            )
            new_linear.weight.data.copy_(weight)
            if bias is not None:
                new_linear.bias.data.copy_(bias)
            return new_linear

        def prune_in(linear: nn.Linear, indices: torch.Tensor, scale_vec: torch.Tensor | None = None) -> nn.Linear:
            weight = linear.weight.data[:, indices].clone()
            if scale_vec is not None:
                weight = weight * scale_vec.view(1, -1)
            new_linear = nn.Linear(indices.numel(), linear.out_features, bias=(linear.bias is not None)).to(
                weight.device, dtype=weight.dtype
            )
            new_linear.weight.data.copy_(weight)
            if linear.bias is not None:
                new_linear.bias.data.copy_(linear.bias.data)
            return new_linear

        self.up_proj = prune_out(self.up_proj, idx)
        self.gate_proj = prune_out(self.gate_proj, idx)
        self.down_proj = prune_in(self.down_proj, idx, scale_keep)

        for attr_name in ("gate_score", "eval_keep_mask", "_rescale_mode", "rescale_alpha"):
            if hasattr(self, attr_name):
                delattr(self, attr_name)


class LlamaAttnGate(nn.Module):
    def __init__(self, cfg, layer_idx: int, src: nn.Module, gate_init: float, tau: float, device: str):
        super().__init__()
        self.config = cfg
        self.layer_idx = layer_idx
        self.head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        self.num_heads = cfg.num_attention_heads
        self.num_key_value_heads = cfg.num_key_value_heads
        self.num_key_value_groups = max(1, self.num_heads // max(1, self.num_key_value_heads))
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = cfg.attention_dropout
        self.is_causal = True
        self.tau = tau

        self.q_proj = src.q_proj
        self.k_proj = src.k_proj
        self.v_proj = src.v_proj
        self.o_proj = src.o_proj
        for parameter in (
            list(self.q_proj.parameters())
            + list(self.k_proj.parameters())
            + list(self.v_proj.parameters())
            + list(self.o_proj.parameters())
        ):
            parameter.requires_grad_(False)

        self.gate_score_kv = nn.Parameter(torch.full((self.num_key_value_heads,), gate_init, device=device))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings,
        attention_mask=None,
        past_key_value=None,
        cache_position=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_value.update(k, v, self.layer_idx, cache_kwargs)

        act_dtype = q.dtype
        if getattr(self, "_rescale_mode", False) and hasattr(self, "eval_keep_mask_kv"):
            mask_kv = self.eval_keep_mask_kv.to(act_dtype)
            alpha = getattr(self, "rescale_alpha", 0.5)
            scale_kv = 1.0 + alpha * torch.tanh(self.gate_score_kv.to(act_dtype))
            z = mask_kv * scale_kv
        elif self.training and hasattr(self, "_batch_mask"):
            hard_mask = self._batch_mask.to(act_dtype)
            soft_prob = torch.sigmoid(self.gate_score_kv / self.tau).to(act_dtype)
            z = hard_mask + (soft_prob - soft_prob.detach())
        elif hasattr(self, "eval_keep_mask_kv"):
            z = self.eval_keep_mask_kv.to(act_dtype)
        else:
            z = torch.sigmoid(self.gate_score_kv / self.tau).to(act_dtype)

        kv_gate = z.view(1, -1, 1, 1)
        k = k * kv_gate
        v = v * kv_gate

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if not (
                self.config._attn_implementation == "sdpa"
                and kwargs.get("output_attentions", False)
            ):
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        out, attn = attention_interface(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        out = out.reshape(*input_shape, -1).contiguous()
        out = self.o_proj(out)
        return out, attn

    @torch.no_grad()
    def register_hard_mask_kv(self, keep_mask_kv: torch.Tensor) -> None:
        if hasattr(self, "eval_keep_mask_kv"):
            self.eval_keep_mask_kv.data = keep_mask_kv.float()
        else:
            self.register_buffer("eval_keep_mask_kv", keep_mask_kv.float())

    @torch.no_grad()
    def materialize_pruning(self) -> None:
        if not hasattr(self, "eval_keep_mask_kv"):
            return

        keep_kv = self.eval_keep_mask_kv.to(torch.bool)
        if keep_kv.sum() == 0:
            keep_kv = keep_kv.clone()
            keep_kv[0] = True

        num_heads = self.num_heads
        num_kv_heads = self.num_key_value_heads
        head_dim = self.head_dim
        group_size = max(1, num_heads // max(1, num_kv_heads))
        keep_heads = keep_kv.repeat_interleave(group_size)

        scale_keep_kv = None
        if getattr(self, "_rescale_mode", False) and hasattr(self, "gate_score_kv"):
            alpha = getattr(self, "rescale_alpha", 0.5)
            full_scale = 1.0 + alpha * torch.tanh(self.gate_score_kv.detach().to(self.k_proj.weight.dtype))
            scale_keep_kv = full_scale[keep_kv]

        def prune_q_out(linear: nn.Linear, keep_mask_h: torch.Tensor) -> nn.Linear:
            weight = linear.weight.data
            out_features, in_features = weight.shape
            assert out_features == num_heads * head_dim
            weight = weight.view(num_heads, head_dim, in_features)[keep_mask_h, :, :]
            new_linear = nn.Linear(in_features, int(weight.numel() // in_features), bias=(linear.bias is not None)).to(
                weight.device, dtype=weight.dtype
            )
            new_linear.weight.data.copy_(weight.reshape(new_linear.weight.shape))
            if linear.bias is not None:
                bias = linear.bias.data.view(num_heads, head_dim)[keep_mask_h, :].reshape(-1)
                new_linear.bias.data.copy_(bias)
            return new_linear

        def prune_kv_out(
            linear: nn.Linear,
            keep_mask: torch.Tensor,
            scale_vec: torch.Tensor | None = None,
        ) -> nn.Linear:
            weight = linear.weight.data
            out_features, in_features = weight.shape
            assert out_features == num_kv_heads * head_dim
            weight = weight.view(num_kv_heads, head_dim, in_features)[keep_mask, :, :]
            if scale_vec is not None:
                weight = scale_vec.view(-1, 1, 1) * weight
            new_linear = nn.Linear(in_features, int(weight.numel() // in_features), bias=(linear.bias is not None)).to(
                weight.device, dtype=weight.dtype
            )
            new_linear.weight.data.copy_(weight.reshape(new_linear.weight.shape))
            if linear.bias is not None:
                bias = linear.bias.data.view(num_kv_heads, head_dim)[keep_mask, :].reshape(-1)
                new_linear.bias.data.copy_(bias)
            return new_linear

        def prune_o_in(linear: nn.Linear, keep_mask_h: torch.Tensor) -> nn.Linear:
            weight = linear.weight.data
            out_features, in_features = weight.shape
            assert in_features == num_heads * head_dim
            weight = weight.view(out_features, num_heads, head_dim)[:, keep_mask_h, :].reshape(out_features, -1)
            new_linear = nn.Linear(weight.shape[1], out_features, bias=(linear.bias is not None)).to(
                weight.device, dtype=weight.dtype
            )
            new_linear.weight.data.copy_(weight)
            if linear.bias is not None:
                new_linear.bias.data.copy_(linear.bias.data)
            return new_linear

        self.q_proj = prune_q_out(self.q_proj, keep_heads)
        self.k_proj = prune_kv_out(self.k_proj, keep_kv, scale_keep_kv)
        self.v_proj = prune_kv_out(self.v_proj, keep_kv, scale_keep_kv)
        self.o_proj = prune_o_in(self.o_proj, keep_heads)

        self.num_heads = int(keep_heads.sum().item())
        self.num_key_value_heads = int(keep_kv.sum().item())
        self.num_key_value_groups = max(1, self.num_heads // max(1, self.num_key_value_heads))

        for attr_name in ("gate_score_kv", "eval_keep_mask_kv", "_rescale_mode", "rescale_alpha"):
            if hasattr(self, attr_name):
                delattr(self, attr_name)


class QwenMLPGate(LlamaMLPGate):
    pass


class QwenAttnGate(nn.Module):
    def __init__(self, cfg, layer_idx: int, src: nn.Module, gate_init: float, tau: float, device: str):
        super().__init__()
        del layer_idx
        self.config = cfg
        self.src = src
        self.tau = tau
        self.num_heads = int(getattr(cfg, "num_attention_heads", getattr(src, "num_heads", 0)))
        self.num_key_value_heads = int(getattr(cfg, "num_key_value_heads", getattr(src, "num_key_value_heads", 0)))
        if self.num_heads <= 0 or self.num_key_value_heads <= 0:
            raise RuntimeError("Unable to infer num_attention_heads / num_key_value_heads for Qwen.")
        self.head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // self.num_heads))

        for parameter in self.src.parameters():
            parameter.requires_grad_(False)

        self.gate_score_kv = nn.Parameter(torch.full((self.num_key_value_heads,), gate_init, device=device))
        self._hook_handles = []
        if hasattr(self.src, "k_proj") and isinstance(self.src.k_proj, nn.Module):
            self._hook_handles.append(self.src.k_proj.register_forward_hook(self._k_hook))
        if hasattr(self.src, "v_proj") and isinstance(self.src.v_proj, nn.Module):
            self._hook_handles.append(self.src.v_proj.register_forward_hook(self._v_hook))

    def _compute_kv_gate(self, dtype: torch.dtype) -> torch.Tensor:
        if getattr(self, "_rescale_mode", False) and hasattr(self, "eval_keep_mask_kv"):
            mask = self.eval_keep_mask_kv.to(dtype)
            alpha = getattr(self, "rescale_alpha", 0.5)
            scale = 1.0 + alpha * torch.tanh(self.gate_score_kv.to(dtype))
            return mask * scale
        if self.training and hasattr(self, "_batch_mask"):
            hard_mask = self._batch_mask.to(dtype)
            soft_prob = torch.sigmoid(self.gate_score_kv / self.tau).to(dtype)
            return hard_mask + (soft_prob - soft_prob.detach())
        if hasattr(self, "eval_keep_mask_kv"):
            return self.eval_keep_mask_kv.to(dtype)
        return torch.sigmoid(self.gate_score_kv / self.tau).to(dtype)

    def _apply_gate(self, out: torch.Tensor) -> torch.Tensor:
        if (not torch.is_tensor(out)) or out.dim() != 3:
            return out
        num_kv_heads, head_dim = int(self.num_key_value_heads), int(self.head_dim)
        expected = num_kv_heads * head_dim
        z = self._compute_kv_gate(out.dtype).view(1, 1, num_kv_heads, 1)
        if out.shape[-1] == expected:
            batch_size, seq_len, channels = out.shape
            gated = out.view(batch_size, seq_len, num_kv_heads, head_dim) * z
            return gated.view(batch_size, seq_len, channels)
        if out.shape[1] == expected:
            batch_size, channels, seq_len = out.shape
            gated = out.view(batch_size, num_kv_heads, head_dim, seq_len) * z.permute(0, 2, 3, 1)
            return gated.view(batch_size, channels, seq_len)
        return out

    def _k_hook(self, module, inputs, output):
        del module, inputs
        return self._apply_gate(output)

    def _v_hook(self, module, inputs, output):
        del module, inputs
        return self._apply_gate(output)

    def forward(self, *args, **kwargs):
        return self.src(*args, **kwargs)

    @torch.no_grad()
    def register_hard_mask_kv(self, keep_mask_kv: torch.Tensor) -> None:
        if hasattr(self, "eval_keep_mask_kv"):
            self.eval_keep_mask_kv.data = keep_mask_kv.float()
        else:
            self.register_buffer("eval_keep_mask_kv", keep_mask_kv.float())

    @torch.no_grad()
    def materialize_pruning(self):
        if not hasattr(self, "eval_keep_mask_kv"):
            return self.src

        keep_kv = self.eval_keep_mask_kv.to(torch.bool)
        if keep_kv.sum() == 0:
            keep_kv = keep_kv.clone()
            keep_kv[0] = True

        num_heads = int(self.num_heads)
        num_kv_heads = int(self.num_key_value_heads)
        head_dim = int(self.head_dim)
        group_size = max(1, num_heads // max(1, num_kv_heads))
        keep_heads = keep_kv.repeat_interleave(group_size)

        scale_keep_kv = None
        if getattr(self, "_rescale_mode", False) and hasattr(self, "gate_score_kv"):
            alpha = getattr(self, "rescale_alpha", 0.5)
            full_scale = 1.0 + alpha * torch.tanh(self.gate_score_kv.detach().to(self.src.k_proj.weight.dtype))
            scale_keep_kv = full_scale[keep_kv]

        def prune_q_out(linear: nn.Linear, keep_mask_h: torch.Tensor) -> nn.Linear:
            weight = linear.weight.data
            out_features, in_features = weight.shape
            assert out_features == num_heads * head_dim
            weight = weight.view(num_heads, head_dim, in_features)[keep_mask_h, :, :]
            new_linear = nn.Linear(in_features, int(weight.numel() // in_features), bias=(linear.bias is not None)).to(
                weight.device, dtype=weight.dtype
            )
            new_linear.weight.data.copy_(weight.reshape(new_linear.weight.shape))
            if linear.bias is not None:
                bias = linear.bias.data.view(num_heads, head_dim)[keep_mask_h, :].reshape(-1)
                new_linear.bias.data.copy_(bias)
            return new_linear

        def prune_kv_out(linear: nn.Linear, keep_mask: torch.Tensor, scale_vec: torch.Tensor | None = None) -> nn.Linear:
            weight = linear.weight.data
            out_features, in_features = weight.shape
            assert out_features == num_kv_heads * head_dim
            weight = weight.view(num_kv_heads, head_dim, in_features)[keep_mask, :, :]
            if scale_vec is not None:
                weight = scale_vec.view(-1, 1, 1) * weight
            new_linear = nn.Linear(in_features, int(weight.numel() // in_features), bias=(linear.bias is not None)).to(
                weight.device, dtype=weight.dtype
            )
            new_linear.weight.data.copy_(weight.reshape(new_linear.weight.shape))
            if linear.bias is not None:
                bias = linear.bias.data.view(num_kv_heads, head_dim)[keep_mask, :].reshape(-1)
                new_linear.bias.data.copy_(bias)
            return new_linear

        def prune_o_in(linear: nn.Linear, keep_mask_h: torch.Tensor) -> nn.Linear:
            weight = linear.weight.data
            out_features, in_features = weight.shape
            assert in_features == num_heads * head_dim
            weight = weight.view(out_features, num_heads, head_dim)[:, keep_mask_h, :].reshape(out_features, -1)
            new_linear = nn.Linear(weight.shape[1], out_features, bias=(linear.bias is not None)).to(
                weight.device, dtype=weight.dtype
            )
            new_linear.weight.data.copy_(weight)
            if linear.bias is not None:
                new_linear.bias.data.copy_(linear.bias.data)
            return new_linear

        self.src.q_proj = prune_q_out(self.src.q_proj, keep_heads)
        self.src.k_proj = prune_kv_out(self.src.k_proj, keep_kv, scale_keep_kv)
        self.src.v_proj = prune_kv_out(self.src.v_proj, keep_kv, scale_keep_kv)
        self.src.o_proj = prune_o_in(self.src.o_proj, keep_heads)

        self.src.num_heads = int(keep_heads.sum().item())
        self.src.num_key_value_heads = int(keep_kv.sum().item())
        self.src.num_key_value_groups = max(1, self.src.num_heads // max(1, self.src.num_key_value_heads))
        self.src.head_dim = head_dim
        if hasattr(self.src, "hidden_size"):
            self.src.hidden_size = int(self.src.num_heads * head_dim)

        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        return self.src


@torch.no_grad()
def attach_pruning_gates(model, gate_init: float, tau: float, device: str, model_family: str = "auto") -> None:
    family = infer_model_family(model) if model_family == "auto" else model_family.lower()
    for layer_idx, block in enumerate(get_transformer_layers(model)):
        if family == "qwen":
            block.mlp = QwenMLPGate(model.config, block.mlp, gate_init=gate_init, tau=tau, device=device)
            block.self_attn = QwenAttnGate(
                model.config,
                layer_idx,
                block.self_attn,
                gate_init=gate_init,
                tau=tau,
                device=device,
            )
        else:
            block.mlp = LlamaMLPGate(model.config, block.mlp, gate_init=gate_init, tau=tau, device=device)
            block.self_attn = LlamaAttnGate(
                model.config,
                layer_idx,
                block.self_attn,
                gate_init=gate_init,
                tau=tau,
                device=device,
            )


def get_decoder_layers(model):
    return get_transformer_layers(model)


def inject_gates(model, gate_init: float, tau: float, device: str, model_family: str = "auto") -> None:
    attach_pruning_gates(model, gate_init=gate_init, tau=tau, device=device, model_family=model_family)
