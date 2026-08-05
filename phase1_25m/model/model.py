"""
Physics-domain Small Language Model (SLM) — from scratch
Decoder-only Transformer | ~26M params | RoPE | SwiGLU | RMSNorm | Pre-LN

Spec locked per config:
  vocab_size=12736, hidden_size=512, n_layers=6, n_heads=8, head_dim=64,
  max_position_embeddings=512, ffn_intermediate=1408, rope_theta=10000.0,
  dropout=0.075, attn_dropout=0.0, weight tying=True, linear bias=False
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclassf
class PhysicsSLMConfig:
    vocab_size: int = 12736
    hidden_size: int = 512
    n_layers: int = 6
    n_heads: int = 8
    head_dim: int = 64  # hidden_size // n_heads == 64
    max_position_embeddings: int = 512
    ffn_intermediate: int = 1408
    rope_theta: float = 10000.0
    dropout: float = 0.075
    attn_dropout: float = 0.0
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02  # GPT-style normal init std
    tie_weights: bool = True
    bias: bool = False  # no linear layer bias anywhere


# --------------------------------------------------------------------------
# RMSNorm
# --------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm.to(dtype)) * self.weight


# --------------------------------------------------------------------------
# Rotary Positional Embeddings (RoPE)
# --------------------------------------------------------------------------
class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_position_embeddings: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position_embeddings = max_position_embeddings
        self._build_cache(max_position_embeddings)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, head_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, head_dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, device, dtype):
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        return (
            self.cos_cached[:seq_len].to(device=device, dtype=dtype),
            self.sin_cached[:seq_len].to(device=device, dtype=dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    # cos, sin: (seq_len, head_dim) -> (1, 1, seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


# --------------------------------------------------------------------------
# Multi-Head Self Attention (fused QKV, no bias, RoPE, causal, KV-cache)
# --------------------------------------------------------------------------
class MHASelfAttention(nn.Module):
    def __init__(self, config: PhysicsSLMConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        assert self.n_heads * self.head_dim == self.hidden_size

        self.qkv_proj = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=config.bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.bias)

        self.rotary = RotaryEmbedding(self.head_dim, config.max_position_embeddings, config.rope_theta)
        self.attn_dropout = config.attn_dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, past_kv=None, use_cache: bool = False):
        B, T, C = x.shape

        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.hidden_size, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            past_len = past_k.shape[2]
        else:
            past_len = 0

        cos, sin = self.rotary(past_len + T, device=x.device, dtype=x.dtype)
        cos_t, sin_t = cos[past_len:past_len + T], sin[past_len:past_len + T]
        q, k = apply_rotary_pos_emb(q, k, cos_t, sin_t)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        present_kv = (k, v) if use_cache else None

        is_causal = past_len == 0  # only causal-mask the first (full) pass
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.o_proj(out)
        out = self.resid_dropout(out)
        return out, present_kv


# --------------------------------------------------------------------------
# SwiGLU FFN
# --------------------------------------------------------------------------
class SwiGLUFFN(nn.Module):
    def __init__(self, config: PhysicsSLMConfig):
        super().__init__()
        h, i = config.hidden_size, config.ffn_intermediate
        self.gate_proj = nn.Linear(h, i, bias=config.bias)
        self.up_proj = nn.Linear(h, i, bias=config.bias)
        self.down_proj = nn.Linear(i, h, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


# --------------------------------------------------------------------------
# Transformer Block (Pre-LN, standard residual)
# --------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, config: PhysicsSLMConfig):
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = MHASelfAttention(config)
        self.post_attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn = SwiGLUFFN(config)

    def forward(self, x, past_kv=None, use_cache=False):
        attn_out, present_kv = self.attn(self.input_norm(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.ffn(self.post_attn_norm(x))
        return x, present_kv


# --------------------------------------------------------------------------
# Full Model
# --------------------------------------------------------------------------
class PhysicsSLM(nn.Module):
    def __init__(self, config: PhysicsSLMConfig):
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_weights:
            self.lm_head.weight = self.tok_embeddings.weight

        self.apply(self._init_weights)
        # scaled init for residual projections (GPT-2 style depth scaling)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=config.initializer_range / math.sqrt(2 * config.n_layers))

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None,
                past_kv_list=None, use_cache: bool = False):
        """
        input_ids: (B, T)
        labels:    (B, T) — already shifted by the caller OR pass unshifted
                   and this function will shift internally (see note below).
        """
        B, T = input_ids.shape
        x = self.tok_embeddings(input_ids)
        x = self.dropout(x)

        new_kv_list = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past_kv = past_kv_list[i] if past_kv_list is not None else None
            x, present_kv = layer(x, past_kv=past_kv, use_cache=use_cache)
            if use_cache:
                new_kv_list.append(present_kv)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # standard next-token shift: predict token[t+1] from position t
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {"logits": logits, "loss": loss, "past_kv_list": new_kv_list}

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=256, temperature=0.7, top_k=100, top_p=0.9,
                 eos_token_id=None):
        self.eval()
        past_kv_list = None
        generated = input_ids

        # prime the cache with the prompt
        out = self.forward(generated, use_cache=True)
        past_kv_list = out["past_kv_list"]
        next_logits = out["logits"][:, -1, :]

        for _ in range(max_new_tokens):
            next_logits = next_logits / max(temperature, 1e-5)

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = -float("inf")

            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cum_probs = torch.cumsum(probs, dim=-1)
                remove = cum_probs > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                sorted_logits[remove] = -float("inf")
                next_logits = torch.full_like(next_logits, -float("inf")).scatter(
                    1, sorted_idx, sorted_logits
                )

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            out = self.forward(next_token, past_kv_list=past_kv_list, use_cache=True)
            past_kv_list = out["past_kv_list"]
            next_logits = out["logits"][:, -1, :]

        return generated

    def num_parameters(self, non_embedding: bool = False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_embeddings.weight.numel()
        return n


if __name__ == "__main__":
    cfg = PhysicsSLMConfig()
    model = PhysicsSLM(cfg)
    total = model.num_parameters()
    print(f"Total parameters: {total:,}  (~{total/1e6:.2f}M)")

    B, T = 2, 64
    x = torch.randint(0, cfg.vocab_size, (B, T))
    out = model(x, labels=x)
    print("logits shape:", out["logits"].shape)
    print("loss:", out["loss"].item(), " | ln(vocab)=", math.log(cfg.vocab_size))

    gen = model.generate(x[:, :8], max_new_tokens=16)
    print("generated shape:", gen.shape)