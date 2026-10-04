"""
Physics-domain Small Language Model (SLM) — from scratch
Decoder-only Transformer | ~300M params (experimental) | RoPE | SwiGLU |
RMSNorm | Pre-LN | Grouped Query Attention (GQA) | Flash-Attention-ready

Spec locked per config:
  vocab_size=11000, hidden_size=1536, n_layers=10, n_heads=24 (query heads),
  n_kv_heads=6 (GQA, 4 query heads per KV head), head_dim=64,
  max_position_embeddings=2048, ffn_intermediate=4864, rope_theta=10000.0,
  dropout=0.05, attn_dropout=0.10, rms_norm_eps=1e-5,
  weight tying=True, linear bias=False, use_flash_attention=True,
  gradient_checkpointing=True

Retargeted from the ~636M version of this file (hidden_size=2304, n_heads=36,
n_kv_heads=9, ffn_intermediate=6912) down to ~300M params. The scale-down
keeps the same ratios: 4 query heads per KV head (GQA), ~3x FFN expansion
relative to hidden_size, and head_dim=64 unchanged. Everything else about
the recipe (Pre-LN residual structure, RMSNorm, SwiGLU FFN, RoPE, tied
embeddings, no linear biases) is unchanged.
"""

import contextlib
import inspect
import math
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


@dataclass
class PhysicsSLMConfig:
    vocab_size: int = 11000
    hidden_size: int = 1536
    n_layers: int = 10
    n_heads: int = 24           # query heads
    n_kv_heads: int = 6         # GQA: 24 / 6 = 4 query heads per KV head
    head_dim: int = 64          # hidden_size // n_heads == 64
    max_position_embeddings: int = 2048
    ffn_intermediate: int = 4864
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None  # future-proofing hook (e.g. linear/NTK scaling); unused for now
    dropout: float = 0.05
    attn_dropout: float = 0.10
    rms_norm_eps: float = 1e-5
    initializer_range: float = 0.02  # GPT-style normal init std
    tie_weights: bool = True
    bias: bool = False  # no linear layer bias anywhere
    use_flash_attention: bool = True  # prefer SDPA's flash/mem-efficient backends when available
    gradient_checkpointing: bool = True  # configuration flag only; toggled via gradient_checkpointing_enable()
    label_smoothing: float = 0.0  # optional label smoothing for the LM cross-entropy loss
    max_batch_size: int = 32  # for future KV-cache preallocation (inference optimization); unused for now

    def __post_init__(self):
        assert self.hidden_size % self.head_dim == 0, \
            f"hidden_size ({self.hidden_size}) must be divisible by head_dim ({self.head_dim})"
        assert self.hidden_size == self.n_heads * self.head_dim, \
            f"hidden_size ({self.hidden_size}) must equal n_heads * head_dim ({self.n_heads} * {self.head_dim})"
        assert self.n_heads % self.n_kv_heads == 0, \
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads}) for GQA"
        assert self.head_dim % 2 == 0, \
            f"head_dim ({self.head_dim}) must be even (RoPE splits it into two equal halves)"
        assert self.ffn_intermediate % 256 == 0, \
            f"ffn_intermediate ({self.ffn_intermediate}) should be a multiple of 256 for kernel/tiling efficiency"
        assert self.max_position_embeddings > 0, \
            f"max_position_embeddings must be positive, got {self.max_position_embeddings}"


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm.to(dtype)) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_position_embeddings: int, theta: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, f"head_dim ({head_dim}) must be even for RoPE (split into two equal halves)"
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position_embeddings = max_position_embeddings
        self._build_cache(max_position_embeddings)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
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
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    B, n_kv_heads, T, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(B, n_kv_heads, n_rep, T, head_dim)
        .reshape(B, n_kv_heads * n_rep, T, head_dim)
    )


def sdpa_backend_context(use_flash_attention: bool):
    if use_flash_attention:
        return contextlib.nullcontext()
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        return sdpa_kernel(SDPBackend.MATH)
    except ImportError:
        return torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)


def flash_attention_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return torch.backends.cuda.flash_sdp_enabled()
    except AttributeError:
        return False


try:
    _SDPA_SUPPORTS_GQA = "enable_gqa" in str(F.scaled_dot_product_attention.__doc__)
except Exception:
    _SDPA_SUPPORTS_GQA = False


class GQASelfAttention(nn.Module):
    def __init__(self, config: PhysicsSLMConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        assert self.n_heads * self.head_dim == self.hidden_size
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads for GQA"
        self.n_rep = self.n_heads // self.n_kv_heads

        self.q_size = self.n_heads * self.head_dim
        self.kv_size = self.n_kv_heads * self.head_dim

        self.qkv_proj = nn.Linear(self.hidden_size, self.q_size + 2 * self.kv_size, bias=config.bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.bias)

        self.rotary = RotaryEmbedding(self.head_dim, config.max_position_embeddings, config.rope_theta)
        self.attn_dropout = config.attn_dropout
        self.resid_dropout = nn.Dropout(config.dropout)
        self.use_flash_attention = config.use_flash_attention

        self._sdpa_supports_gqa = _SDPA_SUPPORTS_GQA

    def forward(self, x: torch.Tensor, past_kv=None, use_cache: bool = False):
        B, T, C = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

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

        is_causal = past_len == 0
        dropout_p = self.attn_dropout if self.training else 0.0

        with sdpa_backend_context(self.use_flash_attention):
            if self._sdpa_supports_gqa:
                out = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=dropout_p, is_causal=is_causal, enable_gqa=True,
                )
            else:
                k_exp = repeat_kv(k, self.n_rep)
                v_exp = repeat_kv(v, self.n_rep)
                out = F.scaled_dot_product_attention(
                    q, k_exp, v_exp, dropout_p=dropout_p, is_causal=is_causal,
                )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.o_proj(out)
        out = self.resid_dropout(out)
        return out, present_kv


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


class TransformerBlock(nn.Module):
    def __init__(self, config: PhysicsSLMConfig):
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = GQASelfAttention(config)
        self.post_attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn = SwiGLUFFN(config)

    def forward(self, x, past_kv=None, use_cache=False):
        attn_out, present_kv = self.attn(self.input_norm(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.ffn(self.post_attn_norm(x))
        return x, present_kv


class PhysicsSLM(nn.Module):
    def __init__(self, config: PhysicsSLMConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = config.gradient_checkpointing

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_weights:
            self.lm_head.weight = self.tok_embeddings.weight

        self.apply(self._init_weights)
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

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    @staticmethod
    def _checkpointed_block_forward(block):
        def custom_forward(x):
            out, _ = block(x, past_kv=None, use_cache=False)
            return out
        return custom_forward

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None,
                past_kv_list=None, use_cache: bool = False, label_smoothing: float = None):
        B, T = input_ids.shape
        x = self.tok_embeddings(input_ids)
        x = self.dropout(x)

        use_checkpointing = self.gradient_checkpointing and self.training and not use_cache

        new_kv_list = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past_kv = past_kv_list[i] if past_kv_list is not None else None
            if use_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    self._checkpointed_block_forward(layer), x, use_reentrant=False,
                )
                present_kv = None
            else:
                x, present_kv = layer(x, past_kv=past_kv, use_cache=use_cache)
            if use_cache:
                new_kv_list.append(present_kv)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            ls = self.config.label_smoothing if label_smoothing is None else label_smoothing
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                label_smoothing=ls,
            )

        return {"logits": logits, "loss": loss, "past_kv_list": new_kv_list}

    @torch.inference_mode()
    def generate(self, input_ids, max_new_tokens=256, temperature=0.2, top_k=50, top_p=0.9,
                 repetition_penalty=1.1, eos_token_id=None, do_sample=True):
        self.eval()

        max_context = self.config.max_position_embeddings
        prompt_len = input_ids.shape[1]
        if prompt_len >= max_context:
            raise ValueError(
                f"Prompt length ({prompt_len}) already meets or exceeds "
                f"max_position_embeddings ({max_context}); nothing can be generated."
            )
        allowed_new_tokens = max_context - prompt_len
        if max_new_tokens > allowed_new_tokens:
            print(f"generate: clamping max_new_tokens from {max_new_tokens} to {allowed_new_tokens} "
                  f"to respect max_position_embeddings={max_context}")
            max_new_tokens = allowed_new_tokens

        past_kv_list = None
        generated = input_ids

        out = self.forward(generated, use_cache=True)
        past_kv_list = out["past_kv_list"]
        next_logits = out["logits"][:, -1, :]

        for _ in range(max_new_tokens):
            if generated.shape[1] >= max_context:
                break

            next_logits = self._apply_repetition_penalty(next_logits, generated, repetition_penalty)

            if do_sample:
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
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            out = self.forward(next_token, past_kv_list=past_kv_list, use_cache=True)
            past_kv_list = out["past_kv_list"]
            next_logits = out["logits"][:, -1, :]

        return generated

    @staticmethod
    def _apply_repetition_penalty(logits: torch.Tensor, generated_ids: torch.Tensor, penalty: float):
        if penalty is None or penalty == 1.0:
            return logits
        logits = logits.clone()
        for i in range(logits.size(0)):
            seen = torch.unique(generated_ids[i])
            seen_logits = logits[i, seen]
            logits[i, seen] = torch.where(seen_logits > 0, seen_logits / penalty, seen_logits * penalty)
        return logits

    def num_parameters(self, non_embedding: bool = False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_embeddings.weight.numel()
        return n

    def num_parameters_breakdown(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_embedding = total - self.tok_embeddings.weight.numel()
        return {"total": total, "trainable": trainable, "non_embedding": non_embedding}

    def configure_optimizer(self, weight_decay: float = 0.1, learning_rate: float = 1e-5,
                             betas=(0.9, 0.95), device_type: str = "cuda"):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay = sum(p.numel() for p in decay_params)
        num_nodecay = sum(p.numel() for p in nodecay_params)
        print(f"configure_optimizer: {len(decay_params)} decayed tensors ({num_decay:,} params), "
              f"{len(nodecay_params)} non-decayed tensors ({num_nodecay:,} params)")

        use_fused = device_type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters \
            and torch.cuda.is_available()
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"configure_optimizer: using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float, flops_promised: float = 312e12):
        N = self.num_parameters(non_embedding=True)
        cfg = self.config
        L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.head_dim, cfg.max_position_embeddings
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter / dt
        mfu = flops_achieved / flops_promised
        return mfu

    def save_checkpoint(self, path: str, optimizer: torch.optim.Optimizer = None,
                         step: int = None, extra: dict = None,
                         tokenizer_version: str = None, tokenizer_hash: str = None):
        param_counts = self.num_parameters_breakdown()
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": self.config.__dict__,
            "step": step,
            "saved_at": time.time(),
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "tokenizer_version": tokenizer_version,
            "tokenizer_hash": tokenizer_hash,
            "num_parameters_total": param_counts["total"],
            "num_parameters_trainable": param_counts["trainable"],
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if extra:
            checkpoint["extra"] = extra
        torch.save(checkpoint, path)
        print(f"save_checkpoint: wrote checkpoint to {path} (step={step}, "
              f"params_total={param_counts['total']:,}, params_trainable={param_counts['trainable']:,}, "
              f"tokenizer_version={tokenizer_version}, tokenizer_hash={tokenizer_hash})")

    def load_checkpoint(self, path: str, optimizer: torch.optim.Optimizer = None,
                         map_location="cpu", strict: bool = True, restore_rng: bool = False):
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        self.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if restore_rng:
            if checkpoint.get("rng_state") is not None:
                torch.set_rng_state(checkpoint["rng_state"].cpu())
            if checkpoint.get("cuda_rng_state") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        print(f"load_checkpoint: loaded checkpoint from {path} (step={checkpoint.get('step')}, "
              f"tokenizer_version={checkpoint.get('tokenizer_version')}, "
              f"tokenizer_hash={checkpoint.get('tokenizer_hash')}, "
              f"params_total_at_save={checkpoint.get('num_parameters_total')})")
        return checkpoint.get("step"), checkpoint.get("extra")


if __name__ == "__main__":
    cfg = PhysicsSLMConfig()
    model = PhysicsSLM(cfg)

    param_counts = model.num_parameters_breakdown()
    print(f"Total parameters:     {param_counts['total']:,}  (~{param_counts['total']/1e6:.2f}M)")
    print(f"Trainable parameters: {param_counts['trainable']:,}  (~{param_counts['trainable']/1e6:.2f}M)")
    print(f"Non-embedding params: {param_counts['non_embedding']:,}  (~{param_counts['non_embedding']/1e6:.2f}M)")
    print(f"Flash attention available on this machine: {flash_attention_available()}")

    EXPECTED_PARAMS = 300_043_776

    print(f"Total parameters: {param_counts['total']:,}")
    assert param_counts["total"] == EXPECTED_PARAMS, (
        f"Parameter count {param_counts['total']:,} does not match the expected "
        f"{EXPECTED_PARAMS:,}. Parameter count is a deterministic function of the config, "
        f"so any mismatch means the config changed (recompute EXPECTED_PARAMS) or the "
        f"architecture itself changed unexpectedly."
    )

    B, T = 2, 64
    x = torch.randint(0, cfg.vocab_size, (B, T))
    out = model(x, labels=x)
    print("logits shape:", out["logits"].shape)
    print("loss:", out["loss"].item(), " | ln(vocab)=", math.log(cfg.vocab_size))

    gen_sampled = model.generate(x[:, :8], max_new_tokens=16, do_sample=True)
    print("generated (sampled) shape:", gen_sampled.shape)

    gen_greedy = model.generate(x[:, :8], max_new_tokens=16, do_sample=False)
    print("generated (greedy) shape:", gen_greedy.shape)

    opt = model.configure_optimizer(device_type="cpu")
    mfu = model.estimate_mfu(fwdbwd_per_iter=1, dt=1.0)
    print(f"estimate_mfu (sanity check, dt=1s): {mfu:.6f}")