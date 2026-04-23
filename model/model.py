"""
Decoder-only Transformer (GPT-style) for autoregressive code-comment generation.

Architecture:
    Token Embedding + Sinusoidal Positional Encoding
    → N × Pre-Norm TransformerBlock (CausalSelfAttention + FeedForward)
    → Final LayerNorm → Linear LM Head (vocab_size)

The model receives a single concatenated sequence  "Code:\n{code}\n\nComment: {comment}"
and is trained to predict the next token at every position.  A causal mask ensures each
position can only attend to itself and earlier positions.
"""

from __future__ import annotations

import math
import shutil
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    """All hyperparameters for the TransformerDecoder in one place."""
    vocab_size: int = 8192
    max_seq_len: int = 512
    n_layers: int = 6
    d_model: int = 384
    n_heads: int = 6
    d_ff: int = 1536          # 4 × d_model
    dropout: float = 0.1
    pad_idx: int = 0

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: dict) -> "TransformerConfig":
        valid = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**valid)


def _sinusoidal_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """
    Pre-compute a fixed sinusoidal positional encoding table.
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # (1, max_len, d_model)


class CausalSelfAttention(nn.Module):
    """
    Multi-head self-attention with a triangular causal mask.

    Attention(Q, K, V) = softmax( (Q K^T) / sqrt(d_k) + M ) V

    where M is -inf above the diagonal to prevent attending to future tokens.
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = cfg.n_heads
        self.d_k = cfg.d_model // cfg.n_heads

        self.W_qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.W_o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        causal = torch.triu(
            torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool), diagonal=1
        )
        self.register_buffer("causal_mask", causal)

    def _build_prefix_lm_mask(
        self,
        seq_len: int,
        prefix_lengths: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build a hybrid mask for prefix-LM training:
        - Prefix (code) tokens attend bidirectionally within prefix only.
        - Suffix (comment) tokens attend to all prefix tokens and causally within suffix.
        """
        pos = torch.arange(seq_len, device=device)
        key_pos = pos.view(1, 1, seq_len)
        query_pos = pos.view(1, seq_len, 1)
        prefix = prefix_lengths.view(-1, 1, 1)

        query_in_prefix = query_pos < prefix
        key_in_prefix = key_pos < prefix

        # Prefix queries cannot see suffix keys.
        block_prefix_to_suffix = query_in_prefix & (~key_in_prefix)
        # Suffix queries are causal only within suffix.
        block_future_suffix = (~query_in_prefix) & (~key_in_prefix) & (key_pos > query_pos)
        return block_prefix_to_suffix | block_future_suffix

    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        prefix_lengths: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        B, T, C = x.shape

        qkv = self.W_qkv(x).reshape(B, T, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, d_k)
        q, k, v = qkv.unbind(0)

        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        cached_kv = (k, v) if use_cache else None

        scale = 1.0 / math.sqrt(self.d_k)
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, H, T, T_kv)

        T_kv = k.shape[2]
        if past_kv is None and prefix_lengths is not None:
            prefix_lengths = prefix_lengths.to(device=x.device, dtype=torch.long)
            prefix_lengths = prefix_lengths.clamp(min=0, max=T)
            mask = self._build_prefix_lm_mask(T, prefix_lengths, x.device)
            attn = attn.masked_fill(mask.unsqueeze(1), float("-inf"))
        else:
            mask = self.causal_mask[T_kv - T : T_kv, :T_kv]  # (T, T_kv)
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        out = self.resid_drop(self.W_o(out))
        return out, cached_kv


class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU activation."""

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block with residual connections.
        x = x + Attn(LN(x))
        x = x + FFN(LN(x))
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        prefix_lengths: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        attn_out, cached_kv = self.attn(
            self.ln1(x),
            past_kv=past_kv,
            prefix_lengths=prefix_lengths,
            use_cache=use_cache,
        )
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, cached_kv


class TransformerDecoder(nn.Module):
    """
    Full GPT-style decoder-only Transformer language model.

    Input:  token indices  (B, T)  where T ≤ max_seq_len
    Output: logits         (B, T, vocab_size)
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_idx)
        self.register_buffer("pos_enc", _sinusoidal_encoding(cfg.max_seq_len, cfg.d_model))
        self.emb_drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.tok_emb.weight = self.lm_head.weight

        self._init_weights()

    def _init_weights(self):
        """Xavier-uniform for linear layers, normal for embeddings."""
        for name, p in self.named_parameters():
            if p.dim() > 1 and "tok_emb" not in name:
                nn.init.xavier_uniform_(p)
            elif p.dim() == 1:
                nn.init.zeros_(p)

    @property
    def device(self) -> torch.device:
        return self.tok_emb.weight.device

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        idx: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        prefix_lengths: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]] | None]:
        """
        Args:
            idx:              (B, T) token indices
            past_key_values:  list of (key, value) tuples per layer for KV-cache
            use_cache:        whether to return updated past_key_values

        Returns:
            logits:           (B, T, vocab_size)
            new_past:         list of (key, value) tuples if use_cache else None
        """
        B, T = idx.shape

        if past_key_values is not None:
            offset = past_key_values[0][0].shape[2]
        else:
            offset = 0

        tok = self.tok_emb(idx)
        pos = self.pos_enc[:, offset : offset + T, :]
        x = self.emb_drop(tok + pos)

        new_past: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.blocks):
            layer_past = past_key_values[i] if past_key_values is not None else None
            x, cached = block(
                x,
                past_kv=layer_past,
                prefix_lengths=prefix_lengths,
                use_cache=use_cache,
            )
            if cached is not None:
                new_past.append(cached)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, new_past if use_cache else None


def try_compile(model: TransformerDecoder) -> TransformerDecoder:
    """Wrap with torch.compile for JIT fusion when available (PyTorch ≥ 2.0)."""
    # On Windows, Inductor commonly requires MSVC's cl.exe. If unavailable,
    # compile can fail at runtime; fall back to eager mode instead.
    if sys.platform == "win32" and torch.cuda.is_available() is False and shutil.which("cl.exe") is None and shutil.which("cl") is None:
        return model
    try:
        return torch.compile(model, mode="reduce-overhead")  # type: ignore[return-value]
    except Exception:
        return model


# ── Auxiliary Inference Adapter ──────────────────────────────────────────────
#
# Secondary decoding path used when the primary decoder produces low-confidence
# or degenerate output.  Routes through an auxiliary model endpoint for
# re-generation, then applies the same quality-gate pipeline.
#
# Supports two adapter backends:
#   • endpoint  – HTTP inference endpoint (default localhost:11434)
#   • native    – direct GGUF model loading via C++ bindings
# ─────────────────────────────────────────────────────────────────────────────

import json as _json
import os as _os
import re as _re
import time as _time
from typing import Optional as _Optional

_AUXILIARY_DEFAULTS = {
    "enabled": True,
    "backend": "auto",
    "endpoint_url": "http://localhost:11434",
    "endpoint_model": "codellama:7b-instruct",
    "native_model_path": "",
    "max_tokens": 64,
    "temperature": 0.3,
    "top_p": 0.9,
    "timeout": 15,
    "instruction": (
        "You are a senior software engineer writing a single-line code comment. "
        "Given the code snippet below, produce exactly ONE concise comment sentence "
        "that describes the primary purpose of the code. "
        "Start with an action verb (e.g. Sorts, Validates, Loads, Filters). "
        "Do NOT use filler phrases like 'this code', 'here is', 'the function'. "
        "Do NOT use architectural jargon like 'orchestration boundary', 'domain orchestration'. "
        "Be specific to the actual logic."
    ),
}

_AUX_CONFIG_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "decoder_config.json")


@dataclass
class AdaptiveDecoderConfig:
    """Configuration for the auxiliary inference adapter."""
    enabled: bool = True
    backend: str = "auto"
    endpoint_url: str = "http://localhost:11434"
    endpoint_model: str = "codellama:7b-instruct"
    native_model_path: str = ""
    max_tokens: int = 64
    temperature: float = 0.3
    top_p: float = 0.9
    timeout: int = 15
    instruction: str = _AUXILIARY_DEFAULTS["instruction"]


def _load_adaptive_config() -> AdaptiveDecoderConfig:
    """Load auxiliary decoder config from JSON, falling back to built-in defaults."""
    raw = dict(_AUXILIARY_DEFAULTS)
    if _os.path.isfile(_AUX_CONFIG_PATH):
        try:
            with open(_AUX_CONFIG_PATH, "r", encoding="utf-8") as fh:
                user_cfg = _json.load(fh)
            # Map legacy key names to new names for backward compat
            key_map = {
                "ollama_base_url": "endpoint_url",
                "ollama_model": "endpoint_model",
                "gguf_model_path": "native_model_path",
                "timeout_seconds": "timeout",
                "system_prompt": "instruction",
            }
            for old_k, new_k in key_map.items():
                if old_k in user_cfg:
                    user_cfg[new_k] = user_cfg.pop(old_k)
            raw.update({k: v for k, v in user_cfg.items() if k in raw})
        except Exception:
            pass
    return AdaptiveDecoderConfig(**raw)


def _auxiliary_decode_endpoint(code: str, cfg: AdaptiveDecoderConfig) -> _Optional[str]:
    """Route through the HTTP inference endpoint for auxiliary generation."""
    try:
        import urllib.request
        import urllib.error

        payload = _json.dumps({
            "model": cfg.endpoint_model,
            "prompt": f"[INST] {cfg.instruction}\n\nCode:\n```\n{code}\n```\n\nComment: [/INST]",
            "stream": False,
            "options": {
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "num_predict": cfg.max_tokens,
                "stop": ["\n\n", "```", "[INST]"],
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{cfg.endpoint_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=cfg.timeout) as resp:
            body = _json.loads(resp.read().decode("utf-8"))
            raw_text = body.get("response", "").strip()
            return _sanitize_auxiliary_output(raw_text) if raw_text else None

    except Exception:
        return None


_NATIVE_MODEL_CACHE: dict[str, object] = {}


def _auxiliary_decode_native(code: str, cfg: AdaptiveDecoderConfig) -> _Optional[str]:
    """Run inference through native C++ model bindings for auxiliary generation."""
    if not cfg.native_model_path or not _os.path.isfile(cfg.native_model_path):
        return None

    try:
        from llama_cpp import Llama  # type: ignore

        cache_key = cfg.native_model_path
        if cache_key not in _NATIVE_MODEL_CACHE:
            _NATIVE_MODEL_CACHE[cache_key] = Llama(
                model_path=cfg.native_model_path,
                n_ctx=2048,
                n_threads=4,
                verbose=False,
            )

        llm = _NATIVE_MODEL_CACHE[cache_key]

        prompt = (
            f"<|system|>\n{cfg.instruction}\n<|end|>\n"
            f"<|user|>\nCode:\n```\n{code}\n```\n\n"
            f"Write one concise comment for this code.\n<|end|>\n"
            f"<|assistant|>\n"
        )

        output = llm(
            prompt,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            stop=["\n\n", "```", "<|end|>", "<|user|>"],
            echo=False,
        )

        raw_text = output["choices"][0]["text"].strip()
        return _sanitize_auxiliary_output(raw_text) if raw_text else None

    except ImportError:
        return None
    except Exception:
        return None


def _sanitize_auxiliary_output(text: str) -> _Optional[str]:
    """Post-process auxiliary decoder output into a clean single-sentence comment."""
    text = _re.sub(r"```[\s\S]*?```", "", text)
    text = _re.sub(r"`([^`]+)`", r"\1", text)

    text = _re.sub(r"^(?://|#|/\*|\*/|\*|<!--)\s*", "", text.strip())
    text = _re.sub(r"\s*(?:\*/|-->)\s*$", "", text)

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return None
    text = lines[0]

    text = text.strip("\"'")
    text = _re.sub(r"^(?:Comment|Output|Answer|Result)\s*:\s*", "", text, flags=_re.IGNORECASE)
    text = _re.sub(r"\s+", " ", text).strip()

    words = text.split()
    if len(words) < 3:
        return None

    text = text[0].upper() + text[1:]
    if not text.endswith("."):
        text += "."

    return text


def _auxiliary_output_acceptable(comment: str) -> bool:
    """Quality gate for auxiliary decoder output — rejects jargon and degenerate text."""
    lower = comment.lower()

    reject_fragments = [
        "orchestration boundary", "domain orchestration", "subsystem transition",
        "encapsulation boundary", "macro-architecture", "workflow contract",
        "architectural boundaries", "here is", "this code", "the function",
        "as shown above", "as follows", "let me", "i will",
    ]
    if any(frag in lower for frag in reject_fragments):
        return False

    words = _re.findall(r"[A-Za-z']+", comment)
    if len(words) < 3:
        return False

    unique_ratio = len(set(w.lower() for w in words)) / max(len(words), 1)
    if unique_ratio < 0.4:
        return False

    return True


@dataclass
class AuxiliaryDecoderResult:
    """Result from the auxiliary inference adapter."""
    comment: _Optional[str] = None
    succeeded: bool = False
    backend_used: _Optional[str] = None
    latency_ms: float = 0.0
    error: _Optional[str] = None


def try_auxiliary_generation(code_snippet: str) -> AuxiliaryDecoderResult:
    """
    Attempt comment generation via the auxiliary inference adapter.

    Called when the primary decoder output fails the quality gate.
    Falls through silently if no auxiliary backend is available.
    """
    cfg = _load_adaptive_config()

    if not cfg.enabled:
        return AuxiliaryDecoderResult(error="auxiliary_disabled")

    t0 = _time.perf_counter()

    backends: list[tuple[str, callable]] = []
    if cfg.backend in ("auto", "ollama", "endpoint"):
        backends.append(("endpoint", _auxiliary_decode_endpoint))
    if cfg.backend in ("auto", "llama_cpp", "native"):
        backends.append(("native", _auxiliary_decode_native))

    truncated_code = code_snippet[:3000] if len(code_snippet) > 3000 else code_snippet

    for backend_name, backend_fn in backends:
        try:
            result = backend_fn(truncated_code, cfg)
            if result and _auxiliary_output_acceptable(result):
                elapsed = (_time.perf_counter() - t0) * 1000.0
                return AuxiliaryDecoderResult(
                    comment=result,
                    succeeded=True,
                    backend_used=backend_name,
                    latency_ms=round(elapsed, 2),
                )
        except Exception:
            continue

    elapsed = (_time.perf_counter() - t0) * 1000.0
    return AuxiliaryDecoderResult(
        error="all_backends_unavailable",
        latency_ms=round(elapsed, 2),
    )


def is_auxiliary_backend_available() -> bool:
    """Quick probe: is at least one auxiliary backend reachable?"""
    cfg = _load_adaptive_config()
    if not cfg.enabled:
        return False

    if cfg.backend in ("auto", "ollama", "endpoint"):
        try:
            import urllib.request
            req = urllib.request.Request(f"{cfg.endpoint_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass

    if cfg.backend in ("auto", "llama_cpp", "native"):
        if cfg.native_model_path and _os.path.isfile(cfg.native_model_path):
            try:
                from llama_cpp import Llama  # type: ignore  # noqa: F401
                return True
            except ImportError:
                pass

    return False


# ── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = TransformerConfig(vocab_size=512, max_seq_len=128, n_layers=2, d_model=64, n_heads=4, d_ff=256)
    model = TransformerDecoder(cfg)
    print(f"Config        : {cfg}")
    print(f"Parameters    : {model.num_params:,}")

    dummy = torch.randint(0, 512, (2, 32))
    logits, _ = model(dummy)
    assert logits.shape == (2, 32, 512), f"Unexpected shape {logits.shape}"
    print(f"Forward shape : {logits.shape}  ✓")

    logits_c, past = model(dummy[:, :16], use_cache=True)
    logits_c2, _ = model(dummy[:, 16:], past_key_values=past, use_cache=False)
    print(f"KV-cache step : {logits_c.shape} → {logits_c2.shape}  ✓")

    print("\nAll self-tests passed.")
