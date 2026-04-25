"""
Training pipeline for the GPT-style Decoder-only Transformer.

Orchestrates:
    - Data loading via BPE tokenizer + FormattingPipe
    - TransformerDecoder construction / checkpoint resumption
    - AdamW optimizer with linear warmup + cosine decay schedule
    - Cross-entropy loss with ignore_index for padding
    - Per-epoch metrics: loss, perplexity, token accuracy, BLEU-4
    - Gradient clipping, early stopping, telemetry logging, matplotlib plots
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from tabulate import tabulate
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from dataset import (
    BPETokenizer,
    FormattingPipe,
    build_dataset,
    PAD_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,
    MAX_SEQ_LEN,
    BPE_VOCAB_SIZE,
    SCRIPT_DIR as DATA_DIR,
)
from model import TransformerConfig, TransformerDecoder, try_compile

# ── Hyperparameters ──────────────────────────────────────────────────────────

BATCH_SIZE       = 48
N_LAYERS         = 4
D_MODEL          = 384
N_HEADS          = 6
D_FF             = 1536
DROPOUT          = 0.12
LEARNING_RATE    = 5e-5
WEIGHT_DECAY     = 0.05
WARMUP_STEPS     = 800
CLIP             = 1.0
VALIDATION_SPLIT = 0.15
PATIENCE         = 10
SEED             = 42
CONTINUE_EPOCHS  = 50
LABEL_SMOOTHING  = 0.1
COMMENT_LOSS_WEIGHT = 2.2

SAVE_DIR  = SCRIPT_DIR
CKPT_DIR  = os.path.join(SAVE_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)
CKPT_PATH = os.path.join(CKPT_DIR, "checkpoint.pt")
LOG_PATH  = os.path.join(SAVE_DIR, "telemetry_log.json")
REGISTRY_PATH = os.path.join(SAVE_DIR, "experiment_registry.jsonl")


# ── Utilities ────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Phase 0: deterministic mode for strict reproducibility
    if os.getenv("DETERMINISTIC", "") == "1":
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def make_batches(
    data: list[list[int]], batch_size: int, shuffle: bool = True
) -> list[torch.Tensor]:
    """Split a flat list of encoded sequences into batched tensors."""
    indices = list(range(len(data)))
    if shuffle:
        random.shuffle(indices)
    batches: list[torch.Tensor] = []
    for i in range(0, len(indices), batch_size):
        batch_idx = indices[i : i + batch_size]
        batch = [data[j] for j in batch_idx]
        batches.append(torch.LongTensor(batch))
    return batches


def find_comment_prefix_lengths(
    input_ids: torch.Tensor,
    comment_marker_ids: list[int],
    pad_idx: int,
) -> torch.Tensor:
    """
    Return per-sample prefix lengths (number of tokens up to and including "Comment:").
    If marker is not found, fall back to non-pad length.
    """
    B, T = input_ids.shape
    non_pad_lens = (input_ids != pad_idx).sum(dim=1)
    if not comment_marker_ids:
        return non_pad_lens

    marker_len = len(comment_marker_ids)
    if marker_len > T:
        return non_pad_lens

    marker = torch.tensor(comment_marker_ids, device=input_ids.device, dtype=input_ids.dtype)
    windows = input_ids.unfold(1, marker_len, 1)  # (B, T-marker_len+1, marker_len)
    matches = (windows == marker.view(1, 1, marker_len)).all(dim=-1)

    has_match = matches.any(dim=1)
    first_match = matches.float().argmax(dim=1).to(non_pad_lens.dtype)
    prefix_lens = non_pad_lens.clone()
    prefix_lens[has_match] = first_match[has_match] + marker_len
    return prefix_lens.clamp(min=0, max=T)


def weighted_comment_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    prefix_lens: torch.Tensor,
    pad_idx: int,
    comment_loss_weight: float,
    label_smoothing: float,
) -> torch.Tensor:
    """
    Up-weight loss for positions that predict comment tokens (after "Comment:").
    """
    B, T, V = logits.shape
    per_token = F.cross_entropy(
        logits.reshape(-1, V),
        target_ids.reshape(-1),
        ignore_index=pad_idx,
        reduction="none",
        label_smoothing=label_smoothing,
    ).view(B, T)

    positions = torch.arange(T, device=logits.device).unsqueeze(0)
    comment_start_pred_idx = (prefix_lens - 1).clamp(min=0, max=T)
    comment_mask = positions >= comment_start_pred_idx.unsqueeze(1)
    valid_mask = target_ids != pad_idx

    token_weights = torch.ones((B, T), device=logits.device, dtype=logits.dtype)
    token_weights = torch.where(comment_mask, token_weights * comment_loss_weight, token_weights)
    token_weights = token_weights * valid_mask.to(logits.dtype)

    weighted = per_token * token_weights
    denom = token_weights.sum().clamp_min(1.0)
    return weighted.sum() / denom


class LinearWarmupCosineDecay:
    """Custom LR schedule: linear warmup then cosine decay to 0."""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        base_lr: float,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            lr = self.base_lr * (self.current_step / max(self.warmup_steps, 1))
        else:
            progress = (self.current_step - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1
            )
            lr = self.base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    @property
    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


def compute_token_accuracy(logits: torch.Tensor, targets: torch.Tensor, pad_idx: int) -> float:
    """Token-level accuracy ignoring padding positions."""
    preds = logits.argmax(dim=-1).reshape(-1)
    targs = targets.reshape(-1)
    mask = targs != pad_idx
    if mask.sum() == 0:
        return 0.0
    return (preds[mask] == targs[mask]).float().mean().item()


def compute_token_metrics(logits: torch.Tensor, targets: torch.Tensor, pad_idx: int) -> tuple[float, float, float]:
    """Precision, recall, F1 at the token level (ignoring padding)."""
    preds = logits.argmax(dim=-1).reshape(-1)
    targs = targets.reshape(-1)
    mask = targs != pad_idx
    pred_m = preds[mask]
    true_m = targs[mask]
    if true_m.numel() == 0:
        return 0.0, 0.0, 0.0
    correct = (pred_m == true_m).float().sum().item()
    precision = correct / max(pred_m.numel(), 1)
    recall = correct / max(true_m.numel(), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return precision, recall, f1


def compute_grad_norm(model: nn.Module) -> float:
    """L2 norm of all gradients (for monitoring stability)."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return math.sqrt(total)


def compute_bleu(
    model: TransformerDecoder,
    val_seqs: list[list[int]],
    tokenizer: BPETokenizer,
    device: torch.device,
    max_samples: int = 200,
) -> float:
    """Sentence-level BLEU-4 on a sample of validation sequences."""
    from collections import Counter

    def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    def sentence_bleu(ref: list[str], hyp: list[str], max_n: int = 4) -> float:
        """Sentence-level BLEU with add-1 smoothing for n>1 (fixes BLEU=0.0 bug)."""
        if len(hyp) == 0:
            return 0.0
        scores = []
        for n in range(1, max_n + 1):
            ref_ng = Counter(ngrams(ref, n))
            hyp_ng = Counter(ngrams(hyp, n))
            clipped = sum(min(hyp_ng[g], ref_ng[g]) for g in hyp_ng)
            total = max(sum(hyp_ng.values()), 1)
            # Add-1 smoothing for n > 1 to avoid zero on short references
            if n == 1:
                scores.append(clipped / total)
            else:
                scores.append((clipped + 1) / (total + 1))
        if any(s == 0 for s in scores):
            return 0.0
        log_avg = sum(math.log(s) for s in scores) / len(scores)
        bp = min(1.0, math.exp(1 - len(ref) / max(len(hyp), 1)))
        return bp * math.exp(log_avg)

    prompt_template_tokens = tokenizer.encode("Comment:")
    eos_id = tokenizer.eos_id
    pad_id = tokenizer.pad_id
    ctx_len = model.cfg.max_seq_len

    model.eval()
    bleu_scores: list[float] = []
    indices = list(range(len(val_seqs)))
    random.shuffle(indices)
    sample_ids = indices[:max_samples]

    with torch.no_grad():
        for idx in sample_ids:
            seq = val_seqs[idx]

            # Find "Comment:" boundary in the token sequence
            comment_start = -1
            seq_no_pad = [t for t in seq if t != pad_id]
            for i in range(len(seq_no_pad) - len(prompt_template_tokens)):
                if seq_no_pad[i : i + len(prompt_template_tokens)] == prompt_template_tokens:
                    comment_start = i + len(prompt_template_tokens)
                    break
            if comment_start < 0 or comment_start >= len(seq_no_pad):
                continue

            # Reference tokens
            ref_ids = [t for t in seq_no_pad[comment_start:] if t != eos_id and t != pad_id]
            ref_tokens = tokenizer.decode(ref_ids).split()
            if not ref_tokens:
                continue

            # Generate
            prompt = seq_no_pad[:comment_start]
            input_ids = torch.LongTensor([prompt]).to(device)
            generated: list[int] = []
            for _ in range(80):
                logits, _ = model(input_ids[:, -ctx_len:])
                next_id = logits[0, -1].argmax().item()
                if next_id == eos_id:
                    break
                generated.append(next_id)
                input_ids = torch.cat([input_ids, torch.LongTensor([[next_id]]).to(device)], dim=1)

            hyp_tokens = tokenizer.decode(generated).split()
            bleu_scores.append(sentence_bleu(ref_tokens, hyp_tokens))

    model.train()
    return sum(bleu_scores) / max(len(bleu_scores), 1)


def generate_plots(epochs_data: list[dict], save_path: str):
    """Generate training curve plots."""
    epochs = [e["epoch"] for e in epochs_data]
    train_loss = [e["train_loss"] for e in epochs_data]
    val_loss = [e["val_loss"] for e in epochs_data]
    train_f1 = [e["train_f1"] for e in epochs_data]
    val_f1 = [e["val_f1"] for e in epochs_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, train_loss, label="Train Loss", marker="o")
    ax1.plot(epochs, val_loss, label="Val Loss", marker="o")
    ax1.set_title("Loss vs. Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend()

    ax2.plot(epochs, train_f1, label="Train F1", marker="o")
    ax2.plot(epochs, val_f1, label="Val F1", marker="o")
    ax2.set_title("F1 Score vs. Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Score")
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ── Main Pipeline ────────────────────────────────────────────────────────────

def run_pipeline():
    """
    End-to-end training orchestration:
        1. Seed + device selection
        2. Build dataset (download → BPE → encode → split)
        3. Construct or resume TransformerDecoder
        4. Train with AdamW + linear warmup + cosine decay
        5. Evaluate per-epoch metrics + BLEU
        6. Checkpoint best model, early stopping, telemetry JSON + plots
    """
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("+" + "=" * 62 + "+")
    print("|  Automatic Comment Generator — Transformer Training v3      |")
    print("+" + "=" * 62 + "+")
    print(f"\n  Device          : {device}")
    print(f"  Epochs          : {CONTINUE_EPOCHS}")
    print(f"  Batch Size      : {BATCH_SIZE}")
    print(f"  Learning Rate   : {LEARNING_RATE}")
    print(f"  Weight Decay    : {WEIGHT_DECAY}")
    print(f"  Warmup Steps    : {WARMUP_STEPS}")
    print(f"  Architecture    : Decoder-only Transformer ({N_LAYERS}L, {D_MODEL}d, {N_HEADS}H)")
    print()

    # ── Data ─────────────────────────────────────────────────────────────
    tokenizer, train_seqs, val_seqs, dataset_stats = build_dataset(
        val_split=VALIDATION_SPLIT,
        bpe_vocab_size=BPE_VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN,
    )
    pad_idx = tokenizer.pad_id
    comment_marker_ids = tokenizer.encode("Comment:")

    # ── Model ────────────────────────────────────────────────────────────
    cfg = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=MAX_SEQ_LEN,
        n_layers=N_LAYERS,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff=D_FF,
        dropout=DROPOUT,
        pad_idx=pad_idx,
    )
    model = TransformerDecoder(cfg).to(device)

    # ── Mixed-Precision (AMP) ────────────────────────────────────────────
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_dtype = torch.float16 if use_amp else torch.float32

    start_epoch = 0
    if os.path.isfile(CKPT_PATH):
        print(f"\n  [*] Resuming from existing checkpoint: {CKPT_PATH}")
        ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
        try:
            # Only load if architecture matches
            saved_cfg = TransformerConfig.from_dict(ckpt.get("config", {}))
            if saved_cfg.vocab_size == cfg.vocab_size and saved_cfg.d_model == cfg.d_model:
                model.load_state_dict(ckpt["model_state_dict"])
                start_epoch = ckpt.get("epoch", 0)
                print(f"    Loaded weights from epoch {start_epoch}")
            else:
                print("    [!] Architecture mismatch, training from scratch")
                start_epoch = 0
        except (RuntimeError, KeyError) as e:
            print(f"    [!] Could not load weights ({e}), training from scratch")
            start_epoch = 0
    else:
        print("\n  [i] No checkpoint found — training from scratch")

    n_params = model.num_params
    print(f"  Trainable parameters: {n_params:,}")
    print(f"  Vocab size          : {cfg.vocab_size}")
    print(f"  Mixed precision     : {'AMP (FP16)' if use_amp else 'FP32 (CPU)'}")

    # ── Optimizer & Schedule ─────────────────────────────────────────────
    no_decay = {"bias", "ln_f.weight", "ln_f.bias", "ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias"}
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim.AdamW(param_groups, lr=LEARNING_RATE, betas=(0.9, 0.95))

    total_steps = (len(train_seqs) // BATCH_SIZE + 1) * CONTINUE_EPOCHS
    scheduler = LinearWarmupCosineDecay(optimizer, WARMUP_STEPS, total_steps, LEARNING_RATE)

    # ── Telemetry ────────────────────────────────────────────────────────
    telemetry = {
        "experiment_id": f"exp_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "model_name": "TransformerDecoder-CodeComment",
        "mixed_precision": use_amp,
        "architecture": {
            "type": "Decoder-only Transformer (GPT-style)",
            **cfg.to_dict(),
            "trainable_params": n_params,
        },
        "hyperparameters": {
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_steps": WARMUP_STEPS,
            "batch_size": BATCH_SIZE,
            "optimizer": "AdamW",
            "lr_scheduler": "LinearWarmupCosineDecay",
            "clip_grad_norm": CLIP,
            "comment_loss_weight": COMMENT_LOSS_WEIGHT,
            "epochs_total": CONTINUE_EPOCHS,
            "resumed_from_epoch": start_epoch,
            "validation_split": VALIDATION_SPLIT,
            "early_stopping_patience": PATIENCE,
            "seed": SEED,
        },
        "dataset": {
            **dataset_stats,
            "tokenizer": "BPE",
            "validation_strategy": "repo_aware_group_split",
            "code_types": dataset_stats.get("code_type_breakdown", {}),
        },
        "device": str(device),
        "started_at": datetime.datetime.now().isoformat(),
        "epochs": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0

    print("\n" + "=" * 64)
    print("  TRAINING STARTED")
    print("=" * 64 + "\n")

    for epoch in range(1, CONTINUE_EPOCHS + 1):
        epoch_start = time.time()
        global_epoch = start_epoch + epoch

        # ── Training ─────────────────────────────────────────────────────
        model.train()
        train_loss_sum = 0.0
        train_prec_sum, train_rec_sum, train_f1_sum = 0.0, 0.0, 0.0
        n_train_batches = 0
        last_grad_norm = 0.0

        batches = make_batches(train_seqs, BATCH_SIZE, shuffle=True)
        pbar = tqdm(batches, desc=f"Epoch {global_epoch:3d} [Train]", leave=False, unit="batch")

        for step, batch in enumerate(pbar, 1):
            batch = batch.to(device)

            # Input: all tokens except the last,  Target: all tokens except the first
            input_ids = batch[:, :-1]
            target_ids = batch[:, 1:]
            prefix_lens = find_comment_prefix_lengths(input_ids, comment_marker_ids, pad_idx)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                logits, _ = model(input_ids, prefix_lengths=prefix_lens)
                loss = weighted_comment_loss(
                    logits=logits,
                    target_ids=target_ids,
                    prefix_lens=prefix_lens,
                    pad_idx=pad_idx,
                    comment_loss_weight=COMMENT_LOSS_WEIGHT,
                    label_smoothing=LABEL_SMOOTHING,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            last_grad_norm = compute_grad_norm(model)

            # Phase 0: stability assertions
            assert not torch.isnan(loss), f"NaN loss at epoch {global_epoch} step {step}"
            assert not torch.isinf(loss), f"Inf loss at epoch {global_epoch} step {step}"
            if math.isnan(last_grad_norm) or math.isinf(last_grad_norm):
                print(f"  [WARN] Skipping step {step} (grad_norm={last_grad_norm})")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss_sum += loss.item()
            p, r, f = compute_token_metrics(logits, target_ids, pad_idx)
            train_prec_sum += p
            train_rec_sum += r
            train_f1_sum += f
            n_train_batches += 1

            if step % 50 == 0:
                pbar.set_postfix({
                    "Loss": f"{loss.item():.4f}",
                    "F1": f"{f:.4f}",
                    "LR": f"{scheduler.lr:.2e}",
                })

            # Live sample every 200 steps
            if step % 200 == 0:
                model.eval()
                with torch.no_grad():
                    ctx_len = model.cfg.max_seq_len
                    sample_seq = batch[0]
                    # Find a rough midpoint to split prompt/generated
                    non_pad = (sample_seq != pad_idx).sum().item()
                    prompt_len = min(non_pad // 2, non_pad - 1)
                    prompt = sample_seq[:prompt_len].unsqueeze(0)
                    gen_ids: list[int] = []
                    inp = prompt
                    for _ in range(60):
                        lg, _ = model(inp[:, -ctx_len:])
                        nxt = lg[0, -1].argmax().item()
                        if nxt == tokenizer.eos_id:
                            break
                        gen_ids.append(nxt)
                        inp = torch.cat([inp, torch.LongTensor([[nxt]]).to(device)], dim=1)
                    gen_text = tokenizer.decode(gen_ids)
                    print(f"\n  [Live Step {step}] Generated: {gen_text[:120]}")
                model.train()

        avg_train_loss = train_loss_sum / max(n_train_batches, 1)
        avg_train_ppl = math.exp(min(avg_train_loss, 100))

        # ── Validation ───────────────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        val_prec_sum, val_rec_sum, val_f1_sum = 0.0, 0.0, 0.0
        n_val_batches = 0

        with torch.no_grad():
            for batch in tqdm(
                make_batches(val_seqs, BATCH_SIZE, shuffle=False),
                desc=f"Epoch {global_epoch:3d} [Val]",
                leave=False,
                unit="batch",
            ):
                batch = batch.to(device)
                input_ids = batch[:, :-1]
                target_ids = batch[:, 1:]
                prefix_lens = find_comment_prefix_lengths(input_ids, comment_marker_ids, pad_idx)

                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    logits, _ = model(input_ids, prefix_lengths=prefix_lens)
                    loss = weighted_comment_loss(
                        logits=logits,
                        target_ids=target_ids,
                        prefix_lens=prefix_lens,
                        pad_idx=pad_idx,
                        comment_loss_weight=COMMENT_LOSS_WEIGHT,
                        label_smoothing=LABEL_SMOOTHING,
                    )
                val_loss_sum += loss.item()

                p, r, f = compute_token_metrics(logits, target_ids, pad_idx)
                val_prec_sum += p
                val_rec_sum += r
                val_f1_sum += f
                n_val_batches += 1

        avg_val_loss = val_loss_sum / max(n_val_batches, 1)
        avg_val_ppl = math.exp(min(avg_val_loss, 100))

        # ── BLEU (computed every epoch for reliable tracking) ────────
        bleu_score = compute_bleu(model, val_seqs, tokenizer, device)

        current_lr = scheduler.lr
        epoch_time = time.time() - epoch_start

        epoch_record = {
            "epoch": global_epoch,
            "local_epoch": epoch,
            "train_loss": round(avg_train_loss, 6),
            "val_loss": round(avg_val_loss, 6),
            "train_perplexity": round(avg_train_ppl, 4),
            "val_perplexity": round(avg_val_ppl, 4),
            "train_precision": round(train_prec_sum / max(n_train_batches, 1), 6),
            "train_recall": round(train_rec_sum / max(n_train_batches, 1), 6),
            "train_f1": round(train_f1_sum / max(n_train_batches, 1), 6),
            "val_precision": round(val_prec_sum / max(n_val_batches, 1), 6),
            "val_recall": round(val_rec_sum / max(n_val_batches, 1), 6),
            "val_f1": round(val_f1_sum / max(n_val_batches, 1), 6),
            "bleu_4": round(bleu_score, 6),
            "learning_rate": current_lr,
            "grad_norm": round(last_grad_norm, 4),
            "epoch_time_sec": round(epoch_time, 2),
        }
        telemetry["epochs"].append(epoch_record)

        # ── CLI Table ────────────────────────────────────────────────────
        headers = ["Epoch", "T-Loss", "V-Loss", "T-PPL", "V-PPL", "T-F1", "V-F1", "BLEU-4", "LR", "Time"]
        row = [
            global_epoch,
            f"{avg_train_loss:.4f}",
            f"{avg_val_loss:.4f}",
            f"{avg_train_ppl:.2f}",
            f"{avg_val_ppl:.2f}",
            f"{epoch_record['train_f1']:.4f}",
            f"{epoch_record['val_f1']:.4f}",
            f"{bleu_score:.4f}" if bleu_score > 0 else "-",
            f"{current_lr:.2e}",
            f"{epoch_time:.1f}",
        ]
        if epoch == 1:
            print("\n" + tabulate([row], headers=headers, tablefmt="simple"))
        else:
            print(tabulate([row], tablefmt="plain"))

        # ── Checkpoint ───────────────────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg.to_dict(),
                    "max_seq_len": MAX_SEQ_LEN,
                    "epoch": global_epoch,
                    "val_loss": avg_val_loss,
                },
                CKPT_PATH,
            )
            print(f"         [*] Saved best checkpoint (val_loss={avg_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  [STOP] Early stopping triggered after {PATIENCE} epochs without improvement")
                break

        # ── Persist telemetry after every epoch ──────────────────────────
        telemetry["completed_at"] = datetime.datetime.now().isoformat()
        telemetry["best_val_loss"] = round(best_val_loss, 6)
        telemetry["total_epochs_run"] = epoch
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(telemetry, f, indent=2)

    # ── Final ────────────────────────────────────────────────────────────
    telemetry["completed_at"] = datetime.datetime.now().isoformat()
    telemetry["best_val_loss"] = round(best_val_loss, 6)
    telemetry["status"] = "completed"
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)

    plot_path = os.path.join(SAVE_DIR, "training_metrics.png")
    if telemetry["epochs"]:
        generate_plots(telemetry["epochs"], plot_path)

    print("\n" + "=" * 64)
    print("  TRAINING COMPLETE")
    print("=" * 64)
    print(f"  Best Val Loss : {best_val_loss:.4f}")
    print(f"  Checkpoint    : {CKPT_PATH}")
    print(f"  Telemetry Log : {LOG_PATH}")
    print(f"  Plots Output  : {plot_path}")
    print()

    # Phase 0: append to experiment registry
    registry_entry = {
        "experiment_tag": telemetry.get("experiment_tag", telemetry["experiment_id"]),
        "timestamp": datetime.datetime.now().isoformat(),
        "architecture": {"n_layers": N_LAYERS, "d_model": D_MODEL, "n_heads": N_HEADS, "d_ff": D_FF},
        "hyperparameters": {"lr": LEARNING_RATE, "batch_size": BATCH_SIZE, "comment_loss_weight": COMMENT_LOSS_WEIGHT, "label_smoothing": LABEL_SMOOTHING, "warmup_steps": WARMUP_STEPS},
        "best_val_loss": round(best_val_loss, 6),
        "best_val_f1": round(max((e["val_f1"] for e in telemetry["epochs"]), default=0), 6),
        "best_bleu4": round(max((e["bleu_4"] for e in telemetry["epochs"]), default=0), 6),
        "total_epochs": telemetry.get("total_epochs_run", 0),
        "status": "completed",
    }
    with open(REGISTRY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(registry_entry, ensure_ascii=True) + "\n")

    return LOG_PATH


def _parse_train_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the code-comment Transformer")
    parser.add_argument("--experiment-tag", default=None, help="Short tag for this experiment run")
    parser.add_argument("--deterministic", action="store_true", help="Enable strict deterministic mode")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_train_args()
    if args.deterministic:
        os.environ["DETERMINISTIC"] = "1"
    run_pipeline()
