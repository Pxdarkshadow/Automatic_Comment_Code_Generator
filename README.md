# Automatic Comment Code Generator

A VS Code extension + local PyTorch Transformer model that generates code comments from selected source code.

The project combines:
- A TypeScript VS Code extension (`src/`) that detects comment targets and inserts comments in-place
- A from-scratch **GPT-style Decoder-only Transformer** (`model/`) that performs local inference

## Project Goals

- Generate comments that are relevant, specific, and action-oriented.
- Reduce vague/boilerplate outputs using repetition penalty and quality gates.
- Keep inference fully local (no external API dependency).
- Achieve < 100ms inference latency on local CPU/GPU.
- Measure quality with repeatable benchmark metrics.

## Repository Layout

```
├── src/
│   ├── extension.ts          # VS Code command (auto-comment.generate), target detection, comment insertion
│   └── aiProvider.ts         # Invokes Python inference process and parses JSON telemetry
├── model/
│   ├── model.py              # GPT-style Decoder-only Transformer (CausalSelfAttention, Pre-Norm Blocks)
│   ├── dataset.py            # BPE tokenizer, FormattingPipe prompt template, CodeSearchNet data pipeline
│   ├── train_pipeline.py     # Training loop (AdamW, linear warmup + cosine decay, early stopping)
│   └── predict.py            # Inference with greedy/top-k/top-p/beam decoding, KV-cache, repetition penalty
├── package.json              # VS Code extension manifest
├── webpack.config.js         # Extension bundler config
└── README.md
```

## Architecture Overview

### Model Architecture

**Decoder-only Transformer (GPT-style)** — ~12M parameters

| Component | Detail |
|---|---|
| **CausalSelfAttention** | Multi-head attention with triangular causal mask |
| **FeedForward** | Linear → GELU → Linear (d_ff = 4 × d_model) |
| **TransformerBlock** | Pre-norm residual: `x + Attn(LN(x))`, `x + FFN(LN(x))` |
| **Embeddings** | Token embedding + sinusoidal positional encoding (weight-tied with LM head) |
| **LM Head** | Linear projection to vocab_size |

Default hyperparameters:
```
n_layers   = 6       d_model = 384       n_heads = 6
d_ff       = 1536    max_seq = 512       dropout = 0.1
vocab_size = 8192 (BPE)
```

### Tokenization

**Custom BPE (Byte-Pair Encoding)** tokenizer trained on the joint code+comment corpus:
- Handles camelCase splitting (`attendanceScore` → `attend`, `ance`, `Score`)
- Handles snake_case splitting
- Shared vocabulary for both code and natural language (8192 tokens)
- Prompt template: `Code:\n{code}\n\nComment: {comment}<eos>`

### Extension Flow

1. User runs command: `Auto-Comment Code`.
2. Extension analyzes selection/file and finds classes, functions, and control-flow blocks.
3. For each target snippet, extension calls Python predictor via `child_process`.
4. Predictor returns JSON containing `comment` + telemetry fields.
5. Extension decides whether to use model output or deterministic rule-based fallback.
6. Comment is inserted with language-aware prefix (`#`, `//`, `<!-- -->`, etc.).

### Decoding Strategies

| Strategy | Description |
|---|---|
| **Greedy** | argmax at each step |
| **Top-k** | Sample from top k=50 highest-probability tokens |
| **Top-p (Nucleus)** | Sample from smallest set where cumulative probability ≥ p=0.95 |
| **Beam Search** | Beam search with length-normalized scoring |

All strategies support:
- **Temperature scaling** for controlling randomness
- **Logit-based repetition penalty** to avoid boilerplate loops
- **KV-cache** for O(1) per-step inference
- **Min/max length** constraints

### Training Pipeline

| Aspect | Configuration |
|---|---|
| **Optimizer** | AdamW (weight decay 0.01, β=(0.9, 0.95)) |
| **LR Schedule** | Linear warmup (2000 steps) + cosine decay |
| **Loss** | Cross-entropy with `ignore_index` for padding |
| **Data** | CodeSearchNet (Python, Java, JavaScript, Go) |
| **Metrics** | Loss, perplexity, token F1, BLEU-4, gradient norm |

## Prerequisites

- Windows/macOS/Linux
- Python 3.11+ with PyTorch for local inference
- Node.js 18+
- VS Code 1.80+

## Setup

### 1) Create and activate virtual environment

```bash
python -m venv venv11
# Windows:
venv11\Scripts\activate
# Linux/macOS:
source venv11/bin/activate
```

### 2) Install Python dependencies

```bash
pip install torch datasets tqdm tabulate matplotlib
```

For GPU training, install PyTorch with CUDA support:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 3) Install Node dependencies

```bash
npm install
```

### 4) Build extension

```bash
npm run compile
```

## Training

```bash
python model/train_pipeline.py
```

Outputs:
- `model/checkpoints/checkpoint.pt` — Best model weights
- `model/bpe_vocab.json` — Trained BPE tokenizer vocabulary

## Inference

### Plain text output

```bash
python model/predict.py "def add(a, b): return a + b"
```

### JSON output with telemetry

```bash
python model/predict.py --b64 "<base64-code>" --json
```

### Decoding controls

```bash
python model/predict.py "..." --mode top_p --top-p 0.95 --temperature 0.7 --repetition-penalty 1.2
python model/predict.py "..." --mode beam --beam-width 5 --min-len 6 --max-len 80
python model/predict.py "..." --mode top_k --top-k 50
python model/predict.py "..." --mode greedy
```

## VS Code Usage

1. Install the extension from a packaged VSIX or the VS Code Marketplace.
2. Ensure the configured Python executable can import `torch`.
3. If needed, set `autoComment.pythonPath` to the Python executable that has PyTorch installed.
4. Open a code file and optionally select code.
5. Run command: `Auto-Comment Code`.
6. Generated comments are inserted above detected targets.

## Marketplace Packaging Notes

The extension is packaged from the compiled `dist/extension.js` bundle plus the local inference assets under `model/`.
Training-only artifacts, telemetry logs, experiment manifests, source maps, virtual environments, and `.env` files are excluded by `.vscodeignore`.

Before publishing, confirm that `publisher` in `package.json` matches the Azure DevOps Marketplace publisher ID that owns the extension.

## Fallback Behavior

If the model checkpoint is missing, corrupted, or produces low-quality output:
- Inference does **not** crash
- Predictor falls back to a local LLM as a measure to generate high-context comments
- Telemetry marks `fallback_reason: model_load_failed`
- The extension continues to work with these fallback mechanisms
- **Note:** For better accuracy, please keep the internet on so the local LLM fallback can function optimally.

The fallback path is also used when the local model emits empty, repetitive, or jargon-heavy text. It is intentionally pattern-aware for common return expressions, filters, predicates, loops, storage, API, and UI code so comments remain specific instead of generic.

## NPM Scripts

| Script | Description |
|---|---|
| `npm run compile` | Build extension bundle |
| `npm run watch` | Incremental webpack build |
| `npm run package` | Production bundle for publish |
| `npm run lint` | Lint TypeScript sources |
| `npm test` | Run extension test entrypoint |

## Version

- Extension: `0.0.1`
- Command ID: `auto-comment.generate`
- Model: Decoder-only Transformer v3

## License

No explicit license file is currently present in the repository. Add a `LICENSE` file if you plan to distribute publicly.


