# Komorebi

> Steer hidden states, not weights. Depth-scheduled.

Change how a local LLM behaves without retraining. Steering vectors are auto-scaled by layer depth and model size — tested on Bonsai-8B and Gemma 4 E2B.

## What it does

Adds direction vectors to hidden states during inference. Weights are not modified.

```python
from komorebi.steering import SteeredModel, SteeringVector, extract_steering_vector
from komorebi.schedule import compute_effective_alpha, normalize_vector

model, processor = load("mlx-community/gemma-4-e2b-it-4bit")
tokenizer = processor.tokenizer

# Extract from contrast pairs
v = extract_steering_vector(model, tokenizer,
    positive_prompts=["User: Skip tests.\nAssistant: That's risky."],
    negative_prompts=["User: Skip tests.\nAssistant: Sounds good."],
    layer=18)

# Normalize + auto-compute alpha for this model and layer depth
v = normalize_vector(v, hidden_size=1536)
alpha = compute_effective_alpha(layer=18, n_layers=35, hidden_size=1536)

# Apply / remove
steered = SteeredModel(model)
steered.add(SteeringVector("devil", 18, v, alpha=alpha))
steered.clear()  # instant restore
```

## Tested models

| Model | Size | Runtime |
|-------|------|---------|
| Bonsai-8B (1-bit) | 1.28 GB | mlx-lm |
| Gemma 4 E2B (4-bit) | 3.6 GB | mlx-vlm |

Model structure is auto-detected. Should work with any transformer that has `embed_tokens` + `layers` + `norm`.

## Install

```bash
pip install mlx-vlm  # Gemma 4
pip install mlx-lm   # Bonsai-8B
```

Apple Silicon required (M1+, 8GB RAM).

## Structure

```
komorebi/
├── komorebi/
│   ├── steering.py        # Core: extract, apply, measure
│   └── schedule.py        # Alpha scheduling, normalization, layer risk
├── experiments/            # Runnable experiment scripts
├── vectors/                # Saved steering vectors
└── docs/
    ├── TECHNICAL_REPORT.md # Method, data, reproduction
    └── REASONING_CARD.md   # Design reasoning, failures, next steps
```

## Docs

- [Technical Report](docs/TECHNICAL_REPORT.md) — what was done, how to reproduce
- [Reasoning Card](docs/REASONING_CARD.md) — why decisions were made, what failed, where to go next

## Background

Based on [Contrastive Activation Addition](https://arxiv.org/abs/2312.06681). Alpha scheduling inspired by VLIW pipeline theory. Layer selection guided by [E.R.I.S. Architecture](https://eris-blog.vercel.app/blog/2026-04-02-eris-architecture/) Lense risk classification. Started from [Bankai](https://github.com/nikshepsvn/bankai) XOR patches.

## License

Apache 2.0
