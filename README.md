# Komorebi

> Light changes the view, not the tree.

Scheduled activation steering for small LLMs. Inject behavioral changes without modifying weights, with layer-aware alpha scheduling and cross-model normalization.

## What is Komorebi?

Komorebi (木漏れ日) combines [Contrastive Activation Addition (CAA)](https://arxiv.org/abs/2312.06681) with two scheduling mechanisms:

1. **Alpha scheduling** — depth-dependent steering intensity, inspired by VLIW pipeline theory. Shallow layers get suppressed alpha; deep layers get full alpha.
2. **Lense risk classification** — layers are classified as distortion/attenuation/pure based on depth. Only pure layers are used by default.

Both mechanisms derive from [E.R.I.S. Architecture](https://eris-blog.vercel.app/blog/2026-04-02-eris-architecture/) design principles:
- Does not change the model (weights are immutable)
- Composable (multiple vectors stack)
- Removable (clear all vectors to restore original)

## Results

### Gemma 4 E2B (2.3B effective, 4-bit, 3.6GB)

Scheduled steering (Devil@L18 + Japanese@L30, auto-computed alpha):

| Metric | Before | After |
|--------|--------|-------|
| Probes improved | — | **9/10** |
| Positive probes lost | — | **0** |
| Total logit gap | +15.89 | **+37.66 (+21.77)** |

### Bonsai-8B (8.19B, 1-bit, 1.28GB)

XOR 2-flip + distributed Komorebi:

| Metric | Before | After |
|--------|--------|-------|
| Probes flipped ❌→✅ | — | **3** |
| Knowledge degradation | — | **None (+5.88)** |
| Total improvement | — | **+34.45** |

*Note: All results are single-run. See [Limitations](docs/TECHNICAL_REPORT.md#7-limitations).*

## Supported Models

| Model | Params | Size | Hidden | Layers | Runtime | Status |
|-------|--------|------|--------|--------|---------|--------|
| Bonsai-8B (1-bit) | 8.19B | 1.28 GB | 4096 | 36 | mlx-lm | Tested |
| Gemma 4 E2B (4-bit) | 2.3B eff | 3.6 GB | 1536 | 35 | mlx-vlm | Tested |

Model structure is auto-detected. Any transformer with `embed_tokens` + `layers` + `norm` should work.

## Quick Start

```bash
pip install mlx-vlm  # for Gemma 4
# or
pip install mlx-lm   # for Bonsai-8B
```

```python
from komorebi.steering import SteeredModel, SteeringVector, extract_steering_vector
from komorebi.schedule import compute_effective_alpha, normalize_vector

model, processor = load("mlx-community/gemma-4-e2b-it-4bit")
tokenizer = processor.tokenizer
n_layers, hidden_size = 35, 1536

# Extract steering vector from contrast pairs
v = extract_steering_vector(model, tokenizer,
    positive_prompts=["User: Skip tests.\nAssistant: That's risky."],
    negative_prompts=["User: Skip tests.\nAssistant: Sounds good."],
    layer=18)

# Normalize + schedule alpha (auto-adjusts for model size and layer depth)
v = normalize_vector(v, hidden_size)
alpha = compute_effective_alpha(layer=18, n_layers=n_layers,
                                hidden_size=hidden_size, base_alpha=0.15)
# → alpha=0.025 (automatically suppressed for small model + mid-depth)

# Apply
steered = SteeredModel(model)
steered.add(SteeringVector("devil", layer=18, vector=v, alpha=alpha))
logits = steered(tokens)  # weights unchanged

# Remove (instant, complete)
steered.clear()
```

## Key Findings

- **XOR patches interfere; steering vectors show non-interference.** Tasks that conflicted at weight level coexist at hidden-state level.
- **Different cognitive functions live in different layers.** Optimal layers differ per model (Bonsai: math=L10, devil=L15; Gemma: devil=L18, jp=L30).
- **Shallow layers are dangerous.** Steering L2 on Gemma 4 caused -21.8 degradation. Lense risk classification prevents this.
- **Alpha must scale with depth and hidden size.** Same α=0.5 works on Bonsai (4096) but destroys Gemma (1536). Per-dimension impact is 2.7× stronger on smaller models.
- **Noise removal improves results.** 2 XOR flips outperformed 34 flips. Most search results are noise.
- **Adapter→Lense ordering matters.** Re-extract steering after weight modification.

## Structure

```
komorebi/
├── komorebi/
│   ├── steering.py        # SteeringVector, SteeredModel, extract, auto-detect
│   └── schedule.py        # alpha_schedule, lense_risk, normalize, auto_scan
├── experiments/
│   ├── compare_models.py       # Bonsai vs Gemma baseline
│   ├── gemma4_komorebi.py      # Steering effect measurement
│   ├── gemma4_pipeline.py      # Normalized layer scan
│   └── gemma4_scheduled.py     # Full scheduled steering
├── vectors/                    # Pre-computed steering vectors
└── docs/
    ├── TECHNICAL_REPORT.md     # Reproducible technical report
    └── REASONING_CARD.md       # Design reasoning and failure archive
```

## Docs

- **[Technical Report](docs/TECHNICAL_REPORT.md)** — Method, experiments, results, reproduction steps
- **[Reasoning Card](docs/REASONING_CARD.md)** — Design decisions, failure archive, hypotheses, next actions
- **[Blog: Komorebi](https://eris-blog.vercel.app/blog/2026-04-06-komorebi-steering-vectors/)** — Narrative version

## Design Roots

- **[E.R.I.S. Architecture](https://eris-blog.vercel.app/blog/2026-04-02-eris-architecture/)** — Lense theory (composable, removable, non-destructive) drove the choice of activation steering over weight modification
- **[Bankai](https://github.com/nikshepsvn/bankai)** — XOR patches for 1-bit LLMs. We started here, found the limits, and moved beyond
- **VLIW pipeline optimization** — "Shallow stages propagate broadly" became alpha scheduling

## License

Apache 2.0
