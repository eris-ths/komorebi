# Komorebi

> Light changes the view, not the tree.

Steering vectors for small LLMs. Inject behavioral changes into language models without modifying weights.

## What is Komorebi?

Komorebi (木漏れ日) adds direction vectors to hidden states during inference, changing model behavior while keeping weights untouched. Named after the Japanese word for sunlight filtering through trees.

**Lense conditions:**
1. Does not change the model (weights are immutable)
2. Composable (multiple vectors can be stacked)
3. Removable (clear all vectors to restore original behavior)

## Supported Models

| Model | Params | Size | Hidden | Layers | Runtime | Status |
|-------|--------|------|--------|--------|---------|--------|
| Bonsai-8B (1-bit) | 8.19B | 1.28 GB | 4096 | 36 | mlx-lm | Tested |
| **Gemma 4 E2B (4-bit)** | 2.3B eff | 3.6 GB | 1536 | 35 | mlx-vlm | **Tested** |

Model structure is auto-detected via `_find_model_parts()`. Any transformer with `embed_tokens` + `layers` + `norm` should work.

## Quick Start

```bash
pip install mlx-vlm  # for Gemma 4
# or
pip install mlx-lm   # for Bonsai-8B
```

```python
from komorebi.steering import (
    SteeredModel, SteeringVector,
    extract_steering_vector, extract_hidden_states
)
from mlx_vlm import load  # or mlx_lm

model, processor = load("mlx-community/gemma-4-e2b-it-4bit")

# Extract a steering vector from contrast pairs
devil_vec = extract_steering_vector(
    model, processor.tokenizer,
    positive_prompts=["User: Skip tests.\nAssistant: That's risky."],
    negative_prompts=["User: Skip tests.\nAssistant: Sounds good."],
    layer=17,
)

# Apply steering (weights unchanged)
steered = SteeredModel(model)
steered.add(SteeringVector("devil", layer=17, vector=devil_vec, alpha=0.5))

# Generate with steering active
logits = steered(tokens)

# Remove steering (instant, complete)
steered.clear()
```

## Key Findings (Bonsai-8B experiments)

- **XOR patches interfere; steering vectors resonate.** Tasks that conflicted with XOR (mul vs div) improved together with steering.
- **Different cognitive functions live in different layers.** Math=L10, Devil=L15, Persona=L25-30.
- **Alpha 0.5 is the sweet spot.** Maximizing any single direction degrades others.
- **Layer distribution reduces interference.** Spreading vectors across layers beats concentrating on one.
- **Composition order matters.** Adapter (XOR) first, then Lense (steering) re-extracted from the modified distribution.

## Structure

```
komorebi/
├── komorebi/          # Core library
│   ├── steering.py    # SteeringVector, SteeredModel, extract, auto-detect
│   └── __init__.py
├── experiments/       # Experiment scripts
│   └── test_gemma4_e2b.py
├── vectors/           # Pre-computed steering vectors (.json + .npz)
└── docs/              # Architecture and findings
```

## Background

- [Blog: Komorebi — 1-bit LLMの行動を光で変える](https://eris-blog.vercel.app/blog/2026-04-06-komorebi-steering-vectors/)
- Inspired by [Bankai](https://github.com/nikshepsvn/bankai) (XOR patches for 1-bit LLMs)
- Built with [E.R.I.S. Architecture](https://eris-blog.vercel.app/blog/2026-04-02-eris-architecture/) design principles

## License

Apache 2.0
