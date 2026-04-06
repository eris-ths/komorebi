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

| Model | Size | Status |
|-------|------|--------|
| Bonsai-8B (1-bit) | 1.28 GB | Tested |
| Gemma 4 E2B (4-bit) | ~3.2 GB | In progress |

## Quick Start

```bash
pip install mlx-vlm  # for Gemma 4
# or
pip install mlx-lm   # for Bonsai-8B

python experiments/baseline.py
```

## Structure

```
komorebi/
├── komorebi/          # Core library
│   ├── steering.py    # SteeringVector, SteeredModel
│   ├── probes.py      # Probe definitions and measurement
│   └── extract.py     # Hidden state extraction
├── experiments/       # Experiment scripts
├── vectors/           # Pre-computed steering vectors
└── docs/              # Architecture and findings
```

## Background

- [Blog: Komorebi — 1-bit LLMの行動を光で変える](https://eris-blog.vercel.app/blog/2026-04-06-komorebi-steering-vectors/)
- Inspired by [Bankai](https://github.com/nikshepsvn/bankai) (XOR patches for 1-bit LLMs)
- Built with [E.R.I.S. Architecture](https://eris-blog.vercel.app/blog/2026-04-02-eris-architecture/) design principles

## License

Apache 2.0
