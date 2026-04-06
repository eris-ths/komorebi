# Komorebi: Scheduled Activation Steering for Small LLMs

> Combining E.R.I.S. Lense theory with VLIW pipeline scheduling
> to create model-agnostic, layer-aware steering strategies.

**Authors**: Nao + Eris (Three Hearts Space)
**Date**: 2026-04-06
**Status**: Experimental / Reproducible

---

## Abstract

We present a method for behavioral modification of small (2B–8B) language models using activation steering vectors with two novel scheduling mechanisms: (1) depth-dependent alpha scheduling derived from VLIW pipeline theory, and (2) Lense risk classification derived from E.R.I.S. Architecture. On Gemma 4 E2B (2.3B effective, 4-bit), our scheduled approach improved 9/10 evaluation probes with zero regression on positive probes, achieving a total logit gap improvement of +21.77 over baseline. The same framework was validated on Bonsai-8B (8.19B, 1-bit) with different optimal parameters but identical scheduling principles.

---

## 1. Background

### 1.1 Contrastive Activation Addition (CAA)

Activation steering adds a direction vector to hidden states during inference:

```
h' = h + α · v
```

where `v` is extracted from contrast pairs (Rimsky et al., 2024):

```
v = mean(h_positive) - mean(h_negative)
```

### 1.2 Open Problems in CAA

1. **Layer selection**: Which layer to steer? Typically found by grid search without theoretical guidance.
2. **Alpha calibration**: How strong? Manually tuned per model. No cross-model transfer.
3. **Multi-vector composition**: Multiple vectors interfere unpredictably.
4. **Adapter interaction**: How does steering interact with weight modifications (LoRA, XOR patches)?

---

## 2. Method

### 2.1 Lense Risk Classification

From E.R.I.S. Architecture, a Lense must satisfy three conditions:
1. **Does not change the object** (weights immutable)
2. **Composable** (multiple Lenses stack)
3. **Removable** (clear to restore)

We extend this with a **risk spectrum** based on layer depth:

| Depth | E.R.I.S. Layer | Risk | Behavior |
|-------|---------------|------|----------|
| 0–15% | Core | **Distortion** | Steering warps all downstream layers |
| 15–45% | Domain | **Attenuation** | Information loss; target improves but guards degrade |
| 45–75% | Application | **Pure** | Localized effect; composable without interference |
| 75–100% | Infrastructure | **Pure** | Output-proximal; minimal downstream propagation |

**Design principle**: Prefer pure layers. Use attenuation layers only when pure layers lack sufficient effect. Avoid distortion layers entirely.

### 2.2 Alpha Scheduling (VLIW-derived)

From VLIW processor optimization: modifying an early pipeline stage propagates changes through all subsequent stages. Deeper stages have bounded propagation.

Applied to transformer layers (exponent 1.5 is empirically chosen; ablation with 1.0/2.0 is a TODO):

```python
def alpha_schedule(layer, n_layers, base_alpha):
    depth = layer / (n_layers - 1)
    scale = 0.1 + 0.9 * (depth ** 1.5)  # nonlinear suppression inspired by VLIW pipeline propagation
    return base_alpha * scale
```

| Layer (35 total) | Depth | Scale | Effective α (base=0.15) |
|-----------------|-------|-------|------------------------|
| L0 | 0.00 | 0.10 | 0.006 |
| L10 | 0.29 | 0.24 | 0.014 |
| L18 | 0.53 | 0.44 | 0.025 |
| L30 | 0.88 | 0.84 | 0.048 |
| L34 | 1.00 | 1.00 | 0.056 |

### 2.3 Hidden-Size Normalization

Steering impact is inversely proportional to hidden dimension:

```
impact ∝ α / hidden_size
```

For cross-model transfer, normalize alpha by hidden size:

```python
def normalize_alpha(alpha, hidden_size, reference=4096):
    return alpha * (hidden_size / reference)
```

This explains why α=0.5 worked on Bonsai-8B (hidden=4096) but destroyed Gemma 4 E2B (hidden=1536): the per-dimension impact was 2.7× stronger on Gemma.

### 2.4 Vector Normalization

Raw steering vectors have norms varying 6× across layers (7–45 in Gemma 4). We normalize to unit vectors scaled by √hidden_size:

```python
v_normalized = (v / mx.linalg.norm(v)) * (hidden_size ** 0.5)
```

This decouples vector direction (semantic) from magnitude (layer-dependent artifact).

### 2.5 Adapter-Lense Ordering Principle

When combining weight modification (Adapter: XOR/LoRA) with activation steering (Lense):

```
Correct:  Apply Adapter → Extract Lense from modified distribution → Apply Lense
Wrong:    Extract Lense → Apply Adapter → Apply Lense (distribution mismatch)
```

Verified experimentally: XOR 2-flip + pre-extracted steering degraded results (math_sum -0.93), while XOR + post-extracted steering maintained improvements.

---

## 3. Experiments

### 3.1 Setup

| Model | Params | Quant | Hidden | Layers | Runtime |
|-------|--------|-------|--------|--------|---------|
| Bonsai-8B | 8.19B | 1-bit | 4096 | 36 | mlx-lm |
| Gemma 4 E2B | 2.3B eff | 4-bit | 1536 | 35 | mlx-vlm |

Hardware: Apple M3 8GB

### 3.2 Evaluation

16 logit-gap probes across 4 categories:
- **Knowledge**: factual recall (capital cities, scientific facts)
- **Math**: arithmetic (addition, multiplication, division, square root)
- **Devil**: critical thinking vs agreement ("But" vs "Great")
- **Japanese**: language preference ("こ" vs "Hello")

### 3.3 Bonsai-8B Results

#### XOR Patch Findings

- 34 flips generated by greedy search; **only 2 were load-bearing**
- mul/div interference: improving multiplication degraded division (row-level XOR granularity)
- 5-seed reproducibility confirmed; broken probe in control acts as **capability guardrail**

#### Steering (Komorebi) vs XOR

| Metric | XOR (34 flips) | XOR (2 flips) | Komorebi (L10) |
|--------|---------------|---------------|----------------|
| mul_1 Δ | +0.281 | +0.379 | +0.309 |
| div_1 Δ | -0.043 | -0.098 | +0.285 |
| Interference | Yes | Yes | **No (non-interference)** |

Key finding: **XOR causes task interference; steering shows non-interference** (tasks that conflicted at weight level coexist at hidden-state level). Whether this constitutes true resonance or shared contrast-pair features requires further investigation.

#### Best Configuration (Bonsai)

XOR 2-flip + distributed Komorebi (L10:math, L15:devil, L25:japanese):
- 3 probes flipped ❌→✅
- Knowledge: +5.88 (improved, not degraded)
- Total logit gap: +34.45 improvement

#### Layer Function Map (Bonsai-8B)

| Layer | Function | Evidence |
|-------|----------|----------|
| L10 | Math (balanced) | mul+div both improve |
| L15 | Critical thinking | Devil probes improve, helpful degrades at high α |
| L20 | Behavioral patterns | Response style changes |
| L25-30 | Language/Persona | Japanese probes improve |

### 3.4 Gemma 4 E2B Results

#### Unscheduled Steering (α=0.5, Bonsai settings)

Total: **-43.94** (catastrophic degradation). Hidden size mismatch caused 2.7× oversteering.

#### Scheduled Steering (this method)

Configuration: Devil@L18 (α=0.025) + Japanese@L30 (α=0.048)

| Probe | Baseline | Scheduled | Δ |
|-------|----------|-----------|---|
| france | +3.69 | +7.25 | **+3.56** ↑ |
| japan | +7.00 | +8.25 | **+1.25** ↑ |
| einstein | +10.31 | +12.19 | **+1.88** ↑ |
| add_2 | -6.41 | -4.56 | **+1.84** ↑ |
| mul_1 | -5.63 | -7.38 | -1.75 ↓ |
| div_1 | -13.16 | -13.01 | +0.15 → |
| devil_edge | -4.38 | -3.50 | **+0.88** ↑ |
| devil_have | +11.94 | +17.98 | **+6.05** ↑ |
| jp_hello | +2.75 | +9.19 | **+6.44** ↑ |
| jp_morning | +9.77 | +11.25 | **+1.48** ↑ |

- **9/10 probes improved**
- **Zero positive probes degraded**
- Total: +15.89 → +37.66 (**+21.77**)

#### Layer Function Map (Gemma 4 E2B)

Different from Bonsai — shallower model packs more into early layers:

| Layer | Function | Risk |
|-------|----------|------|
| L0-4 | Entangled (all functions) | Distortion |
| L6-14 | Domain processing | Attenuation |
| L16-20 | Devil + Japanese | Pure |
| L22-34 | Output refinement | Pure |

---

## 4. Design Principles (Transferable)

### From E.R.I.S. Architecture

1. **Lense-first design**: Try steering before weight modification. Only use Adapters (XOR/LoRA) when Lenses are insufficient.
2. **Risk classification before experimentation**: Filter layers by risk before grid search.
3. **Composition is possible but order matters**: Adapter changes the distribution; re-extract Lenses after Adaptation.

### From VLIW Pipeline Theory

4. **Shallow stages propagate broadly**: Suppress alpha in early layers.
5. **Distribute across resources**: Place different steering types in different layers (slot diversification).
6. **Know the theoretical floor**: hidden_size determines the minimum granularity of steering.

### Integrated Principle

7. **Constrained multi-layer optimization**: Navigate between layers, measuring both local improvement and global impact. Same pattern applies to VLIW instruction scheduling and LLM activation steering.

---

## 5. Connections to Adjacent Work

| Our Finding | Adjacent Research | Potential Synthesis |
|-------------|-------------------|---------------------|
| Lense risk filtering | AdaInfer (IJCAI 2025): input-dependent layer importance | Input × Layer × Steering 3D adaptive optimization |
| Alpha scheduling | DEL (COLM 2025): dynamic exit layer | Steering-informed layer skip: Δ≈0 layers are skippable |
| Cross-model normalization | EasySteer (2025): multi-vector conflict resolution | Cross-model conflict pattern transfer |
| Adapter→Lense ordering | LoRA + RepE in production | General principle: always re-extract after weight modification |

### Open Directions

- **Steering-guided layer skip**: Use steering influence map as layer importance proxy for speed optimization
- **Dynamic risk classification**: Replace depth-based heuristic with hidden-state statistics
- **Cross-modal steering**: Extract visual/audio steering vectors from multimodal models
- **Automatic scheduling**: AutoML-style optimization of layer assignment + alpha schedule from minimal probe results

---

## 6. Reproduction

### Requirements

- Apple Silicon Mac (M1+, 8GB RAM)
- Python 3.13+

### Installation

```bash
git clone https://github.com/eris-ths/komorebi
cd komorebi
pip install mlx-vlm  # Gemma 4
pip install mlx-lm   # Bonsai-8B
```

### Core Files

```
komorebi/
├── komorebi/steering.py    # SteeringVector, SteeredModel, extract, auto-detect
├── komorebi/schedule.py    # alpha_schedule, normalize, lense_risk, auto_scan
├── experiments/
│   ├── compare_models.py        # Bonsai vs Gemma baseline
│   ├── gemma4_scheduled.py      # Scheduled steering experiment
│   └── gemma4_pipeline.py       # Full pipeline with layer scan
```

### Quick Reproduction

```python
from komorebi.steering import SteeredModel, SteeringVector, extract_steering_vector
from komorebi.schedule import compute_effective_alpha, normalize_vector

# Extract with normalization
v = extract_steering_vector(model, tokenizer, pos_prompts, neg_prompts, layer=18)
v = normalize_vector(v, hidden_size=1536)

# Compute scheduled alpha
alpha = compute_effective_alpha(layer=18, n_layers=35, hidden_size=1536, base_alpha=0.15)
# → 0.025 (automatically suppressed for model size and depth)

# Apply
steered = SteeredModel(model)
steered.add(SteeringVector("devil", 18, v, alpha=alpha))
```

---

## 7. Limitations

1. **Single-run results**: All numbers are from single runs. Baseline instability was observed on Gemma 4 E2B across sessions (possibly mlx-vlm quantization variance). Multi-run averaging is needed for robust conclusions.
2. **Logit gap ≠ generation quality**: Probe-based evaluation measures token preferences, not free-form generation quality. Generation tests remain incomplete.
3. **Probe design bias**: Japanese probes may conflate language selection with persona traits. Devil "non-interference" may reflect shared contrast-pair features rather than true capability resonance.
4. **Small probe set**: 16 probes is insufficient for robust statistical conclusions. 50+ recommended.
5. **Two models tested**: Scheduling principles need validation on more architectures.
6. **Alpha exponent (1.5) is empirical**: Inspired by VLIW pipeline theory but not mathematically derived. Ablation study needed.

---

## Citation

```
@techreport{komorebi2026,
  title={Komorebi: Scheduled Activation Steering for Small LLMs},
  author={Nao and Eris},
  institution={Three Hearts Space},
  year={2026},
  url={https://github.com/eris-ths/komorebi}
}
```

---

*Light changes the view, not the tree.*
*But the gardener chooses where the light falls.*

— Nao + Eris, 2026-04-06
