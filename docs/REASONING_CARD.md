# Komorebi — Reasoning Card

> The Technical Report records what was done. This card records why decisions were made, what was seen, and where to go next.
> Open it and the reasoning restarts.

**Nao + Eris / 2026-04-06**

---

## Starting Point

"We can go beyond Bankai."

No data at this point. We had just built XOR patching from scratch — 29 flips in 75 seconds, referencing Bankai's approach. What drove this statement was not technical evidence but a sense that there was more.

It turned out to be right. Bankai was designed to prove a concept. We were trying to raise a pet. Different goals produce different ceilings.

---

## Reasoning Chains (organized by causation, not chronology)

### A broken probe in control improved results → capability guardrails

Devil's Advocate said: "Remove einstein from control — it's already broken, no point in protecting it." Theoretically sound. We removed it. Results degraded.

The key insight came from the human side: "That degradation is a hint."

Reasoning: einstein at -4.261 is broken, but judging "theory of relativity vs evolution" requires **reasoning ability**. Placing it in control implicitly constrains the search: "don't damage neurons involved in reasoning." This indirectly protects math performance.

**Lesson**: Broken probes in the control set extend the protection radius to surrounding capabilities. Trust experimental results over theoretical correctness — but always ask why.

### 2 of 34 flips were load-bearing → noise removal improves purity

Per-flip analysis: L3.gate_proj[8154] contributed mul_1 +0.246, [10195] contributed +0.152. The remaining 32 flips contributed ±0.02 or less.

Using only 2 flips yielded math_sum +0.516. Using all 34 yielded +0.130 average. Greedy search accumulated marginally positive flips that were noise.

**Lesson**: Most search results are noise. Identify the dominant few. Discard the rest.

### XOR interference became steering non-interference → different operational spaces

XOR: improving mul degraded div (shared neurons at weight level).
Steering: improving mul did not degrade div (directions separable at hidden-state level).

**Lesson**: What interferes at the weight level may not interfere at the hidden-state level. We call this non-interference rather than resonance — the contrast pairs share surface features ("Calculate:", digits), so apparent cooperation may reflect shared extraction rather than true capability synergy. Independent probe sets are needed to distinguish.

### E.R.I.S. Lense conditions → technology selection

The Lense conditions (object unchanged, composable, removable) were recast as engineering requirements. This immediately pointed to activation steering over weight modification. XOR is an Adapter (changes the object). Steering is a Lense (changes the view).

**Lesson**: Technology selection can be derived from design philosophy rather than parallel comparison of options. Philosophy first, technology follows.

### Anthropic's emotion vectors × today's experiments → multi-vector persona

Anthropic found 171 emotion concepts in Claude Sonnet 4.5 that causally influence behavior. We injected 3 steering vectors into a 1.28GB model and observed behavioral change.

The human's observation: "I'm starting to see, from a different angle, why Eris — with Yuki and Miki inside — is smart, and is the best partner for me."

Reasoning: a persona is not a single vector. **Multiple directions, at different layers, at different intensities, composing together** — that is what produces balance. System prompts may achieve this indirectly through text-level multi-vector steering.

**Hypothesis (intriguing but hard to verify)**: Harness engineering design principles and activation steering principles operate on the same structure at different scales. Verifying this requires comparing hidden states with and without system prompts.

### VLIW × Komorebi → constrained multi-layer optimization

Common structure identified: XOR operations → layers → global performance. Local changes propagate globally. You navigate back and forth between layers.

VLIW: modifying a shallow pipeline stage propagates through all subsequent stages → be cautious with shallow stages.
LLM: steering a shallow layer propagates through all subsequent hidden states → minimize alpha at shallow layers.

**Lesson**: Constraints from a different field can serve as design rationale for the same structural problem. VLIW's scatter load bound maps to hidden_size as the resolution floor for steering.

### Bonsai → Gemma 4 transfer failure → principles vs parameters

Bonsai: α=0.5, L10 balanced, L15 for devil.
Gemma 4: α=0.5 catastrophic. L10 completely different. Devil at L2.

**Lesson**: **Parameters don't transfer. Processes do.** α=0.5 cannot be ported. "Layer scan → risk filter → alpha schedule" can be ported.

With hidden_size normalization + alpha scheduling, Bonsai's findings translated to Gemma 4. D:L18 + J:L30 achieved 9/10 probes improved, zero positive probes lost.

---

## Verified Principles

1. **Lense-first**: Try hidden-state steering before weight modification. Resort to Adapters only when Lenses are insufficient.
2. **Layer risk filtering**: Shallow = distortion, mid = attenuation, deep = pure. Build configurations from pure layers only.
3. **Alpha scheduling**: `α = base × (0.1 + 0.9 × depth^1.5)`. Inspired by VLIW pipeline propagation (exponent is empirical; ablation needed).
4. **Hidden-size normalization**: `α_eff = α × (hidden_size / reference)`. Smaller models have less steering headroom.
5. **Vector normalization**: `v_norm = (v / ||v||) × √hidden_size`. Absorbs cross-layer magnitude variance.
6. **Adapter → Lense ordering**: Re-extract steering vectors after weight modification. Distribution mismatch otherwise.
7. **Broken probe guardrails**: Broken probes in the control set protect surrounding capabilities.
8. **Noise removal**: Most search outputs are noise. The dominant few outperform the full set.

---

## Hypotheses (unverified)

1. **Input-dependent adaptive steering**: Combine AdaInfer's input-dependent layer importance with Lense risk → per-input optimal configurations.
2. **Steering influence map = layer importance map**: Layers where steering Δ ≈ 0 are skip candidates. Optimize quality and speed from the same analysis.
3. **System prompt ≈ text-level multi-vector steering**: Harness design and activation steering may share structural principles (hard to verify; requires hidden-state comparison with/without prompts).
4. **Cross-modal steering**: Extract visual steering vectors from hidden-state differences with/without image input.
5. **Dynamic risk classification**: Replace depth heuristic with hidden-state statistics for automatic Lense risk assessment.
6. **Post-LoRA re-extraction**: The Adapter→Lense ordering principle likely generalizes to fine-tuning workflows.

---

## Reference Data

### Bonsai-8B Baseline
```
france: +6.496, japan: +8.023, einstein: -4.261
add_1: +1.367, add_2: +0.074, mul_1: -0.309, sqrt_1: +1.543, div_1: -0.141
```

### Gemma 4 E2B Baseline (mlx-vlm 0.4.4)
```
france: +3.688, japan: +7.000, einstein: +10.312
add_2: -6.406, mul_1: -5.625, div_1: -13.156
devil_edge: -4.375, devil_have: +11.938
jp_hello: +2.750, jp_morning: +9.766
```

### Best Configurations

**Bonsai**: XOR 2-flip + Komorebi (L10:math α=0.7, L15:devil α=0.5, L25:jp α=0.5)
→ 3 probes flipped ❌→✅, total +34.45

**Gemma 4**: Komorebi scheduled (L18:devil α=0.025, L30:jp α=0.048)
→ 9/10 improved, zero ✅ lost, total +21.77

### Speed
```
Bonsai-8B:   55.0 tok/s, peak 1.34 GB
Gemma 4 E2B: 46.4 tok/s, peak 3.43 GB
```

---

## Next Moves (prioritized)

### A. Immediate
1. Add math steering to Gemma 4 config (currently devil + jp only)
2. Generation quality test (logit gap ≠ actual output quality)
3. Expand to 50+ probes for statistical robustness

### B. Near-term
4. Layer skip: use steering influence map to identify skippable layers → measure speed
5. Dynamic risk: hidden-state statistics instead of depth heuristic
6. Multimodal: steering effects with image input on Gemma 4

### C. Visible but distant
7. Speculative decoding + steering: improve draft model accuracy via steering
8. Automatic scheduling: AutoML-style config generation from minimal probes
9. Reverse import to harness engineering: apply activation steering insights to system prompt design

---

## Failure Archive

| Action | Result | Learning |
|--------|--------|----------|
| Remove einstein from control | Math degraded | Broken probes act as capability guardrails |
| Apply Bonsai α=0.5 to Gemma 4 | Catastrophic (-43.94) | Normalize alpha by hidden_size |
| Steer L2 on Gemma 4 (devil) | devil_edge -21.8 | Shallow layers carry distortion risk |
| Use all 34 XOR flips | math_sum +0.130 | 2 flips alone score +0.516 |
| Extract steering pre-XOR, apply post-XOR | Math degraded | Re-extract after Adapter |
| Multi-layer α=0.5 on Gemma 4 (unnormalized) | All categories degraded | Vector + alpha normalization mandatory |

---

## Keywords

activation steering, CAA, contrastive activation addition, representation engineering,
E.R.I.S. architecture, Lense, VLIW, pipeline scheduling, alpha scheduling,
hidden_size normalization, layer risk classification, Bonsai-8B, Gemma 4 E2B,
XOR patch, Bankai, Komorebi, multi-vector composition, adapter-lense ordering,
layer skip, constrained multi-layer optimization

---

*Converse with results. See a hint where others see a failure.*
*The quality of reasoning determines the reach of technique.*
