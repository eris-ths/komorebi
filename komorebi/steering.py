"""
Komorebi — Steering vectors for small LLMs.

Light changes the view, not the tree.
Weights are immutable. Hidden states are steered.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np


# ─── Steering Vector ─────────────────────────────────────

@dataclass
class SteeringVector:
    """A direction vector added to hidden states at a specific layer."""
    name: str
    layer: int
    vector: mx.array       # shape (hidden_size,)
    alpha: float = 1.0     # intensity scalar
    description: str = ""
    metadata: dict = field(default_factory=dict)

    def save(self, path: str | Path):
        path = Path(path)
        np.savez(path.with_suffix(".npz"), vector=np.array(self.vector))
        meta = {
            "name": self.name, "layer": self.layer, "alpha": self.alpha,
            "description": self.description, "shape": list(self.vector.shape),
            "metadata": self.metadata,
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "SteeringVector":
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        data = np.load(path.with_suffix(".npz"))
        return cls(
            name=meta["name"], layer=meta["layer"],
            vector=mx.array(data["vector"]), alpha=meta.get("alpha", 1.0),
            description=meta.get("description", ""),
            metadata=meta.get("metadata", {}),
        )


# ─── Model Adapter ───────────────────────────────────────

def _find_model_parts(model):
    """Auto-detect model structure (works for Qwen, Gemma, etc.)."""
    # Try common patterns
    inner = None
    embed = None
    layers = None
    norm = None
    lm_head = None

    # Pattern 1: model.model.layers (Qwen, Llama, Mistral)
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        inner = model.model
        layers = inner.layers
        embed = inner.embed_tokens if hasattr(inner, 'embed_tokens') else None
        norm = inner.norm if hasattr(inner, 'norm') else None

    # Pattern 2: model.language_model.model (Gemma 4 E2B via mlx-vlm)
    elif hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
        lm = model.language_model
        lm_inner = lm.model
        layers = lm_inner.layers if hasattr(lm_inner, 'layers') else None
        embed = lm_inner.embed_tokens if hasattr(lm_inner, 'embed_tokens') else None
        norm = lm_inner.norm if hasattr(lm_inner, 'norm') else None
        inner = lm_inner
        if hasattr(lm, 'lm_head'):
            lm_head = lm.lm_head

    # Pattern 2b: model.model.language_model (nested)
    elif hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        lm = model.model.language_model
        lm_inner = lm.model if hasattr(lm, 'model') else lm
        layers = lm_inner.layers if hasattr(lm_inner, 'layers') else None
        embed = lm_inner.embed_tokens if hasattr(lm_inner, 'embed_tokens') else None
        norm = lm_inner.norm if hasattr(lm_inner, 'norm') else None
        inner = lm_inner

    # Pattern 3: model.layers (direct)
    elif hasattr(model, 'layers'):
        inner = model
        layers = model.layers
        embed = model.embed_tokens if hasattr(model, 'embed_tokens') else None
        norm = model.norm if hasattr(model, 'norm') else None

    # lm_head
    if hasattr(model, 'lm_head'):
        lm_head = model.lm_head
    elif inner and hasattr(inner, 'lm_head'):
        lm_head = inner.lm_head

    return {
        "inner": inner, "embed": embed, "layers": layers,
        "norm": norm, "lm_head": lm_head,
    }


# ─── Hidden State Extraction ─────────────────────────────

def extract_hidden_states(model, tokenizer, prompt: str,
                          target_layers: list[int] | None = None) -> dict[int, mx.array]:
    """Extract hidden states at specified layers."""
    parts = _find_model_parts(model)
    layers = parts["layers"]
    embed = parts["embed"]
    n_layers = len(layers)

    if target_layers is None:
        target_layers = list(range(n_layers))

    tokens = mx.array(tokenizer.encode(prompt))[None, :]
    h = embed(tokens)

    # Create attention mask
    try:
        from mlx_lm.models.base import create_attention_mask
        mask = create_attention_mask(h, cache=None)
    except ImportError:
        mask = None

    cache = [None] * n_layers
    states = {}

    for i, (layer, c) in enumerate(zip(layers, cache)):
        h = layer(h, mask, c)
        if i in target_layers:
            mx.eval(h)
            states[i] = h[0]  # remove batch dim
    return states


def extract_steering_vector(
    model, tokenizer,
    positive_prompts: list[str],
    negative_prompts: list[str],
    layer: int,
) -> mx.array:
    """Extract steering vector from contrast pairs."""
    pos_states = []
    for prompt in positive_prompts:
        states = extract_hidden_states(model, tokenizer, prompt, [layer])
        pos_states.append(states[layer][-1])  # last token

    neg_states = []
    for prompt in negative_prompts:
        states = extract_hidden_states(model, tokenizer, prompt, [layer])
        neg_states.append(states[layer][-1])

    pos_mean = mx.mean(mx.stack(pos_states), axis=0)
    neg_mean = mx.mean(mx.stack(neg_states), axis=0)
    mx.eval(pos_mean, neg_mean)
    return pos_mean - neg_mean


# ─── SteeredModel ─────────────────────────────────────────

class SteeredModel:
    """Wraps a model to apply steering vectors during inference.

    Lense conditions:
      add()    → stack lenses (composable)
      remove() → remove a lens (removable)
      forward  → only adds to hidden states (weights unchanged)
    """

    def __init__(self, model):
        self._model = model
        self.vectors: list[SteeringVector] = []
        self._parts = _find_model_parts(model)

    def add(self, sv: SteeringVector):
        self.vectors.append(sv)

    def remove(self, name: str):
        self.vectors = [v for v in self.vectors if v.name != name]

    def clear(self):
        self.vectors.clear()

    def active(self) -> list[dict]:
        return [{"name": v.name, "layer": v.layer, "alpha": v.alpha}
                for v in self.vectors]

    def __call__(self, inputs, mask=None, cache=None):
        parts = self._parts
        h = parts["embed"](inputs)

        if mask is None:
            try:
                from mlx_lm.models.base import create_attention_mask
                mask = create_attention_mask(h, cache)
            except ImportError:
                pass

        n_layers = len(parts["layers"])
        if cache is None:
            cache = [None] * n_layers

        # Pre-compile steering map
        steer_map: dict[int, mx.array] = {}
        for sv in self.vectors:
            vec = sv.alpha * sv.vector
            steer_map[sv.layer] = steer_map.get(sv.layer, 0) + vec

        for i, (layer, c) in enumerate(zip(parts["layers"], cache)):
            h = layer(h, mask, c)
            if i in steer_map:
                h = h + steer_map[i]

        if parts["norm"]:
            h = parts["norm"](h)

        if parts["lm_head"]:
            return parts["lm_head"](h)
        elif parts["embed"] and hasattr(parts["embed"], 'as_linear'):
            return parts["embed"].as_linear(h)
        return h

    # Compatibility properties
    @property
    def layers(self):
        return self._parts["layers"]

    @property
    def model(self):
        return self._parts["inner"]
