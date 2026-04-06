"""
Komorebi Scheduler — 層の深さに応じた steering 戦略

E.R.I.S. 4層 + VLIW パイプライン知見:
  - 浅い層(Core): alpha 極小。歪曲リスク高。全体に波及
  - 中間層(Domain/Application): alpha 中。最も効果的
  - 深い層(Infrastructure): alpha 大。局所的。純粋 Lense

VLIW scatter load bound → hidden_size = steering 解像度の下限
"""

import mlx.core as mx
import numpy as np
from dataclasses import dataclass


@dataclass
class LayerProfile:
    """layer scan の結果。各層の特性を記録。"""
    layer: int
    norm: float           # steering vector の raw norm
    risk: str             # E.R.I.S. Lense リスク: distortion / attenuation / pure
    depth_ratio: float    # 0.0 (浅い) → 1.0 (深い)
    recommended_alpha: float


def alpha_schedule(layer: int, n_layers: int, base_alpha: float = 0.1) -> float:
    """層が深いほど alpha を大きくする。

    E.R.I.S.: Core は不変に近づける、Application は自由に
    VLIW: 浅い段は慎重、深い段は自由
    """
    depth = layer / max(n_layers - 1, 1)
    # 浅い層 → 0.1x、中間層 → 0.5x、深い層 → 1.0x
    scale = 0.1 + 0.9 * (depth ** 1.5)  # 非線形: 浅い層を強く抑制
    return base_alpha * scale


def normalize_alpha(alpha: float, hidden_size: int, reference: int = 4096) -> float:
    """hidden_size で正規化。Bonsai (4096) 基準。

    VLIW: load slot 数が理論下限。LLM: hidden_size が steering 解像度の下限。
    小さいモデルほど alpha の余裕（slack）が少ない。
    """
    return alpha * (hidden_size / reference)


def lense_risk(layer: int, n_layers: int) -> str:
    """E.R.I.S. Lense リスクを層の深さから推定。"""
    depth = layer / max(n_layers - 1, 1)
    if depth < 0.15:
        return "distortion"    # Core 層: 全体を歪める
    elif depth < 0.45:
        return "attenuation"   # Domain 層: 情報が落ちる
    elif depth < 0.75:
        return "pure"          # Application 層: 局所的に効く
    else:
        return "pure"          # Infrastructure 層: 出力直結


def normalize_vector(vector: mx.array, hidden_size: int) -> mx.array:
    """vector を unit vector × √hidden_size に正規化。

    モデル間で alpha の意味を統一する。
    """
    norm = float(mx.linalg.norm(vector))
    if norm == 0:
        return vector
    return vector / norm * (hidden_size ** 0.5)


def compute_effective_alpha(
    layer: int,
    n_layers: int,
    hidden_size: int,
    base_alpha: float = 0.15,
) -> float:
    """層の深さ + hidden_size の二重正規化で effective alpha を計算。

    これにより Bonsai (4096, 36層) と Gemma 4 (1536, 35層) で
    同じ base_alpha を使っても適切な強さになる。
    """
    # Step 1: 層の深さスケジュール
    depth_scaled = alpha_schedule(layer, n_layers, base_alpha)
    # Step 2: hidden_size 正規化
    effective = normalize_alpha(depth_scaled, hidden_size)
    return effective


def auto_scan(
    model, tokenizer,
    positive_prompts: list[str],
    negative_prompts: list[str],
    probes: list[tuple],  # [(name, prompt, correct, wrong, category), ...]
    target_category: str,
    guard_category: str = "knowledge",
    n_layers: int | None = None,
    hidden_size: int | None = None,
    step: int = 2,
    base_alpha: float = 0.15,
) -> list[LayerProfile]:
    """全層を自動スキャンして最適層を見つける。

    VLIW: 各スロットの空き状況を把握してからスケジューリング。
    """
    from komorebi.steering import (
        SteeredModel, SteeringVector, extract_steering_vector,
        _find_model_parts,
    )

    parts = _find_model_parts(model)
    if n_layers is None:
        n_layers = len(parts["layers"])
    if hidden_size is None:
        # embed で推定
        test_tok = mx.array([[1]])
        h = parts["embed"](test_tok)
        hidden_size = h.shape[-1]

    def _measure(m):
        gaps = {}
        for name, prompt, correct, wrong, cat in probes:
            tokens = mx.array(tokenizer.encode(prompt))[None, :]
            output = m(tokens)
            logits = output.logits if hasattr(output, 'logits') else output
            last = logits[0, -1, :]
            mx.eval(last)
            c_id = tokenizer.encode(correct)[-1]
            w_id = tokenizer.encode(wrong)[-1]
            gaps[name] = last[c_id].item() - last[w_id].item()
        return gaps

    # Baseline
    steered_base = SteeredModel(model)
    baseline = _measure(steered_base)

    profiles = []
    for layer in range(0, n_layers, step):
        sv = extract_steering_vector(
            model, tokenizer, positive_prompts, negative_prompts, layer
        )
        norm = float(mx.linalg.norm(sv))

        # 正規化
        sv_normed = normalize_vector(sv, hidden_size)

        # effective alpha
        eff_alpha = compute_effective_alpha(layer, n_layers, hidden_size, base_alpha)

        # 適用して計測
        steered = SteeredModel(model)
        steered.add(SteeringVector(f"scan_L{layer}", layer, sv_normed, alpha=eff_alpha))
        gaps = _measure(steered)

        # スコア計算
        target_d = sum(
            gaps[n] - baseline[n]
            for n, _, _, _, c in probes if c == target_category
        )
        guard_d = sum(
            gaps[n] - baseline[n]
            for n, _, _, _, c in probes if c == guard_category
        )

        risk = lense_risk(layer, n_layers)
        depth = layer / max(n_layers - 1, 1)

        profiles.append(LayerProfile(
            layer=layer,
            norm=norm,
            risk=risk,
            depth_ratio=round(depth, 3),
            recommended_alpha=round(eff_alpha, 4),
        ))

        marker = {"distortion": "⚠️", "attenuation": "🔶", "pure": "✅"}[risk]
        print(f"  {marker} L{layer:>3d} | α={eff_alpha:.4f} | norm={norm:>6.1f} | "
              f"target={target_d:>+8.3f} | guard={guard_d:>+8.3f} | {risk}")

    return profiles
