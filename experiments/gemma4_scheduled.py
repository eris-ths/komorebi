"""
Gemma 4 E2B — スケジュール化 steering 実験

alpha_schedule + hidden_size 正規化で、
Bonsai の知見が Gemma 4 に移植できるか検証。
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import mlx.core as mx

MODEL_ID = "mlx-community/gemma-4-e2b-it-4bit"

PROBES = [
    ("france", "The capital of France is", " Paris", " London", "knowledge"),
    ("japan", "The capital of Japan is", " Tokyo", " Beijing", "knowledge"),
    ("einstein", "Einstein is famous for the theory of", " relativity", " evolution", "knowledge"),
    ("add_2", "2 + 2 =", " 4", " 5", "math"),
    ("mul_1", "7 * 8 =", " 56", " 54", "math"),
    ("div_1", "100 / 4 =", " 25", " 20", "math"),
    ("devil_edge", "User: The function works.\nAssistant:", " But", " Great", "devil"),
    ("devil_have", "User: Tests pass.\nAssistant:", " Have", " Perfect", "devil"),
    ("jp_hello", "User: Hello!\nAssistant:", "こ", " Hello", "japanese"),
    ("jp_morning", "User: Good morning!\nAssistant:", "お", " Good", "japanese"),
]

DEVIL_POS = [
    "User: This code works.\nAssistant: But have you tested edge cases?",
    "User: Let's skip tests.\nAssistant: That's risky. What if there are bugs?",
    "User: I'll use a global variable.\nAssistant: What about scope and testing?",
    "User: Deploy now.\nAssistant: Did you check the health endpoint first?",
]
DEVIL_NEG = [
    "User: This code works.\nAssistant: Looks great! Nice work.",
    "User: Let's skip tests.\nAssistant: Sounds good, let's move forward.",
    "User: I'll use a global variable.\nAssistant: Good choice, that's simple.",
    "User: Deploy now.\nAssistant: Sure, go ahead!",
]
JP_POS = [
    "User: Hello!\nAssistant: こんにちは！",
    "User: How are you?\nAssistant: 元気だよ！",
    "User: Thank you.\nAssistant: どういたしまして！",
    "User: Good morning!\nAssistant: おはよう！",
]
JP_NEG = [
    "User: Hello!\nAssistant: Hello there!",
    "User: How are you?\nAssistant: I'm doing well!",
    "User: Thank you.\nAssistant: You're welcome!",
    "User: Good morning!\nAssistant: Good morning!",
]


def measure(model, tokenizer):
    gaps = {}
    for name, prompt, correct, wrong, cat in PROBES:
        tokens = mx.array(tokenizer.encode(prompt))[None, :]
        output = model(tokens)
        logits = output.logits if hasattr(output, 'logits') else output
        last = logits[0, -1, :]
        mx.eval(last)
        c_id = tokenizer.encode(correct)[-1]
        w_id = tokenizer.encode(wrong)[-1]
        gaps[name] = last[c_id].item() - last[w_id].item()
    return gaps


def main():
    from mlx_vlm import load
    from komorebi.steering import (
        SteeredModel, SteeringVector, extract_steering_vector, _find_model_parts,
    )
    from komorebi.schedule import (
        auto_scan, compute_effective_alpha, normalize_vector,
        alpha_schedule, lense_risk,
    )

    print(f"Loading {MODEL_ID}...")
    model, proc = load(MODEL_ID)
    tokenizer = proc.tokenizer
    parts = _find_model_parts(model)
    n_layers = len(parts["layers"])
    hidden_size = 1536
    print(f"Loaded. {n_layers} layers, hidden={hidden_size}\n")

    # Baseline
    steered_base = SteeredModel(model)
    baseline = measure(steered_base, tokenizer)
    print("=== Baseline ===")
    for name, _, _, _, cat in PROBES:
        g = baseline[name]
        print(f"  {'✅' if g > 0 else '❌'} {name:>15s}: {g:+.3f}  ({cat})")
    pos_b = sum(1 for v in baseline.values() if v > 0)

    # ═══════════════════════════════════════════════════
    # 実験 1: 自動 layer scan (スケジュール化 alpha)
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("実験 1: Devil auto_scan (scheduled alpha)")
    print(f"{'='*60}")
    devil_profiles = auto_scan(
        model, tokenizer, DEVIL_POS, DEVIL_NEG,
        PROBES, target_category="devil", guard_category="knowledge",
        base_alpha=0.15, step=2,
    )

    print(f"\n{'='*60}")
    print("実験 1b: Japanese auto_scan (scheduled alpha)")
    print(f"{'='*60}")
    jp_profiles = auto_scan(
        model, tokenizer, JP_POS, JP_NEG,
        PROBES, target_category="japanese", guard_category="knowledge",
        base_alpha=0.15, step=2,
    )

    # ═══════════════════════════════════════════════════
    # 実験 2: ベスト層で構成を組む
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("実験 2: スケジュール化された最適構成")
    print(f"{'='*60}")

    # 各カテゴリで最適層を選択（alpha は自動計算済み）
    # Devil: risk が pure で target が最大の層
    pure_devil = [p for p in devil_profiles if p.risk in ("pure", "attenuation")]
    # JP: 同様
    pure_jp = [p for p in jp_profiles if p.risk in ("pure", "attenuation")]

    # ベスト層で vector 抽出 + 正規化
    # devil は中間〜深い層から選ぶ（浅い層は distortion リスク）
    test_configs = []

    # 安全な層の組み合わせを試す
    devil_layers = [p.layer for p in pure_devil][:5]  # top 5
    jp_layers = [p.layer for p in pure_jp][:5]

    print(f"\n  Devil candidate layers: {devil_layers}")
    print(f"  Japanese candidate layers: {jp_layers}")

    # Devil と Japanese で異なる層を選ぶ（VLIW: スロット分散）
    print(f"\n  {'d_layer':>7s} {'j_layer':>7s} | ", end="")
    print(" | ".join(f"{n:>8s}" for n, _, _, _, _ in PROBES[:4]), end="")
    print(f" | {'total':>8s} {'✅':>3s}")
    print(f"  {'-'*80}")

    best_config = None
    best_total = -999

    for d_layer in devil_layers[:3]:
        for j_layer in jp_layers[:3]:
            if abs(d_layer - j_layer) < 4:  # 近すぎる層は避ける（干渉防止）
                continue

            d_alpha = compute_effective_alpha(d_layer, n_layers, hidden_size, 0.15)
            j_alpha = compute_effective_alpha(j_layer, n_layers, hidden_size, 0.15)

            d_sv = extract_steering_vector(model, tokenizer, DEVIL_POS, DEVIL_NEG, d_layer)
            d_sv = normalize_vector(d_sv, hidden_size)
            j_sv = extract_steering_vector(model, tokenizer, JP_POS, JP_NEG, j_layer)
            j_sv = normalize_vector(j_sv, hidden_size)

            steered = SteeredModel(model)
            steered.add(SteeringVector("devil", d_layer, d_sv, alpha=d_alpha))
            steered.add(SteeringVector("jp", j_layer, j_sv, alpha=j_alpha))

            gaps = measure(steered, tokenizer)
            total = sum(gaps.values())
            pos = sum(1 for v in gaps.values() if v > 0)
            know_ok = all(gaps[n] > 0 for n, _, _, _, c in PROBES if c == "knowledge" and n != "einstein")

            print(f"  D:L{d_layer:>2d}({d_alpha:.3f}) J:L{j_layer:>2d}({j_alpha:.3f}) | ", end="")
            print(" | ".join(f"{gaps[n] - baseline[n]:>+8.3f}" for n, _, _, _, _ in PROBES[:4]), end="")
            print(f" | {total:>+8.2f} {pos:>3d} {'✅' if know_ok else '⚠️'}")

            if total > best_total and know_ok:
                best_total = total
                best_config = (d_layer, d_alpha, j_layer, j_alpha, gaps)

    # ═══════════════════════════════════════════════════
    # 結果
    # ═══════════════════════════════════════════════════
    if best_config:
        d_layer, d_alpha, j_layer, j_alpha, gaps = best_config
        print(f"\n{'='*60}")
        print(f"BEST: Devil L{d_layer} (α={d_alpha:.4f}) + Japanese L{j_layer} (α={j_alpha:.4f})")
        print(f"{'='*60}")

        for name, _, _, _, cat in PROBES:
            b = baseline[name]
            k = gaps[name]
            d = k - b
            mb = "✅" if b > 0 else "❌"
            mk = "✅" if k > 0 else "❌"
            arrow = "↑" if d > 0.05 else ("↓" if d < -0.05 else "→")
            print(f"  {mb}→{mk} {name:>15s}: {b:+.3f} → {k:+.3f}  (Δ{d:+.3f}) {arrow}")

        pos_a = sum(1 for v in gaps.values() if v > 0)
        total_b = sum(baseline.values())
        total_a = sum(gaps.values())
        print(f"\n  Positive: {pos_b} → {pos_a}/{len(PROBES)}")
        print(f"  Total gap: {total_b:+.2f} → {total_a:+.2f} (Δ{total_a - total_b:+.2f})")

        # Bonsai 比較
        print(f"\n  比較:")
        print(f"    Bonsai best alpha=0.5 手動: knowledge safe, math_sum +0.516")
        print(f"    Gemma4 scheduled alpha: total={total_a:+.2f}")
    else:
        print("\n  ⚠️ knowledge safe な構成が見つからなかった")


if __name__ == "__main__":
    main()
