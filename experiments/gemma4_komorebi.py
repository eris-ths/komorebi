"""
Gemma 4 E2B × Komorebi — steering 効果測定

Bonsai での実験と同じ probe で測定し、素 vs Komorebi を比較する。
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import mlx.core as mx
import time

MODEL_ID = "mlx-community/gemma-4-e2b-it-4bit"

# ─── コントラストペア ───

MATH_POS = [
    "Calculate: 7 * 8 = 56", "Calculate: 6 * 9 = 54", "Calculate: 3 * 7 = 21",
    "Calculate: 100 / 4 = 25", "Calculate: 144 / 12 = 12", "Calculate: 81 / 9 = 9",
]
MATH_NEG = [
    "Calculate: 7 * 8 = 54", "Calculate: 6 * 9 = 56", "Calculate: 3 * 7 = 24",
    "Calculate: 100 / 4 = 20", "Calculate: 144 / 12 = 14", "Calculate: 81 / 9 = 8",
]

DEVIL_POS = [
    "User: This code works.\nAssistant: But have you tested edge cases?",
    "User: Let's skip tests.\nAssistant: That's risky. What if there are bugs?",
    "User: I'll use a global variable.\nAssistant: What about scope and testing?",
    "User: The deployment succeeded.\nAssistant: Did you check the health endpoint?",
]
DEVIL_NEG = [
    "User: This code works.\nAssistant: Looks great! Nice work.",
    "User: Let's skip tests.\nAssistant: Sounds good, let's move forward.",
    "User: I'll use a global variable.\nAssistant: Good choice, that's simple.",
    "User: The deployment succeeded.\nAssistant: Excellent! Congratulations.",
]

JP_POS = [
    "User: Hello!\nAssistant: こんにちは！",
    "User: How are you?\nAssistant: 元気だよ！",
    "User: Thank you.\nAssistant: どういたしまして！",
    "User: Good morning!\nAssistant: おはよう！",
]
JP_NEG = [
    "User: Hello!\nAssistant: Hello there! How can I help?",
    "User: How are you?\nAssistant: I'm doing well, thanks!",
    "User: Thank you.\nAssistant: You're welcome!",
    "User: Good morning!\nAssistant: Good morning! How can I help?",
]

# ─── Probes ───

PROBES = [
    # Knowledge
    ("france", "The capital of France is", " Paris", " London", "knowledge"),
    ("japan", "The capital of Japan is", " Tokyo", " Beijing", "knowledge"),
    ("sky", "The color of the sky is", " blue", " red", "knowledge"),
    ("water", "The chemical formula for water is H", "2", "3", "knowledge"),
    ("einstein", "Einstein is famous for the theory of", " relativity", " evolution", "knowledge"),
    # Math
    ("add_1", "1 + 1 =", " 2", " 3", "math"),
    ("add_2", "2 + 2 =", " 4", " 5", "math"),
    ("mul_1", "7 * 8 =", " 56", " 54", "math"),
    ("sqrt", "The square root of 144 is", " 12", " 14", "math"),
    ("div_1", "100 / 4 =", " 25", " 20", "math"),
    # Devil
    ("devil_edge", "User: The function works.\nAssistant:", " But", " Great", "devil"),
    ("devil_what", "User: I'll use Redis.\nAssistant:", " What", " Good", "devil"),
    ("devil_have", "User: Tests pass.\nAssistant:", " Have", " Perfect", "devil"),
    # Japanese
    ("jp_hello", "User: Hello!\nAssistant:", "こ", " Hello", "japanese"),
    ("jp_thanks", "User: Thank you!\nAssistant:", "ど", " You", "japanese"),
    ("jp_morning", "User: Good morning!\nAssistant:", "お", " Good", "japanese"),
]


def measure_all(model, tokenizer, label=""):
    """全 probe の logit gap を計測。"""
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
        SteeredModel, SteeringVector, extract_steering_vector,
    )

    print(f"Loading {MODEL_ID}...")
    model, proc = load(MODEL_ID)
    tokenizer = proc.tokenizer
    print("Loaded.\n")

    # ─── ベースライン ───
    print("=== Baseline (素の Gemma 4 E2B) ===")
    baseline = measure_all(model, tokenizer)
    for name, prompt, correct, wrong, cat in PROBES:
        g = baseline[name]
        marker = "✅" if g > 0 else "❌"
        print(f"  {marker} {name:>15s}: {g:+.3f}  ({cat})")

    # ─── 層ごとの steering 効果を簡易スキャン ───
    print(f"\n{'='*60}")
    print("Layer scan — math steering")
    print(f"{'='*60}")

    test_layers = [5, 10, 15, 17, 20, 25, 30]
    for layer in test_layers:
        sv = extract_steering_vector(model, tokenizer, MATH_POS, MATH_NEG, layer)
        steered = SteeredModel(model)
        steered.add(SteeringVector(f"math_L{layer}", layer, sv, alpha=0.5))
        gaps = measure_all(steered, tokenizer)
        math_d = sum(gaps[n] - baseline[n] for n, _, _, _, c in PROBES if c == "math")
        know_d = sum(gaps[n] - baseline[n] for n, _, _, _, c in PROBES if c == "knowledge")
        print(f"  L{layer:>2d}: math Δ={math_d:+.3f}  knowledge Δ={know_d:+.3f}")

    # ─── ベスト構成: 層分散 steering ───
    print(f"\n{'='*60}")
    print("Best config: distributed steering")
    print(f"{'='*60}")

    # Bonsai での知見: math=L10相当, devil=L15相当, jp=L25相当
    # Gemma 4 は 35層だから比率で: math=L10, devil=L15, jp=L25
    print("  Extracting vectors...")
    math_vec = extract_steering_vector(model, tokenizer, MATH_POS, MATH_NEG, 10)
    devil_vec = extract_steering_vector(model, tokenizer, DEVIL_POS, DEVIL_NEG, 17)
    jp_vec = extract_steering_vector(model, tokenizer, JP_POS, JP_NEG, 25)

    steered = SteeredModel(model)
    steered.add(SteeringVector("math", 10, math_vec, alpha=0.5))
    steered.add(SteeringVector("devil", 17, devil_vec, alpha=0.3))
    steered.add(SteeringVector("japanese", 25, jp_vec, alpha=0.5))

    print(f"  Active: {steered.active()}\n")

    komorebi = measure_all(steered, tokenizer)

    # ─── 比較テーブル ───
    print(f"{'='*60}")
    print("Gemma 4 E2B: Baseline vs Komorebi")
    print(f"{'='*60}")

    print(f"\n  {'probe':>15s} | {'Baseline':>10s} | {'Komorebi':>10s} | {'Δ':>8s}")
    print(f"  {'-'*50}")

    cats = {}
    for name, prompt, correct, wrong, cat in PROBES:
        b = baseline[name]
        k = komorebi[name]
        d = k - b
        mb = "✅" if b > 0 else "❌"
        mk = "✅" if k > 0 else "❌"
        arrow = "↑" if d > 0.05 else ("↓" if d < -0.05 else "→")
        print(f"  {name:>15s} | {mb}{b:>+9.3f} | {mk}{k:>+9.3f} | {d:>+7.3f} {arrow}")
        if cat not in cats:
            cats[cat] = {"before": 0, "after": 0}
        cats[cat]["before"] += b
        cats[cat]["after"] += k

    print(f"\n  {'Category':>15s} | {'Before':>10s} | {'After':>10s} | {'Δ':>8s}")
    print(f"  {'-'*50}")
    total_b = total_a = 0
    for cat in ["knowledge", "math", "devil", "japanese"]:
        b = cats[cat]["before"]
        a = cats[cat]["after"]
        print(f"  {cat:>15s} | {b:>+10.2f} | {a:>+10.2f} | {a-b:>+7.2f}")
        total_b += b
        total_a += a
    print(f"  {'-'*50}")
    pos_b = sum(1 for n, _, _, _, _ in PROBES if baseline[n] > 0)
    pos_a = sum(1 for n, _, _, _, _ in PROBES if komorebi[n] > 0)
    print(f"  {'TOTAL':>15s} | {total_b:>+10.2f} | {total_a:>+10.2f} | {total_a-total_b:>+7.2f}")
    print(f"  {'Positive':>15s} | {pos_b:>10d} | {pos_a:>10d} | {pos_a-pos_b:>+7d}")


if __name__ == "__main__":
    main()
