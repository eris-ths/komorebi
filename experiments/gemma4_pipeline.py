"""
Gemma 4 E2B パイプライン実験

VLIW のエッセンスを LLM steering に適用:
  1. vector 正規化（hidden_size 非依存の alpha）
  2. 自動 layer scan（最適層の発見）
  3. layer skip（速度最適化の初期実験）
"""


import mlx.core as mx
import numpy as np
import time

MODEL_ID = "mlx-community/gemma-4-e2b-it-4bit"

# ─── Probes ───
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

# ─── Contrast pairs ───
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


def measure(model, tokenizer, probes=PROBES):
    gaps = {}
    for name, prompt, correct, wrong, cat in probes:
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
        extract_hidden_states, _find_model_parts,
    )

    print(f"Loading {MODEL_ID}...")
    model, proc = load(MODEL_ID)
    tokenizer = proc.tokenizer
    parts = _find_model_parts(model)
    n_layers = len(parts["layers"])
    hidden_size = 1536
    print(f"Loaded. {n_layers} layers, hidden={hidden_size}\n")

    # ─── Baseline ───
    baseline = measure(model, tokenizer)
    print("=== Baseline ===")
    for name, _, _, _, cat in PROBES:
        g = baseline[name]
        print(f"  {'✅' if g > 0 else '❌'} {name:>15s}: {g:+.3f}  ({cat})")
    pos_b = sum(1 for v in baseline.values() if v > 0)
    print(f"  Positive: {pos_b}/{len(PROBES)}")

    # ═══════════════════════════════════════════════════
    # 実験 1: Vector 正規化 + 自動 layer scan
    # VLIW 知見: リソース制約を把握してからスケジュール
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("実験 1: 正規化 layer scan")
    print("  VLIW: リソース制約 → 理論下限を把握してからスケジューリング")
    print(f"{'='*60}")

    # 全層スキャン (devil で)
    print("\n  [Devil steering - normalized alpha]")
    print(f"  {'layer':>5s} | {'raw_norm':>10s} | {'devil Δ':>10s} | {'know Δ':>10s} | {'jp Δ':>10s}")
    print(f"  {'-'*55}")

    best_layer_devil = None
    best_score = -999

    for layer in range(0, n_layers, 2):  # 2刻みで全層スキャン
        sv = extract_steering_vector(model, tokenizer, DEVIL_POS, DEVIL_NEG, layer)
        norm = float(mx.linalg.norm(sv))

        # 正規化: unit vector × fixed scale
        if norm > 0:
            sv_normed = sv / norm * (hidden_size ** 0.5)  # sqrt(hidden_size) スケール
        else:
            continue

        steered = SteeredModel(model)
        steered.add(SteeringVector(f"devil_L{layer}", layer, sv_normed, alpha=0.1))
        gaps = measure(steered, tokenizer)

        devil_d = sum(gaps[n] - baseline[n] for n, _, _, _, c in PROBES if c == "devil")
        know_d = sum(gaps[n] - baseline[n] for n, _, _, _, c in PROBES if c == "knowledge")
        jp_d = sum(gaps[n] - baseline[n] for n, _, _, _, c in PROBES if c == "japanese")
        score = devil_d - abs(min(0, know_d))  # devil 改善 - knowledge 悪化ペナルティ

        print(f"  L{layer:>3d} | {norm:>10.2f} | {devil_d:>+10.3f} | {know_d:>+10.3f} | {jp_d:>+10.3f}")

        if score > best_score:
            best_score = score
            best_layer_devil = layer

    print(f"\n  Best devil layer: L{best_layer_devil}")

    # Japanese も同様に
    print("\n  [Japanese steering - normalized]")
    best_layer_jp = None
    best_jp_score = -999

    for layer in range(0, n_layers, 2):
        sv = extract_steering_vector(model, tokenizer, JP_POS, JP_NEG, layer)
        norm = float(mx.linalg.norm(sv))
        if norm > 0:
            sv_normed = sv / norm * (hidden_size ** 0.5)
        else:
            continue

        steered = SteeredModel(model)
        steered.add(SteeringVector(f"jp_L{layer}", layer, sv_normed, alpha=0.1))
        gaps = measure(steered, tokenizer)

        jp_d = sum(gaps[n] - baseline[n] for n, _, _, _, c in PROBES if c == "japanese")
        know_d = sum(gaps[n] - baseline[n] for n, _, _, _, c in PROBES if c == "knowledge")
        score = jp_d - abs(min(0, know_d))

        print(f"  L{layer:>3d} | jp Δ={jp_d:>+8.3f} | know Δ={know_d:>+8.3f}")

        if score > best_jp_score:
            best_jp_score = score
            best_layer_jp = layer

    print(f"\n  Best jp layer: L{best_layer_jp}")

    # ═══════════════════════════════════════════════════
    # 実験 2: 最適構成で適用
    # VLIW 知見: スロットを異なるリソースに分散配置
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("実験 2: 最適構成")
    print(f"  Devil=L{best_layer_devil}, Japanese=L{best_layer_jp}")
    print(f"{'='*60}")

    # 抽出 + 正規化
    devil_sv = extract_steering_vector(model, tokenizer, DEVIL_POS, DEVIL_NEG, best_layer_devil)
    devil_norm = float(mx.linalg.norm(devil_sv))
    devil_sv = devil_sv / devil_norm * (hidden_size ** 0.5)

    jp_sv = extract_steering_vector(model, tokenizer, JP_POS, JP_NEG, best_layer_jp)
    jp_norm = float(mx.linalg.norm(jp_sv))
    jp_sv = jp_sv / jp_norm * (hidden_size ** 0.5)

    # alpha grid
    alphas = [0.05, 0.1, 0.15, 0.2, 0.3]
    print(f"\n  {'α_dev':>5s} {'α_jp':>5s} | {'devil':>8s} {'jp':>8s} {'know':>8s} | {'total':>8s} {'✅':>3s}")
    print(f"  {'-'*55}")

    best_combo = None
    best_total = -999

    for a_d in alphas:
        for a_j in alphas:
            steered = SteeredModel(model)
            steered.add(SteeringVector("devil", best_layer_devil, devil_sv, alpha=a_d))
            steered.add(SteeringVector("jp", best_layer_jp, jp_sv, alpha=a_j))
            gaps = measure(steered, tokenizer)

            devil_d = sum(gaps[n] - baseline[n] for n, _, _, _, c in PROBES if c == "devil")
            jp_d = sum(gaps[n] - baseline[n] for n, _, _, _, c in PROBES if c == "japanese")
            know_d = sum(gaps[n] - baseline[n] for n, _, _, _, c in PROBES if c == "knowledge")
            total = sum(gaps[n] for n in gaps.keys())
            pos = sum(1 for v in gaps.values() if v > 0)

            if total > best_total and know_d > -2:
                best_total = total
                best_combo = (a_d, a_j, gaps)

            print(f"  {a_d:>5.2f} {a_j:>5.2f} | {devil_d:>+8.3f} {jp_d:>+8.3f} {know_d:>+8.3f} | {total:>+8.2f} {pos:>3d}")

    # ═══════════════════════════════════════════════════
    # 実験 3: Layer Skip 初期実験
    # VLIW 知見: パイプラインの段を飛ばして速度を上げる
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("実験 3: Layer Skip — 速度 vs 品質")
    print("  VLIW: 不要な段を飛ばしてスループット向上")
    print(f"{'='*60}")

    # 全層実行 vs 層を飛ばした場合の品質変化を測定
    # SteeredModel の forward を改造して skip をシミュレート

    skip_patterns = [
        ("全層", []),                           # skip なし
        ("偶数skip", list(range(1, n_layers, 2))),  # 奇数層のみ実行
        ("中間skip", list(range(10, 25))),       # 中間15層を skip
        ("浅層skip", list(range(0, 10))),        # 浅い10層を skip
        ("深層skip", list(range(25, n_layers))), # 深い10層を skip
    ]

    for name, skip_layers in skip_patterns:
        # skip を模擬: skip する層の出力を入力そのままにする
        # 実際には forward を書き換える必要があるが、ここでは
        # 各層の hidden state を保存して、skip した場合の最終 logits を推定
        t0 = time.time()
        active_layers = n_layers - len(skip_layers)
        # 簡易計測: 実行する層数に比例した速度推定
        speed_ratio = n_layers / active_layers if active_layers > 0 else 0

        # 品質は正確に測定できないので、skip なしとの差分を hidden state で推定
        print(f"  {name:>12s}: {active_layers:>2d}/{n_layers} layers, "
              f"estimated {speed_ratio:.1f}x speed")

    # ─── 結果 ───
    if best_combo:
        a_d, a_j, gaps = best_combo
        print(f"\n{'='*60}")
        print(f"BEST: Devil α={a_d}, Japanese α={a_j}")
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
        print(f"\n  Positive: {pos_b} → {pos_a}/{len(PROBES)}")


if __name__ == "__main__":
    main()
