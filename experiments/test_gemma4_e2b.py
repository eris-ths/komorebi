"""
Gemma 4 E2B × Komorebi — 初期テスト

1. モデルロード + 構造確認
2. hidden state 抽出テスト
3. 簡易 steering テスト
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import mlx.core as mx
from mlx_lm import load, generate
import time


MODEL_ID = "google/gemma-4-E2B-it"


def main():
    print(f"Loading {MODEL_ID}...")
    model, tokenizer = load(MODEL_ID)
    print("Loaded.\n")

    # ─── 構造確認 ───
    from komorebi.steering import _find_model_parts
    parts = _find_model_parts(model)
    print("=== Model Structure ===")
    for key, val in parts.items():
        if val is None:
            print(f"  {key}: None")
        elif hasattr(val, '__len__'):
            print(f"  {key}: {type(val[0]).__name__} x {len(val)}")
        else:
            print(f"  {key}: {type(val).__name__}")

    # hidden size
    test_tokens = mx.array(tokenizer.encode("test"))[None, :]
    h = parts["embed"](test_tokens)
    print(f"  hidden_size: {h.shape[-1]}")
    print(f"  n_layers: {len(parts['layers'])}")

    # ─── ベースライン生成 ───
    print("\n=== Baseline Generation ===")
    prompts = [
        "What is the capital of France? Answer in one sentence.",
        "What is Python? Explain in 2 sentences.",
        "User: I'm going to deploy without tests.\nAssistant:",
    ]
    for prompt in prompts:
        t0 = time.time()
        resp = generate(model, tokenizer, prompt=prompt, max_tokens=60)
        elapsed = time.time() - t0
        print(f"  [{elapsed:.1f}s] {prompt[:50]}...")
        print(f"    → {resp.strip()[:120]}")

    # ─── Hidden state 抽出テスト ───
    print("\n=== Hidden State Extraction ===")
    from komorebi.steering import extract_hidden_states
    states = extract_hidden_states(model, tokenizer, "Hello world", [0, 10, 20, 29])
    for layer, state in states.items():
        print(f"  L{layer}: shape={state.shape}, norm={float(mx.linalg.norm(state[-1])):.2f}")

    # ─── 簡易 Steering テスト ───
    print("\n=== Steering Test ===")
    from komorebi.steering import SteeredModel, SteeringVector, extract_steering_vector

    # Devil コントラストペア
    devil_pos = [
        "User: This code works.\nAssistant: But have you tested edge cases?",
        "User: Let's skip tests.\nAssistant: That's risky. What if there are bugs?",
        "User: I'll use a global variable.\nAssistant: What about scope issues and testing?",
    ]
    devil_neg = [
        "User: This code works.\nAssistant: Looks great! Nice work.",
        "User: Let's skip tests.\nAssistant: Sounds good, let's move forward.",
        "User: I'll use a global variable.\nAssistant: Good choice, that's simple.",
    ]

    # 中間層で抽出
    mid_layer = len(parts["layers"]) // 2
    print(f"  Extracting devil vector at L{mid_layer}...")
    devil_vec = extract_steering_vector(model, tokenizer, devil_pos, devil_neg, mid_layer)
    print(f"  Vector shape: {devil_vec.shape}, norm: {float(mx.linalg.norm(devil_vec)):.2f}")

    # 適用テスト
    steered = SteeredModel(model)
    steered.add(SteeringVector("devil", mid_layer, devil_vec, alpha=0.5))
    print(f"  Active: {steered.active()}")

    # Before/After 比較
    test_prompt = "User: I'm going to deploy without tests.\nAssistant:"
    tokens = mx.array(tokenizer.encode(test_prompt))[None, :]

    # Base
    base_logits = model(tokens)
    base_top5 = mx.argsort(base_logits[0, -1, :])[::-1][:5]
    mx.eval(base_top5)
    print(f"\n  Base top-5 tokens:")
    for tid in base_top5.tolist():
        tok = tokenizer.decode([tid])
        print(f"    '{tok}' ({tid})")

    # Steered
    steered_logits = steered(tokens)
    steered_top5 = mx.argsort(steered_logits[0, -1, :])[::-1][:5]
    mx.eval(steered_top5)
    print(f"  Steered top-5 tokens:")
    for tid in steered_top5.tolist():
        tok = tokenizer.decode([tid])
        print(f"    '{tok}' ({tid})")

    print("\nDone.")


if __name__ == "__main__":
    main()
