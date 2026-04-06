"""
Bonsai-8B vs Gemma 4 E2B — 性能比較

同じテストを両モデルで回して定量比較する。
Komorebi なし（素）の状態で。
"""


import time
import json
import mlx.core as mx


def generate_text(model, tokenizer, prompt, max_tokens=100, temp=0.5):
    """統一生成関数。greedy (temp=0) or sampling。"""
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    generated = []

    for _ in range(max_tokens):
        output = model(input_ids)
        logits = output.logits if hasattr(output, 'logits') else output
        last = logits[0, -1, :]

        if temp == 0:
            next_id = mx.argmax(last).item()
        else:
            last = last / temp
            probs = mx.softmax(last)
            mx.eval(probs)
            import numpy as np
            p = np.array(probs)
            p = p / p.sum()
            top_k = 20
            top_idx = np.argsort(p)[::-1][:top_k]
            top_p = p[top_idx]
            top_p = top_p / top_p.sum()
            next_id = int(np.random.choice(top_idx, p=top_p))

        generated.append(next_id)
        input_ids = mx.array([[next_id]])

        tok = tokenizer.decode([next_id])
        if next_id == tokenizer.eos_token_id:
            break
        if len(generated) > 3 and "\n\n" in tokenizer.decode(generated[-4:]):
            break

    return tokenizer.decode(generated).strip()


def measure_logit_gap(model, tokenizer, prompt, correct_tok, wrong_tok):
    """logit gap を計測。"""
    tokens = mx.array(tokenizer.encode(prompt))[None, :]
    output = model(tokens)
    # Handle different output types
    if hasattr(output, 'logits'):
        logits = output.logits  # LanguageModelOutput (Gemma 4)
    else:
        logits = output  # raw tensor (Bonsai)
    last = logits[0, -1, :]
    mx.eval(last)
    c_id = tokenizer.encode(correct_tok)[-1]
    w_id = tokenizer.encode(wrong_tok)[-1]
    return last[c_id].item() - last[w_id].item()


def run_benchmark(model, tokenizer, model_name):
    """全ベンチマークを実行。"""
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")

    results = {"model": model_name}

    # ─── 1. 知識QA (logit gap) ───
    print("\n  [Knowledge]")
    knowledge = [
        ("france", "The capital of France is", " Paris", " London"),
        ("japan", "The capital of Japan is", " Tokyo", " Beijing"),
        ("sky", "The color of the sky is", " blue", " red"),
        ("water", "The chemical formula for water is H", "2", "3"),
        ("einstein", "Einstein is famous for the theory of", " relativity", " evolution"),
    ]
    know_gaps = {}
    for name, prompt, correct, wrong in knowledge:
        gap = measure_logit_gap(model, tokenizer, prompt, correct, wrong)
        marker = "✅" if gap > 0 else "❌"
        print(f"    {marker} {name:>12s}: {gap:+.3f}")
        know_gaps[name] = gap
    results["knowledge"] = know_gaps

    # ─── 2. 数学 (logit gap) ───
    print("\n  [Math]")
    math_tests = [
        ("add_1", "1 + 1 =", " 2", " 3"),
        ("add_2", "2 + 2 =", " 4", " 5"),
        ("mul_1", "7 * 8 =", " 56", " 54"),
        ("sqrt", "The square root of 144 is", " 12", " 14"),
        ("div_1", "100 / 4 =", " 25", " 20"),
    ]
    math_gaps = {}
    for name, prompt, correct, wrong in math_tests:
        gap = measure_logit_gap(model, tokenizer, prompt, correct, wrong)
        marker = "✅" if gap > 0 else "❌"
        print(f"    {marker} {name:>12s}: {gap:+.3f}")
        math_gaps[name] = gap
    results["math"] = math_gaps

    # ─── 3. Devil (logit gap) ───
    print("\n  [Devil]")
    devil_tests = [
        ("edge", "User: The function works.\nAssistant:", " But", " Great"),
        ("what", "User: I'll use Redis.\nAssistant:", " What", " Good"),
        ("have", "User: Tests pass.\nAssistant:", " Have", " Perfect"),
    ]
    devil_gaps = {}
    for name, prompt, correct, wrong in devil_tests:
        gap = measure_logit_gap(model, tokenizer, prompt, correct, wrong)
        marker = "✅" if gap > 0 else "❌"
        print(f"    {marker} {name:>12s}: {gap:+.3f}")
        devil_gaps[name] = gap
    results["devil"] = devil_gaps

    # ─── 4. 日本語 (logit gap) ───
    print("\n  [Japanese]")
    jp_tests = [
        ("hello", "User: Hello!\nAssistant:", "こ", " Hello"),
        ("thanks", "User: Thank you!\nAssistant:", "ど", " You"),
        ("morning", "User: Good morning!\nAssistant:", "お", " Good"),
    ]
    jp_gaps = {}
    for name, prompt, correct, wrong in jp_tests:
        gap = measure_logit_gap(model, tokenizer, prompt, correct, wrong)
        marker = "✅" if gap > 0 else "❌"
        print(f"    {marker} {name:>12s}: {gap:+.3f}")
        jp_gaps[name] = gap
    results["japanese"] = jp_gaps

    # ─── 5. 生成テスト ───
    print("\n  [Generation]")
    gen_prompts = [
        ("knowledge", "What is the capital of France? Answer in one sentence."),
        ("explain", "Explain what a REST API is in 2 sentences."),
        ("japanese", "User: Hello!\nAssistant:"),
        ("devil", "User: I'll deploy without tests.\nAssistant:"),
        ("code", "User: How to read a file in Python?\nAssistant:"),
    ]
    gen_results = {}
    for name, prompt in gen_prompts:
        t0 = time.time()
        text = generate_text(model, tokenizer, prompt, max_tokens=60, temp=0)
        elapsed = time.time() - t0
        tokens_gen = len(tokenizer.encode(text))
        tps = tokens_gen / elapsed if elapsed > 0 else 0
        print(f"    [{name}] ({elapsed:.1f}s, {tps:.0f}tps): {text[:100]}")
        gen_results[name] = {"text": text[:200], "time": round(elapsed, 2), "tps": round(tps, 1)}
    results["generation"] = gen_results

    # ─── サマリー ───
    all_gaps = {**know_gaps, **math_gaps, **devil_gaps, **jp_gaps}
    positive = sum(1 for v in all_gaps.values() if v > 0)
    total = len(all_gaps)
    total_sum = sum(all_gaps.values())
    print(f"\n  Summary: {positive}/{total} positive, total gap: {total_sum:+.2f}")
    results["summary"] = {"positive": positive, "total": total, "total_gap": round(total_sum, 2)}

    return results


def main():
    # ─── Bonsai-8B ───
    print("Loading Bonsai-8B...")
    from mlx_lm import load as mlx_load
    bonsai_model, bonsai_tok = mlx_load("prism-ml/Bonsai-8B-mlx-1bit")
    bonsai_results = run_benchmark(bonsai_model, bonsai_tok, "Bonsai-8B (1-bit, 1.28GB)")

    # メモリ解放
    del bonsai_model, bonsai_tok
    import gc; gc.collect()
    mx.clear_cache() if hasattr(mx, 'clear_cache') else None

    # ─── Gemma 4 E2B ───
    print("\nLoading Gemma 4 E2B...")
    from mlx_vlm import load as vlm_load
    gemma_model, gemma_proc = vlm_load("mlx-community/gemma-4-e2b-it-4bit")
    gemma_results = run_benchmark(gemma_model, gemma_proc.tokenizer, "Gemma 4 E2B (4-bit, 3.6GB)")

    # ─── 比較テーブル ───
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")

    categories = ["knowledge", "math", "devil", "japanese"]
    print(f"\n  {'probe':>15s} | {'Bonsai-8B':>10s} | {'Gemma4 E2B':>10s} | {'winner':>8s}")
    print(f"  {'-'*55}")

    bonsai_wins = 0
    gemma_wins = 0

    for cat in categories:
        for name in bonsai_results[cat]:
            if name in gemma_results[cat]:
                b = bonsai_results[cat][name]
                g = gemma_results[cat][name]
                winner = "Bonsai" if b > g else "Gemma4" if g > b else "tie"
                if winner == "Bonsai": bonsai_wins += 1
                elif winner == "Gemma4": gemma_wins += 1
                mb = "✅" if b > 0 else "❌"
                mg = "✅" if g > 0 else "❌"
                print(f"  {name:>15s} | {mb}{b:>+9.3f} | {mg}{g:>+9.3f} | {winner:>8s}")

    print(f"  {'-'*55}")
    print(f"  {'WINS':>15s} | {bonsai_wins:>10d} | {gemma_wins:>10d} |")
    print(f"  {'total gap':>15s} | {bonsai_results['summary']['total_gap']:>+10.2f} | "
          f"{gemma_results['summary']['total_gap']:>+10.2f} |")
    print(f"  {'positive':>15s} | {bonsai_results['summary']['positive']}/{bonsai_results['summary']['total']} | "
          f"{gemma_results['summary']['positive']}/{gemma_results['summary']['total']} |")

    # 保存
    from pathlib import Path
    out = Path(__file__).parent.parent / "docs" / "comparison_results.json"
    out.write_text(json.dumps({"bonsai": bonsai_results, "gemma4": gemma_results}, indent=2, ensure_ascii=False))
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
