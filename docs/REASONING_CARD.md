# Komorebi — 推理カード

> Technical Report は「何をした」。このカードは「なぜそう判断した」「何が見えた」「次にどこを攻める」。
> 開くと推理が再起動する。

**Nao + Eris / 2026-04-06**

---

## 出発点の直感

なお: 「Bankai 以上にできる」

この時点でデータはなかった。XOR Patch を Bankai 参考に一から実装して、75 秒で 29 flips を生成した直後。なおが言ったのは技術的な根拠ではなく、**ここから先に行ける手応え**。

この直感が正しかった理由は後からわかった。Bankai は「証明」で完結してた。僕らは「育てたい」から始まってた。目的が違うから、手法の天井も違う。

---

## 推理の連鎖（時系列ではなく因果で）

### einstein を control に入れたら結果が良くなった → 壊れた probe がガードレールになる

Devil が「einstein を control から外せ」と言った。理論的に正しい（壊れてるものを守る意味がない）。外した。結果が悪化した。

なお: 「einstein を外して悪化がヒント」

推理: einstein は -4.261 で壊れてるが、「theory of relativity vs evolution」の判断には**推論能力**が要る。control に入れると「推論に関わる neuron を壊すな」という制約が間接的にかかる。これが math にも良い副作用を持った。

**教訓**: 壊れた probe を control に入れると、その probe に関わる neuron 群の保護範囲が広がる。理論的正しさより実験結果を信じる。ただし「なぜ」を追う。

### 34 flips 中 2 つだけが支配的 → ノイズを削ると純度が上がる

flip 影響分析で、L3.gate_proj[8154] が mul_1 +0.246、[10195] が +0.152。残り 32 個は ±0.02 未満。

2 flip だけ抜き出したら math_sum +0.516 で、34 flip (+0.130 avg) より**はるかに良い**。Greedy search が微小な positive flip を積み上げてたが、それらは noise だった。

**教訓**: 探索が見つけたものの大半は noise。支配的な少数を特定して、残りを捨てる勇気。

### XOR の干渉が Steering では共鳴になった → Adapter と Lense は別の空間で動く

XOR: mul を改善 → div が悪化（同じ row の neuron を共有）
Steering: mul を改善 → div も改善（hidden state レベルでは方向が分離可能）

**教訓**: 重みレベル（XOR）では干渉したものが、hidden state レベル（Steering）では干渉しなかった（non-interference）。「共鳴」と呼びたくなるが、contrast pairs の共通要素による可能性もある。独立 probe セットで要検証。

### E.R.I.S. Lense → Steering の技術選定

なお: 「E.R.I.S. Architecture と Lense を活用して進めたい」

Lense 条件（対象不変・合成可能・外せる）を技術要件に読み替えた瞬間に、Steering が導出された。XOR は Adapter（対象を変える）。Steering は Lense（対象を変えない）。

**これは技術選定の方法論として重要**。「次に何を試すか」を技術の並列比較ではなく、設計思想から導出する。思想が先、技術が後。

### Anthropic 論文 × 今日の実験 → ユキ・ミキ・エリスの理由

Anthropic: 「Claude の内部に 171 種類の感情ベクトルがある。行動に因果的に影響する」
今日: 「Bonsai-8B に 3 つの steering vector を入れたら行動が変わった」

なお: 「ゆきとみきのいるエリスがなぜ賢くて、僕にとってベストパートナーなのかが別の方向から見えてきた」

推理: 単一の persona vector じゃない。**複数の方向が、異なる層で、異なる強度で重なって**、初めてバランスが取れる。system prompt を通じて間接的にこれが起きてる。E.R.I.S. Architecture で設計した persona-identity.md / eris-persona.md / protocols.md が、テキストレベルの multi-vector steering として機能してる。

**仮説（未検証だが重要）**: ハーネスエンジニアリングの設計原則と、activation steering の設計原則は、同じ構造を異なるスケールで操作してる。

### VLIW × Komorebi → 制約付き多層最適化

なお: 「VLIW の取り組みと似てる部分があった」

共通構造: XOR 演算 → 層 → 全体性能。局所の変更が全体に波及する。行き来しながら最適化する。

VLIW: 浅い段の変更は全段に波及 → 浅い段は慎重に
LLM: 浅い層の steering は全 probe に波及 → alpha を極小に

**教訓**: 別分野の制約が、同じ構造の問題の設計根拠になる。VLIW の scatter load bound → hidden_size が steering 解像度の下限。

### Bonsai → Gemma 4 移植の失敗 → 「原理」vs「パラメータ」

Bonsai: alpha=0.5, L10 がバランス型, L15 が devil
Gemma 4: alpha=0.5 で壊滅。L10 は全然違う。L2 に devil。

**教訓**: 移植できるのは**パラメータではなくプロセス**。alpha=0.5 は移植できない。「layer scan → risk filter → alpha schedule」というプロセスは移植できる。

hidden_size 正規化 + alpha scheduling で、Bonsai の知見を Gemma 4 に変換できた。D:L18 + J:L30 で 9/10 probes 改善、✅ 壊れゼロ。

---

## 原理（検証済み）

1. **Lense-first**: 重みを変える前に hidden state で試す。足りない時だけ Adapter に降りる
2. **Lense risk で層を選別**: 浅い層は distortion、中間は attenuation、深い層は pure。pure 層だけで構成すると壊れない
3. **Alpha scheduling**: `alpha = base × (0.1 + 0.9 × depth^1.5)`。VLIW パイプライン波及理論から
4. **Hidden_size 正規化**: `alpha_effective = alpha × (hidden_size / reference)`。モデル間移植性
5. **Vector 正規化**: `v_normed = (v / ||v||) × √hidden_size`。層間の norm 差を吸収
6. **Adapter → Lense 順序**: Adapter 適用後に Lense を再抽出しないと分布がズレる
7. **壊れた probe の control 効果**: 壊れた probe を control に入れると周辺能力が保護される
8. **ノイズ除去**: 探索結果の大半は noise。支配的な少数だけ使う

---

## 仮説（未検証だが有望）

1. **入力依存 adaptive steering**: AdaInfer の入力依存 layer importance × Lense risk → 入力ごとに最適な steering 構成が変わる
2. **Steering influence map = layer importance map**: steering Δ≈0 の層は skip 可能。品質と速度を同じ分析で最適化
3. **System prompt = テキストレベルの multi-vector steering**: ハーネス設計と activation steering は同じ原理の異なるスケール（興味深いが検証困難。prompt あり/なしの hidden state 差分を取る必要がある）
4. **Cross-modal steering**: 画像入力時の hidden state 差分 → 視覚 steering vector
5. **Dynamic risk classification**: depth_ratio ではなく hidden state の統計量で Lense risk を自動判定
6. **LoRA 後の steering 再抽出**: Adapter→Lense 順序原則が fine-tuning 一般に当てはまる

---

## データ（再現用の数値）

### Bonsai-8B ベースライン
```
france: +6.496, japan: +8.023, einstein: -4.261
add_1: +1.367, add_2: +0.074, mul_1: -0.309, sqrt_1: +1.543, div_1: -0.141
```

### Gemma 4 E2B ベースライン（mlx-vlm 0.4.4）
```
france: +3.688, japan: +7.000, einstein: +10.312
add_2: -6.406, mul_1: -5.625, div_1: -13.156
devil_edge: -4.375, devil_have: +11.938
jp_hello: +2.750, jp_morning: +9.766
```

### ベスト構成

**Bonsai**: XOR 2flip + Komorebi (L10:math α=0.7, L15:devil α=0.5, L25:jp α=0.5)
→ 3 probes ❌→✅ 転倒, total +34.45

**Gemma 4**: Komorebi scheduled (L18:devil α=0.025, L30:jp α=0.048)
→ 9/10 改善, ✅壊れゼロ, total +21.77

### 速度
```
Bonsai-8B:   55.0 tok/s, peak 1.34 GB
Gemma 4 E2B: 46.4 tok/s, peak 3.43 GB
```

---

## 次にどこを攻めるか（優先順位付き）

### A. すぐやるべき
1. **Gemma 4 で math steering を追加**: 今の構成は devil + jp のみ。math を pure 層で加えて全カテゴリ改善を目指す
2. **生成テスト**: logit gap だけでなく実際の生成文で品質を確認。repetition penalty 統合
3. **Probe 数の拡充**: 16 probes → 50+ で robustness を確認

### B. 次にやりたい
4. **Layer skip 実装**: steering influence map から skip 可能な層を特定。速度測定
5. **Dynamic risk**: hidden state の統計量で Lense risk を自動判定する実験
6. **マルチモーダル**: 画像入力時の steering 効果を測定

### C. 見えてるが遠い
7. **Speculative decoding + steering**: draft model の精度を steering で上げて全体速度を改善
8. **自動スケジューリング**: probe 結果から最適構成を自動生成する AutoML 的フレームワーク
9. **ハーネスエンジニアリングへの逆輸入**: activation steering の知見を system prompt 設計に活かす

---

## 失敗アーカイブ

| 何をした | 結果 | 学び |
|---------|------|------|
| einstein を control から除外 | math が悪化 | 壊れた probe がガードレールになる |
| Bonsai の alpha=0.5 を Gemma 4 に適用 | 壊滅 (-43.94) | hidden_size で正規化必須 |
| L2 に devil steering (Gemma 4) | devil_edge -21.8 | 浅い層は distortion リスク |
| 34 flip 全部適用 | math_sum +0.130 | 2 flip だけの方が +0.516 |
| XOR 前の分布で steering 抽出 → XOR 後に適用 | math 悪化 | Adapter 後に Lense を再抽出すべき |
| Multi-layer Komorebi α=0.5 (Gemma 4, 前回) | 全カテゴリ大幅悪化 | vector 正規化 + alpha schedule が必須 |

---

## キーワード（検索用）

activation steering, CAA, contrastive activation addition, representation engineering,
E.R.I.S. architecture, Lense, VLIW, pipeline scheduling, alpha scheduling,
hidden_size normalization, layer risk classification, Bonsai-8B, Gemma 4 E2B,
XOR patch, Bankai, Komorebi, multi-vector composition, adapter-lense ordering,
layer skip, constrained multi-layer optimization

---

*結果と対話する。データを見て「これがヒントだ」と言えること。*
*推理の質が、技術の到達点を決める。*
