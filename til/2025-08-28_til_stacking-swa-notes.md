<!-- Filename: 2025-08-28_til_stacking-swa-notes.md -->

What
- Why stacking sliding window attention does not let models "see" very far in practice.

Context
- I found an article I liked and decided to write short notes in simple steps.
- Article: Why Stacking Sliding Windows Can't See Very Far by Guangxuan Xiao.

Notes
1) The window
- Each layer only looks back W tokens.
- Example with W=10: at token 100, the layer sees tokens 91–100.
- Tiny check: With W=100, token 1000 directly sees tokens 901–1000.

2) Layers: reach vs influence
- Stacking L layers suggests info could hop W at a time, so reach ≈ L×W.
- What matters is influence, not just reach. Far info becomes a faint whisper.
- Words to keep: reach = could travel; influence = actually affects the output.
- Tiny check: With L=3 and W=100, nominal reach is 300 tokens; influence that far back is small.

3) Without residuals: spreading like a blur
- Averaging the last W tokens each layer is repeated blurring.
- Repeated blurs spread slowly, like diffusion.
- Rule: effective spread grows like sqrt(L) windows, not L windows.
- Sanity example: L=100, W=100 → useful spread ≈ 10×100 = about 1000 tokens.

4) With residuals: the exponential barrier
- Two paths per layer: residual keeps most of the current token (α ~ 0.9–0.99);
  attention adds a small slice (1−α) from the window.
- To carry info from distance d, it must hop k ≈ ceil(d/W) times through that small slice.
- Each hop multiplies influence by (1−α). After k hops: influence ≈ (1−α)^k.
- Numbers you can feel: α=0.95 → per window 0.05; then 0.05, 0.0025, 0.000125.

5) Rule of thumb horizon
- Influence at distance d (tokens) ≈ C · (1−α)^(d/W).
- Practical horizon is where influence falls below your tolerance (e.g., 0.1%).
- Example: α=0.95, W=100 → a few windows (≈300–400 tokens) already very small.

6) If you need longer reach
- Increase W (costlier), soften the residual (lower effective α), add non‑local routes
  (global/memory tokens, retrieval, sparse long‑range heads), or use persistent state models
  (state space models) to carry information without exponential loss.

Pitfalls
- Confusing reach with influence; assuming depth alone solves long range.
- Ignoring α — when α is high, distant info drops exponentially.
- Forgetting tolerance — "how small is too small" is task dependent.

Links
- https://guangxuanx.com/blog/stacking-swa.html

