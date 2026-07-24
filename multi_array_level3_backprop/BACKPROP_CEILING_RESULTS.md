# Backprop ceiling rig — results (2026-07-11)

Separable rig `backprop_rig.py` (torch). Fixed L0 (frozen 576-dim ReLU'd PCA, the same
features every test uses), trains L1/L2/L3 by backprop on the **same connectivity** as the
small model (12/4/2 chips, block-grouped, leaky α=0.1, no hidden bias). Isolates the learning
rule and the architecture, against our forwards-only rig on the identical setup.

## The decomposition (EMNIST letters, 20,800 test)

| Configuration | Test acc | What the step costs |
|---|---|---|
| Linear classifier on the input (PCA-576) | 70.28% | (reference) |
| **Dense MLP, full connectivity, same sizes (192/64/32), backprop** | **79.46%** | **depth+mixing ADDS +9.2** vs linear |
| Chip-factored (matched connectivity), backprop, **float** | **59.18%** | **−20.3 = chip-factoring / no cross-chip mixing** |
| Chip-factored, backprop, **8-bit STE weights** | **58.46%** | **−0.72 = quantisation (nearly FREE)** |
| Our forwards-only rule on this small topology (SM-1, GHA+bounded-E) | ~32.1% | **~−26 = the learning rule** |

## What this tells us (reframes both open problems)

1. **Depth is NOT the enemy — lack of cross-chip mixing is.** A dense MLP of the *same depth
   and layer sizes* reaches **79.46%**, BEATING linear-on-input (70.28) by +9. So depth helps
   *when information can mix across the layer*. "Burning out at four layers" is really
   "burning out from block-grouped connectivity with no cross-chip mixing": each chip sees
   only a local 48-feature window and can't combine with others until the next narrow
   aggregation, and the 576→192→64→32 funnel over non-overlapping local windows discards
   global structure. **Fix = mixing (overlapping/richer routing, permutation between layers,
   or width), not fewer layers.**

2. **Quantisation is nearly free (−0.72 pp).** 8-bit STE backprop ≈ float backprop. The chip's
   8-bit weights are NOT a meaningful bottleneck for accuracy. (This retires the worry from
   the L0_A same-scale-transform finding: quantisation blocks a clean *identity init*, but
   costs almost nothing for *trained* accuracy.)

3. **The learning rule is the single biggest lever (~−26 pp).** On the identical chip-factored
   architecture, backprop reaches ~59% but our forwards-only GHA+bounded-E rule reaches ~32%.
   The rule captures only ~54% of what the architecture allows. This is the headroom the
   forwards-only-from-random work targets.

4. **Two independent gaps, now separated:**
   - *Architecture (mixing):* dense 79.46 → chip-factored 59.18 = **−20** (the BIG topology
     mitigates this with richer routing + width → it reached 67.97; the small 12/4/2 with
     block-diag routing is the worst case).
   - *Learning rule:* chip-factored backprop ~59 → our rule ~32 = **−27**.
   Quantisation is negligible; both real gaps are addressable independently.

## The benchmark for the forwards-only rule
On this small chip-factored architecture the achievable ceiling is **~59% (float) / ~58%
(8-bit, the matched number)**. The next experiment — `--rand_init --local_only` (forwards-only
from random, local-delta as sole signal) — is measured against **58.46%**: how close can the
forwards-only rule get to backprop on the identical architecture, starting from random?

## Caveats
- Dense has ~12× more params than chip-factored (block-sparse), so the −20 conflates
  connectivity pattern + parameter count; both are real, and the actionable lever (add
  cross-chip mixing) recovers connectivity without going fully dense.
- Backprop uses Adam, 150 epochs, batch 256, lr 1e-3, no hidden bias (matches chip),
  readout bias (matches fit_clf). Reproduce: `python3 backprop_rig.py [--dense|--quant]`.
