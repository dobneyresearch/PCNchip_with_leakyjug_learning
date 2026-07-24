# Forward↔Transpose weight-skew — the transpose-at-source async test

**Question:** transpose-at-source makes the chip compute Wᵀ locally from its own live W-SRAM. In an
async, no-global-clock design the weights can move between when the forward ran and when the δ comes
back to be transposed. **Does that skew hurt, and how much?**

**Test:** `pcn_jug_skew.py` (a COPY of `pcn_jug.py`; the validated sim is untouched). `--skew_folds k`
makes the two backward W.T reads (L3→L2 and L2→L1) use a W that differs from the forward's W by `k`
folds of drift. `--skew_mode lag` = the faithful model (the SAME fires, time-shifted → correlated
drift). `--skew_mode rand` = a cruder independent bound (±1 code on ~2%·k of synapses, uncorrelated).
The forward, the E-accumulation, and the sweep all use the LIVE W — as the hardware does.

`--skew_folds 0` reproduces `pcn_jug.py` **exactly** (67.47% == 67.47% at ep=2 ch=4000: identical
accuracy, fire rate, frozen%). So the copy is faithful and skew=0 is a true baseline.

## Result (fast config: ep=2 ch=4000 seed=42, baseline 67.47%)

| skew | mode | accuracy | Δ vs baseline |
|-----:|------|---------:|--------------:|
| 0 | — (invariant holds) | 67.47% | — |
| 1 | lag | **67.47%** | **0.00** |
| 2 | lag | 66.33% | −1.14 |
| 4 | lag | 65.75% | −1.72 |
| 8 | lag | 66.34% | −1.13 |
| 8 | rand | 67.13% | −0.34 |

## Reading it

- **skew = 1 fold is FREE** (identical to baseline). And **1 fold is the realistic worst case**: the
  sweep-barrier (fires only at end-of-fold; the sweep can't run until the fold's backward is done)
  keeps a chip's forward and backward inside one weight-epoch by construction.
- **The penalty is bounded and does NOT grow with skew.** skew = 2/4/8 lag all sit ~1–1.7pp below
  baseline and are within the ~0.7pp noise floor of *each other* — skew=8 is no worse than skew=2. So
  even a **gross** invariant violation (transpose reading 8-fold-stale weights) costs ~1–2pp and
  **does not diverge**.
- **The uncorrelated bound (rand) is gentler still** (−0.34pp at k=8).

## Why it's benign (mechanism)

The transpose produces the δ that drives E, and **E is a slow, residue-preserving integrator**. A
slightly-stale transpose perturbs the *magnitude* of a δ but preserves its *direction*; the jug
averages that over folds. This is the SAME robustness that makes a 20%-wrong comparator cost only
0.34pp — the learning signal lives in an accumulator that forgives per-event noise.

## Verdict

**Temporal weight-skew is NOT a blocker to transpose-at-source.** At the realistic operating point
(≤1 fold, enforced by the sweep barrier) it is free; gross violations are bounded to ~1–2pp and
non-diverging. The sweep barrier is a sufficient invariant, and the protocol only needs to keep the
backward wavefront inside ~1 weight-epoch (fold-tag δ), which it gets naturally.

## Caveats / escalation

- This is the fast, under-converged config (67% baseline, not the 81.96% headline). The QUALITATIVE
  result (free at k=1, bounded, non-diverging) should be robust, but the absolute penalty near
  convergence is being confirmed at a more-converged config (`SKEW_ESCALATE.txt`, ep=3 ch=8000).
- Run to reproduce: `./run_skew_focused.sh` (writes `SKEW_FOCUSED.txt`).
