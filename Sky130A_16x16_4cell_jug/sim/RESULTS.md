# pcn_jug.py — the evidence

Bit-accurate model of the jug chip. BIG topology (24/8/16 chips, 1152 split-sign L0, 39,450
params), EMNIST letters, 6 epochs, `--agc rms`.

> **⚠ NOISE FLOOR ≈ 0.7pp — nothing under ~1pp in this rig is a result.**
> It is **NOT just seed variation.** The float matmuls and the classifier fit are BLAS-dependent, so
> **the same config at a different thread count sums in a different order** and training amplifies
> it: seed 42 gives **82.03%** at `OMP_NUM_THREADS=10` and **81.55%** at `4`. Identical code,
> identical seed. **Do not read a 0.5pp difference as a result.**

---

## ✅ VERIFICATION OF THE FINAL SINGLE-TAIL DESIGN (MN3_e deleted, `pcn_jug.py` defaults)

| run | best |
|---|---|
| seed 3 | **82.17%** |
| seed 1 | 81.63% |
| **20% wrong-signed fires** | 81.62% |
| seed 42 | 81.55% |
| **COMBINED device stress** — 10× log-normal leak spread **AND** ×2 θ mismatch, together | **81.06%** |
| *float ceiling* | *82.13%* |

The combined-stress arm is harsher than anything tested individually and costs **~0.9pp** without
destabilising. **Everything survives the removal of MN3_e.**

Lineage / audit trail: `../../hw_buildcheck/` (`THE_JUG.md`, `RTL_RECONCILIATION.md`,
`HW_BUILD_CHECK.md`). Float reference: `../../multi_array_level3_BIGspec/pcn_bigspec.py` (82.50%).

---

## ★ HEADLINE

| | best |
|---|---|
| old chip absorb rule (`W += lr_slow·E`, block/epoch) | **64.09%** |
| the current rule, written naively to an 8-bit cell | 75.25% |
| **the current rule via the JUG** | **81.96%** |
| float ceiling (same harness) | 82.13% |
| **residual** | **0.17pp — a quarter of the noise floor** |

**And it DELETES hardware:** no digital master, no velocity register, no per-synapse digital store.
Fires at **0.91%/fold** — mean weight motion 0.0091 LSB/fold, **less than half** the old rule's
0.019. It is not out-spending the old rule; it is out-**allocating** it.

**It is NOT the learning rate.** The old rule across a 5× lr sweep is **flat**:
3e-4 → 75.25 · 6e-4 → 74.82 · 9e-4 → 75.59 · 1.5e-3 → 74.56. It cannot be tuned out of its ceiling.

---

## SEEDS (n=4) — it is not a lucky run

| seed | 1 | 2 | 3 | 42 |
|---|---|---|---|---|
| best | 81.63% | 81.38% | 82.17% | 81.96% |

**mean 81.79, spread 0.79pp** (= the noise floor). The best seed exceeds the float control.

---

## θ — the jug's learning rate. A clean inverted-U, optimum at 8

| θ | 1 | 2 | 4 | **8** | 16 | 32 |
|---|---|---|---|---|---|---|
| fire rate | 8.5%/fold | 6.3% | 1.9% | **0.91%** | 0.42% | 0.15% |
| best | 74.22% | 78.36% | 81.08% | **81.96%** | 80.37%¹ | 73.95% |

¹ given 8 epochs (78.18% at ep4) — it genuinely learns slower, but **even with the extra time it
does not catch θ=8**. The optimum is real, not an artefact of stopping early.

**★ FREE LR ANNEALING:** the fire rate FALLS as the network converges (2.69% → 1.85%/fold). The
gradients shrink, the jug fills more slowly, crossings get rarer. **An lr schedule emerging from
the physics.**
⚠⚠ **THEREFORE DO NOT SERVO THE FIRING RATE.** A loop holding it constant by adjusting θ would
**cancel the annealing.** It is the first thing anyone would reach for, and it is a trap.

---

## ⚠⚠ E LIVE IN THE FORWARD MAC — the dual-tail question. **THE ONE THING THAT BREAKS IT.**

`mac_cell_emx` is **dual-tail**: `I_ntail = I_tail_W + I_tail_E`. Ce is wired into the tail, so the
MAC computes **W + E**. **You cannot switch that off in silicon.** The sim's forward has always
used W alone — legitimately, because in fold mode E is tiny (`|E|/|W| ≈ 0.004`, measured at
`pcn_bigspec.py:269`) and folded away every 128 samples. **But the jug lets E accumulate ~110×
longer.**

| E's contribution to the MAC (weight codes at threshold) | best | fire rate |
|---|---|---|
| **0 — no E-tail** | **81.96%** | 0.91%/fold |
| 0.1 | 82.06% | 1.03% |
| 0.2 | 81.32% | 1.06% |
| **0.4 — the OLD design point (gm_E ≈ gm_W)** | **80.46%** | 1.03% |
| 1.0 | 76.83% | — |
| 2.0 | **COLLAPSE** — best 47.54%, ends 12.13% | **14.8%** |
| 6.7 — the E-cap clamp (E_MAX_V = 55 mV) | **COLLAPSE** — 10.47% | **19.3%** |

**It is a POSITIVE FEEDBACK RUNAWAY.** E feeds the MAC → activations grow → deltas grow → E grows
faster → fires more. The fire rate explodes 0.91% → 14.8% → 19.3%.

⇒ **SAFE ZONE ≤ 0.2 CODES.** The old `gm_E ≈ gm_W` target sits at ~0.4 — **already lossy, and only
~2.5× from runaway**, on a gm ratio that drifts with process and temperature.

> ### ✅ **DONE — MN3_e IS DELETED** (approved 2026-07-14). The cell is SINGLE-TAIL.
> Ce only ever needs to be **compared against θ** — it never needs to inject current into the MAC.
> Removing MN3_e:
> * recovers the full **81.96%** (vs 80.46%),
> * returns the MAC core to the original **single-tail 5T** topology (+ a comparator),
> * **dissolves the entire V_bias_e problem** — the constraint that forced V_bias_e to 0.65 V
>   (at 0.9 V the dual-tail saturation window and the diff-pair constraint were *simultaneously
>   impossible*, both tails went into triode at ntail ≈ 22 mV, and it cost a SPICE iteration)
>   exists **only** because MN3_e shares `ntail`. It goes with the transistor.
>   ⚠ **V_bias_e is now a FREE variable — re-derive it against the comparator; do NOT inherit
>   0.65 V by default.**
> * **shrinks the SPICE job**: gm_E characterisation and the `gm_E ≈ gm_W` matching requirement are
>   deleted (old §6.1). §6.4 (the θ comparator) becomes the key analog deliverable.
> * **and makes the silicon and the model compute the same thing** — the sim's forward has always
>   used `W` alone, while the chip was computing `W + E`. That divergence was harmless only while E
>   was tiny and folded away every 128 samples; under the jug it would have become a **runaway
>   instability**. We caught it by reading the RTL, not by simulating.

---

## ★ COMPARATOR SIGN ERRORS — FREE. The spec is very loose.

| wrong-signed fires | 5% | 10% | 20% |
|---|---|---|---|
| best | 81.42% | 82.06% | 81.62% |

**One fire in five going the wrong way costs NOTHING.** (I predicted this would be the killer. It
isn't.)

**Why — and it is a direct consequence of the residue-preserving subtraction:** a wrong-signed fire
moves W the wrong way **and subtracts θ in the wrong direction — i.e. it ADDS the charge back.**
The next correct fire undoes both. **Charge is conserved, so the error self-corrects.**
The subtraction is not just what makes the jug *learn*; it is what makes it *immune to a sloppy
comparator*.

---

## ★★ PHYSICAL CAPACITORS — NO CONTROL LOOP NEEDED

Subthreshold leakage goes **exponentially** with Vt; Vt mismatch is Gaussian ⇒ the leak current is
**log-normal** and varies many-fold across a die. Modelled per-synapse, drawn once (fabrication,
not noise). Mean inter-fire interval at θ=8 ≈ **110 folds**.

| devices | τ across the die (p5–p95) | best | cost |
|---|---|---|---|
| ideal | ∞ | **81.96%** | — |
| τ=1000 folds, 3× spread | 167 – 6,190 | 81.68% | −0.28 |
| τ=300, 3× spread | 50 – 1,857 | 81.62% | −0.34 |
| **τ=100, 3× spread** (median drains ≈ as fast as it fills; worst 5% leak **6× faster than they fire**) | 17 – 619 | **81.53%** | −0.43 |
| **τ=1000, 10× spread** | 23 – 45,638 | **81.38%** | −0.58 |
| **threshold mismatch ×2** | — | 81.20% | −0.76 |

**ALL INSIDE THE NOISE FLOOR**, and it degrades *gracefully* — no cliff.

**Why (a property, not luck):** a leaky cell **does not get the wrong answer**. It needs more charge
to reach threshold, so it **fires less often** = a **reduced per-synapse learning rate**, and SGD is
famously indifferent to that. Every fire it does make still carries the correct sign, and the
residue subtraction keeps it unbiased. Threshold mismatch is free for the *same* reason — it is the
same perturbation by a different physical route.

### ⇒ AND THE REAL CAPACITOR IS FAR BETTER THAN ANY OF THESE
Ce ≈ 100 fF with 5–10 fA leakage ⇒ **≈ 50 µV/s, about 140 s per E-LSB.** The inter-fire interval is
**tens of milliseconds**.

```
   τ_leak / (inter-fire interval)  ≈  10³ – 10⁴
```

**The physical Ce IS a pure integrator on the firing timescale** — which is exactly the
best-performing configuration (`--jug_leak 0`). **The physics hands us the optimum for free.**

⇒ **NO servo, NO replica bias, NO switched-capacitor leak.** The requirement is a loose **one-sided
bound** (τ ≫ inter-fire), met by ~10³. **Matching is a non-requirement.**

---

## The leak (= momentum): real, positive, never large enough to claim

| θ | `--jug_leak 6` | `--jug_leak 0` (pure) | leak's contribution |
|---|---|---|---|
| 1 | 74.85% | 74.22% | +0.63 |
| 2 | 78.85% | 78.36% | +0.49 |
| 4 | 81.37% | 81.08% | **+0.29** |

All inside noise, and **the benefit SHRINKS as θ improves** ⇒ the leak and the threshold are two
mechanisms for the *same* job (rejecting weak evidence), and **the threshold is the better one**.
The win is the **charge integrator**, not the momentum. The leak costs nothing and has never hurt,
so keep it — but do not rely on it.

⚠ **UNEXPLAINED:** `--jug_leak 8` (a *weaker* leak, 256-fold memory) = 70.46% at θ=1 — WORSE than
both `leak 6` (74.85%) and pure (74.22%). **Non-monotonic in memory length.** Noise, a bug in the
integer leak, or something real we do not understand. **Do not design around a curve you cannot
explain.**

---

## Still open

1. **Does the jug beat 82.50% in the FLOAT sim?** ⇒ is it a better **learning rule**, or only a
   better **circuit**? *(Must be done in a COPY — never edit `pcn_bigspec.py`.)*
2. **Per-sample leak** instead of the fold-grid approximation — required before a circuit spec.
3. **Temperature DRIFT during training** (we modelled static mismatch, not drift).
4. The `leak8` non-monotonicity.
5. Task #15 - the per-layer ADC gain. **NOT a blocker**: the spread is **8x** (`[>>3, >>6, >>6]`), a 2-3 bit config register. The '100:1' figure was WRONG (three units). Keep the task as a **SCALING** experiment: ||W|| drives BOTH the forward activation scale AND the backward delta attenuation, so normalising the frozen W per chip may remove the requirement *and* fix attenuation at depth.

---

## ★ CLOSE-OUT 1: does the jug beat the fold rule in FLOAT? — **NO.**

The question: is the jug a better **learning rule**, or only a better **circuit**?
Same topology, same everything, **float weights / float activations / float deltas** — the ONLY
difference is the update rule.

| float, update rule | best |
|---|---|
| **the FOLD rule** (`W += lr·sign(E)` + momentum) — the reference | **82.13%** |
| jug, write quantum **1/64** — *what the chip does* | **73.00%** |
| jug, write quantum 1/256 | 73.11% |
| jug, write quantum 1/1024 (finest) | **81.60%** |

**The jug NEVER beats the fold rule in float**, and the pattern is unambiguous: **as the write
quantum gets FINER, the jug climbs toward the fold rule and stops just short of it.** At the CHIP's
quantum it is **9 points worse**.

### ⇒ **THE JUG IS A HARDWARE ENABLER, NOT A BETTER LEARNING RULE.**
Its entire advantage is about the **COARSENESS OF THE WRITE**:
- give it **fine** writes and it has nothing to offer — the fold rule already spends a fine budget
  well (82.13%);
- give it **only whole-LSB** writes — which is all an 8-bit analog cell can do — and evidence-gating
  becomes the right way to spend them: **75.25% → 81.96%.**

**This is the honest claim to make in any write-up.** It is not a new learning algorithm. It is the
correct way to run the existing rule on a coarse analog weight.
*(⚠ `--float_ref` was BROKEN before this test: with the legacy fold stripped, float mode ran the
JUG fire as `W += ±1.0` — the entire weight range in one step. Fixed via `--jug_step`.)*

---

## ★ CLOSE-OUT 2 (TASK #15): can normalising the frozen W remove the per-layer ADC gain? — **NO.**

The hypothesis was that ‖W‖ drives BOTH the forward activation scale AND the backward delta
attenuation, so making each chip's W norm-preserving would fix both. **`--norm_w` kills it:**

```
norm W  L1: x  4.360 (chip spread 0.159–15.139)  |W|max=9.188  ⚠ CLIPS THE 8-BIT RAIL
norm W  L2: x  0.653 (chip spread 0.632–0.686)   |W|max=0.638
norm W  L3: x  0.720 (chip spread 0.570–0.891)   |W|max=0.634
```

L2 and L3 normalise happily and stay inside the rail. **L1 wants weights up to 9.19 against a rail
of 0.95 — TEN TIMES OVER.**

**WHY:** every layer except L1 receives an input that has been **renormalised to ~42 by the ADC
above it**. **L1's input is the RAW split-sign L0 encoding** — the one input scale that is not under
the converter's control. So L1 is the only layer whose scale cannot be fixed by scaling W, and it is
exactly the layer that needs it.

### ⇒ **THE PER-LAYER ADC GAIN IS NOT A WART — IT IS THE CORRECT PLACE FOR THE GAIN.**
**The ADC reference is programmable. The weight rail is not.** Moving that gain into W blows the
rail tenfold at L1. Keep the **2–3 bit config register** (`[>>3, >>6, >>6]` — an **8×** spread, not
the "100:1" an earlier note wrongly claimed).

**And the backward half of the argument was already handled elsewhere:** the delta channel has its
own **per-layer AGC** (`channel()`), so backward attenuation is compensated on the backward path,
not by ‖W‖.

**⇒ TASK #15 IS CLOSED.** The genuine depth question — whether ‖W‖ limits how deep this can go — is
a separate programme against a **deeper topology**, and the BIG rig (3 layers) cannot answer it.
