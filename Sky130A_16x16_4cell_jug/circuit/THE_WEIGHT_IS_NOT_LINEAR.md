# ⚠⚠⚠ THE ANALOG WEIGHT IS NON-MONOTONIC AND INVERTED ABOVE CODE ~119

**2026-07-14. Found by the first SPICE sweep of the FULL weight range.**
**This predates the jug. It affects EMX, FABLE, and every accuracy number in the project.**

> ## THE FINDING
> The MAC's effective weight is `W_eff = gm(V_w) − gm(V_ref)`. **`gm` PEAKS at V_w ≈ 0.90 V and
> FALLS above it.** WGT_ZERO (code 132) sits at **V_w = 0.927 V — right on the peak.**
>
> ⇒ **Codes 71–119 work.** (weight rises with code)
> ⇒ **Codes 119–192 are DEAD or INVERTED** — that is the entire *positive* weight range.
>    A fire that should push the weight UP pushes it slightly **DOWN**.
>
> **The silicon does not implement the linear signed weight that every simulation assumes.**

---

## 1. The measurement (`tb_mac_cell_jug_weight.spice` + `analyse_weight.py`)

Sweep V_w across the full range; at each point extract `dI_out/dx` — the actual weight the MAC
applies. The weight is the *difference* against a reference cell at WGT_ZERO (this is how the
architecture gets signed weights — `tb_wgt_zero_emx.spice` instantiates `Xref`).

| code | V_w | ntail | gm (µA/V) | **W_eff = gm − gm(132)** | sim assumes |
|---|---|---|---|---|---|
| 71 | 0.500 | 222 mV | 4.26 | **−64.2** | −0.95 |
| 95 | 0.668 | 131 mV | 29.49 | −38.9 | −0.58 |
| **119** | 0.837 | 43 mV | **68.66** ← **PEAK** | **+0.24** | −0.20 |
| **132** | 0.927 | 31 mV | 68.42 | **0** ✓ | 0 ✓ |
| 155 | 1.090 | 21 mV | 67.10 | **−1.32** | +0.36 |
| **191** | 1.342 | 15 mV | 65.79 | **−2.63** | **+0.92** |

**Code 191 should be the MOST POSITIVE weight available. It is slightly NEGATIVE.**

`dW/dcode` at the bottom of the range is **0.162**; at the top it is **−0.028**. Not merely
compressed — **sign-inverted.**

---

## 2. ⚠⚠ THE EXISTING TEST PASSED WHILE THE SIGN WAS BACKWARDS

`tb_wgt_zero_emx.spice` (FABLE, 2026-07-09) reports:

```
   code 128 (V_w=0.900):  +0.0038705 V   <- 'byte-centre' zero (WRONG)
   code 132 (V_w=0.929):   1E-07    V   <- DAC mid-scale zero (RIGHT)
   code 136 (V_w=0.957):  -0.0038092 V
   P2a-B PASS: I_diff flips sign across code 132
```

The sim's convention is `weight = code − 132`. So **code 136 is a POSITIVE weight** — and the
silicon delivers a **NEGATIVE current**. **THE SIGN IS INVERTED.**

**And the test passed anyway**, because `P2a-B` computes:

```
   sflip = I_diff(128) * I_diff(136)   and PASSES if sflip < 0
```

**That is a PRODUCT.** It is satisfied by a correct response **and by an inverted one.** It confirms
the two codes straddle a zero; it never checks **which side gets which sign.**

> ### ★ "Is there a sign flip" is NOT the same test as "is the sign RIGHT."

---

## 3. WHY IT WAS NEVER CAUGHT — four independent reasons

**1. The cell was characterised WHERE IT WORKS.** Every functional testbench pins **V_w = 0.75 V**:

```
tb_mac_cell_emx_dc / _e_sweep / tb_e_inject_pulse / tb_ce_readback / tb_temporal_emx
    -> Vw_char = 0.75      (code 107 — on the HEALTHY RISING side of the gm peak)
```

The weight range runs to **1.35 V**. **Nobody ever looked at the top half.**

**2. The WGT_ZERO work asked "WHERE is the zero?", not "WHAT SHAPE is the curve?"** It answered its
own question **correctly** — 132 genuinely is the zero-current code. But **finding the zero of a
function tells you nothing about its slope.**

**3. The sweep window hid it.** `tb_wgt_zero_emx` sweeps only **0.80–1.05 V** — ±0.125 V around the
peak at 0.90. On the falling side, *locally*, it looks like a clean monotonic sign-flipping
response. Which is exactly what it reported.

**4. ⚠ THE RTL↔SIM CROSS-CHECK CANNOT CATCH THIS, EVER.** `CROSS_CHECK.md` A1 records
*"Forward MAC … EXACT — 1024/1024, max|Δ|=0"*. That compares the **Python sim** against the
**behavioural RTL**. **Both use the same linear code→weight map.**

> ### ★★ TWO MODELS THAT SHARE AN ASSUMPTION CANNOT TEST THAT ASSUMPTION.
> This is the **THIRD** time this exact pattern has bitten in one day:
> * `cap_array.v` does a continuous `real` add the physical circuit cannot do — RTL↔sim can't see it.
> * the `peak`-vs-`rms` AGC default (8pp) — the doc and the code disagreed and nothing compared them.
> * **this.**
> **The only thing that could have caught any of them was the measurement nobody made.**

**⇒ THE MISSING MEASUREMENT: sweep the full code range, extract `dI_out/dx`, and ask
"WHAT WEIGHT DOES EACH CODE ACTUALLY PRODUCE?"**

---

## 4. ROOT CAUSE — the cell is starved at the top of the range

**Two mechanisms, both from the same cause:**

**(a) MN3_w goes into TRIODE.** At w=10 it demands more current than the w=4 PMOS mirror can supply.
`ntail` collapses **222 mV → 15 mV**. MN3_w needs `V_ds > V_gs − V_th ≈ 0.45 V` to act as a current
source; at WGT_ZERO it has **31 mV**. `I_tail` saturates (64.5 → 72 µA across the whole top).

**(b) THE INPUT PAIR goes into triode too.** At high V_w the output node sits at ~0.195 V (pulled
down by the 100 kΩ row load), leaving **MN2 with V_ds = 0.18 V against an overdrive of 0.405 V.**
The diff pair is starved of headroom as well.

Both degrade `gm` at the top of the range. **This is why weakening MN3_w fixes BOTH:** less tail
current ⇒ `iout` sits higher ⇒ MN2 recovers headroom.

---

## 5. THE FIX — MN3_w = 2 µm (from 10)

`sweep_mn3w.spice`, `W_eff = dI_out/dx` (µA/V):

| MN3_w | Vw=0.60 | 0.75 | 0.90 | 1.05 | 1.20 | 1.35 | monotonic? |
|---|---|---|---|---|---|---|---|
| **10 (current)** | 9.4 | 63.8 | **68.6** | 67.4 | 66.4 | 65.8 | ✗ **peaks, then FALLS** |
| 6 | 6.9 | 50.4 | 68.0 | 68.7 | 68.1 | 67.4 | ✗ |
| 4 | 5.7 | 35.7 | 64.4 | 68.3 | 68.7 | 68.5 | ✗ |
| **2** | 4.6 | 17.4 | 50.0 | 61.8 | 65.9 | **67.6** | **✓ MONOTONIC** |
| 1 | 4.1 | 9.9 | 29.1 | 45.4 | 54.1 | 59.1 | ✓ |
| 0.6 | 4.1 | 8.5 | 21.0 | 34.1 | 43.3 | 49.4 | ✓ |

**MN3_w = 2 restores monotonicity across the whole range**, at the cost of a gentler gm span.
Smaller MN3_w ⇒ more linear, but less weight range. **2 is the knee.**

⚠ **The curve is STILL COMPRESSIVE** — low codes far more sensitive than high ones. Monotonic and
correctly signed, but **NOT LINEAR**, which the sim still assumes.

---

## 6. ⚠ CONSEQUENCES — what has to be redone

1. **Re-size MN3_w to 2 µm** in `mac_cell_jug.spice`.
2. **RE-DERIVE WGT_ZERO.** It is a *property of the cell*, and the cell just changed. The design
   note is explicit: *"If you change the MAC cell (V_th, tail bias, mirror ratio) … you MUST
   re-measure the zero-differential-current code in SPICE and update WGT_ZERO to match."*
3. **REWRITE the sign test to be DIRECTIONAL.** Not `I_diff(lo) × I_diff(hi) < 0`, but
   **`I_diff(code) must INCREASE with code`** across the whole range.
4. **FEED THE MEASURED CURVE BACK INTO THE SIM.** `pcn_jug.py` assumes `W = (code − 132)/64`.
   The silicon gives a compressive, non-linear map. **Every accuracy number in the project
   (64.09%, 81.96%, all of them) assumes a weight the silicon does not deliver.**
   ⚠ It may well be harmless — a per-weight learning-rate variation is the SAME CLASS of
   perturbation we measured as FREE (10× leak spread: −0.58pp; ×2 θ mismatch: −0.76pp), and
   gradient descent is famously indifferent to a spread of learning rates.
   **BUT AFTER TODAY IT MUST BE MEASURED, NOT ASSUMED.**

---

## 7. The method lesson (the expensive one)

**The design was characterised at its nominal operating point, and the failure lives at the
extremes of the range.** V_w = 0.75 V is a perfectly good cell. V_w = 1.35 V is a broken one. Every
test looked at the first and none at the second.

**Characterise the RANGE, not the OPERATING POINT.**

---

# ✅ RESOLVED (same day). THREE FIXES, AND THE PROJECT'S NUMBERS SURVIVE.

## FIX 1 — `MN3_w: 10 → 2 µm`.  The sign inversion is GONE.
Re-measured with w=2: **MONOTONIC across the whole range.** `ntail` now spans **54–227 mV**
(was 15–222 — MN3_w in deep triode). The entire positive weight range works again.

## FIX 2 — `WGT_ZERO: 132 → 117` (V_w = 0.823 V).
**WGT_ZERO is a property of the CELL, and the cell changed.** Re-derived from the measurement so
the ± excursions are symmetric: **−31.8 / +32.0 µA/V**. (The design note said exactly this would be
necessary: *"If you change the MAC cell … you MUST re-measure the zero-differential-current code."*)

## FIX 3 ★ — **A PRE-DISTORTED (CALIBRATED) WEIGHT DAC.**  This is the one that matters.

Even fixed, the map is **SIGMOIDAL**: ~0.05 µA/V per code at the rails, ~1.42 in the middle — a
**28× sensitivity variation**. The jug's mechanism is *"one fire = one code"*, so a fire is worth
28× more weight in the middle of the range than at the rails.

**Measured cost of the raw sigmoid: −1.45pp** (80.51% at its own best θ=16, vs 81.96%).
It also forces θ upward (8 → 16), because the compression changes the effective step size.

### The fix: make the DAC non-linear so the WEIGHT is linear.
`weight_dac` maps code → V_w **linearly**; the cell maps V_w → weight **sigmoidally**. Compose them
and you get a sigmoid. **Pre-distort the DAC** (a calibration LUT: choose V_w per code so the
*weight* steps are uniform) and the composition is linear again.

**Is there room?** Linearising the WHOLE sigmoid is impossible — at the rails a uniform weight step
would need ~74 mV of V_w per code, far beyond the budget. But you do not need the whole sigmoid:

| uniform ±range | V_w needed | fits in 850 mV? |
|---|---|---|
| ±32 (100%) | 896 mV | ✗ |
| **±28 (88%)** | **427 mV** | **✓** |
| ±24 (75%) | 292 mV | ✓ |
| ±20 (63%) | 210 mV | ✓ |

**±28 units — 88% of the sigmoid's span — fits in 427 mV of the 850 mV budget, with FULL 121-code
resolution.** In the sim's units that is a weight rail of **±0.83** (vs ±0.95).

### MEASURED — and it recovers essentially everything

| design | weight | best |
|---|---|---|
| linear DAC ⇒ **sigmoid** weight, full range (θ=16, its best) | non-linear, 28× | **80.51%** |
| **★ PRE-DISTORTED DAC ⇒ LINEAR weight, ±0.83** | **linear** | **81.46%** |
| pre-distorted, ±0.71 | linear | 81.30% |
| pre-distorted, ±0.59 | linear | 81.22% |
| *linear map, full ±0.95 rail — the old, **UNACHIEVABLE** assumption* | — | *81.96%* |

> ## ⇒ **81.46% vs 81.96% is INSIDE THE 0.7pp NOISE FLOOR.**
> **A calibration LUT in the weight DAC recovers essentially everything, and the sim's
> linear-weight assumption stops being a fiction and becomes TRUE OF THE SILICON.**

**Bonus:** shrinking the rail from ±0.83 to ±0.59 costs only **0.24pp** — the weight *range* barely
matters (the frozen PCA weights sit well inside it). So there is generous margin: range can be
traded for linearity even more aggressively if a circuit constraint demands it.

---

## ⇒ THE SPEC CHANGES

| | was | **now** |
|---|---|---|
| `MN3_w` | w=10 | **w=2** |
| `WGT_ZERO` | 132 | **117** (V_w = 0.823 V) |
| `weight_dac` | linear code → V_w | **PRE-DISTORTED (calibration LUT)**: code → V_w chosen so the **WEIGHT** is uniform |
| weight rail | ±0.95 (never achievable) | **±0.83** (88% of the span, 427 mV of 850) |
| the sim's linear-weight assumption | a fiction | **TRUE — once the DAC is calibrated** |

⚠ **The three must stay in lock-step**, exactly as the WGT_ZERO design note warns. And the sign
test must be **DIRECTIONAL** (`I_diff must INCREASE with code`), never a product.
