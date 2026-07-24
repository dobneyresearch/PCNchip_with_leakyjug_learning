# CAN THE SPEC ACTUALLY BE BUILT? — bit-accurate hardware check

**Rig:** `pcn_hw.py` (integer-only model of BIG, 24/8/16 chips).
**Last updated: 2026-07-14.**

> ## ★★★ THE GAP IS CLOSED — and it closed by DELETING hardware. **→ READ `THE_JUG.md` FIRST.**
>
> | | |
> |---|---|
> | float control (this rig) | **82.13%** |
> | hardware model, OLD fold rule | 75.25% |
> | **hardware model, THE LEAKY JUG** | **81.96%** — residual gap **0.17pp** |
>
> The 6.9pp was **the UPDATE PATH**, exactly as the elimination argument below predicted. The old
> fold rule nudges EVERY synapse ±0.019 LSB EVERY fold — which in silicon is **smaller than the
> cell's LSB, i.e. a no-op**. The **18-bit digital master was 72 KB of SRAM to make a bad rule
> executable — a PROSTHETIC FOR A BAD RULE.**
>
> **The jug replaces it: E is a leaky capacitor; a threshold crossing fires a whole ±1 LSB.
> No digital master, no velocity register, no per-synapse digital storage. And it is ROBUST to
> real device mismatch (a 10× log-normal spread in leakage costs 0.58pp) ⇒ NO CONTROL LOOP.**
>
> **§8's "analog cell cannot be the accumulator" requirement is SUPERSEDED. Everything below
> about the QUANTISERS still stands** — they were all innocent, and that is what found the answer.

---

## 1. THE BASELINE IS NOW CONFIRMED

`--agc rms` reproduces the recorded headline: **75.25%** (recorded: 75.30%).
Float control: **82.13%** (the float sim on BIG: 82.50%). The rig is honest in both modes.

**⚠ NOISE FLOOR ≈ 0.7pp.** Two runs differing only in an inert parameter (`--e_acc_bits 20`
vs `28`) land 0.67pp apart. **Nothing below ~1pp in this rig is a result.**

---

## 2. RESULTS — every constraint, on the confirmed baseline (6 epochs, `--agc rms`)

| run | what it REMOVES | best |
|---|---|---|
| `float_ctl` | **everything** (float control) | **82.13%** |
| `base_rms` | nothing — **THE SPEC** | **75.25%** |
| `act16_e32` | the ADC (16-bit ≈ lossless) | 75.68% |
| `allmaster_rms` | **ALL** weight quantisation, forward **and** backward | 75.65% |
| `mulaw_peak2` | delta-code loss, peak AGC | 75.31% |
| `act8_e28` | E accumulator width (**inert control**) | 74.58% |
| `mulaw_rms` | delta-code loss (24–40% zeroed → 2–5%) | 74.50% |
| `bwdmaster_rms` | weight quant in the **backward path only** (MISMATCH) | 73.88% |
| `act12_e28` | the ADC (12-bit) | 73.45% |
| `dither_rms` | — (**adds** noise; does NOT test the stochastic-write option) | 73.11% |
| *(linear + peak AGC — the STALE DEFAULT)* | — | *66.69%* |

**Read the top of that table carefully. `float_ctl` is 6.5pp above EVERYTHING else.** No single
ablation moves the needle. That is the whole finding.

---

## 3. WHAT IS NOW RULED OUT (with numbers, on the confirmed baseline)

### ✅ The 8-bit analog weight cell is FREE — and needs NO precision at all
`all_master` gives BOTH the forward pass and the router's `W.T` an infinitely-fine weight.
**Gain: 0.4pp — inside the noise floor.**
⇒ The weight's extra bits do **NO REPRESENTATIONAL WORK**. They are *purely an accumulator*.
This is the single most important structural fact in this document: see §7.

### ✅ The quantised `W.T` is FREE — and MATCHING beats PRECISION
| forward | backward `W.T` | best |
|---|---|---|
| 8-bit cell | 8-bit cell (**matched**) | 75.25% |
| fine master | fine master (**matched**) | 75.65% |
| 8-bit cell | fine master (**MISMATCHED**) | **73.88%** ← worst |

Giving the backprojection a *finer* `W.T` than the operator that actually ran makes things
**WORSE**. The credit-assignment path does not want accuracy; it wants to be the **TRUE TRANSPOSE
OF THE FORWARD OPERATOR**. ⇒ **the router's 8-bit shadow copy is CORRECT, not a compromise.**
It also only needs refreshing when a cell CARRIES (~2% of synapses per fold) — low traffic.

### ✅ The 8-bit ADC is FREE (properly tested this time)
16-bit ADC (essentially lossless): **75.68%** — no gain.
⚠ **The first ADC sweep was INVALID.** Widening the ADC scales the activations, which scales
`E = D.T @ F`, which **saturates the 20-bit accumulator**. `--act_bits` MUST be swept with
`--e_acc_bits` or it measures saturation, not the ADC. (act8 needs a 20-bit E; act12 needs 24;
act16 needs 28.)

### ✅ The E accumulator is FREE at 8-bit activations — *measured, not assumed*
`--delta_stats`: peak |E| = **44,196** against a 20-bit rail of 524,287. **17 bits used of 20.
0.00% saturated.** Widening it to 28 bits: no change (74.58% vs 75.25%, inside noise).

### ✅ The delta channel is NOT the accuracy gap
Companding it from 24–40% zeroed down to 2–5% zeroed changes **nothing** (74.50% vs 75.25%).
**A hypothesis that looked excellent and was killed by the data. See §6.**

---

## 4. ⚠ WHAT IS *NOT* RULED OUT — the integer UPDATE path

`all_master` removes the weight quantisation but **keeps everything else** — including the
integer `E → sign(E) → momentum → master` chain. It scores 75.65%. So does every other arm.

**The update path is switched ON in every run except `float_ctl`.** It is the only component
never isolated. Differences from the float control that live there:

- `E` is accumulated as an **integer** and `.astype(np.int64)` **truncates** it;
- momentum decay is an integer **sign-magnitude shift** (μ = 0.875) vs float 0.9;
- the write is `hw.lr = 20` **master-LSB** (an integer) vs a float `lr`;
- `leaky()` is an **arithmetic shift** (α = 0.125, floors toward −∞) vs float 0.1;
- the per-layer ADC gain is an **arithmetic shift** (`acc >> s`, a floor) vs float `acc / 64`.

**None of these has been isolated.** Any of them could be the gap; so could an interaction.

---

## 5. ★ THE NEXT INSTRUMENT — a LADDER FROM THE FLOAT SIDE (not more guessing)

**The method that failed:** ablating ONE constraint at a time from the hardware side. Every arm
still has every *other* hardware feature switched on, so an interaction is invisible and a single
innocent constraint looks the same as a single guilty one. Three hypotheses were tested this way
and all three were wrong (§6).

**The method that will work:** start at the **float control (82.13%)** and switch **ON** exactly
ONE hardware feature at a time:

| rung | feature enabled | expected |
|---|---|---|
| 0 | none (float control) | 82.13% |
| 1 | integer **activation** datapath (ADC + per-layer shift + shift-leaky) | ? |
| 2 | quantised **delta** channel | ? |
| 3 | quantised **weights** (the 8-bit cell) | ~82% *(predicted free)* |
| 4 | integer **update** path (E → sign → momentum → master) | ? **← prime suspect** |
| 5 | all four (= the spec) | 75.25% |

**The four rungs MUST sum to 6.9pp. If they do not sum, the residual IS the interaction, and
that is the finding.** Requires splitting `--float_ref` into four independent flags
(`q_act` / `q_delta` / `q_wgt` / `q_upd`).

---

## 6. ⚠⚠ THE STALE-DEFAULT BUG — and three wrong hypotheses

### The bug (worth **8pp**, sat undetected in a doc claiming near-validation)
`pcn_hw.py` had **`peak` AGC** as its default. The recorded 75.30% headline was produced with
**`rms`**. The code comment next to it asserted the opposite of the document. Peak AGC scores
**66.69%**.

`--delta_stats` shows why: **peak AGC quantises 71–75% of all deltas to ZERO.**

| delta code | AGC | zeroed | clipped |
|---|---|---|---|
| linear (6-bit int) | **peak** | **71–75%** | 0% |
| linear | rms | 24–40% | 1–2% |
| companded (sign+3exp+2mant) | peak | 6–21% | 0% |
| companded | rms | **2–5%** | 1–2% |

**LESSON: a default that changes after the benchmark is a silent 8pp lie. Every "ruled out"
claim in the previous version of this document had uncertain provenance and had to be re-run.**

### Three hypotheses that were wrong
1. **"The gap is the delta channel's crest factor."** The channel IS lossy (above) and a linear
   6-bit code genuinely cannot span a crest factor of 12–22. But **fixing it buys 0 accuracy.**
2. **"The gap is the ADC."** No: a 16-bit ADC gives 75.68%.
3. **"The gap is the analog cell / needs an 18-bit master for precision."** No: `all_master`
   gives 75.65%. The precision is worth nothing.

---

## 7. ★ COMPANDED DELTA — keep it, but as ROBUSTNESS, not as a fix

| delta code | peak AGC | rms AGC |
|---|---|---|
| linear | **66.69%** | 75.25% |
| **companded** | **75.31%** | 74.50% |

Companding does **not** raise the ceiling. It makes the delta channel **INSENSITIVE TO THE AGC
SETTING** — it deletes an 8-point calibration cliff.

That cliff is not hypothetical: **it is exactly the trap we just fell into.** On silicon it is a
per-layer gain reference that must be right and stay right across process and temperature.
Companding removes the parameter for the price of a **priority encoder and a shift** — the SAME
six cross-chip wires, no multiplier. The decoded value is wider only *locally*, where width is
cheap.

**RECOMMENDATION: build the companded delta.** Not for accuracy — to remove a fragile
calibration that has already cost us 8pp once.

---

## 8. ★★ THE WEIGHT ACCUMULATOR — the arithmetic, and why the "18-bit master" framing was wrong

### The requirement is DYNAMIC RANGE, not precision
- weight range ≈ ±1.0; analog cell LSB = **1/64 = 0.0156**
- `lr = 3e-4` ⇒ **one write moves the weight 0.019 LSB** — a deterministic round-to-nearest
  discards it, ~51 times in a row
- even the **momentum steady-state velocity** `lr/(1−μ)` = 2.4e-3 = **0.15 of a cell LSB**

**The entire learning dynamic lives BENEATH the analog cell's resolution.** The cell only ever
sees the carry-out. This part stands.

### But it is an ODOMETER, not a measurement
Nothing is *transported* at 18 bits and nothing is *computed* at 18 bits:

| quantity | width |
|---|---|
| delta on the wire | **6 bits** |
| the fold's decision | **1 bit** — `sign(E)` |
| weight used by the crossbar | **8 bits** |
| weight used by the router's `W.T` | **8 bits** |
| the master's extra bits | **never leave the synapse; never touch the forward path** |

`all_master` (§3) **proves** this empirically: infinite weight precision is worth 0.4pp. The
extra bits do no representational work. They are **an integrator of one-bit decisions**, and
nothing more.

⇒ **The requirement is a TIME CONSTANT, not a precision.** And a time constant can be realised in
four substrates:

| substrate | mechanism | per-synapse cost |
|---|---|---|
| **bits** | digital residue (what we built) | ~10 bits SRAM + velocity |
| **time** | fold rarely, write a whole ±1 LSB | **none** |
| **charge** | a capacitor integrates it (ΣΔ / integrate-and-fire) | **none (analog)** |
| **probability** | stochastic write at the right rate | **none (an LFSR)** |

### ⚠ Two corrections to the previous version of this document
1. **"An 18-bit digital master, ~72 KB."** Wrong framing. The analog cell already **IS** the top
   8 bits. Only the **fractional residue** need be digital (~10 bits) — a sigma-delta /
   error-feedback quantiser, with a carry incrementing the cell by ±1 LSB. **Half the SRAM**, and
   an increment rather than a rewrite.
2. **The width is inflated by MOMENTUM, not by `lr`.** To count `±lr` steps to one cell-LSB you
   need to count to 52 — **a ~6-bit counter plus a sign**. `frac_bits=6` collapsed (66.9%) not
   because the counter failed but because **the momentum velocity register then has a resolution
   of 1** (`v = 0.875v + 1` saturates at 8) and momentum degenerates.

### ★ THE DECISION THAT UNLOCKS EVERYTHING: does momentum earn its keep in hardware?
With μ = 0.875 the steady-state velocity is **0.15 of a cell LSB**. **A leaky integrator with
that much leak can NEVER reach the threshold to fire a carry.**

⇒ **"Analog integrator (cap + comparator + charge pump)" and "drop momentum" are THE SAME
DECISION.** A local analog accumulator is only physically possible with a near-pure integrator —
i.e. no momentum, or a much longer time constant (μ ≈ 0.99, a ~128-fold memory instead of an
8-fold one). That is a different learning algorithm and must be earned, not assumed.

**Momentum was inherited from the float recipe and has NEVER been ablated against the constrained
rule.** If it does not pay, the architectural block dissolves: a ~7-bit local counter (or a
capacitor) replaces a 10-bit SRAM residue **plus** a 12-bit velocity register, and there is **no
boss-side per-synapse store at all.**

### ⚠ `--dither` does NOT test the stochastic-write option
The implemented `--dither` dithers the carry **while keeping the master** — it adds noise on top
of an accumulator that is already exact, so of course it costs (73.11%). **The real option —
stochastic write with NO residue storage — is UNTESTED.** Do not cite 73.11% against it.

---

## 9. THE FOUR REQUIREMENTS THE FLOAT SIM HID (these still stand)

1. **Hidden activations must be SIGNED** (bipolar ADC). leaky-ReLU emits negatives; a unipolar
   0..255 ADC clips them all to zero (57% zeros, −9pp). L0 **is** non-negative (that is what
   split-sign encoding is for); L1..L3 are **not**.
2. **PER-LAYER ADC GAIN** is mandatory (calibrated shifts on BIG: `[>>3, >>6, >>6]`).
   ⚠ **CORRECTED 2026-07-14: that is an 8× spread, NOT the "100:1" this document used to claim.**
   The 100:1 came from three measurements recorded in three DIFFERENT units. The real requirement
   is a **2–3 bit config register**, written by the boss at load time. **NOT a blocker.**
   The task survives for a different reason: **‖W‖ drives BOTH the forward activation scale AND
   the backward delta attenuation**, so normalising the frozen W per chip may remove this *and*
   fix attenuation at depth — one knob, two problems. A **scaling** experiment.
3. **The analog cell cannot be the accumulator** — but see §8 for the *correct* framing and the
   four substrates. (Worth ~8pp; the mechanism is open.)
4. **Momentum decay must be SIGN-SYMMETRIC.** `v -= v >> 3` is asymmetric about zero (`+1 >> 3 =
   0` never decays; `-1 >> 3 = -1` dies at once) ⇒ every weight biased POSITIVE. Use
   sign-magnitude.

---

## 10. WHAT IS FREE (all re-confirmed on the corrected baseline)

| primitive | cost |
|---|---|
| 8-bit SIGNED activation ADC | **0** (16-bit buys nothing) |
| 6-bit delta broadcast | **0** |
| 20-bit E accumulator | **0** (17 bits actually used) |
| 8-bit analog weight cell | **0** (0.4pp — inside noise) |
| quantised `W.T` in the router | **0** — and matched beats fine |
| **leaky α as a SHIFT (1/8)** | **0** ⇒ **no multiplier in the cell** |
| **momentum μ as a SHIFT** | **0** |
| digital readout | trivially buildable (`--err_mode mse` ⇒ no `exp()` needed) |

---

## 11. METHOD NOTES (the expensive lessons)

1. **A hardware model is worthless until its float control reproduces the reference sim.**
   Forcing that match caught two harness bugs (an lr in the wrong units, 64× too large; the
   asymmetric momentum decay).
2. **A default that changes after the benchmark is a silent lie.** The `peak` AGC cost 8pp and
   sat undetected inside a document asserting near-validation. **Pin the config that produced
   every recorded number.**
3. **Never sweep one width in isolation.** `--act_bits` silently saturates `--e_acc_bits`.
   Check what a width change does DOWNSTREAM before believing the result.
4. **Establish the noise floor before interpreting small deltas.** It is ~0.7pp here. Three
   "findings" in the previous version were inside it.
5. **One-at-a-time ablation from the constrained side cannot find an interaction.** Ladder from
   the *unconstrained* side (§5).
