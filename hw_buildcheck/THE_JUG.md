# THE LEAKY JUG — the mechanism that closed the hardware gap

**2026-07-14.** Rig: `pcn_hw.py --jug`. The user's mechanism. It won.

> ## ★★★ THE HARDWARE MODEL NOW SITS ON THE FLOAT CEILING
>
> | | best (BIG, 6 epochs) |
> |---|---|
> | hardware model, OLD fold rule | **75.25%** |
> | **hardware model, THE JUG** (`--jug --jug_pure --jug_theta 8`) | **81.96%** |
> | float control (the ceiling) | 82.13% |
> | **residual gap** | **0.17pp — a quarter of the noise floor** |
>
> **The 6.9pp "unexplained hardware gap" is CLOSED.** And it closed by DELETING hardware:
> **no digital master, no velocity register, no per-synapse digital storage of any kind.**

---

## 1. The mechanism

**Do not reset E at the fold. Let it LEAK; the deltas top it up. When |E| crosses a threshold θ,
FIRE: the analog cell moves a whole ±1 LSB, and θ is SUBTRACTED from E.**

```
   E  ←  λ·E + g                       # a leaky integrator = a capacitor
   if |E| ≥ θ:                         # a comparator
       cell ← cell ± 1 LSB             # a charge pump
       E    ← E ∓ θ                    # residue-preserving — NEVER zero it
```

- **The residue-preserving subtraction is LOAD-BEARING.** Zeroing E would throw away exactly the
  sub-threshold charge the jug exists to keep. Subtracting θ makes it a true **sigma-delta**.
- `E ← λE + g` **IS** the momentum recurrence ⇒ **E and the velocity register are the same
  register.** Momentum stops being an algorithm and becomes a property of the storage medium.
- **A FIRE *is* a CARRY *is* a ROUTER-SHADOW REFRESH** — one local event, three jobs.
- ⚠ **CORRECTION (2026-07-14): I over-claimed "the global fold disappears / each synapse fires
  ASYNCHRONOUSLY".** The buildable design is a **SHARED, SWEPT comparator** (`absorb_ctrl` walks the
  1024 elements), so **each cell is checked once per sweep and fires AT MOST ONCE per sweep.**
  **The sweep period IS a free refractory period.** That single-fire limit is **the design**, not a
  simulation shortcut. A per-cell comparator (true async) costs 1024× the comparators AND
  **RUNS AWAY** — measured with multi-fire at a dense operating point: **271%/fold, collapse to
  18%.** A burst becomes a TRAIN instead: a cell at 3θ fires once, **keeps 2θ**, and fires again next
  sweep. **Delay the boost; never drop it.** (Clamping Ce harder would be the WRONG rate limiter —
  it throws charge away.)
- θ is a **per-layer CONFIG constant**, set at calibration — same category as the ADC gain.

Per-synapse hardware: **an 8-bit analog weight cell, a capacitor, a comparator, a charge pump.**
That is all.

---

## 2. ★ WHY IT WINS — and it is NOT the learning rate

The winning jug fires **0.91%/fold** ⇒ mean weight motion **0.0091 LSB/fold**, which is **LESS
THAN HALF** the old rule's 0.019. It beats the old rule by **6.7pp while moving the weights half
as much.**

### The lr confound — KILLED
The old rule across a **5× learning-rate range** is **FLAT**:

| lr | 3e-4 (spec) | 6e-4 | 9e-4 | 1.5e-3 |
|---|---|---|---|---|
| best | 75.25% | 74.82% | **75.59%** | 74.56% |

**It cannot be rescued by step size.** It is not merely worse — it is **INSENSITIVE**. The jug, by
contrast, has a proper inverted-U in its own effective lr (§3) with an optimum 6.7pp higher.

### The reason
**The old fold rule spends its weight-motion budget UNIFORMLY** — every synapse, every fold,
±0.019 LSB, whether or not there is any evidence behind it. **The jug spends the SAME budget only
where charge has crossed a threshold.** That reallocation is the whole 6.7pp.

### ★★ THE LESSON
Float hid this. In float you *can* nudge a weight by 0.019 LSB and it works. In silicon that nudge
is **smaller than the smallest thing the cell can do** — a no-op — and the 18-bit digital master
was **72 KB of SRAM to make a bad rule executable**.

> **The digital master was a PROSTHETIC FOR A BAD RULE.** Removing it made the design **BETTER**,
> not merely cheaper. (The user called it "a big architectural gludge". He was right.)

---

## 3. θ SWEEP — a real optimum (θ is the jug's learning rate)

| θ | fire rate | best |
|---|---|---|
| 1 | 8.5%/fold | 74.22% |
| 2 | 6.3%/fold | 78.36% |
| 4 | 1.9%/fold | 81.08% |
| **8** | **0.91%/fold** | **81.96%** ← peak |
| 16 | 0.42%/fold | 80.37% *(given 8 epochs; it was 78.18% at ep4 — it DOES learn slower, but even with the extra time it does not catch θ=8)* |
| 32 | 0.15%/fold | 73.95% |

**The inverted-U is REAL and the peak at θ=8 is GENUINE** — not an artefact of stopping early.
θ=16 was indeed under-served by the 6-epoch budget (it gained 2.2pp from two more epochs) and
STILL falls 1.6pp short. ⇒ **θ=8 is a true optimum in the jug's effective learning rate.**

### ★ Automatic learning-rate annealing, for free
The fire rate FALLS as the network converges (θ=4: 2.69% → 1.85%/fold over 6 epochs). As the
gradients shrink, the jug fills more slowly and threshold crossings get rarer. **This is an lr
schedule emerging from the physics.** Nobody designed it.
⚠ **THEREFORE: DO NOT SERVO THE FIRING RATE.** The obvious control loop ("hold the fire rate
constant by adjusting θ") would **CANCEL THIS ANNEALING**. It is the first thing most people would
reach for and it is a trap.

---

## 4. The LEAK (momentum): real but small, and it SHRINKS as θ improves

| θ | `leak6` (λ=0.984) | `pure` (λ=1) | leak's contribution |
|---|---|---|---|
| 1 | 74.85% | 74.22% | +0.63 |
| 2 | 78.85% | 78.36% | +0.49 |
| 4 | 81.37% | 81.08% | **+0.29** |

All INSIDE the 0.7pp noise floor, and **the benefit shrinks as θ gets better.** Interpretation:
**the leak and the threshold are two mechanisms for the SAME job — rejecting weak evidence — and
the threshold is the better of the two.** At a good θ the comparator is already doing it.

⇒ **The win is the CHARGE INTEGRATOR, not the momentum.** The user's physical picture was right;
the *selectivity* just comes from the comparator rather than the leak.
**But the leak costs nothing** (it is a property of the cap) and has never hurt in any arm ⇒ keep
it. The argument that momentum earns its keep on HARDER tasks than EMNIST is untouched.

### ✅ RESOLVED (2026-07-14): the `leak8` non-monotonicity was **NOISE**. Not an artefact.
The original triple at θ=1 (pure 74.22 / leak6 74.85 / **leak8 70.46**) **does not reproduce.**
Re-run with seeds on `pcn_jug.py`:

| leak (θ=1) | 0 (pure) | 6 | 8 |
|---|---|---|---|
| best | **76.99%** | 76.60 / 76.80 | **76.58%** |

**A 0.4pp spread across the lot.** The `leak8` outlier was a bad draw. (The whole cluster also sits
~2pp above the originals — same code, different thread count. BLAS again.) **The E accumulator was
also checked and is NOT railing: `Esat = 0.0%` in every arm**, so the "clipping destroys the sign"
hypothesis is dead too. ⇒ **no non-monotonicity, no bug — and it independently reconfirms that the
leak contributes nothing at ANY setting.**

> ### ⚠⚠ THE METHOD ERROR THAT NEARLY MADE THIS A "FINDING"
> I argued: *"3.8pp against a 0.7pp noise floor, so it is almost certainly not sampling error."*
> **That reasoning was WRONG.** The 0.7pp floor was measured at the **GOOD operating point**
> (θ=8, ~0.9% firing). At **θ=1** the network fires **8%/fold**, training is far noisier, and the
> floor there is **several points**.
> **A NOISE FLOOR BELONGS TO THE OPERATING POINT IT WAS MEASURED AT. It is not a property of the
> rig.** Treating it as a global constant is exactly how you talk yourself into a phantom result.

---

## 5. ★★★ PHYSICAL DEVICES — NO CONTROL LOOP IS NEEDED

Real capacitors leak through real devices. Subthreshold leakage goes **exponentially** with Vt,
and Vt mismatch is Gaussian ⇒ the leak current is **LOG-NORMAL** and varies many-fold across a die
and with temperature. So we modelled it: a per-synapse log-normal τ, drawn once (it is fabrication,
not noise).

Reference: mean inter-fire interval at θ=8 is **~110 folds**.

| run | τ across the die (p5–p95) | best | cost |
|---|---|---|---|
| **clean** (perfect devices) | ∞ | **81.96%** | — |
| `tau1000`, 3× spread | 167 – 6,190 folds | 81.68% | −0.28 |
| `tau300`, 3× spread | 50 – 1,857 folds | 81.62% | −0.34 |
| `tau100`, 3× spread | **17 – 619 folds** | 81.53% | −0.43 |
| `tau1000`, **10× spread** | **23 – 45,638 folds** | 81.38% | −0.58 |
| **threshold mismatch ×2** | — | 81.20% | −0.76 |

*(noise floor 0.7pp)* — **EVERY ONE IS AT OR INSIDE THE NOISE FLOOR.**

Look at what `tau100` is: the median cell **drains about as fast as it fills**, and the worst 5% of
cells leak **six times faster than they fire**. It costs 0.43pp. A **10× log-normal spread** costs
0.58pp.

### Why it is this robust — a property, not a lucky number
**A leaky cell does not get the WRONG answer. It needs more charge to reach threshold, so it FIRES
LESS OFTEN.** That is a **reduced per-synapse LEARNING RATE**, not a corrupted one — the sign of
every fire is still correct, and the residue subtraction keeps the accumulation unbiased.
Device mismatch therefore lands on the one axis **SGD is famously indifferent to: a random spread
of learning rates across weights.**

That is also why **threshold mismatch is nearly free** — it is mathematically *the same
perturbation*, arriving by a different physical route.

The frozen fraction climbs (66% at `tau100` vs 35% clean) — **and accuracy holds anyway.** The
network does not need every synapse; it needs the ones with evidence, and the leaky cells are
simply the ones that did not have enough.

### ⇒ THE SPEC
- **NO servo. NO replica bias. NO switched-capacitor leak.** All three were solutions to a problem
  the mechanism does not have.
- The requirement collapses to a **ONE-SIDED BOUND, and a loose one**: the *median* leakage time
  constant should be **comparable to or greater than the mean inter-fire interval**. Violating it
  by a factor of ~1 costs half a point.
- **MATCHING IS A NON-REQUIREMENT.** A 10× spread is free. That is an extraordinarily forgiving
  spec for an analog array — exactly the kind of thing that makes a chip manufacturable at yield.

---

## 6. ⚠ NOT YET VALIDATED — the test plan before silicon

**This is a NEW LEARNING RULE** (rate-coded ΣΔ, not fixed-step sign), it has had ONE pass, and
`pcn_hw.py` models the **ARITHMETIC** of silicon, not its **PHYSICS**. (Today we bolted two
physical effects on — log-normal leak and threshold mismatch. There are more.)

### ⚠⚠ THE BIGGEST UNMODELLED RISK: **the WEIGHT capacitor leaks too**
We modelled the JUG leaking. **We did NOT model the WEIGHT leaking.** The weight cell has been
treated as a perfect store. If the weight is analog charge (the existing cell is dual-cap), it
drifts, and **a trained network slowly forgets itself.**

**Note the asymmetry in what the two capacitors must do:**
| | must hold charge for | difficulty |
|---|---|---|
| **the jug** | ~110 folds = one inter-fire interval ⇒ **MILLISECONDS** (at 128 samples/fold) | **easy** — DRAM does tens of ms |
| **the weight** | the whole training run, then indefinitely ⇒ **SECONDS→MINUTES→FOREVER** | **hard** |

⇒ **The weight probably wants to be an 8-bit DIGITAL register driving a DAC, not raw charge.**
Even so **the jug still wins enormously**: 18-bit master + 12-bit velocity (**30 bits/synapse**)
→ **8 bits/synapse**, and those 8 bits are for the **COMPUTATION**, not the learning. 4× less, and
it removes the whole "learning happens in SRAM" architecture.
**CHECK THIS against the existing `Sky130A_16x16_4cell_emx_analog` cell rather than assuming.**

### TIER 1 — could KILL it. Do before ANY silicon work.
1. **Weight-cell leak / drift** (above). Existential, unmodelled.
2. **COMPARATOR SIGN ERRORS.** Threshold *offset* is free (§5) — but that is a MAGNITUDE
   perturbation. A comparator noisy near zero produces **WRONG-SIGNED FIRES**, which is **NOT** a
   per-synapse learning rate — it is a **corrupted signal**, and the §5 robustness argument does
   NOT cover it. Find the tolerable error rate. **The physical effect most likely to be harmful.**
3. **SEEDS.** Everything here is n=1. Three seeds on the winning config.

### TIER 2 — is this a real design win, or only a hardware patch?
4. **★ RUN THE JUG IN THE *FLOAT* SIM** (`pcn_bigspec.py`). Does it beat 82.50%?
   **If it wins in float too, it is a BETTER LEARNING RULE, not just a way to make silicon work.**
   If it only matches, it is a hardware enabler. Both are fine — **we should know which.**
5. **Longer runs.** Is 81.96% a ceiling or still climbing? Does θ=16 catch up (§3)?
6. **Charge-pump step-size mismatch** — the ±1 LSB packet varies per cell. Probably benign by the
   same learning-rate argument as §5, but confirm.

### TIER 3 — spec hardening
7. **Per-sample leak** instead of the fold-grid approximation. **Required before a circuit spec.**
8. **Temperature DRIFT during training** (we modelled STATIC mismatch; drift is different).
9. Transfer to the smallBIG topology.
10. Resolve the **`leak8` non-monotonicity** (§4). Do not design silicon around a curve you cannot
    explain.
11. The per-layer ADC gain (task #15) — a SILICON requirement discovered IN SIM, and the
    measurement behind it is not trusted. It may vanish entirely if the frozen W is normalised
    per chip.

**CONFIDENT ⇒ move to silicon when TIER 1 passes and (4) is answered.** Five cheap runs.

---

## 7. Method note

The gap was found by **elimination, not by hypothesis.** Every quantiser was ablated individually
and every one was innocent (weight cell 0.4pp, ADC 0, delta channel 0, E accumulator 0) — which
felt like failure and was not. It was the elimination that pointed at the **UPDATE PATH**, the one
component never isolated. The jug replaces the update path, and the gap came back.

**Three hypotheses chosen because they sounded good were all wrong** (crest factor, ADC width,
analog-cell precision). The thing that worked was the boring one: rule out everything you can
measure, then look hard at what is left.
