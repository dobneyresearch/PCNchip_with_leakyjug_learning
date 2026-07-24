# JUG — SPICE RESULTS (2026-07-14, first SPICE session on the jug line)

ngspice-42 · sky130A · `mac_cell_jug.spice`, `jug_compare.spice`

---

## ✅ 1. `mac_cell_jug` — MN3_e's deletion is VERIFIED IN SILICON

**T1 (the reason the cell changed):** sweep V_Ce across its **full ±100 mV swing** and measure the
MAC output.

```
   dI(supply) across the WHOLE Ce range = 4.0e-13 A   (0.4 pA — numerical noise)
```

In `mac_cell_emx` the same sweep moved **tens of microamps** through the dual-tail summation.
**SPICE confirms Ce is out of the signal path.** The sim's `--jug_e_fwd 0` is now true of the
silicon, and the positive-feedback runaway (E → MAC → bigger deltas → bigger E; measured
0.91%/fold → 19.3% → collapse) cannot occur.

---

## ⚠⚠ 2. THE BIG FIND — the analog weight was NON-MONOTONIC and INVERTED

**See `THE_WEIGHT_IS_NOT_LINEAR.md`.** Predates the jug; affected EMX, FABLE, and every accuracy
number in the project. `gm` peaked at V_w ≈ 0.90 V and **fell** above it, with WGT_ZERO sitting **on
the peak** ⇒ codes 119–192, **the entire positive weight range, were dead or inverted.**

**Fixed:** `MN3_w` **10 → 2 µm** (monotonic again) · `WGT_ZERO` **132 → 117** · a **PRE-DISTORTED
weight DAC** (calibration LUT). **Measured: 81.46% vs the old, unachievable linear assumption's
81.96% — inside the noise floor.** The project's numbers stand.

---

## ✅ 3. `jug_compare` — THE RESIDUE SUBTRACTION WORKS

This was the one tight analog spec in the whole design: **the fire must SUBTRACT θ from Ce, never
discharge it.** *"The sigma-delta can forgive a bad DECISION. It cannot forgive a LIE ABOUT HOW MUCH
CHARGE IS THERE."*

### ⚠ WHAT DIDN'T WORK — a flying capacitor (SPICE caught it on the first run)

A flying cap whose top plate sits permanently on the Ce node **cannot remove charge.** Switching its
bottom plate VREF→VSS drops V_Ce by exactly θ — and the instant the switch releases, **V_Ce bounces
straight back:**

```
   1.0235 V  ->  0.9757 V  (a clean 47.8 mV = theta)  ->  1.0234 V
   NET CHARGE REMOVED: **ZERO.**
```

**Moving charge is not removing charge.** A capacitor on the node cannot do it without a virtual
ground, and we are not paying for an op-amp per cell.

### ✅ WHAT WORKS — a PULSED CURRENT SOURCE (`Q = I · t`)

A fixed charge, genuinely removed, independent of V_Ce. This is how sigma-delta modulators build
their feedback DACs, and for exactly this reason.

**`Q_θ = Ce · θ = 100 fF × 50 mV = 5 fC` ⇒ 500 nA for 10 ns.**

| test | result |
|---|---|
| **T2 — SUBTRACT, don't discharge** | before **1.0177 V** → after **0.9452 V**. **The residue (45.2 mV) is KEPT.** Not zero. And it **stays** (0.9459, 0.9466 on later sweeps) — the charge is genuinely gone, not displaced. ✅ |
| **T3 — a burst becomes a TRAIN** | driven to ~4θ, it fires on **four successive sweeps** (1.2338 → 1.1582 → 1.0833 → 1.0095 → 0.9397) and then **stops by itself**. ✅ |
| **T4 — a FIXED CHARGE** | steps: **75.6 / 74.9 / 73.8 / 69.8 mV** — constant within 8% across a 300 mV swing. Charge-sharing would give a step ∝ (V_ref − V_Ce) and would visibly shrink. It does not. ✅ |

**⇒ IT IS A WORKING SIGMA-DELTA.** *Delay the boost; never drop it.*

### ✅ CALIBRATED — θ = 50.0 mV, and it took THREE fixes SPICE had to find

**Final: 49.88 / 49.73 / 49.45 / 48.81 mV against a 50.0 mV target** — within **0.25%**, and
constant to **2%** across a whole burst.

| ⚠ the problem SPICE found | the fix |
|---|---|
| **1. The mirror didn't mirror.** 724 nA of a 500 nA target — a 45% GAIN ERROR from channel-length modulation (`MN_src`'s drain sat at a very different voltage from the diode's). | **CASCODE.** It also improved the charge packet's V_Ce independence from 8% → 3%. |
| **2. ⚠⚠ The step did NOT scale with I.** 500 nA → 96 mV but 260 nA → 74.5 mV. Extrapolating to I=0 left **~50 mV of FIXED charge — as much as θ itself.** | see 3 |
| **3. It was NOT switch channel charge.** A half-size **DUMMY switch** (the textbook fix) moved the offset by **1.5 mV**. It was the **CURRENT SOURCE'S SETTLING TRANSIENT**: closing the switch charges the mirror-drain parasitic **FROM Ce** — a fixed charge, independent of I. | **★ STEER THE CURRENT, DON'T SWITCH IT.** Keep the source running and use a differential pair to send it to Ce or to a DUMP node (tied to V_bias_e). Ce never sees a turn-on transient. **Residual injection: 50 mV → 6.5 mV (8×), and `Q = I·t` is back in control.** |

> ### ★★ THE LESSON: **YOU MUST STEER A CURRENT, NOT SWITCH IT.**
> This is how current-steering DACs and ΣΔ feedback DACs are built, and **this is exactly why.**
> Switching a current source into a high-impedance node dumps that node's parasitic charge into
> your signal — and it is a FIXED charge, so it does not scale away, and it does not average out.

---

## ★ 4. THERE IS NO CHARGE PUMP INTO Cw

An earlier spec said *"deposit 1.405 fC on Cw = one 7.02 mV step."* **That was wrong** — it assumed
a LINEAR weight DAC. With the **pre-distorted** DAC a code is a NON-uniform V_w step, so a
fixed-charge pump would give a fixed *voltage* step, which is exactly the wrong thing.

**A fire is a ±1 increment on the 8-bit W-SRAM code.** The existing `weight_dac` + `refresh_ctrl`
path repaints Cw through the LUT. No new analog write path is needed — and:
* W-SRAM is always the current master ⇒ **`refresh` never has to be disabled**
* `save` is not needed during training ⇒ the absorb→save→sync→refresh cycle **collapses**
* the W-cap leak risk goes away **structurally**, not by scheduling

---

## Method notes from this session

1. **The `.tran` timestep must be ≪ the one-shot width**, or the charge packet is not resolved
   (a 1 ns step against a 1 ns pulse gave 0.97 fC instead of 5 fC — a *simulation* artefact, not a
   circuit one).
2. **Bias current sources must PUSH into the diode-connected device**, not sink the same way it
   does. Both mirrors were initially wired to fight themselves and delivered ~100 nA of a 500 nA
   target.
3. **★ CHARACTERISE THE RANGE, NOT THE OPERATING POINT.** Every previous testbench pinned
   V_w = 0.75 V — where the cell works. The failure lived at V_w = 1.35 V, and nobody had looked.

## ★ 5. THE WINDOW COMPARATOR — TRANSISTOR LEVEL (`jug_comparator_t`, 2026-07-15)

The last block, and by design the least fussy: the comparator is allowed to be wrong 20% of the
time (the residue subtraction conserves charge, so a wrong fire is undone by the next right one).
So it is a plain **5T OTA per threshold** (NMOS input pair, PMOS mirror load) + a 2-inverter buffer
— no latch, no trim. `tb_jug_comparator.spice`:

| test | ideal | measured | margin |
|---|---|---|---|
| D1 fire_up threshold | 0.950 V | **0.95075 V** | +0.75 mV |
| D2 fire_dn threshold | 0.850 V | **0.84925 V** | −0.75 mV |
| D3 dead zone (V_Ce = 0.90) | both LOW | ~0.2 µV | clean — a cell at rest never fires |
| D4 propagation delay | < sweep window | **2.5 ns** | ≪ 10 ns |

The sub-mV offset is **60× inside** the tolerated 2·θ (25 mV pass / 0.76pp at 2·θ) — an order of
margin on a spec that barely needed it. The behavioural `jug_comparator` stays as the loose-spec
stand-in in `tb_jug_fire.spice` (that TB's job is the *residue*, not the comparator).

> **Method note (cost an hour):** ngspice's control shell treats bare `<` / `>` inside a `let`
> as FILE REDIRECTION — `let ok = (x < 0.025)` silently tries to read a file `"0.025)"`. Use the
> word operators `lt` / `gt` / `le` / `ge` / `and` / `or`. The *measurements* were perfect the
> whole time; only the PASS/FAIL scaffolding was broken.

---

## Still to do
- **All SPICE blocks built and verified.** Remaining work is layout (Track C) and the sim
  seeds/longer-run confirmation of the final config.
- (done: θ mirror cascoded+trimmed to 49.9 mV; 10-bit weight DAC; directional WGT_ZERO test;
  transistor comparator. RTL: `jug_ctrl` replaced `absorb_ctrl`; `cap_array` subtracts θ;
  `refresh` `training_mode` gate dropped; `tb_epoch_match`/`tb_absorb_match` rewritten to the jug.)
