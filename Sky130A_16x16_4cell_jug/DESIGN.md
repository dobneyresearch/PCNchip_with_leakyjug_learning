# Sky130A_16x16_4cell_jug — Design Document

> ## ⚠⚠⚠ READ `circuit/THE_WEIGHT_IS_NOT_LINEAR.md` FIRST (2026-07-14)
> The first SPICE sweep of the **full** weight range found a bug that **predates the jug** and
> affects EMX, FABLE and **every accuracy number in the project**: the MAC's effective weight
> `gm(V_w)` **PEAKED at V_w ≈ 0.90 V and FELL above it**, with WGT_ZERO (code 132) sitting **on the
> peak** — so **codes 119–192, the entire POSITIVE weight range, were dead or INVERTED.**
>
> It survived because **every testbench pinned V_w = 0.75 V** (the healthy rising side), the
> WGT_ZERO test's sign check was a **product** (`I_diff(128)·I_diff(136) < 0` — which passes for an
> inverted response too), and **the RTL↔sim cross-check shares the linear-weight assumption, so it
> cannot test it.**
>
> ### ✅ RESOLVED — THREE SPEC CHANGES
> | | was | **now** |
> |---|---|---|
> | `MN3_w` | w=10 (in TRIODE; ntail collapsed to 15 mV) | **w=2** — weight is MONOTONIC again |
> | `WGT_ZERO` | 132 | **117** (V_w = 0.823 V) — *a property of the CELL, and the cell changed* |
> | `weight_dac` | linear code → V_w | **PRE-DISTORTED (calibration LUT)** so the **WEIGHT** is uniform |
> | weight rail | ±0.95 (never achievable) | **±0.83** (88% of span; 427 mV of the 850 mV budget) |
>
> **Measured: the pre-distorted DAC gives 81.46% vs the old (unachievable) linear assumption's
> 81.96% — INSIDE the 0.7pp noise floor.** The raw sigmoid costs 1.45pp (80.51%).
> ⇒ **A calibration LUT in the weight DAC recovers essentially everything, and the sim's
> linear-weight assumption becomes TRUE OF THE SILICON rather than a fiction.**


**W+E backprop with the LEAKY JUG absorb, on a SINGLE-TAIL cell.** Rev 2026-07-14.
Supersedes `DESIGN_DEPRECATED_2026-07-14.md` (the block/epoch absorb, headline 64.09%).

Analog circuit design (SPICE) is the primary work item.

---

## 0. What changed, and why

The learning rule changed. The old rule consolidated E into W on a **global schedule**
(`W += lr_slow·E`, then discharge Ce), and a long series of epoch/phasing experiments never got it
past **64.09%**. The current rule — frequent updates, sign-at-fold, momentum — reaches **82.50%**
in float, against a full-backprop ceiling of 82.85% on the identical topology.

**But the new rule cannot be written into this cell as it stands.** Its weight update is
`lr·sign(E)` with `lr = 3e-4`, which is **0.019 of one weight LSB** (LSB = 1/64). A single write
moves the cell by one-fiftieth of the smallest change it can physically make: **the write is a
no-op, ~51 times in a row.** The old absorb never had this problem because it wrote the *magnitude*
of E, clamped to ±6.7 codes — **a factor of ~350 larger.** The chip was built for a rule we have
replaced.

**The leaky jug is how the new rule runs on this cell.**

> ### THE JUG, in one block
> ```
>   Ce accumulates δ·x injections and IS NOT DISCHARGED.
>   When |V_Ce − V_bias| ≥ θ:            # a COMPARATOR
>       W_code ← W_code ± 1               # one whole LSB — all the cell can do
>       V_Ce   ← V_Ce ∓ θ                 # SUBTRACT θ. The residue is KEPT.
> ```
> **The subtraction is load-bearing.** Discharging Ce (the old Step 4) throws away exactly the
> sub-threshold charge the mechanism exists to accumulate. Subtracting θ makes this a true
> **sigma-delta**: no charge, and therefore no learning signal, is ever lost.

**Measured (bit-accurate model, BIG, 24/8/16 chips):**

| | best |
|---|---|
| old absorb rule (block/epoch) | 64.09% |
| new rule, naively written to an 8-bit cell | 75.25% |
| **new rule via the JUG** | **81.96%** |
| float ceiling | 82.13% |

**And it is a circuit SIMPLIFICATION** — see §5. It deletes the Ce-readback path entirely.

---

## 1. Design intent

Both W and E stored as analog voltages on physical capacitors.

- **WeightMx (W)** — stable, slowly-updated weight, analog voltage on **Cw**. Backed by W-SRAM,
  refreshed against leakage.
- **ErrorMx (E)** — the learning accumulator, analog voltage on **Ce**. Under the jug it is a
  **charge integrator with a threshold**, not a per-sample residual that gets discharged.

### ★ Why the leak is a feature, and why the physics gives us the best case for free

A leaking capacitor and momentum are **the same equation**: `E ← λ·E + g` *is* the momentum
recurrence. So the E-cap's leak *is* the momentum term — momentum stops being an algorithm we
implement and becomes a property of the storage medium.

**And the numbers say we land in the best regime automatically.** From §9: Ce ≈ 100 fF with
subthreshold leakage 5–10 fA gives **≈ 50 µV/s droop** — about **140 s to drift one E-LSB** (7 mV).
The jug's mean inter-fire interval is ~110 folds ≈ **tens of milliseconds**. So

```
   τ_leak / (inter-fire interval)  ≈  10³ – 10⁴
```

**The E-cap is, on the jug's timescale, a pure integrator.** That matters because a pure integrator
(no leak) is exactly the configuration that scored **best** in simulation (81.96%, vs 81.37% with a
64-fold leak — the difference is inside the noise floor).

**⇒ NO leak trim, NO servo, NO replica bias, NO switched-capacitor leak is required.** The
requirement is a loose **one-sided bound**: τ_leak ≫ the inter-fire interval. We have it by ~10³.

Simulated device mismatch confirms the margin is not fragile (clean = 81.96%):

| devices | best | cost |
|---|---|---|
| τ = 1000 folds, 3× log-normal spread | 81.68% | −0.28 |
| τ = 100 folds (drains ≈ as fast as it fills), 3× spread | 81.53% | −0.43 |
| τ = 1000 folds, **10× spread** | 81.38% | −0.58 |
| **threshold (θ) mismatch ×2** | 81.20% | −0.76 |

All inside the 0.7pp noise floor. **Matching is a non-requirement.** The reason is structural: a
leaky or high-threshold cell does not get the *wrong* answer — it simply **fires less often**,
which is a reduced *per-synapse learning rate*, and gradient descent is famously indifferent to
that. Every fire it does make still carries the correct sign.

---

## 2. Analog cell architecture: `mac_cell_jug` — **★ THE E-TAIL IS REMOVED**

> ### ★★ MN3_e IS DELETED. The cell is SINGLE-TAIL.
> The old cell was **dual-tail** — `I_ntail = I_tail_W + I_tail_E` — so Ce injected current into
> the MAC and the effective weight was `W + E`. That was deliberate: under the old rule E was a
> **live residual** the MAC was *meant* to see.
>
> **Under the jug, E is an ACCUMULATOR, not a residual, and its presence in the forward path is
> PURELY HARMFUL.** Measured (weight codes a full-threshold E contributes to the MAC):
>
> | E in the MAC | 0 | 0.1 | 0.2 | **0.4 (the old gm_E ≈ gm_W point)** | 1.0 | 2.0 | 6.7 (the clamp) |
> |---|---|---|---|---|---|---|---|
> | best | **81.96%** | 82.06% | 81.32% | **80.46%** | 76.83% | **COLLAPSE** 47→12% | **COLLAPSE** 10% |
>
> It is a **POSITIVE-FEEDBACK RUNAWAY**: E feeds the MAC → activations grow → deltas grow → E grows
> faster → fires more (fire rate 0.91%/fold → 14.8% → **19.3%**). The old design point sits at
> ~0.4 codes — **already lossy, and only ~2.5× from runaway**, on a gm ratio that drifts with
> process and temperature.
>
> **⇒ Ce never needs to inject current into the MAC. It only needs to be COMPARED against θ.**
> Removing MN3_e:
> * recovers the full **81.96%** (vs 80.46%),
> * returns the MAC core to the original **single-tail 5T** topology,
> * **and DISSOLVES the entire V_bias_e problem** — see §2.3.

### 2.1 Transistor inventory

| Transistor | Role | Gate drive | Notes |
|-----------|------|-----------|-------|
| MN1 | Diff pair (positive input) | inp | unchanged |
| MN2 | Diff pair (negative input) | inn | unchanged |
| MP1 | PMOS mirror (reference leg) | nmp1 (diode) | unchanged |
| MP2 | PMOS mirror (output leg) | nmp1 | unchanged |
| MN3_w | **the ONLY tail**: I_tail ∝ (V_w − V_th)² | Cw (200 fF) | unchanged |
| ~~MN3_e~~ | ~~E-tail~~ | — | **★ DELETED** |
| MN4_w / MP4_w | CMOS TG for Cw write access | we_w / we_w_n | unchanged |
| MN4_e / MP4_e | CMOS TG for Ce **inject** access | we_e / we_e_n | retained — Ce is still WRITTEN |
| **CMP** | **θ comparator on Ce** | — | **★ NEW** (§6.4) |

**Ce is now a STORAGE-AND-COMPARE node only.** It is written by the inject circuit and read by a
comparator. It is **not in the signal path.**

### 2.2 Single-tail MAC — Ce is NOT in the signal path

```
I_ntail = I_tail_W = k_W(V_w − V_th)²                    ← ONE tail. Ce contributes nothing.
I_out  ≈ gm_W(V_w) × (inp − Vcm)
```

The differential pair splits a tail current set **solely** by the weight capacitor. This is the
original `mac_cell` behaviour, unchanged since before the EMX work.

### 2.3 Ce biasing — **★ THE CONSTRAINT IS GONE**

The deprecated design was forced to **V_bias_e = VPI = 0.65 V** (not VCM = 0.9 V) because MN3_e
shared `ntail`: at 0.9 V the MN3_e saturation requirement `V(ntail) > 0.42 V` and the diff-pair
requirement `V(ntail) < inp − V_th = 0.42 V` were **simultaneously impossible** — both tails went
into triode at V(ntail) ≈ 22 mV (SPICE, 2026-06-27). That cost a design iteration.

**That constraint existed ONLY because MN3_e shared `ntail`. With MN3_e deleted, it vanishes.**

Ce's bias is now free, and is chosen on two much easier criteria:
1. the **comparator's** input common-mode range, and
2. the **inject circuit's** output compliance.

V_bias_e may stay at 0.65 V (it is already generated by `bias_gen`) or move — **it is now a free
design variable, not a hard constraint.** ⚠ Re-derive it against the comparator (§6.4); do not
inherit 0.65 V by default.

### 2.4 Ce sizing — **100 fF, and the leakage is now a POSITIVE result**

- **Leakage ≈ 50 µV/s** (5–10 fA on 100 fF) ⇒ **~140 s per E-LSB** (7 mV).
- The jug's mean inter-fire interval is ~110 folds ≈ **tens of milliseconds**.

```
   τ_leak / (inter-fire interval)  ≈  10³ – 10⁴
```

**The physical Ce is a PURE INTEGRATOR on the firing timescale** — which is exactly the
best-performing configuration in simulation (`--jug_leak 0`, 81.96%). **The physics hands us the
optimum for free.** See §1.

⚠ **Ce sizing is now driven by the COMPARATOR, not by gm matching.** The old constraint
"gm_E(V_bias) ≈ gm_W(V_w_mid)" is **deleted** along with MN3_e. Size Ce for charge resolution
against the inject quantum and the comparator's offset — see §6.4.

---

## 3. Forward pass (data upward) — **W ONLY**

```
act_sram → inp_dac (R-2R, 8-bit) → V(inp_node) = Vcm ± x_j
        → MN1/MN2 split I_ntail = I_tail_W           ← Ce contributes NOTHING
        → I_out_j = gm_W(V_w_j) × x_j
        → KCL row sum: I_row_i = Σ_j I_out_j
        → Rload → V(iout_i) → SAR ADC (8-bit) → act code a_i
        → act_sram → router → next-layer inp_dac
```

**The MAC computes `W`, not `W + E`.** This now matches the simulation exactly (`--jug_e_fwd 0`),
which is the point of removing MN3_e: **the silicon and the model finally compute the same thing.**

⚠ **The hidden-layer ADC must be SIGNED (bipolar).** leaky-ReLU emits negatives (α·y for y<0); a
unipolar 0..255 converter clips them all to zero, silently turning leaky-ReLU into hard-ReLU
(measured: 57% zeros, −9pp on the untrained baseline). The **L0 input** is non-negative by
construction — that is what split-sign encoding is for — but **L1..L3 are not.**

⚠ **PER-LAYER ADC GAIN** is required (calibrated shifts on BIG: `[>>3, >>6, >>6]`). This is
**config, not runtime discovery**: a ~3-bit shift register the boss writes at load time, like the
weights and the routing. ⚠ **CORRECTED 2026-07-14:** the calibrated shifts are `[>>3, >>6, >>6]` = an **8x spread**, **NOT the '100:1' claimed earlier** (that came from three measurements in three DIFFERENT units). The real requirement is a **2-3 bit config register**. Cheap, and **NOT a blocker**. Task #15 survives for a different reason: **||W|| drives BOTH the forward activation scale AND the backward delta attenuation**, so normalising the frozen W per chip may remove this *and* fix attenuation at depth. A SCALING experiment, not a jug blocker.

---

## 4. Downward pass (E injection) — **UNCHANGED**

```
Host/boss: label, prediction → delta[0..15], boss_h[0..7]
        → WB regs: delta_flat (N_CELLS × N_ROWS × 8-bit), bh, lr_shift
        → e_inject_ctrl FSM: 256 cycles (N_ROWS × N_COLS)
             inject_delta = delta[r]      (signed 8-bit, row direction)
             inject_x     = x_latch[c]    (captured from inp_dac in the forward pass)
             ΔV_e = δ · x · (bh+1) / 2^lr_shift
        → [ANALOG] signed charge injection onto Ce[r×N_COLS+c]:  V_Ce += ΔV_e
        → clamp V_Ce to [V_bias ± E_MAX_V]
```

`lr_shift` scales the injection and, together with **θ** (§5), sets the effective learning rate.

---

## 5. ★ ABSORB — THE LEAKY JUG (replaces the old §5 entirely)

**The old absorb is gone.** It read every Ce with an ADC, computed `round(V_w + V_e)`, repainted Cw
through the DAC, and **discharged Ce**. The jug replaces all of that:

```
absorb_ctrl sweeps all 1024 elements:

  Step 1 — COMPARE:  is |V_Ce − V_bias_e| ≥ θ ?          ← a COMPARATOR. No ADC. No Ce readback.
  Step 2 — if YES:
             W_code ← clip(W_code ± 1, WGT_MIN, WGT_MAX)  ← ±1 code. No full DAC repaint.
             V_Ce   ← V_Ce ∓ θ                            ← SUBTRACT θ. DO NOT DISCHARGE.
  Step 3 — if NO:  do nothing. The charge stays on Ce.
```

### ★★ THE COMPARATOR IS **SHARED AND SWEPT** — and the sweep is a FREE REFRACTORY PERIOD

`absorb_ctrl` walks the 1024 elements **sequentially**, so **each cell is checked once per sweep and
can therefore fire AT MOST ONCE per sweep.** That single-fire limit is **not** a simulation
shortcut — **it is the design**, and it is what makes the mechanism stable.

⚠ **DO NOT put a comparator in every cell.** It costs 1024× the comparators, it lets a synapse fire
repeatedly the instant it crosses, and **it runs away**: with multi-fire enabled at a dense
operating point, synapses fired at **271%/fold** and the network collapsed to **18%**. A per-cell
comparator would then need an explicit **refractory timer** bolted on to undo exactly the freedom it
just bought. **The shared swept comparator is cheaper AND inherently rate-limited. It is strictly
better.**

### A burst becomes a TRAIN — delay the boost, never drop it

A cell that overshot to 3θ **fires once, keeps 2θ** (the residue subtraction), and fires again on the
next two sweeps. **The charge is conserved and the boost is spread over time.**

⇒ This is why **clamping Ce harder is the WRONG rate limiter** — it *throws charge away*, which is
the one thing the whole mechanism exists to prevent. The sweep is the right one: it **delays**.

### Why this is *simpler* than what it replaces

The deprecated §6.3 required **one** of two unattractive things:

- **Option A** — a shared SAR ADC time-multiplexed to every Ce node: **256 ADC conversions per cell
  per absorb**, plus a MUX and `adc_src_sel`.
- **Option B** — a **12-bit digital accumulator per synapse**, which that document itself conceded
  *"re-introduces a digital register file for E"*.

**The jug needs NEITHER.** A comparator against a θ reference, a ±1 increment, and a charge
subtraction. **§6.3 is deleted.** The all-analog principle is preserved rather than compromised.

### θ — a per-layer CONFIG constant

θ is a comparator reference voltage, **written once at configuration time** — the same category as
the weight codes, the routing, and the ADC gain. **A chip never discovers θ at runtime.** Simulation
optimum: **θ = 8 × the per-fold rms of |E|** (a clean inverted-U; θ=4 → 81.08%, **θ=8 → 81.96%**,
θ=16 → 80.37%, θ=32 → 73.95%).

### ⚠⚠ DO NOT servo the firing rate

The firing rate **falls on its own** as the network converges (2.69% → 1.85%/fold over a run): the
gradients shrink, Ce fills more slowly, crossings get rarer. **That is an automatic learning-rate
schedule, emerging from the physics.** A control loop that held the firing rate constant by
adjusting θ would **cancel it**. It is the first thing anyone would reach for and it is a mistake.

### ★ Consequence: `refresh` can run continuously

If a fire is a **±1 increment on the 8-bit W-SRAM entry**, then W-SRAM is **always** the current
master, Cw is a pure slave, and `refresh_ctrl` never needs disabling:

- the `training_mode` gate on `refresh_ctrl.v` can be **removed**
- the `absorb → save → SHADOW_W_SYNC → refresh-enable` coherence cycle **collapses**
- `save_ctrl` is no longer needed *during* training (W-SRAM is always current)
- **the W-cap leak risk is removed structurally, not by scheduling**

---

## 6. SPICE circuit work required

### 6.1 `circuit/mac_cell_jug.spice` — **SINGLE-TAIL** MAC cell **[PRIORITY 1]** *(SIMPLIFIED)*
Characterise **gm_W only**. Verify the single tail stays in saturation across the W range.
**★ The gm_E characterisation and the gm_E ≈ gm_W matching requirement are DELETED with MN3_e** —
this is a strictly smaller SPICE job than the deprecated §6.1, and the dual-tail saturation
conflict (§2.3) no longer exists.

### 6.2 `circuit/e_inject_emx.spice` — analog charge injection **[PRIORITY 2]** *(unchanged)*
Signed charge injection ΔV_e = f(δ, x, bh, lr_shift) onto Ce via MN4_e/MP4_e. Clamp at ±E_MAX_V.

### 6.3 ~~`circuit/ce_readback.spice`~~ — **DELETED. Not required.**
The jug does not read V_Ce. It compares it.

### 6.4 `circuit/jug_compare.spice` — **the θ comparator + ±1 charge pump [PRIORITY 3 — NEW]**

**★ ONE SHARED comparator, time-multiplexed across the array by `absorb_ctrl` — NOT one per cell.**
1024× cheaper, and the sweep period is a **free refractory period** (single fire per sweep). A
per-cell comparator fires repeatedly on a crossing and **runs away** (measured: 271%/fold, network
collapses to 18%), then needs a refractory timer added to undo the freedom it bought. See §5.
Must implement:
- a comparator on `|V_Ce − V_bias_e|` against a programmable reference **θ**;
- on a crossing: a **±1 W-code** update (digital increment on W-SRAM is the cheaper route — see §5)
  and a **charge subtraction of θ from Ce** (not a discharge to V_bias);
- ⚠ **comparator offset/noise near the crossing must be characterised.** An offset shifts the
  *magnitude* threshold and is harmless (mismatch ×2 costs 0.76pp — §1). **Noise that flips the
  SIGN of a fire is NOT harmless** — that is a corrupted gradient, not a reduced learning rate, and
  the robustness argument in §1 does **not** cover it. **This is the key SPICE deliverable.**

### 6.5 SPICE testbenches
`tb_mac_cell_emx_dc`, `tb_mac_cell_emx_e_sweep`, `tb_e_inject_pulse`, **`tb_jug_fire`** (new:
threshold crossing, ±1 code step, residue retention), `tb_temporal_emx`.

---

## 7. RTL status and the changes the jug requires

RTL-T1–T4 pass **against the superseded rule** (`tb_absorb_match`, `tb_epoch_match` encode
`W += lr_slow·E`). They must be rewritten.

| block | change |
|---|---|
| `absorb_ctrl.v` | **REWRITE** — comparator sweep, not ADC-read/DAC-write; ±1 code increment; θ register |
| `cap_array.v` | **E-cap: SUBTRACT θ, do NOT discharge to zero** (the residue is load-bearing) |
| `refresh_ctrl.v` | `training_mode` gate can likely be **REMOVED** (§5) |
| `save_ctrl.v` | not needed *during* training (W-SRAM always current) |
| `pcn_wb_regs_*.v` | add per-layer **θ** register |
| testbenches | `tb_absorb_match` / `tb_epoch_match` **rewrite to the jug**; add `tb_jug_fire` |

### ⚠ Known RTL gaps (carried, and one new)
1. **The absorb rule itself** — RTL implements the superseded rule (**NEW**, this document).
2. **`cap_array.v` does a continuous `real` add** that the described physical circuit could not
   perform (deprecated §5 rounded to 8 bits and repainted via the DAC). **The two documents
   contradicted each other, and an RTL↔sim cross-check cannot catch it** — both sides do float
   adds. See `../hw_buildcheck/RTL_RECONCILIATION.md`.
3. `CODE_MID` **128 → 132** (`CROSS_CHECK.md` B1) — pending coordinated change.
4. **`W.T` backprojection not implemented in RTL** (`CROSS_CHECK.md` A8).

---

## 8. Code range and upward-bias fix — **UNCHANGED**

The [71, 192] range with WGT_ZERO = 128 is asymmetric (57 codes below zero, 64 above), which biases
GHA eigenvectors positive. **Target WGT_ZERO = 132** (V_w = 0.927 V): 61 codes below, 60 above —
near-symmetric across the same physical capacitor range.

Required: `cap_array.v` reset value → `(132.0 − WGT_MIN)/SPAN`; Python sims `CODE_MID = 132`;
re-run all testbenches. The DAC and mac_cell analog circuits are unaffected (a code-to-voltage
mapping convention only).

---

## 9. Key parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Process | Sky130A 130 nm | |
| VDD / VCM | 1.8 V / 0.9 V | |
| WGT_MIN / WGT_MAX | 71 / 192 | V_w = 0.500 V / 1.350 V |
| **WGT_ZERO** | **117** ⚠ | V_w = 0.823 V. **RE-DERIVED** for MN3_w=2 (was 132 @ 0.927 V). A property of the CELL — re-measure if the cell changes. |
| SPAN / CODE_SCALE | 121 / 64 | weight LSB = 1/64 |
| Cw / Ce | 200 fF / ~100 fF | Ce to be SPICE-confirmed |
| **Ce leakage** | **≈ 50 µV/s** (5–10 fA) | **≈ 140 s per E-LSB ⇒ a pure integrator on the firing timescale (§1)** |
| E_MAX_V | 0.055 V | ±55 mV. **No longer a MAC constraint** — Ce is out of the signal path; this is now only the inject clamp / comparator range. θ ≈ 0.4 codes sits ~17× inside it. |
| **V_bias_e** | **now a FREE variable** | ⚠ the dual-tail triode constraint is DELETED with MN3_e (§2.3). Re-derive against the comparator — do NOT inherit 0.65 V by default. |
| **θ (jug threshold)** | **8 × per-fold rms\|E\| ≈ 0.4 weight codes** | **per-layer CONFIG constant. ~17× headroom to the E clamp.** |
| **fire rate at θ=8** | **~0.9 %/fold** | mean inter-fire interval ~110 folds |
| N_CELLS / N_ROWS / N_COLS / N_ELEMS | 4 / 16 / 16 / 256 | |

---

## 10. Validation status

| | |
|---|---|
| Bit-accurate model (BIG, 24/8/16), jug | **81.96%** |
| Float ceiling | 82.13% |
| Old absorb rule (superseded) | 64.09% |
| Device mismatch (10× leak spread, ×2 θ spread) | **free** (§1) |
| **E live in the forward path** | ✅ **RESOLVED — MN3_e DELETED.** It was the one thing that broke it (runaway collapse ≥2 codes; the old design point already cost 1.5pp). §2. |
| **Comparator sign errors** | ✅ **FREE.** 5% → 81.42 · 10% → 82.06 · **20% → 81.62**. The residue-preserving subtraction CONSERVES CHARGE, so a wrong-signed fire is undone by the next correct one. **The comparator spec is LOOSE.** |
| **Seeds (n=4)** | ✅ 81.38 / 81.63 / 81.96 / **82.17** — mean **81.79**, spread 0.79pp (= the noise floor). The best seed EXCEEDS the float control. |
| Does the jug beat 82.50% in *float*? | ⚠ open — is it a better **rule**, or only a better **circuit**? |

**Remaining before a tape-out spec:**
1. Does the jug beat **82.50%** in the **float** sim? ⇒ is it a better **learning rule**, or only a
   better **circuit**? *(Do it in a COPY — never edit `pcn_bigspec.py`.)*
2. **Per-sample leak** instead of the fold-grid approximation — required before a circuit spec.
3. **Temperature DRIFT during training** (we modelled STATIC mismatch, not drift).
4. The `leak8` non-monotonicity — do not design around a curve we cannot explain.
5. Task #15 - the per-layer ADC gain. **NOT a blocker**: the spread is **8x** (`[>>3, >>6, >>6]`), a 2-3 bit config register. The '100:1' figure was WRONG (three units). Keep the task as a **SCALING** experiment: ||W|| drives BOTH the forward activation scale AND the backward delta attenuation, so normalising the frozen W per chip may remove the requirement *and* fix attenuation at depth.
6. **SPICE**: §6.1 (single-tail, simplified) and **§6.4 (the θ comparator — now the key deliverable)**.

---

## 11. Relationship to other folders

- **`/emx/`** — NOT pursued (E as 12-bit digital SRAM; 65.8%). No new work.
- **`../hw_buildcheck/`** — the bit-accurate model (`pcn_hw.py --jug`), `THE_JUG.md`,
  `RTL_RECONCILIATION.md`, `HW_BUILD_CHECK.md`. **The accuracy reference.**
- **`../multi_array_level3_BIGspec/`** — `pcn_bigspec.py`, the float reference (82.50%).
  **Validated — do not edit; copy if you need to experiment.**

---

## 12. Re-parameterisation (cell size) and the RTL regression — 2026-07-17

### `rtl/run_all_tb.sh` — run EVERY testbench
There was no harness: TBs had only ever been run individually, which is how the four
issues below survived. `./run_all_tb.sh [substring]` elaborates and runs all of them.
**Silence is a FAIL** (a TB with no verdict is reported as NO-VERDICT).
**Status: 17/17 pass.**

### ⚠ Four defects it found immediately (all pre-existing, none caught by the old flow)

| | defect | why it survived |
|---|---|---|
| **1** | **`LR_CFG` bit aliasing (REAL BUG).** `bh <= wb_dat_i[7:5]` and `lr_shift <= wb_dat_i[5:0]` **both claimed bit 5**, while readback packed 9 bits `{23'h0, bh, lr_shift}`. `lr_shift` had been widened 5→6 bits without fixing the packing, the readback, or the header. You could not set `LR_SHIFT[5]` without corrupting `BH[0]` — **in the register that sets the learning rate.** ✅ Fixed: layout is now `[8:6]=BH, [5:0]=LR_SHIFT` (9-bit, matching the readback). | `tb_pcn_wb_regs` declared `wire [4:0] lr_shift` against a 6-bit output ⇒ **silently truncated** ⇒ the write half of the test passed. The read half had been failing, unnoticed, because nobody ran it. |
| **2** | **`tb_refresh_ctrl` T3 asserted a deleted requirement** — that `training_mode` disables refresh. The jug **removed that gate on purpose** (W-SRAM is always the master; the coherence cycle collapses). ✅ Test INVERTED: it now asserts refresh keeps running, i.e. it guards against the gate coming back. | never re-run after the jug rewrite |
| **3** | **Three dead TBs** driving `absorb_*` ports retired with `absorb_ctrl`: `tb_cap_array` (deleted — its subject is covered by tb_jug_fire/tb_cap_theta/tb_epoch_match/tb_absorb_match/tb_e_inject_match), `tb_cap_array_mac` (**Part B tested the dual-tail `(W+E)f` MAC — deleted with MN3_e, so it asserted behaviour now wrong BY DESIGN**; rewritten to Part A + a new **A2** that proves E stays OUT of the signal path), `tb_save_ctrl` (ports fixed). | leftovers of the retirement |
| **4** | **`tb_pcn_digital_top_emx`** referenced `pcn_digital_top_emx` — **the wrong design's** top, left behind when this folder was copied from `/emx`. ✅ Deleted. | never elaborated |

### ✅ The N≠16 hardcodes are fixed
All elaborated cleanly and would have produced **wrong silicon**:

| location | was | now |
|---|---|---|
| `pcn_digital_top_jug.v` | `localparam RTR_CELL_AW = 4;` | `$clog2(N_ROWS)` |
| `pcn_transpose.v` | `N_ELEMS` an **independent parameter** (could disagree with `N`) | **DERIVED** `localparam N_ELEMS = N*N` — cannot disagree |
| `pcn_transpose.v` | `cell_sel [1:0]`, `sram_addr [9:0]`, `rd [9:0]` | `$clog2(N_CELLS)`, `$clog2(N_CELLS)+$clog2(N*N)`, `ELEM_AW` |
| `router_chip_emx.v` | `ABSORB_TIMEOUT = 1280` (literal) | `N_CELLS*N_ROWS*N_COLS + MARGIN` |

Also added: an **elaboration-time `$fatal` if `N_ROWS != N_COLS`** — `router_backproj` transposes a
SQUARE block, so a rectangular cell needs separate row/col dims through the whole δ path. Fail loudly
rather than silently.

### ★ `rtl/tb_param_smoke.v` — the test that proves it re-parameterises
Every other TB runs at the 16×16×4 default, so they prove only that **N=16 still works** — by
construction they **cannot** catch a hardcoded width. `tb_param_smoke` instantiates at
**32×32, 8 cells** and checks the geometry-derived properties (every element read exactly once, in
order; a **cell_sel of 5** lands in the right SRAM region; `done` after exactly N² reads;
`ABSORB_TIMEOUT` spans the sweep). It deliberately does **not** re-derive `router_backproj`'s
arithmetic — a test that mirrors the implementation proves nothing; the maths is
`tb_pcn_transpose`'s job (golden vectors).

**Negative control run (do this again if you touch the widths):** restoring the old `[1:0] cell_sel`
makes it fail with **1024 reads to the wrong cell** — cell 5 pruned to cell 1, "elaborates cleanly,
wrong silicon", exactly as predicted. At 32×32×8 the old literal `ABSORB_TIMEOUT=1280` would have
covered 1280 of **8192** elements, truncating the sweep and starving ~85% of the array.

### ⚠ Still hardcoded (NOT yet fixed — N_CELLS-related, in the router/act path)
`loop_mask [24:0]` (= (N_CELLS+1)²), `input/output_cell_mask [4:0]`, `spi_dest_cell_w`/
`spi_src_cell_w`/`peer_*_cell_w` `[3:0]`, `sender_id_w [N_CELLS*2-1:0]`, `rtr_page_sel` 2-bit,
`cap_array`'s `jug_cell [1:0]` / `jug_addr [7:0]`, and the golden-vector TBs
(`tb_pcn_transpose`, `tb_transpose_hop`) whose Python stimulus is N=16-only.
**⇒ The δ/transpose datapath re-parameterises; the router/activation path does not yet.**
