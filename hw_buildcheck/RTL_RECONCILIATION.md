# RTL ↔ SIM RECONCILIATION — what the silicon implements vs what we now run

**2026-07-14.** Triggered by a direct question from Saul: *"in the temporal design, we use the SRAM
refresh to maintain the weight matrix — is that accurate?"* **It is accurate**, and checking it
turned up five more things.

> ## THE ONE-LINE VERSION
> **The RTL implements a rule we have SUPERSEDED.** The chip's absorb was designed for the
> block/epoch W+E rule (headline **64.09%**). The current rule (**82.50%** float) writes weight
> changes **350× smaller than the chip's absorb was built for**, and they underflow the 8-bit cell.
> **THE JUG IS THE HARDWARE REALISATION OF THE NEW RULE** — and its absorb circuit is *simpler*
> than the one currently specced.

---

## 1. ✅ CONFIRMED: the W-cap leak is already solved (`refresh_ctrl.v`)

```
// Background DRAM-style W-cap refresh from W-SRAM.
// Must be disabled during TRAINING_MODE: W-SRAM is stale while training
// (only updated at explicit save), so refreshing from SRAM would overwrite
// learned cap voltages with old codes.
// With 1 pA leakage, 100 fF cap drifts 1 LSB in ~400 ms.
// At 50 MHz, a full 1024-entry scan takes 2048 cycles = 41 µs.
```

Three-tier storage (`PCN_control_and_management.md` §2), all three FSMs exist:

| tier | storage | volatility | role |
|---|---|---|---|
| **E-cap** | analog voltage | ~ms; **zeroed every absorb** | per-sample learning correction |
| **W-cap** | analog voltage | leaks; needs periodic **refresh** | the live weight the MAC uses |
| **W-SRAM** (1024×8b/chip) | digital SRAM | volatile | on-die master copy |

⇒ **"The weight capacitor leaks and the network will forget itself" is NOT a risk.** It was closed
long ago. (I had it as the #1 unmodelled risk. It was already handled.)

---

## 2. ★★ THE SUB-LSB PROBLEM IS A PROPERTY OF THE **NEW RULE**, NOT OF THE CHIP

| | write size per absorb |
|---|---|
| **chip's absorb** (`CROSS_CHECK.md` A5): `W += lr_slow·E`, E clamped to **±6.7 codes** (A3) | up to **6.7 LSB** |
| **the new rule** (`--sign_at_fold`, lr 3e-4): `W += lr·sign(E)` | **0.019 LSB** |

**A factor of ~350.**

The chip's absorb has **no underflow problem** — it was designed for a rule that writes in whole
codes. The new rule writes far below the cell's resolution and vanishes.

⇒ **"The analog cell cannot be the accumulator, therefore an 18-bit digital master" was WRONG as
stated.** It is not a chip defect. It is the specific, real obstacle to running the **better** rule
on the existing silicon. **The jug removes it.**

### ⚠ DO NOT re-run the old rule as a "fair baseline"
Saul's steer (2026-07-14): *"We have moved from an original design of W+E … then a long series of
epoch and phasing tests which didn't raise the bar. … Our newest models should be the reference
point."* The old block/epoch rule plateaued at **64.09%**; the new rule is **82.50%**. An 18pp gap
is not a close call. **The newest models are the reference. Do not re-litigate.**

### ★ And it explains why all the phasing work never paid
The epoch/phasing experiments were searching for the right **GLOBAL** absorb cadence.
**The jug says there isn't one — the cadence should not be global at all.** Each synapse
consolidates when *it* has crossed threshold. Those experiments were tuning a parameter that should
not exist. That is why they could not raise the bar.

---

## 3. ⚠ RTL ↔ DESIGN.md INCONSISTENCY (a real bug, independent of the jug)

**`cap_array.v` (the behavioural model) — a CONTINUOUS analog add, no quantisation:**
```verilog
if (absorb_en) begin
    real v_w;
    v_w = W_cap[idx] + E_cap[idx];    // `real`. NO rounding.
    W_cap[idx] = v_w;
    E_cap[idx] = 0.0;
end
```

**`DESIGN.md` §5 (the physical sequence) — ROUNDS to 8 bits and repaints through the DAC:**
```
Step 1 — Read Ce (SAR ADC)
Step 2 — new_W_code = round(V_eff_normalized × SPAN) + WGT_MIN     ← 8-BIT QUANTISATION
Step 3 — Write new_W_code to Cw via weight_dac
Step 4 — Discharge Ce to V_bias
```

**The behavioural model performs an operation the described circuit cannot.**
⚠ **An RTL↔sim cross-check CANNOT catch this** — both sides do float adds, so both agree and both
are wrong about the silicon. **Resolve which is intended.** (The jug's absorb supersedes both, but
the doc bug should still be fixed.)

---

## 4. ⚠ THE RTL IS CROSS-CHECKED AGAINST THE SUPERSEDED RULE

`CROSS_CHECK.md` A5 validated `W += lr_slow·E` against the **FABLE/bigv2** sim — the 64.09% era.
**`sign_at_fold` (worth +19pp, part of the 82.50% recipe) is NOT in the RTL and has NEVER been
cross-checked.**

Known RTL gaps, now three:
1. **The absorb rule itself** (this document) — NEW.
2. `CODE_MID` **128 → 132** (`CROSS_CHECK.md` B1, pending coordinated change).
3. **`W.T` backprojection not implemented in RTL** (`CROSS_CHECK.md` A8 — GAP).

---

## 5. ★ THE JUG IS A **CIRCUIT SIMPLIFICATION**, not merely a learning win

`DESIGN.md` §6.3 says the absorb needs E-cap readback, and offers two options — **both bad**:

- **Option A — shared SAR ADC:** one ADC conversion **per E-cap** (256 per cell per absorb).
- **Option B — shadow register:** a **12-bit digital accumulator per synapse**, which the document
  itself concedes *"re-introduces a digital register file for E"*.

**THE JUG NEEDS NEITHER.** Its absorb is:

```
   compare V_Ce against ±θ          # a COMPARATOR — no ADC read of E
   if crossed:
       W_code ± 1                    # a +-1 code increment — no full DAC repaint
       V_Ce  -= sign · θ             # charge SUBTRACTION — the residue is KEPT, not discharged
```

⇒ **It DELETES §6.3** (a PRIORITY 3 SPICE item with two unattractive options) **and preserves the
all-analog principle** the document says it wants.

### ★ And it may let `refresh` run CONTINUOUSLY
If a fire is a **±1 increment on the 8-bit W-SRAM entry**, then **W-SRAM is always the current
master**, the W-cap is a pure slave, and **`refresh` never has to be disabled**.
⇒ collapses the `absorb → save → SHADOW_W_SYNC → refresh-enable` coherence cycle
⇒ removes the `training_mode` gate on `refresh_ctrl.v`
⇒ **removes the W-cap-leak concern STRUCTURALLY rather than by scheduling.**

---

## 6. What the jug changes in the RTL (the actionable list)

| block | change |
|---|---|
| `absorb_ctrl.v` | comparator sweep instead of ADC-read + DAC-write; ±1 code increment; θ register |
| `cap_array.v` | E-cap: **subtract θ**, do NOT discharge to zero (the residue is load-bearing) |
| `refresh_ctrl.v` | `training_mode` gate can likely be **removed** (see §5) |
| `save_ctrl.v` | no longer needed *during* training (W-SRAM is always current) |
| WB regs | add per-layer **θ** (a config constant, like the ADC gain) |
| `DESIGN.md` §6.3 | **DELETE** — no Ce readback path required |

---

## 7. Still open (unchanged by this)

- **Tier 1:** comparator **sign errors** (a noisy comparator near zero gives WRONG-SIGNED fires —
  NOT a per-synapse lr, a corrupted signal; the robustness argument does not cover it) · **seeds**
  (everything is n=1).
- **Tier 2:** does the jug beat **82.50%** in the **float** sim? ⇒ is it a better LEARNING RULE, or
  only a better circuit? *(⚠ Must be done in a COPY — never edit the validated `pcn_bigspec.py`.)*
- Per-sample (not fold-grid) leak, before any circuit spec.
- The `leak8` non-monotonicity.
- Task #15 — the per-layer ADC gain.
