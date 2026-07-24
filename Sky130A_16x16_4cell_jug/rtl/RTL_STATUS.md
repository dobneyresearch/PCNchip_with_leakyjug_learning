# JUG RTL — status

Ported from `Sky130A_16x16_4cell_emx_analog_fable/rtl/`. **The jug touches THREE blocks and adds
TWO. Everything else — the router, the hop engine, the WB fabric, the controller — ports UNCHANGED.**

`iverilog -g2012` : **the full top elaborates, 0 errors** (incl. `router_ctrl`, WB fabric, shared).

---

## ★ NEW / CHANGED

| block | change |
|---|---|
| **`jug_ctrl.v`** ★NEW | **Replaces `absorb_ctrl`.** The swept comparator + the ±1 W-SRAM code increment + the fixed-width fire one-shot. |
| **`wgt_lut.v`** ★NEW | The **pre-distortion LUT** (code → 10-bit DAC drive) so the WEIGHT is linear. **Loadable ⇒ the per-die calibration hook.** |
| `cap_array.v` | **MAC = W only** (MN3_e deleted; Ce is out of the signal path). **Absorb → SUBTRACT θ, never discharge.** Comparator + residue ports. **WGT_ZERO 132 → 117.** Ce swing ±55 mV → ±150 mV (free — Ce no longer drives the MAC). |
| `refresh_ctrl.v` | **`training_mode` gate REMOVED.** W-SRAM is always the master, so refresh runs continuously. The absorb→save→sync→refresh coherence cycle **collapses**, and the W-cap leak risk goes away **structurally**. |
| `pcn_wb_regs_*.v` | + `WGT_LUT_ADDR` / `WGT_LUT_DATA` (auto-incrementing, so the boss streams 256 entries) + `JUG_THETA`. |
| `pcn_digital_top_jug.v` | rewired: `jug_ctrl`, `wgt_lut`, cap_array's new ports. |
| `save_ctrl.v` | **unchanged** — but no longer needed *during* training. Checkpoints only. |

## PORTED UNCHANGED
`router_ctrl.v` (771 lines) · `router_node.v` · `router_backproj.v` · `router_chip_emx.v` ·
`hop_engine.v` · `wb_ctrl_fabric.v` · `rtr_adc_bridge.v` · `routing_hebb.v` · `e_inject_ctrl.v`
and the shared `act_sram` / `weight_fsm` / `sar_adc` / `sram_if` / `hebb_ctrl` / `power_fsm`.

**The jug changes only how E becomes W. It does not touch how data moves.**

---

## ✅ `tb_jug_fire.v` — ALL PASS

| test | result |
|---|---|
| T1 the swept comparator fires at \|E\| ≥ θ | ✅ **fired exactly once** — the sweep IS the refractory period |
| **T2 ★ SUBTRACT, don't discharge** | E: **0.080 → 0.030** — ★ **the residue survived** |
| **T3 ★ a burst becomes a TRAIN** | E: 0.170 → 0.120 → 0.070 → 0.020 (**stops**); code 117→118→119→120. **3 fires, charge conserved.** |
| **T4 ★ a fire is a ±1 CODE INCREMENT** | W-SRAM 117 → 118. **No charge pump into Cw.** |
| T5 sub-threshold must not fire | ✅ did not fire, **and kept its 0.040 V of evidence** |

---

## ⚠⚠ A TEST THAT CANNOT FAIL IS NOT A TEST

T3/T4 **initially reported PASS on an `x`.** An off-by-one on the W-SRAM read latency made the
code X, and `if (X != 118)` evaluates to **X**, which Verilog treats as **FALSE** — so the failure
branch was never taken.

**Fixed twice over:** an `RDW` wait state for the SRAM read, **and every code comparison changed to
`!==`** so an X can never pass.

> **This is the SAME DEFECT CLASS as the WGT_ZERO sign check** (`I_diff(128)·I_diff(136) < 0` — a
> product, which passes for an INVERTED weight too) that let a broken weight through for months.
> **Three hours apart, in my own code.**
> ★ **A test must be able to distinguish right from wrong. Check that it CAN FAIL.**

---

## ✅ `tb_epoch_match.v` / `tb_absorb_match.v` — REWRITTEN TO THE JUG, ALL PASS (2026-07-15)

The old ones drove the dead `absorb_ctrl` and asserted `W = clip(W+E); E = 0`. Rewritten to
drive `jug_ctrl` + `cap_array` over the sigma-delta fire-sweep. `absorb_ctrl` is no longer
referenced by either (kept only for the legacy `tb_absorb_ctrl`).

| tb | what it pins down | result |
|---|---|---|
| **`tb_epoch_match`** | full multi-sweep EPOCH parity vs the Python jug model (`gen_epoch_stimulus.py`): 127 injects + 10 sweeps, bursts drained as trains | **1024/1024 W codes EXACT** and **1024/1024 Ce residues within 2 µV** — the sub-threshold charge the old absorb threw away all SURVIVES |
| **`tb_absorb_match`** | value-correctness **AT THE RAIL** (the one case the epoch + `tb_jug_fire` don't isolate) | ✅ code clamps at WGT_MIN/MAX **but θ is still drained** (anti-windup); sub-threshold holds; burst-into-rail drains as a train |

> ★ The rail case exposed a stale comment in `jug_ctrl` (claimed "the charge STAYS on Ce" at a
> rail). The RTL was RIGHT — it drains θ unconditionally, matching the validated sim
> (`pcn_jug.py` clips `Wm` at line 804 but does `E = e − s·th` at line 807). Comment fixed.

## ✅ Transistor-level comparator — `jug_comparator_t` + `tb_jug_comparator.spice`, ALL PASS
Two 5T OTAs (one per threshold) + 2-inverter buffers. The last and least-fussy block.

| test | result |
|---|---|
| D1 fire_up threshold (ideal 0.950 V) | **0.95075 V — +0.75 mV** offset |
| D2 fire_dn threshold (ideal 0.850 V) | **0.84925 V — −0.75 mV** offset |
| D3 dead zone (both low at V_Ce = 0.90) | ✅ ~0.2 µV — neither fires at rest |
| D4 speed (step into fire band) | **2.5 ns** propagation ≪ 10 ns sweep window |

Sub-mV offset is **60× inside** the tolerated 2·θ (0.76pp) mismatch. The behavioural
`jug_comparator` is kept as the loose-spec stand-in used by `tb_jug_fire.spice`.

## ✅ CODE_MID tied to WGT_ZERO — single source of truth (2026-07-15)
`router_backproj.v` shipped `CODE_MID = 132` (the OLD cell) against the jug cell's zero of **117**
— a 15-code offset that would put a spurious `+15·Σδ` bias on every backprojected output. Fixed and
**made undriftable**: new `pcn_weight_params.vh` holds ``\`define PCN_WGT_ZERO 117`` as the ONE place.
- `cap_array.v` (forward MAC zero) and `router_backproj.v` (`CODE_MID`, the transpose reference)
  both `` `include `` it — they can no longer disagree.
- `gen_backproj_stim.py` READS the `.vh` and overrides the twin's global, so the co-sim golden is
  tied too. `tb_router_backproj`: **RTL-B1 PASS** at 117. Full top elaborates 0 errors.
- The shared multi-array twin (`hw_multi_array_l3_fable`) is left at 132 (its own validated line).

## ✅ `pcn_transpose.v` — TRANSPOSE-AT-SOURCE, item 3a done (2026-07-15)
The on-chip transpose: reads a cell's 16×16 W block **from the chip's own W-SRAM** (1-code/cycle,
issue→wait→capture) and runs the validated `router_backproj` on it. Replaces the router-held
`SHADOW_W` — weights never leave the die; the read is always-current, no coherence. `CODE_MID` =
`PCN_WGT_ZERO` (117) via the shared `.vh`, so it's the true transpose of the forward operator.

`tb_pcn_transpose.v` (same golden as `tb_router_backproj`, sourced through a mock W-SRAM):
**64 cases × 16 elems, worst |err| = 1 LSB, 0 over ±2 — ALL PASS.** Proves the read path, not just
the arithmetic. (⚠ gotcha: `cell` is a reserved word in `-g2012` SV → renamed `cell_sel`.)

## ✅ item 3b sub-step 1 — `pcn_transpose` wired into the top (2026-07-15)
- **`pcn_digital_top_jug`**: instantiated `u_transpose`, δ = `delta_flat[cell]` slice (the SAME δ that
  drives E-inject), read port into the SRAM arbiter, partial → WB.
- **SRAM arbiter rebuilt**: priority JUG(RMW) > save > weight_fsm > transpose(read) > refresh; WB
  writes when idle. ⚠ **This also FIXED a pre-existing bug**: `jug_sram_addr/we/wdata` were connected
  to `jug_ctrl` but **never routed into the SRAM mux** — so in the assembled top, jug fires never
  reached the W-SRAM. Now they do.
- **WB regs** (`pcn_wb_regs_4cell_emx`): added **BP_TRIG (0xA0)** (write: start_transpose + cell,
  reset DST ptr; read: transpose_busy) and **BP_DST (0xA4)** (read: partial byte, auto-inc ptr).
- **Verified**: full top **elaborates 0 errors**; `tb_wb_transpose.v` ALL PASS (BP_TRIG pulse, cell
  latch, BP_DST returns the 16 partial bytes in order, busy read). `tb_pcn_transpose.v` still passes.

## ✅ "FINISH THE TOP" — skeleton gaps closed (2026-07-15)
The top elaborated 0 errors but several EMX control paths were **declared-but-undriven** (iverilog
doesn't flag floating nets — that's why they hid). All now closed and **tested**:
- **LUT load path**: `wgt_lut_addr/wdata/we` connected to `u_regs`. ★ AND fixed a bug — the
  `wgt_lut_we` deassert + addr auto-increment was **misplaced in the reset branch**, so in normal
  operation `wgt_lut_we` stuck high and the addr never advanced. Moved to the operational deassert.
  `tb_wb_transpose` now streams 3 LUT entries: **we pulses, addr 5→6→7. PASS.**
- **Runtime θ**: `cap_array` now takes `jug_theta` (mV; 0 ⇒ `THETA` param), `th_eff` drives the
  comparator + residue subtraction; wired to `u_regs` + `u_caps`. `tb_cap_theta` proves it MOVES the
  threshold (θ=100 ⇒ no fire, θ=70 ⇒ fire). `jug_theta` also given a reset default (50). PASS.
- (earlier: the jug SRAM RMW arbiter gap — fixed.)

**Verified state:** top elaborates **0 errors**; **all 7 block TBs pass** (`tb_jug_fire`,
`tb_epoch_match`, `tb_absorb_match`, `tb_cap_theta`, `tb_pcn_transpose`, `tb_wb_transpose`,
`tb_router_backproj`). Two real bugs found and fixed by finishing the wiring (jug-SRAM, LUT-reset).

## ✅ item 3b sub-step 2 core — `router_gather.v` built + verified (2026-07-15)
The gather-only router: `hop_engine` MINUS the `router_backproj` transpose (that moved to the chip
in `pcn_transpose`). Takes **pre-computed partials** from the source chips, accumulates per dest
(one source/cycle so the NBA sum is correct), then **avg_bp = fixed-divide by nominal fanin** +
sat_int8 — no shadow, no weights, no wait-for-all (C1/C2/C3 of `../INTERCONNECT_PROTOCOL.md`).
`tb_router_gather.v` (self-checking vs an independent in-TB avg_bp reference): **200 cases × 32
outputs, 0 mismatches — ALL PASS.**

**Both transpose-at-source datapaths now exist and pass:** `pcn_transpose` (chip: Wᵀδ from own
W-SRAM) + `router_gather` (router: accumulate + avg_bp).

## ✅ item 3b — RETIREMENT DONE (2026-07-15). The transpose-at-source RTL is CLEAN.
The shadow-based / transpose-in-router architecture is gone:
- **DELETED**: `hop_engine.v`, `router_node.v` (held `shadow[256]` + `WLOAD` + an in-router
  `router_backproj`); their TBs `tb_hop_chain`/`tb_big_chain`/`tb_hop_integration`; their generators
  `gen_hopchain_stim.py`/`gen_bigchain_stim.py`; and the now-dead hex (`hop2_w`, `l1_expected`, all `big_*`).
- **KEPT**: `router_backproj.v` — the transpose primitive, now used ONLY by `pcn_transpose`. Renamed its
  input `shadow_w_flat` → `w_flat` (no longer a shadow; fed from the chip's own W-SRAM).
- **ADDED (composition proof)**: `tb_transpose_hop.v` + `gen_transhop_stim.py` — 4× `pcn_transpose`
  (chip) → `router_gather` (router) reproduces the old fused hop (`twin_hop_generic`, 117-tied):
  **worst 1 LSB, PASS.** Replaces `tb_hop_chain`'s coverage under the new split.

**FINAL CLEAN STATE — verified:** top elaborates **0 errors**; **all 9 tests pass** (`tb_jug_fire`,
`tb_epoch_match`, `tb_absorb_match`, `tb_cap_theta`, `tb_pcn_transpose`, `tb_router_backproj`,
`tb_router_gather`, `tb_transpose_hop`, `tb_wb_transpose`). Weights never leave the chip; the router
holds no weights; transpose = `pcn_transpose` (chip), gather = `router_gather` (router).

## ✅ absorb_ctrl RETIREMENT DONE (2026-07-15) — the pre-jug rule is gone
- **DELETED**: `absorb_ctrl.v` (the dead `W += E; E := 0` rule), `tb_absorb_ctrl.v`, and
  `tb_integration.v` (the latter was ALREADY dead — it drove cap_array's removed `.absorb_*` ports;
  its coverage is subsumed by `tb_epoch_match` + `tb_e_inject_match`).
- **FIXED a silently-broken test**: `tb_e_inject_match.v` also drove the removed `.absorb_*` ports AND
  verified injects via `mac_eff_code` — but the jug MAC is **W-only** (E is out of the signal path),
  so injects don't move it. Rewritten to verify via `E_cap` directly (as the other jug TBs do):
  the outer-product injects match the Python formula exactly — **5/5 PASS**. `e_inject_ctrl` coverage
  restored (it's a live module; it had no working test).

**FINAL: top elaborates 0 errors; all 10 block TBs pass; no `absorb_ctrl` anywhere.** The jug is the
only weight-update rule in the RTL.

## Still to do
- Sim: seeds / longer runs on the final config; per-sample (not fold-grid) leak before a circuit spec.
- (done: directional WGT_ZERO test; 10-bit `weight_dac`; cascoded+trimmed θ mirror — see `../circuit/SPICE_RESULTS.md`.)
