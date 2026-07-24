#!/usr/bin/env python3
"""
gen_epoch_stimulus.py  —  RTL-T4 stimulus generator, THE JUG.

★ REWRITTEN FOR THE JUG (2026-07-15).  The old version encoded the DEAD absorb rule
  (`W_cap += E_cap; E_cap = 0`).  The jug does NOT discharge — it SUBTRACTS theta and keeps
  the residue, and the weight moves as a ±1 CODE INCREMENT on the W-SRAM, once per sweep.
  See ../circuit/tb_jug_fire.spice and jug_ctrl.v.

Produces FOUR files for tb_epoch_match.v:
  w_init.hex      — initial W codes (1024 × 8-bit hex, $readmemh format)
  stim.hex        — the SCHEDULE: a mix of injects and SWEEP markers (32-bit packed hex)
  w_expected.hex  — expected FINAL W codes after the whole schedule (1024 × 8-bit hex)
  e_expected.hex  — expected FINAL Ce residues, in signed micro-volts (1024 × 32-bit hex)
                    ★ this is the load-bearing check: the residue must SURVIVE, not vanish.

Schedule word packing (32-bit):
  bit  [31]    — OP: 1 = SWEEP the whole array once ; 0 = INJECT one element
  bits [25:24] — inject_cell  [1:0]     (inject only)
  bits [23:16] — inject_addr  [7:0]     (inject only)
  bits [15: 8] — inject_delta [7:0]     (signed, two's-complement)
  bits [ 7: 0] — inject_x     [7:0]     (unsigned)
Fixed for all injects: bh=3 (bh_r=4), lr_shift=20.

Python arithmetic MIRRORS cap_array.v + jug_ctrl.v EXACTLY (both are IEEE-double):
  inject:  E = clip(E + d*x*bh_r/2^lr_shift, -E_MAX_V, +E_MAX_V)
  sweep (each element, once, in order 0..1023):
      if   E >= +THETA:  code = min(code+1, WGT_MAX);  E = clip(E - THETA, ±E_MAX_V)
      elif E <= -THETA:  code = max(code-1, WGT_MIN);  E = clip(E + THETA, ±E_MAX_V)
      else:              nothing — the charge STAYS on Ce.
  ⚠ ONE fire per element per sweep (the sweep is a refractory period). A 3*theta element
    needs THREE sweeps to drain — "a burst becomes a train."
"""

import os

# ── Constants matching cap_array.v / jug_ctrl.v EXACTLY ─────────────────────
WGT_MIN  = 71
WGT_MAX  = 192
SPAN     = WGT_MAX - WGT_MIN      # 121
E_MAX_V  = 0.150                  # ⚠ 0.150, NOT 0.055 — MN3_e is gone, Ce is out of the MAC
THETA    = 0.050                  # the fire threshold, in Ce volts

N_CELLS  = 4
N_ELEMS  = 256
TOTAL    = N_CELLS * N_ELEMS      # 1024

BH       = 3                     # bh_r = BH + 1 = 4
LR_SHIFT = 20                    # 2^20 = 1048576

SEED     = 42
OUT_DIR  = os.path.dirname(os.path.abspath(__file__))

SWEEP_WORD = 0x80000000          # bit[31] set = "sweep the array once"


def clip(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def inject(E, idx, delta, x):
    """delta signed, x unsigned. Mirror cap_array.v line 149-152."""
    dv = delta * x * (BH + 1) / float(1 << LR_SHIFT)
    E[idx] = clip(E[idx] + dv, -E_MAX_V, E_MAX_V)


def sweep(code, E):
    """One pass, element 0..TOTAL-1, at most one ±1 fire each. Mirror jug_ctrl + cap_array."""
    for i in range(TOTAL):
        if E[i] >= THETA:
            code[i] = min(code[i] + 1, WGT_MAX)
            E[i] = clip(E[i] - THETA, -E_MAX_V, E_MAX_V)   # SUBTRACT — keep the residue
        elif E[i] <= -THETA:
            code[i] = max(code[i] - 1, WGT_MIN)
            E[i] = clip(E[i] + THETA, -E_MAX_V, E_MAX_V)   # add theta back
        # else: no fire, charge stays on Ce


def main():
    # small deterministic PRNG so the .hex is reproducible without numpy
    import random
    rng = random.Random(SEED)

    # ── Initial W codes ─────────────────────────────────────────────────────
    w_init = [rng.randint(WGT_MIN, WGT_MAX) for _ in range(TOTAL)]
    code   = list(w_init)                 # the WEIGHT lives as an integer SRAM code
    E      = [0.0] * TOTAL                 # Ce starts empty

    schedule = []   # list of 32-bit words (inject or sweep)

    def emit_inject(cell, addr, delta, x):
        idx = cell * N_ELEMS + addr
        schedule.append(((cell & 0x3) << 24) | ((addr & 0xFF) << 16) |
                        ((delta & 0xFF) << 8) | (x & 0xFF))
        inject(E, idx, delta if delta < 128 else delta - 256, x)

    def emit_sweep():
        schedule.append(SWEEP_WORD)
        sweep(code, E)

    # ── ROUND 0: corner cases that OVERSHOOT past threshold ─────────────────
    # These build BURSTS (multi-theta), which later bare sweeps must drain as TRAINS.
    # dv per inject = delta * x * 4 / 2^20.  To reach ~3*theta = 0.150 need dv≈0.15:
    #   delta=120, x=255  -> 120*255*4/1048576 = 0.1167   (≈2.3*theta, saturates near E_MAX)
    corner = [
        (0,   0, 120, 255),   # cell0 addr0: big +   -> burst up
        (0,   0, 120, 255),   # again -> clamps toward +E_MAX (0.150) = 3*theta => TRAIN of 3
        (0,   1, -120, 255),  # cell0 addr1: big -   -> burst down
        (0,   1, -120, 255),  # -> -E_MAX => train of 3 down
        (1,   0,  60, 200),   # cell1 addr0: ~1.05*theta (0.0458) -> just under, NO fire first
        (2,   5,  90, 255),   # cell2 addr5: ~1.75*theta -> fires once, keeps 0.75*theta
        (3, 200, -90, 255),   # cell3 addr200: ~-1.75*theta
    ]
    for (c, a, d, x) in corner:
        emit_inject(c, a, d, x)
    emit_sweep()              # first sweep: each burst fires ONCE

    # ── ROUNDS 1..3: BARE sweeps (no inject) — drain the bursts as TRAINS ────
    for _ in range(3):
        emit_sweep()

    # ── ROUNDS 4..9: random injects (a few per round) then a sweep ──────────
    for _ in range(6):
        for _ in range(20):
            idx   = rng.randrange(TOTAL)
            cell  = idx >> 8
            addr  = idx & 0xFF
            delta = rng.randint(-127, 127)
            x     = rng.randint(0, 255)
            emit_inject(cell, addr, delta, x)
        emit_sweep()

    # ── Final expected state ────────────────────────────────────────────────
    w_expected = list(code)
    e_micro    = [int(round(e * 1e6)) for e in E]     # signed micro-volts

    # ── Statistics ──────────────────────────────────────────────────────────
    n_sweeps  = sum(1 for w in schedule if w == SWEEP_WORD)
    n_inj     = len(schedule) - n_sweeps
    n_changed = sum(1 for i in range(TOTAL) if w_expected[i] != w_init[i])
    n_resid   = sum(1 for e in E if abs(e) > 1e-9)
    e_max     = max(abs(e) for e in E)
    print(f"gen_epoch_stimulus.py  (seed={SEED})  — THE JUG")
    print(f"  schedule: {n_inj} injects + {n_sweeps} sweeps = {len(schedule)} ops")
    print(f"  {n_changed}/{TOTAL} W codes changed after the epoch")
    print(f"  {n_resid}/{TOTAL} elements hold a non-zero RESIDUE (the jug's whole point)")
    print(f"  max |Ce| residue: {e_max*1e3:.2f} mV  (E_MAX_V = {E_MAX_V*1e3:.0f} mV, "
          f"theta = {THETA*1e3:.0f} mV)")

    # ── Write files ─────────────────────────────────────────────────────────
    with open(os.path.join(OUT_DIR, 'w_init.hex'), 'w') as f:
        for c in w_init:
            f.write(f'{c:02x}\n')

    with open(os.path.join(OUT_DIR, 'stim.hex'), 'w') as f:
        for word in schedule:
            f.write(f'{word:08x}\n')

    with open(os.path.join(OUT_DIR, 'w_expected.hex'), 'w') as f:
        for c in w_expected:
            f.write(f'{c:02x}\n')

    # e_expected: signed micro-volts as 32-bit two's-complement hex
    with open(os.path.join(OUT_DIR, 'e_expected.hex'), 'w') as f:
        for e in e_micro:
            f.write(f'{e & 0xFFFFFFFF:08x}\n')

    print(f"  Wrote w_init.hex, stim.hex ({len(schedule)} ops), "
          f"w_expected.hex, e_expected.hex")
    print(f"  N_OPS = {len(schedule)}  (set the tb localparam to this)")
    print(f"  Run: iverilog -g2012 -Wall -o /tmp/tb_epoch_match.vvp "
          f"tb_epoch_match.v jug_ctrl.v cap_array.v && vvp /tmp/tb_epoch_match.vvp")


if __name__ == '__main__':
    main()
