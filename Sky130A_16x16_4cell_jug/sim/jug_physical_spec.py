#!/usr/bin/env python3
"""jug_physical_spec.py — THE UNITS BRIDGE. Turns the sim's abstract numbers into VOLTS and
FEMTOCOULOMBS, so the SPICE work has something to design against.

The sim works in integer accumulator units: theta is "8 x the per-fold rms of |E|", where E counts
(delta_code x activation_code) products. **A circuit designer cannot build a comparator against
that.** This script emits what they actually need:

    * theta          — the comparator reference, in mV on Ce
    * dV_e           — the charge injected per sample, in uV on Ce
    * dQ_pump        — the charge the +-1 LSB pump must deposit on Cw, in fC
    * Ce swing       — how far Ce must range without clamping (theta + the real overshoot)
    * DYNAMIC RANGE  — the ratio that decides whether Ce is BIG ENOUGH to beat kT/C noise

The last one is the constraint that actually bites, and it is the reason this script exists:
**kT/C noise on a 100 fF capacitor is ~203 uV rms.** If the smallest meaningful injection lands
below that, it is buried in thermal noise and the jug cannot integrate it.

NOTE the two parameters that MN3_e's deletion FREED (they were MAC constraints, and Ce is no longer
in the signal path):
    * V_bias_e  — was pinned to 0.65 V by the dual-tail triode conflict. NOW FREE: set it from the
                  comparator's input common-mode range.
    * E_MAX_V   — was +-55 mV because Ce drove a tail. NOW FREE: bounded only by the inject
                  circuit's compliance and the comparator's range.
Both must be re-derived HERE, against theta — not inherited.

    python3 jug_physical_spec.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('OMP_NUM_THREADS', '8')

import pcn_jug as J                       # the sim (argparse is __main__-guarded)
B = J.B

# ── PHYSICAL CONSTANTS (Sky130A, from DESIGN.md §9) ──────────────────────────────────────
VDD        = 1.8
CW         = 200e-15      # W-cap
CE         = 100e-15      # E-cap  (SIZING IS AN OUTPUT OF THIS SCRIPT, not an input)
V_W_MIN    = 0.500        # WGT_MIN = code 71
V_W_MAX    = 1.350        # WGT_MAX = code 192
SPAN       = 121          # codes
KB, TEMP   = 1.380649e-23, 300.0
SIG_TOL    = 0.10         # ★ MEASURED tolerance: sigma(kT/C on Ce) <= 0.1 * theta. See main().

V_PER_CODE = (V_W_MAX - V_W_MIN) / SPAN          # volts per ONE weight LSB
DQ_PUMP    = CW * V_PER_CODE                     # charge the +-1 LSB pump must deposit on Cw


def ktc_noise(C):
    """sqrt(kT/C) — the thermal noise floor on a switched capacitor. THE hard limit on how small
    an injection can be and still mean anything."""
    return np.sqrt(KB * TEMP / C)


def main():
    print("\n" + "=" * 78)
    print("JUG — PHYSICAL SPEC  (the units bridge: sim units -> volts & femtocoulombs)")
    print("=" * 78)

    # ── run a few folds of the real sim to get the ACTUAL E statistics ──────────────────
    class A:                                  # the design point
        epochs, chunk, fold_every = 1, 4000, 128
        act_bits, delta_bits, e_acc_bits, leaky_shift = 8, 6, 20, 3
        float_ref, seed, agc, delta_code = False, 42, 'rms', 'linear'
        mant_bits, exp_off = 2, -3
        jug_leak, jug_theta, jug_multifire, norm_w = 0, 8.0, False, False
        jug_tau, jug_tau_spread, jug_theta_spread = 0.0, 1.0, 1.0
        jug_e_fwd, jug_sign_err, jug_step, delta_stats = 0.0, 0.0, 1 / 64, False
        frac_bits, lr, v_bits, mom_shift, w_mode, dither, jug = 0, 0.0, 12, 3, 'cell', False, True

    a = A()
    hw = J.HW(a)
    F_tr, F_te, y_tr, y_te = B.get_emnist_l0_features()
    Xc = B.float_to_act_code(F_tr).astype(np.int64)
    w1, w2 = B.load_l1_weights(), B.load_l2_weights()
    w3 = B.init_l3_weights_random_ortho(seed=2)
    Wm = [[[np.rint(pg.astype(np.float64) * hw.wscale).astype(np.int64) for pg in ch]
           for ch in L] for L in (w1, w2, w3)]
    hw.calibrate(Wm, Xc[:2000])

    # accumulate E over ONE fold, exactly as the chip would, and look at the numbers
    A_acts = J.forward(hw, Wm, Xc[:a.fold_every])
    F1, F2, F3 = A_acts
    W_f, b_f = B.fit_clf(J.forward(hw, Wm, Xc[:8000])[-1].astype(np.float64), y_tr[:8000])
    sc = F3.astype(np.float64) @ W_f.T + b_f
    z = sc - sc.max(1, keepdims=True); pr = np.exp(z); pr /= pr.sum(1, keepdims=True)
    S = -pr; S[np.arange(a.fold_every), y_tr[:a.fold_every]] += 1.0
    D3 = hw.channel((S @ W_f) * hw.leaky_d(F3), 'L3')

    # the per-sample injection quantum: one (delta x activation) product
    inj = np.abs(np.outer(D3[0], F2[0]).ravel())
    inj = inj[inj > 0]
    # E after one fold, at L3
    E1f = np.abs((D3.T @ F2[:, :J.N_ROWS])).ravel()

    theta_sim = 8.0 * float(np.sqrt(np.mean(E1f ** 2)))
    inj_rms   = float(np.sqrt(np.mean(inj ** 2)))
    inj_min   = float(np.percentile(inj, 5))
    inj_max   = float(inj.max())

    print(f"\n── SIM UNITS (integer accumulator) ─────────────────────────────────────────")
    print(f"  one injection (|delta x act|):  p5={inj_min:8.1f}  rms={inj_rms:9.1f}  "
          f"max={inj_max:10.1f}")
    print(f"  |E| after ONE fold (128 samples):            rms={np.sqrt(np.mean(E1f**2)):9.1f}")
    print(f"  theta = 8 x that                          theta={theta_sim:9.1f}")
    print(f"\n  ⚠ DO NOT compute a 'dynamic range' as theta / smallest-representable-injection.")
    print(f"    p5={inj_min:.0f} is the smallest non-zero INTEGER product — a rounding convention of")
    print(f"    this sim, NOT a physical injection that must be resolved. Dividing by it manufactures")
    print(f"    a ~14-bit requirement out of nothing and hands the analog team an impossible spec.")
    print(f"    THE RIGHT QUESTION IS MEASURED, NOT DERIVED: how much kT/C noise does the jug")
    print(f"    tolerate on Ce?  ⇒ see --jug_ce_noise. ANSWER: sigma <= 0.1 * theta.")

    # ── the CHARGE PUMP (this one is absolute — no free scale) ───────────────────────────
    print(f"\n── THE ±1 LSB CHARGE PUMP into Cw (ABSOLUTE — no free parameter) ───────────")
    print(f"  V_w range          : {V_W_MIN} .. {V_W_MAX} V over {SPAN} codes")
    print(f"  ONE weight LSB     : {V_PER_CODE*1e3:.2f} mV on Cw")
    print(f"  Cw                 : {CW*1e15:.0f} fF")
    print(f"  ⇒ dQ_pump          : {DQ_PUMP*1e15:.3f} fC   ← THE CHARGE-PUMP SPEC")
    print(f"  (a fire must move Cw by exactly one code: {V_PER_CODE*1e3:.2f} mV)")

    # ── Ce SIZING — from the MEASURED noise tolerance ────────────────────────────────────
    print(f"\n── Ce SIZING — from the MEASURED kT/C tolerance ────────────────────────────")
    print(f"  MEASURED (--jug_ce_noise sweep, clean ref 81.96%):")
    print(f"    sigma/theta = 0.05 -> ~81.0%   (~1pp — at the noise floor)")
    print(f"    sigma/theta = 0.10 -> ~81.0%   (~1pp)")
    print(f"    sigma/theta = 0.25 ->  78.55%  (-3.4pp)")
    print(f"    sigma/theta = 0.50 ->  71.39%  (-10.6pp)")
    print(f"    sigma/theta = 1.00 ->  COLLAPSE (51.8 -> 12.8%)")
    print(f"  ⇒ **SPEC: sigma <= 0.1 * theta.**  (0.05 for margin.)")
    print(f"\n  WHY THIS IS THE ONE TIGHT SPEC IN AN OTHERWISE SLOPPY DESIGN:")
    print(f"    Everything else that is free perturbs a DECISION (when a synapse fires, or")
    print(f"    occasionally which way). Thermal noise corrupts the EVIDENCE. A wrong-signed fire")
    print(f"    is undone by the next correct one because the residue subtraction CONSERVES")
    print(f"    CHARGE — but charge that was never real cannot be subtracted back out.")
    print(f"    ★ The sigma-delta can forgive a bad DECISION. It cannot forgive a LIE ABOUT")
    print(f"      HOW MUCH CHARGE IS THERE.")
    print(f"\n  kT/C is sampled EVERY time the inject switch closes, so over a fold of N injections")
    print(f"  it accumulates as sqrt(N)*sqrt(kT/C):")
    print(f"\n        sqrt(N) * sqrt(kT/Ce)  <=  {SIG_TOL} * theta_volts")
    print(f"        ⇒   theta_volts  >=  (sqrt(N)/{SIG_TOL}) * sqrt(kT/Ce)")
    print(f"\n  ⚠ N (how often the switch samples kT/C) is a CIRCUIT-TOPOLOGY question, not a")
    print(f"    modelling one. THE RELATION is the deliverable; N is the analog designer's to fill")
    print(f"    in. Below assumes N = {a.fold_every} (one sample per injection):")
    N = a.fold_every
    print(f"\n    {'Ce':>8} {'sqrt(kT/C)':>12} {'min theta':>12} {'Ce swing (3x theta)':>22}")
    for C in (50e-15, 100e-15, 200e-15, 500e-15, 1e-12):
        n = ktc_noise(C)
        th_min = (np.sqrt(N) / SIG_TOL) * n
        print(f"    {C*1e15:6.0f} fF {n*1e6:9.0f} uV {th_min*1e3:9.1f} mV {th_min*3*1e3:18.0f} mV"
              f"   {'✓' if th_min*3 < 0.4 else '⚠ check inject compliance'}")
    print(f"\n  ⇒ **Ce = 100 fF IS ADEQUATE**: theta ~ 23 mV, Ce swing ~ +-100 mV. Comfortable in")
    print(f"    a 1.8 V supply. And this is ONLY POSSIBLE because MN3_e is gone — the old")
    print(f"    E_MAX_V = +-55 mV was a MAC constraint (Ce drove a tail). Ce is out of the signal")
    print(f"    path now, so it is FREE TO SWING. With the E-tail still there, this would NOT fit.")

    # ── the spec the SPICE work starts from ──────────────────────────────────────────────
    print(f"\n── ⇒ WHAT SPICE MUST HIT ──────────────────────────────────────────────────")
    print(f"  1. CHARGE PUMP   : deposit {DQ_PUMP*1e15:.3f} fC on Cw = one {V_PER_CODE*1e3:.2f} mV step. "
          f"Bidirectional.")
    print(f"  2. COMPARATOR    : trip on |V_Ce - V_bias| >= theta.")
    print(f"     ⚠ THE SPEC IS EXTRAORDINARILY LOOSE — tell the analog designer:")
    print(f"       · offset may be a LARGE FRACTION of theta   (x2 threshold mismatch = -0.76pp)")
    print(f"       · it may get the SIGN WRONG 20% of the time (= -0.34pp; the residue")
    print(f"         subtraction CONSERVES CHARGE, so a wrong fire is undone by the next right one)")
    print(f"       · Ce matching is a NON-REQUIREMENT           (10x log-normal spread = -0.58pp)")
    print(f"     It is a coarse, sloppy, 1-bit decision device. It is ALLOWED to be wrong.")
    print(f"  3. Ce            : ⚠ THE ONE TIGHT SPEC. sigma(kT/C) <= {SIG_TOL} x theta.")
    print(f"                     Ce = 100 fF works: theta ~ 23 mV, swing ~ ±100 mV (table above).")
    print(f"  4. RESIDUE       : the fire must SUBTRACT theta from Ce, NOT discharge it.")
    print(f"                     ★ THIS IS THE ONE THING THAT MUST BE EXACT. Everything else is")
    print(f"                       loose; this is not. Discharging Ce destroys the mechanism.")
    print(f"\n── ⚠ TWO PARAMETERS THAT MN3_e's DELETION FREED — RE-DERIVE, DO NOT INHERIT ──")
    print(f"  V_bias_e : was pinned to 0.65 V by the dual-tail triode conflict. Ce no longer drives")
    print(f"             a tail ⇒ NOW SET BY THE COMPARATOR's input common-mode range.")
    print(f"  E_MAX_V  : was ±55 mV because Ce drove the MAC. Ce is out of the signal path ⇒ NOW")
    print(f"             BOUNDED ONLY by inject compliance and comparator range. Re-choose against")
    print(f"             theta + the real overshoot.")
    print("=" * 78 + "\n")


if __name__ == '__main__':
    main()
