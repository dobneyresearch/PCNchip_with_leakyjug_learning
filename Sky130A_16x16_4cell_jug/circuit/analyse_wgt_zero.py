#!/usr/bin/env python3
"""analyse_wgt_zero.py — the DIRECTIONAL WGT_ZERO check.

⚠ The FABLE test used  sflip = I_diff(128)*I_diff(136) < 0  — a PRODUCT. It passes for an
  INVERTED weight too, and an inverted weight is exactly what it let through, for months.
  ★ "Is there a sign flip" is NOT the test "is the sign RIGHT."
"""
import numpy as np, sys

RL, V_MIN, V_LSB, CMIN, CMAX, CZERO = 100e3, 0.500, 0.00702, 71, 192, 117
d = np.loadtxt('output/wgt_zero_jug.csv', skiprows=1)
vw, iout, ioutr = d[:, 0], d[:, 1] / RL, d[:, 2] / RL
code = np.rint((vw - V_MIN) / V_LSB).astype(int) + CMIN
idiff = (iout - ioutr) * 1e6            # uA — the SIGNED weight, vs the reference cell

ok = (code >= CMIN) & (code <= CMAX)
code, idiff = code[ok], idiff[ok]
fails = []

print("\n" + "=" * 76)
print("  WGT_ZERO — the DIRECTIONAL test (the FABLE one was a PRODUCT, and sign-blind)")
print("=" * 76)
print(f"\n  {'code':>5} {'I_diff(uA)':>12}   {'expected sign':>14}")
print("  " + "-" * 40)
for c in (71, 90, 105, 117, 130, 155, 191):
    i = int(np.argmin(np.abs(code - c)))
    want = "NEGATIVE" if code[i] < CZERO else ("~zero" if code[i] == CZERO else "POSITIVE")
    print(f"  {code[i]:5d} {idiff[i]:12.3f}   {want:>14}")

# ── D1: MONOTONIC across the FULL range (not a ±4-code window) ────────────────
mono = np.all(np.diff(idiff) > -1e-9)
print(f"\n  D1  MONOTONIC across the FULL code range 71..192 ?   "
      f"{'*** YES ***' if mono else '*** NO — FATAL ***'}")
if not mono: fails.append("D1 non-monotonic")

# ── D2: ★ DIRECTIONAL — the right sign on the right side ─────────────────────
below = idiff[code < CZERO]
above = idiff[code > CZERO]
d2 = np.all(below < 0) and np.all(above > 0)
print(f"  D2  ★ DIRECTIONAL: I_diff < 0 BELOW zero and > 0 ABOVE it ?  "
      f"{'*** YES ***' if d2 else '*** NO — THE WEIGHT IS INVERTED ***'}")
print(f"        below (codes<{CZERO}): max = {below.max():+.3f} uA   (must be < 0)")
print(f"        above (codes>{CZERO}): min = {above.min():+.3f} uA   (must be > 0)")
if not d2: fails.append("D2 INVERTED")

# ── D3: the zero really is a zero ─────────────────────────────────────────────
iz = idiff[code == CZERO][0]
d3 = abs(iz) < 0.5
print(f"  D3  |I_diff(code {CZERO})| ~ 0 ?  {iz:+.4f} uA   "
      f"{'*** YES ***' if d3 else '*** NO ***'}")
if not d3: fails.append("D3 zero is not zero")

print("\n" + "=" * 76)
if fails:
    print(f"  *** FAIL: {', '.join(fails)}")
    sys.exit(1)
print("  ALL PASS — monotonic, correctly SIGNED, and the zero is where we say it is.")
print("  (The FABLE product-test would have passed even if the sign were BACKWARDS.)")
print("=" * 76 + "\n")
