#!/usr/bin/env python3
"""analyse_weight.py — is the code -> weight map LINEAR, as the sim assumes?

Reads output/w_transfer.csv (from tb_mac_cell_jug_weight.spice): a 2-D DC sweep with
inp = {0.890, 0.900, 0.910} inner and Vw = 0.500 .. 1.350 in 7.02 mV (ONE CODE) steps outer.

    W_eff(Vw) = dI_out/dx   — the REAL weight the MAC applies
    W_lin     = (code - 132)/64   — what the SIM assumes

★ THE JUG'S ENTIRE MECHANISM IS "ONE FIRE = ONE CODE."  If a code is worth 10x more weight
  at one end of the range than the other, the effective LEARNING RATE depends on where the
  weight currently sits.
"""
import numpy as np

RL = 100e3
CODE_MIN, CODE_MID, CODE_MAX = 71, 132, 192
V_MIN, V_LSB = 0.500, 0.00702

d = np.loadtxt('output/w_transfer.csv', skiprows=1)
inp, viout, ntail = d[:, 0], d[:, 1], d[:, 2]

# regroup: 3 inner points (inp) per outer point (Vw)
n = len(inp) // 3
inp, viout, ntail = inp[:n*3].reshape(n, 3), viout[:n*3].reshape(n, 3), ntail[:n*3].reshape(n, 3)
vw = V_MIN + np.arange(n) * V_LSB
code = np.rint((vw - V_MIN) / V_LSB).astype(int) + CODE_MIN

# the REAL weight = dI_out/dx, in uA/V
w_eff = (viout[:, 2] - viout[:, 0]) / RL / 0.020 * 1e6
w_lin = (code - CODE_MID) / 64.0                       # what the sim assumes
ok = (code >= CODE_MIN) & (code <= CODE_MAX)
vw, code, w_eff, w_lin, nt = vw[ok], code[ok], w_eff[ok], w_lin[ok], ntail[ok, 1]

print("\n" + "=" * 84)
print("  CODE -> WEIGHT TRANSFER FUNCTION  (does the silicon match the sim?)")
print("=" * 84)
print(f"\n  {'Vw(V)':>7} {'code':>5} {'ntail(mV)':>10} {'W_eff(uA/V)':>12} "
      f"{'dW/dcode':>10} {'W_lin(sim)':>11}")
print("  " + "-" * 74)
dwdc = np.gradient(w_eff)
for i in range(0, len(code), 6):
    print(f"  {vw[i]:7.3f} {code[i]:5d} {nt[i]*1e3:10.1f} {w_eff[i]:12.2f} "
          f"{dwdc[i]:10.3f} {w_lin[i]:11.3f}")

# ── the two questions ────────────────────────────────────────────────────────────────
mono = np.all(np.diff(w_eff) > 0) or np.all(np.diff(w_eff) < 0)
lo, hi = np.abs(dwdc[3]), np.abs(dwdc[-4])
print("\n" + "=" * 84)
print(f"  MONOTONIC?            {'YES' if mono else '*** NO — FATAL: two codes, one weight ***'}")
print(f"  dW/dcode at the BOTTOM of the range : {lo:.4f}")
print(f"  dW/dcode at the TOP of the range    : {hi:.4f}")
print(f"  ⇒ RATIO (bottom/top)                : {lo/max(hi,1e-12):.1f}x")
print()
print(f"  The jug's mechanism is ONE FIRE = ONE CODE. A ratio far from 1 means one fire is")
print(f"  worth {lo/max(hi,1e-12):.0f}x more weight at one end of the range than the other —")
print(f"  i.e. THE EFFECTIVE LEARNING RATE DEPENDS ON WHERE THE WEIGHT SITS.")
print()
print(f"  ⚠ Is that fatal? NOT NECESSARILY. A per-synapse, value-dependent learning rate is")
print(f"    the SAME CLASS of perturbation as the device mismatch we already measured as")
print(f"    FREE (10x log-normal leak spread: -0.58pp; x2 threshold mismatch: -0.76pp).")
print(f"    Gradient descent is famously indifferent to a spread of learning rates.")
print(f"    ** BUT IT MUST BE MEASURED, NOT ASSUMED. ** Feed this curve back into pcn_jug.py")
print(f"    as a non-linear code->weight map and re-run. That is the honest next step.")
print("=" * 84 + "\n")

np.savetxt('output/w_map.csv', np.c_[code, vw, w_eff],
           header='code,vw,w_eff_uA_per_V', delimiter=',', comments='')
print("  wrote output/w_map.csv  (feed this into the sim)\n")
