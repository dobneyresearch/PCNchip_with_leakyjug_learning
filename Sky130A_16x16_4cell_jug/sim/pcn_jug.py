#!/usr/bin/env python3
"""pcn_jug.py — bit-accurate model of the JUG chip (Sky130A_16x16_4cell_jug).

THE JUG. Ce is NOT discharged. It accumulates charge; when |V_Ce| crosses a threshold the
synapse FIRES: the analog weight moves one whole +-1 LSB, and theta is SUBTRACTED from Ce.

    Ce  <-  lambda*Ce + delta*x            # a capacitor. The LEAK **IS** momentum.
    if |Ce| >= theta:                       # a COMPARATOR (no ADC, no readback)
        W_code <- W_code +- 1               # one whole LSB - all the cell can do
        Ce     <- Ce -+ theta               # SUBTRACT. The residue is KEPT.

**The residue-preserving subtraction is LOAD-BEARING.** Discharging Ce throws away exactly the
sub-threshold charge the mechanism exists to accumulate. Subtracting theta makes this a true
sigma-delta: no charge, and therefore no learning signal, is ever lost. It is also what makes the
mechanism immune to a sloppy comparator (a wrong-signed fire ADDS the charge back, and the next
correct fire undoes it - measured: 20% wrong fires cost NOTHING).

WHY IT EXISTS: the current learning rule writes lr*sign(E) with lr=3e-4 = **0.019 of one weight
LSB**. Written naively to an 8-bit analog cell that is a NO-OP, ~51 times in a row. The old chip
absorb (`W += lr_slow*E`, E clamped to +-6.7 codes) never hit this because it wrote ~350x larger -
but that rule plateaued at 64.09%. The jug is how the BETTER rule (82.50% float) runs on the cell.

    hardware model, old absorb rule .............. 64.09%
    new rule naively written to an 8-bit cell .... 75.25%
    ** new rule via the JUG ...................... 81.96% **
    float ceiling ................................ 82.13%

NO digital master. NO velocity register. NO per-synapse digital store. The global fold disappears:
each synapse fires when IT has the evidence.

EVERYTHING HERE IS AN INTEGER. Declared widths. Only primitives real silicon has.

    python3 pcn_jug.py                       # the jug spec (theta=8, pure integrator)
    python3 pcn_jug.py --float_ref           # float ceiling control
    python3 pcn_jug.py --jug_e_fwd 0.4       # E live in the forward MAC (the DUAL-TAIL question)
    python3 pcn_jug.py --jug_tau 100 --jug_tau_spread 10   # real, mismatched capacitors

Lineage: ../../hw_buildcheck/  (pcn_hw.py, THE_JUG.md, RTL_RECONCILIATION.md) - the audit trail.
Float reference: ../../multi_array_level3_BIGspec/pcn_bigspec.py (82.50%) - DO NOT EDIT; copy.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'multi_array_level3_BIGspec'))
import pcn_bigspec as B                      # topology, weights, EMNIST L0 (float reference)

N_ROWS, N_COLS = B.N_ROWS, B.N_COLS
CODE_MIN, CODE_MID, CODE_MAX = B.CODE_MIN, B.CODE_MID, B.CODE_MAX   # 71 / 132 / 196
CODE_SCALE = 64                                                     # weight LSB = 1/64
N_CLASSES = B.N_CLASSES

# ⚠ THE MEASURED CODE -> WEIGHT MAP (from SPICE). None = the LINEAR map the sim always assumed.
W_MAP = None        # the measured weight per code (in sim code units)
W_MAP_SIM = None    # the same, in the sim's FLOAT weight units (for the inverse map)


def w_to_code(w_float):
    """INVERSE MAP: which CODE produces the weight we actually want?

    The weight_dac maps code -> V_w LINEARLY (that is the hardware). The CELL maps V_w -> weight
    SIGMOIDALLY (that is the physics). So the weight is NOT linear in the code, and loading a
    frozen PCA weight by `code = rint(w*64) + CODE_MID` puts it on the WRONG WEIGHT.

    A real chip fixes this in software at load time: pick the code whose MEASURED weight is
    closest to the one you want. A calibration LUT. No hardware change, and free.
    (What it CANNOT fix is the LEARNING step: the jug moves the code by +-1, and the resulting
    weight change still varies ~28x across the range. That is the thing being tested.)"""
    if W_MAP is None:
        return np.rint(w_float * CODE_SCALE).astype(np.int64)          # the old linear assumption
    grid = W_MAP_SIM                                                    # weight per code offset
    idx = np.abs(w_float.reshape(-1, 1) - grid.reshape(1, -1)).argmin(axis=1)
    return (idx + CODE_MIN - CODE_MID).astype(np.int64).reshape(w_float.shape)


def load_w_map(path):
    """Load the SPICE-measured code->weight curve (circuit/output/w_map.csv).

    Columns: code, vw, gm_uA_per_V, w_eff_vs_zero.  We use w_eff_vs_zero (the weight the MAC
    actually applies, referenced to the zero-current code) and rescale it to the same span as the
    linear map, so the ADC calibration and theta calibration land in the same place. The SHAPE is
    what matters — the absolute scale is absorbed by the per-layer ADC gain."""
    global W_MAP, W_MAP_SIM, CODE_MID
    d = np.loadtxt(path, delimiter=',', skiprows=1)
    codes, w = d[:, 0].astype(int), d[:, 3]
    full = np.zeros(CODE_MAX - CODE_MIN + 1)
    full[codes - CODE_MIN] = w
    # rescale to the linear map's span so the ADC / theta calibration lands in the same place.
    # THE SHAPE is what matters; the absolute scale is absorbed by the per-layer ADC gain.
    lin_span = max(abs(CODE_MIN - 132), abs(CODE_MAX - 132))
    W_MAP = full / np.abs(full).max() * lin_span
    # ⚠⚠ WGT_ZERO IS A PROPERTY OF THE CELL, AND THE CELL CHANGED (MN3_w 10 -> 2).
    # Re-derive it: the zero-current code is where the measured weight crosses zero.
    zero_code = int(codes[np.argmin(np.abs(w))])
    old_mid = CODE_MID
    CODE_MID = zero_code
    W_MAP_SIM = W_MAP / CODE_SCALE          # in the sim's float-weight units, for the inverse map
    print(f"  ** MEASURED W MAP ** {path}")
    print(f"     ★ WGT_ZERO RE-DERIVED: {old_mid} -> {zero_code}  "
          f"(a property of the CELL, and the cell changed)")
    dw = np.abs(np.gradient(W_MAP))
    mid = len(dw) // 2
    print(f"     dW/dcode: rails ~{dw[3]:.3f} / {dw[-4]:.3f}   middle ~{dw[mid]:.3f}"
          f"   => {dw[mid]/max(dw[3], 1e-9):.0f}x SENSITIVITY VARIATION")
    print(f"     monotonic: {bool(np.all(np.diff(W_MAP) >= -1e-9))}")


# ── the hardware datapath ─────────────────────────────────────────────────────
class HW:
    def __init__(self, a):
        self.frac = a.frac_bits          # sub-LSB bits held in the DIGITAL master weight
        self.wscale = CODE_SCALE << self.frac    # master units per unit of W_float
        self.dbits = a.delta_bits        # cross-chip delta broadcast width (signed)
        self.dmax = (1 << (a.delta_bits - 1)) - 1
        self.eacc = a.e_acc_bits         # E accumulator width (signed, saturating)
        self.emax = (1 << (a.e_acc_bits - 1)) - 1
        self.vbits = a.v_bits            # momentum velocity width (signed)
        self.vmax = (1 << (a.v_bits - 1)) - 1
        self.mom_sh = a.mom_shift        # mu = 1 - 2^-mom_shift   (0.875 at shift 3)
        self.lky_sh = a.leaky_shift      # alpha = 2^-leaky_shift  (0.125 at shift 3)
        self.lr = max(1, int(round(a.lr * self.wscale)))   # lr in MASTER units (>=1 LSB of master)
        self.agc = {}                    # layer -> current power-of-2 gain (a SHIFT)
        self.agc_mode = a.agc            # 'peak' | 'rms'  (the doc and the code disagreed — settle it)
        self.float_ref = a.float_ref
        self.act_bits = a.act_bits
        self.w_mode = a.w_mode           # TEST: which W the fwd/bwd datapaths see
        self.dither = a.dither           # TEST: stochastic carry into the analog cell
        self.rng = np.random.default_rng(a.seed + 1)
        self.delta_code = a.delta_code   # 'linear' | 'mulaw' (companded delta broadcast)
        self.mbits = a.mant_bits         # mantissa bits inside the companded delta
        self.eoff = a.exp_off            # exponent offset (sets the code's lowest step)
        self.dstats = {} if a.delta_stats else None
        self.e_fwd = a.jug_e_fwd     # E's live contribution to the forward MAC
        # ── PER-LAYER ADC GAIN (a power-of-2 shift) ───────────────────────────────────
        # THE FLOAT SIM HID THIS: the layers sit on different scales, so a single fixed 8-bit
        # mapping cannot serve all three. ⇒ THE CHIP NEEDS A PER-LAYER ADC REFERENCE / GAIN.
        # It is CONFIG, not runtime discovery: the boss writes a shift register at load time,
        # exactly like the weights and the routing. A chip never learns what layer it is in.
        # ⚠ CORRECTED 2026-07-14: the calibrated shifts are [>>3, >>6, >>6] = an **8x** spread,
        # NOT the "100:1" an earlier note claimed. That figure was WRONG — it came from three
        # measurements recorded in three DIFFERENT units (code units / ADC volts / raw). The real
        # requirement is a 2-3 bit register. Cheap, and NOT a blocker.
        self.ashift = [0, 0, 0]        # calibrated in calibrate()

    # ---- the analog weight cell: 8-bit. THERE IS NO FINER COPY. ----
    def cell(self, Wm):
        """The 8-bit analog weight. The jug writes it +-1 CODE at a time, so there is NO sub-LSB
        residue to store and NO digital master anywhere.

        ⚠⚠ --w_map: THE CODE -> WEIGHT MAP IS NOT LINEAR.
        The sim has always assumed  W = (code - CODE_MID).  **SPICE says otherwise.**
        The real weight is gm(V_w) - gm(V_zero), and gm is SIGMOIDAL in V_w: nearly FLAT at both
        rails (~0.05 uA/V per code) and STEEP in the middle (~1.46 uA/V per code) — a ~30x
        sensitivity variation.  In the flat regions a +-1-code fire BARELY MOVES THE WEIGHT —
        which is exactly the failure mode the jug exists to avoid.
        (At the INHERITED MN3_w=10 it was worse than non-linear: it was NON-MONOTONIC, and the
        entire positive half of the code range was INVERTED. Fixed by MN3_w=2. See
        ../circuit/THE_WEIGHT_IS_NOT_LINEAR.md.)

        NOTE the quantisation is in the CODE, not in the weight VALUE: the cell stores an 8-bit
        code, and the resulting ANALOG weight is a continuous function of it. So the map returns a
        FLOAT."""
        code = np.clip(Wm + CODE_MID, CODE_MIN, CODE_MAX).astype(np.int32)
        if W_MAP is None:
            return code - CODE_MID                       # the LINEAR map the sim always assumed
        return W_MAP[code - CODE_MIN]                    # the MEASURED map, from SPICE

    def wgt(self, Wm, path):
        return Wm if self.float_ref else self.cell(Wm)

    def shift(self, acc, s):
        if np.issubdtype(np.asarray(acc).dtype, np.integer):
            return acc >> s
        return np.rint(acc / float(1 << s)).astype(np.int64)

    def adc(self, y):
        """SIGNED 8-bit ADC on the hidden layers.

        ⚠ THE HIDDEN ACTIVATIONS MUST BE SIGNED. leaky-ReLU emits NEGATIVE values (alpha*y for
        y<0). A unipolar 0..255 ADC clips them all to ZERO — which silently turns leaky-ReLU into
        hard-ReLU and throws the whole negative half away (measured: 57% zeros, -9pp on the GHA
        baseline alone). The L0 INPUT is non-negative by construction (that is exactly what the
        split-sign encoding is for), but the hidden layers are not.
        ⇒ THE CHIP NEEDS A SIGNED (bipolar) ADC / activation bus on L1..L3.

        MUST return an INTEGER — np.rint gives float, which silently breaks the shift chain at
        the next layer (also caught the hard way)."""
        if self.float_ref:
            return y
        hi = (1 << (self.act_bits - 1)) - 1          # +127 for 8-bit
        lo = -(1 << (self.act_bits - 1))             # -128
        return np.clip(np.rint(y), lo, hi).astype(np.int64)

    def calibrate(self, Wm, Xc):
        """Set each layer's ADC shift so the activation rms lands at ~1/4 of full scale."""
        if self.float_ref:
            return
        full = (1 << (self.act_bits - 1)) - 1      # SIGNED full-scale (+127)
        want = full / 3.0
        f = Xc.astype(np.int64)
        for L in range(3):
            raws = []
            for c in range(len(Wm[L])):
                acc = 0
                for p in range(len(Wm[L][c])):
                    Ws = self.wgt(Wm[L][c][p], 'fwd')
                    seg = self._slice(f, L, c, p)
                    acc = acc + seg @ Ws.T
                raws.append(acc)
            R = np.concatenate(raws, axis=1).astype(np.float64)
            rms = float(np.sqrt(np.mean(R * R))) + 1e-12
            self.ashift[L] = int(np.clip(np.round(np.log2(rms / want)), 0, 24))
            f = self.adc(self.leaky(np.concatenate(
                [self.shift(r, self.ashift[L]) for r in raws], axis=1)))
        print(f"  ADC gain    : per-layer shifts {self.ashift} "
              f"(>>{self.ashift[0]}, >>{self.ashift[1]}, >>{self.ashift[2]})", flush=True)

    def _slice(self, f, L, c, p):
        if L == 0:
            st = c * B.N_FEATS_PER_L1
            return f[:, st + p*N_COLS: st + (p+1)*N_COLS]
        if L == 1:
            st = c * B.N_L1_PER_L2 * N_ROWS
            return f[:, st + p*N_ROWS: st + (p+1)*N_ROWS]
        li = B.L3_ROUTING[c][p]
        return f[:, li*N_ROWS:(li+1)*N_ROWS]

    def leaky(self, y):
        """alpha = 2^-shift — a SHIFT, no multiplier in the cell."""
        if self.float_ref:
            return np.where(y >= 0, y, 0.1 * y)
        y = np.asarray(y, dtype=np.int64)
        return np.where(y >= 0, y, -((-y) >> self.lky_sh))

    def leaky_d(self, f):
        if self.float_ref:
            return np.where(f >= 0, 1.0, 0.1)
        return np.where(f >= 0, 1.0, 2.0 ** -self.lky_sh)

    # ---- the 6-bit delta channel: power-of-2 AGC, then quantise to a FIXED range ----
    def channel(self, D, layer):
        if self.float_ref:
            return D
        # PEAK AGC by default, not rms AGC. The delta has a CREST FACTOR of ~15, so setting the
        # gain from the rms (aiming rms at full-scale/4) CLIPS the large deltas — and the large
        # deltas are the informative ones. Set the gain from the PEAK instead (a peak detector,
        # which is what a real AGC uses anyway). Measured: rms-AGC cost ~5pp; peak-AGC is free.
        # NB the SLEW LIMIT below is load-bearing: an UNSLEWED peak AGC is unstable (one outlier
        # delta sets the gain and crushes everything else). --agc rms re-tests the alternative.
        if self.agc_mode == 'rms':
            ref = float(np.sqrt(np.mean(D * D))) * 3.0 + 1e-12
        else:
            ref = float(np.max(np.abs(D))) + 1e-12
        # ── THE CREST-FACTOR TRAP ────────────────────────────────────────────────────────
        # Within a fold the gain is a single scalar, and at the fold only sign(E) survives — and
        # sign is invariant to a positive scale. So the AGC GAIN ITSELF barely matters. The delta
        # channel can only cost us through ROUNDING (small deltas -> 0) and CLIPPING (big deltas
        # -> +-dmax). With a crest factor of ~15, a LINEAR 6-bit code cannot avoid both: peak-AGC
        # spends 4 of its 6 bits on headroom and leaves ~2 bits on the bulk of the signal;
        # rms-AGC recovers the resolution and clips the informative tails.
        # ⇒ --delta_code mulaw: a COMPANDED code (sign + exponent + mantissa). A priority encoder
        #   and a shift. No multiplier. Spans the tails AND keeps relative precision on the bulk.
        sh = int(np.clip(np.floor(np.log2(self.dmax / ref)), -30, 30))
        prev = self.agc.get(layer)
        # SLOW AGC: move the shift by at most 1 per fold (a real AGC has a time constant)
        if prev is None:
            self.agc[layer] = sh
        else:
            self.agc[layer] = prev + int(np.sign(sh - prev)) if sh != prev else prev
        X = D * (2.0 ** self.agc[layer])
        Q, qmax = self.compand(X) if self.delta_code == 'mulaw' else \
            (np.clip(np.rint(X), -self.dmax, self.dmax), self.dmax)
        if self.dstats is not None:
            crest = float(np.max(np.abs(X))) / (float(np.sqrt(np.mean(X * X))) + 1e-12)
            self.dstats.setdefault(layer, []).append(
                (float(np.mean(Q == 0)), float(np.mean(np.abs(Q) >= qmax)), crest))
        return Q

    def compand(self, X):
        """mu-law style COMPANDED delta: sign + exponent + mantissa, in `dbits` WIRES.

        A LINEAR code must choose between resolution and headroom, and with a crest factor of ~15
        it cannot have both in 6 bits. A COMPANDED one does not have to choose: the magnitude is
        coded as (exp, mantissa), giving CONSTANT RELATIVE precision across a huge range. In
        silicon this is a priority encoder and a shift — no multiplier.

        NOTE the asymmetry that makes this cheap: the CROSS-CHIP BROADCAST still carries only
        `dbits` bits (sign + exp + mant), but the value DECODED locally at the receiver is wider.
        Local register width is cheap; the inter-chip wire is the expensive thing, and it does not
        grow.

        Returns an INTEGER in units of the smallest quantiser step. The absolute scale is
        irrelevant: within a fold the gain is a constant, only sign(E) survives the fold, and the
        next layer's AGC re-normalises anyway."""
        mb = self.mbits
        emax = (1 << (self.dbits - 1 - mb)) - 1        # sign + exponent + mantissa = dbits
        a = np.abs(X)
        e = np.clip(np.floor(np.log2(a + 1e-30)).astype(np.int64) - self.eoff, 0, emax)
        m = np.clip(np.rint(a / (2.0 ** (e + self.eoff - mb))),
                    0, (1 << (mb + 1)) - 1).astype(np.int64)
        Q = np.sign(X).astype(np.int64) * (m << e)
        return Q, (((1 << (mb + 1)) - 1) << emax)


def normalise_frozen_w(hw, Wf, Xc, target=None):
    """★ TASK #15 — NORM-PRESERVING FROZEN W. Scale each CHIP's frozen W by a SCALAR.

    ‖W‖ DRIVES TWO THINGS AT ONCE:
      1. the FORWARD activation scale  — which is the ONLY reason the layers need DIFFERENT ADC
         gains at all (calibrated shifts today: [>>3, >>6, >>6] = an 8x spread);
      2. the BACKWARD delta magnitude  — `d_{l-1} = W.T @ d_l`, so the delta attenuates by ~‖W‖
         per layer. THAT is the vanishing-gradient wall that limits DEPTH.

    **ONE KNOB, TWO PROBLEMS.** If each chip's W is scaled so its output rms matches a common
    target, then (a) one GLOBAL ADC gain should suffice — the per-layer gain requirement may
    DISAPPEAR — and (b) the delta should stop shrinking with depth.

    It is a per-chip SCALAR: it preserves the PCA/GHA directions and fixes only the gain.
    A rotation would destroy the learned basis; a scalar cannot."""
    f = Xc.astype(np.float64)
    for L in range(3):
        in_rms = float(np.sqrt(np.mean(f * f))) + 1e-12      # NORM-PRESERVING: match THIS.
        accs, alphas = [], []
        for c in range(len(Wf[L])):
            acc = 0.0
            for p in range(len(Wf[L][c])):
                acc = acc + hw._slice(f, L, c, p) @ Wf[L][c][p].T
            r = float(np.sqrt(np.mean(acc * acc))) + 1e-12
            a = in_rms / r                                    # scale so OUT rms == IN rms
            for p in range(len(Wf[L][c])):
                Wf[L][c][p] = Wf[L][c][p] * a
            alphas.append(a); accs.append(acc * a)
        y = np.concatenate(accs, axis=1)
        f = np.where(y >= 0, y, 0.1 * y)                      # leaky, as in the real forward
        wmax = max(float(np.abs(pg).max()) for ch in Wf[L] for pg in ch)
        print(f"  norm W  L{L+1}: x{np.mean(alphas):7.3f} "
              f"(chip spread {min(alphas):.3f}–{max(alphas):.3f})  "
              f"in_rms={in_rms:8.2f} -> out_rms={in_rms:8.2f}   |W|max={wmax:.3f}"
              f"{'  ⚠ CLIPS THE 8-BIT RAIL' if wmax > 0.94 else ''}", flush=True)
    return Wf


def e_fwd_codes(hw, E, THETA):
    """The E-cap's contribution to the FORWARD MAC, in weight-code units.

    THE CHIP'S CELL IS DUAL-TAIL: I_ntail = I_tail_W + I_tail_E, so the MAC computes W + E.
    Ce is wired into the tail — this is NOT optional in silicon. Scaled so that a synapse sitting
    exactly at the fire threshold contributes `--jug_e_fwd` weight codes."""
    if hw.e_fwd <= 0.0 or THETA[0] == 0:
        return None
    return [[[pg.astype(np.float64) * (hw.e_fwd / THETA[L]) for pg in ch] for ch in E[L]]
            for L in range(3)]


def forward(hw, Wm, Xc, Ef=None):
    """Integer forward. Xc: (N, 1152) act codes. Returns the per-layer activations (codes).

    Ef: the E-cap's forward contribution (see e_fwd_codes). None = E invisible to the forward,
    which is what the sim has always done and what the CHIP CANNOT DO."""
    A = []
    f = Xc.astype(np.int64)
    # L1: 24 chips x 3 pages, 48 -> 16
    out = np.empty((len(f), B.N_L1_FEATS), dtype=np.int64 if not hw.float_ref else np.float64)
    for c in range(B.N_L1_CHIPS):
        acc = 0
        for p in range(B.N_PAGES):
            Ws = hw.wgt(Wm[0][c][p], 'fwd')
            if Ef is not None:
                Ws = Ws + Ef[0][c][p]      # DUAL-TAIL: the MAC sees W + E
            seg = f[:, c*B.N_FEATS_PER_L1 + p*N_COLS : c*B.N_FEATS_PER_L1 + (p+1)*N_COLS]
            acc = acc + seg @ Ws.T
        y = acc / CODE_SCALE if hw.float_ref else hw.shift(acc, hw.ashift[0])
        out[:, c*N_ROWS:(c+1)*N_ROWS] = hw.adc(hw.leaky(y))
    A.append(out); f = out
    # L2: 8 chips x 3 pages
    out = np.empty((len(f), B.N_L2_FEATS), dtype=out.dtype)
    for g in range(B.N_L2_CHIPS):
        acc = 0
        st = g * B.N_L1_PER_L2 * N_ROWS
        for p in range(B.N_PAGES):
            Ws = hw.wgt(Wm[1][g][p], 'fwd')
            if Ef is not None:
                Ws = Ws + Ef[1][g][p]      # DUAL-TAIL: the MAC sees W + E
            seg = f[:, st + p*N_ROWS : st + (p+1)*N_ROWS]
            acc = acc + seg @ Ws.T
        y = acc / CODE_SCALE if hw.float_ref else hw.shift(acc, hw.ashift[1])
        out[:, g*N_ROWS:(g+1)*N_ROWS] = hw.adc(hw.leaky(y))
    A.append(out); f = out
    # L3: 16 chips x 2 pages, OVERLAPPING routing
    out = np.empty((len(f), B.N_L3_FEATS), dtype=out.dtype)
    for g in range(B.N_L3_CHIPS):
        acc = 0
        for p, li in enumerate(B.L3_ROUTING[g]):
            Ws = hw.wgt(Wm[2][g][p], 'fwd')
            if Ef is not None:
                Ws = Ws + Ef[2][g][p]      # DUAL-TAIL: the MAC sees W + E
            seg = f[:, li*N_ROWS:(li+1)*N_ROWS]
            acc = acc + seg @ Ws.T
        y = acc / CODE_SCALE if hw.float_ref else hw.shift(acc, hw.ashift[2])
        out[:, g*N_ROWS:(g+1)*N_ROWS] = hw.adc(hw.leaky(y))
    A.append(out)
    return A


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--chunk', type=int, default=32000)
    p.add_argument('--fold_every', type=int, default=128)
    p.add_argument('--act_bits', type=int, default=8, help='ADC width at EVERY layer')
    p.add_argument('--delta_bits', type=int, default=6, help='cross-chip delta broadcast')
    p.add_argument('--e_acc_bits', type=int, default=20, help='E accumulator width (saturating)')
    p.add_argument('--leaky_shift', type=int, default=3, help='alpha = 2^-s  (3 => 0.125)')
    p.add_argument('--float_ref', action='store_true', help='float control: the CEILING (~82.1%%)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--agc', default='rms', choices=['rms', 'peak'],
                   help="delta-channel AGC reference. **rms IS THE DEFAULT AND MUST STAY THAT "
                        "WAY.** A `peak` default silently cost 8pp once (it quantises 71-75%% of "
                        "all deltas to ZERO) and sat undetected in a doc claiming near-validation. "
                        "See ../../hw_buildcheck/HW_BUILD_CHECK.md §6.")
    p.add_argument('--delta_code', default='linear', choices=['linear', 'mulaw'],
                   help='CROSS-CHIP DELTA CODE. linear = a plain signed integer (what we have). '
                        'mulaw = COMPANDED (sign + exp + mantissa) in the SAME number of wires. '
                        'The delta has a crest factor of ~15, which a linear 6-bit code cannot '
                        'span: peak-AGC leaves ~2 bits on the bulk of the signal, rms-AGC clips '
                        'the tails. Companding removes the choice. Priority encoder + shift.')
    p.add_argument('--mant_bits', type=int, default=2, help='mantissa bits in the companded delta')
    p.add_argument('--exp_off', type=int, default=-3, help='exponent offset (lowest coded step)')
    # ── THE LEAKY JUG (the user's mechanism, 2026-07-14) ─────────────────────────────────
    # E is NOT reset at the fold. It LEAKS, and the deltas top it up. When |E| crosses a
    # threshold the synapse FIRES: the analog cell moves a WHOLE +-1 LSB (all it can do), and
    # theta is SUBTRACTED from E — the residue is KEPT, which is what makes this a true
    # sigma-delta rather than a lossy reset.
    #   * `E <- lambda*E + g` IS the momentum recurrence. The LEAK **IS** MOMENTUM. E and the
    #     velocity register are THE SAME REGISTER. Momentum stops being an algorithm we
    #     implement and becomes a property of the storage medium (a capacitor).
    #   * --jug FORCES frac_bits = 0: the weight state IS the 8-bit analog cell. NO digital
    #     master, NO sub-LSB residue, NO velocity register. NO per-synapse digital store at all.
    #   * The GLOBAL FOLD disappears in the real thing (each synapse fires when IT has evidence).
    #     Here we still evaluate the threshold on the fold grid — a batched approximation.
    #   * A FIRE is a CARRY is a ROUTER-SHADOW REFRESH: one local event, three jobs.
    # ⚠ THE RISK IS THE DEADZONE. A leaky integrator saturates at E_ss = g/(1-lambda). To fire at
    #   the CURRENT effective lr a synapse needs theta ~ 51*g, so it only ever fires if
    #   1/(1-lambda) > 51, i.e. lambda > 0.98 (--jug_leak >= 6). With the current momentum's
    #   lambda = 0.875 (leak 3), E_ss = 8g < theta and EVERY SYNAPSE FREEZES. Hence the tension to
    #   measure: the weaker the leak must be for the jug to fire, the less momentum it provides.
    p.add_argument('--jug_leak', type=int, default=0,
                   help='DESIGNED leak (a switched-capacitor: lambda = 1 - 2^-N per fold). '
                        '0 = NO leak, a PURE INTEGRATOR — **THIS IS THE DESIGN POINT.** The '
                        'physical Ce leaks at ~50 uV/s (~140 s per E-LSB) against an inter-fire '
                        'interval of tens of ms, so tau_leak / inter-fire ~ 1e3-1e4: the real '
                        'capacitor IS a pure integrator on the firing timescale. The leak (= '
                        'momentum) is worth at most +0.3pp and is inside the noise floor.')
    p.add_argument('--jug_theta', type=float, default=8.0,
                   help='fire threshold, as a multiple of the calibrated per-fold rms of |E| '
                        '(per layer, set once at calibration — CONFIG, like the ADC gain). This is '
                        'the learning rate: effective lr = firing_rate x 1 cell-LSB. **8 IS THE '
                        'MEASURED OPTIMUM** — a clean inverted-U: 4 -> 81.08, 8 -> 81.96, '
                        '16 -> 80.37, 32 -> 73.95.')

    # ── ⚠⚠ THE DUAL-TAIL QUESTION — E IS PHYSICALLY LIVE IN THE FORWARD MAC ──────────────
    # mac_cell_emx is DUAL-TAIL: I_ntail = I_tail_W + I_tail_E. Ce is wired into the tail, so the
    # MAC computes W + E. You CANNOT switch that off in silicon. The sim's forward uses W alone.
    # MEASURED (weight codes that a full-threshold E contributes to the MAC):
    #     0.0 -> 81.96   0.1 -> 82.06   0.2 -> 81.32   0.4 -> 80.46   1.0 -> 76.83
    #     2.0 -> COLLAPSE (47% then 12%)     6.7 (the E-cap clamp) -> COLLAPSE (10%)
    # It is a POSITIVE FEEDBACK RUNAWAY: E feeds the MAC -> activations grow -> deltas grow ->
    # E grows faster -> fires more (0.9%/fold -> 14.8% -> 19.3%).
    # ⇒ **SAFE ZONE IS <= 0.2 CODES.** The old design's gm_E ~ gm_W target puts it at ~0.4 —
    #   ALREADY LOSSY, and only ~2.5x from runaway. ⇒ REMOVE THE E-TAIL (MN3_e), or size gm_E so
    #   a full-threshold E contributes <= 0.2 codes. See ../DESIGN.md §3.1.
    p.add_argument('--jug_e_fwd', type=float, default=0.0,
                   help="weight codes that a full-threshold E contributes to the FORWARD MAC. "
                        "0 = no E-tail (the RECOMMENDED design). 0.4 = the old dual-tail design "
                        "point (costs 1.5pp). >=1.0 destabilises; >=2.0 runs away completely.")
    p.add_argument('--jug_sign_err', type=float, default=0.0,
                   help="probability a fire has the WRONG SIGN (comparator noise near the zero "
                        "crossing). **MEASURED FREE: 5%% -> 81.42, 10%% -> 82.06, 20%% -> 81.62.** "
                        "The residue-preserving subtraction self-corrects: a wrong fire ADDS the "
                        "charge back, and the next correct fire undoes it. Charge is conserved. "
                        "⇒ THE COMPARATOR SPEC IS VERY LOOSE.")

    # ── PHYSICAL (mismatched) CAPACITORS — what silicon actually hands you ────────────────
    # Subthreshold leakage goes EXPONENTIALLY with Vt and Vt mismatch is Gaussian => the leak
    # current is LOG-NORMAL and varies many-fold across a die. MEASURED (clean = 81.96):
    #   tau=1000 folds, 3x spread -> 81.68 | tau=100 (drains ~as fast as it fills), 3x -> 81.53
    #   tau=1000, 10x spread      -> 81.38 | threshold mismatch x2               -> 81.20
    # ALL INSIDE THE 0.7pp NOISE FLOOR. A leaky cell does not get the WRONG answer — it needs more
    # charge to fire, so it FIRES LESS OFTEN = a reduced PER-SYNAPSE LEARNING RATE, which SGD is
    # famously indifferent to. ⇒ **NO SERVO, NO REPLICA BIAS. MATCHING IS A NON-REQUIREMENT.**
    p.add_argument('--jug_tau', type=float, default=0.0,
                   help='PARASITIC leak: median leakage time constant IN FOLDS. 0 = ideal. The '
                        'REAL Ce gives tau/inter-fire ~ 1e3-1e4, so this is a stress test.')
    p.add_argument('--jug_tau_spread', type=float, default=1.0,
                   help='per-synapse LOG-NORMAL spread of tau (geometric sigma). 10x is free.')
    p.add_argument('--jug_theta_spread', type=float, default=1.0,
                   help='per-synapse LOG-NORMAL spread of the FIRE THRESHOLD (comparator offset + '
                        'capacitor mismatch) = a random per-synapse learning rate. x2 is free.')
    p.add_argument('--jug_ce_noise', type=float, default=0.0,
                   help='kT/C THERMAL NOISE on Ce, as a fraction of theta. sqrt(kT/C) is real and '
                        'unavoidable (203 uV rms on 100 fF; 64 uV on 1 pF), and it sets the '
                        'MINIMUM CAPACITOR SIZE. Sweep this to find how much noise the jug '
                        'tolerates, then read Ce off the answer. This is the LAST number SPICE '
                        'needs.')
    p.add_argument('--jug_multifire', action='store_true',
                   help='fire n = floor(|E|/theta) times per check instead of at most ONCE. The '
                        'correct sigma-delta, and closer to the chip (which fires ASYNCHRONOUSLY '
                        'on every crossing). Without it a high-gradient synapse that races past '
                        'theta mid-fold is throttled to one LSB per fold and E piles up.')
    p.add_argument('--w_rail', type=float, default=0.0,
                   help='★ THE PRE-DISTORTED (CALIBRATED) WEIGHT DAC. Clip the weight rail to '
                        '±X and keep the LINEAR map. The cell maps V_w -> weight SIGMOIDALLY, so '
                        'a LINEAR code->V_w DAC gives a sigmoidal weight (--w_map). A '
                        'PRE-DISTORTED DAC (an LUT: code -> V_w chosen so the WEIGHT is uniform) '
                        'makes the weight linear again — at the cost of range. MEASURED: ±28 of '
                        'the sigmoid\'s ±31.9 span (88%%) fits in 427 mV of the 850 mV Vw budget, '
                        'with FULL 121-code resolution. In sim units that is a rail of ±0.83 '
                        '(vs ±0.95). This flag measures what that 12%% of range costs.')
    p.add_argument('--w_map', default=None,
                   help='⚠ USE THE SPICE-MEASURED CODE->WEIGHT MAP instead of the linear one the '
                        'sim has always assumed. e.g. ../circuit/output/w_map.csv. The real map is '
                        'SIGMOIDAL (~30x sensitivity variation between the rails and the middle), '
                        'so a +-1-code fire is worth very different amounts of weight depending on '
                        'where the weight sits. See ../circuit/THE_WEIGHT_IS_NOT_LINEAR.md.')
    p.add_argument('--norm_w', action='store_true',
                   help='★ TASK #15: scale each CHIP\'s frozen W by a SCALAR so it is '
                        'NORM-PRESERVING (output rms == input rms). ‖W‖ drives BOTH the forward '
                        'activation scale (the ONLY reason the layers need different ADC gains) '
                        'AND the backward delta magnitude (d_{l-1} = W.T @ d_l — the '
                        'vanishing-gradient wall that limits DEPTH). ONE KNOB, TWO PROBLEMS. '
                        'Preserves the PCA/GHA directions; fixes only the gain.')
    p.add_argument('--jug_step', type=float, default=1.0 / CODE_SCALE,
                   help='FLOAT MODE ONLY: how far one fire moves the weight, in FLOAT units. '
                        'Default 1/64 = the float equivalent of ONE 8-bit code (what the chip '
                        'does). Make it SMALLER to ask whether a finer write quantum would help — '
                        'a question the silicon cannot ask. ⚠ It confounds with the effective '
                        'learning rate (eff_lr = fire_rate x step), so retune --jug_theta with it.')
    p.add_argument('--delta_stats', action='store_true',
                   help='DIAGNOSTIC: report what the delta channel is actually doing — the '
                        'fraction of deltas quantised to ZERO, the fraction CLIPPED at the rail, '
                        'and the crest factor. Then exit.')
    a = p.parse_args()
    a.jug = True          # THIS IS THE JUG CHIP. It is not an option.
    if a.w_map:
        load_w_map(a.w_map)
    a.jug_pure = (a.jug_leak == 0)      # leak 0 => pure integrator (the design point)
    a.frac_bits = 0       # the weight state IS the 8-bit analog cell. No master, no residue.
    a.lr = 0.0; a.v_bits = 12; a.mom_shift = 3; a.w_mode = 'cell'; a.dither = False
    hw = HW(a)

    F_tr, F_te, y_tr, y_te = B.get_emnist_l0_features()
    Xc_tr = B.float_to_act_code(F_tr).astype(np.int64)
    Xc_te = B.float_to_act_code(F_te).astype(np.int64)
    _d = np.concatenate([np.where(y_te == c)[0][:40] for c in range(N_CLASSES)])

    # weights -> integer MASTER (float weight * 64 * 2^frac)
    w1, w2 = B.load_l1_weights(), B.load_l2_weights()
    w3 = B.init_l3_weights_random_ortho(seed=2)
    if a.norm_w:
        # ★ TASK #15 — do this on the FLOAT weights, BEFORE they are quantised to 8-bit codes.
        w1, w2, w3 = normalise_frozen_w(hw, [w1, w2, w3], Xc_tr[:4000])
    if a.float_ref:
        Wm = [w1, w2, w3]
    else:
        # ⚠ THE FROZEN WEIGHTS MUST GO THROUGH THE **INVERSE MAP**.
        # `rint(w * wscale)` assumes the weight is LINEAR in the code. It is not (SPICE).
        # A real chip picks the code whose MEASURED weight is closest to the one it wants —
        # a calibration LUT, done in software at load time. Free, and correct.
        Wm = [[[w_to_code(pg.astype(np.float64)) for pg in ch]
               for ch in L] for L in (w1, w2, w3)]
    w_lo = -a.w_rail if a.w_rail > 0 else B.W_FLOAT_MIN
    w_hi = +a.w_rail if a.w_rail > 0 else B.W_FLOAT_MAX
    if a.w_rail > 0:
        print(f"  ** PRE-DISTORTED DAC ** weight rail ±{a.w_rail} (linear map, reduced range)")
    if a.float_ref:
        WMIN, WMAX = w_lo, w_hi                            # the FLOAT weight rail
    else:
        WMIN = int(np.rint(w_lo * hw.wscale))              # the 8-bit CODE rail
        WMAX = int(np.rint(w_hi * hw.wscale))

    E = [[[np.zeros_like(pg, dtype=np.int64) for pg in ch] for ch in L] for L in Wm]

    print(f"\n{'='*78}")
    print("BIT-ACCURATE HARDWARE MODEL — BIG (24/8/16 chips)")
    if a.float_ref:
        print("  *** FLOAT REFERENCE (control) ***")
    else:
        print(f"  activations : {a.act_bits}-bit ADC at EVERY layer")
        print(f"  weight cell : 8-bit analog (CODE_MID={CODE_MID}) + {a.frac_bits} SUB-LSB bits")
        print(f"                => {8+a.frac_bits}-bit digital master, {hw.lr} master-LSB per write")
        print(f"                => the analog cell moves 1 LSB every ~{(1<<a.frac_bits)//hw.lr} folds")
        print(f"  delta       : {a.delta_bits}-bit signed + power-of-2 AGC (a SHIFT)")
        print(f"  E accum     : {a.e_acc_bits}-bit saturating")
        print(f"  velocity    : {a.v_bits}-bit,  mu = 1-2^-{a.mom_shift} = {1-2**-a.mom_shift}")
        print(f"  leaky       : alpha = 2^-{a.leaky_shift} = {2**-a.leaky_shift}  (a SHIFT)")
        print(f"  AGC ref     : {a.agc}")
        if a.w_mode != 'cell' or a.dither:
            print(f"  ** PROBE **   w_mode={a.w_mode}  dither={a.dither}")
    hw.calibrate(Wm, Xc_tr[:2000])
    print(f"{'='*78}\n", flush=True)

    # ── JUG: calibrate the fire threshold, and set up the firing statistics ───────────────
    # theta is a PER-LAYER constant fixed at calibration — exactly the same category as the ADC
    # gain: the boss writes it in at load time. A chip never has to discover it at runtime.
    # ── THE WRITE QUANTUM ────────────────────────────────────────────────────────────────
    # The chip can only move the weight by ONE CODE. In float mode there is no code grid, so the
    # step must be given explicitly — the float EQUIVALENT of one code is 1/CODE_SCALE = 1/64.
    # --jug_step lets the FLOAT arm ask the question the chip cannot: does a FINER write quantum
    # help? (If it does, the 1-LSB quantum is costing us. If not, the coarse write is FREE.)
    STEP = (a.jug_step if a.float_ref else 1)
    if a.float_ref:
        print(f"  ** FLOAT JUG **  weight is a FLOAT; one fire moves it {a.jug_step:.5f} "
              f"(= {a.jug_step * CODE_SCALE:.2f} code equivalents)")

    THETA = [0, 0, 0]
    FIRES = [None, None, None]     # per-synapse fire counts, for the FROZEN fraction
    LAM = THS = None               # per-synapse PHYSICAL device mismatch
    if a.jug:
        FIRES = [[[np.zeros_like(pg, dtype=np.int64) for pg in ch] for ch in L] for L in Wm]
        # ── PHYSICAL DEVICES: log-normal mismatch, drawn ONCE (it is fabrication, not noise) ──
        drng = np.random.default_rng(a.seed + 99)
        if a.jug_tau > 0:
            s = np.log(a.jug_tau_spread)
            LAM = [[[np.exp(-1.0 / (a.jug_tau * np.exp(drng.normal(0, s, pg.shape))))
                     for pg in ch] for ch in L] for L in Wm]
            allt = np.concatenate([(-1.0 / np.log(x)).ravel() for L in LAM for ch in L for x in ch])
            print(f"  ** PARASITIC LEAK **  tau median {a.jug_tau:.0f} folds, spread "
                  f"x{a.jug_tau_spread}  => tau p5..p95 = "
                  f"{np.percentile(allt, 5):.0f}..{np.percentile(allt, 95):.0f} folds")
        if a.jug_theta_spread != 1.0:
            s = np.log(a.jug_theta_spread)
            THS = [[[np.exp(drng.normal(0, s, pg.shape)) for pg in ch] for ch in L] for L in Wm]
            print(f"  ** THRESHOLD MISMATCH **  log-normal x{a.jug_theta_spread} "
                  f"(= a random per-synapse learning rate)")
        print(f"  ** LEAKY JUG **  E leaks (the leak IS momentum); a threshold crossing fires")
        print(f"                   a whole +-1 cell-LSB.  frac_bits FORCED to 0:")
        print(f"                   NO digital master, NO velocity register, NO per-synapse SRAM.")
        if a.jug_pure:
            print(f"     leak       : NONE (pure sigma-delta integrator — the CONTROL)")
        else:
            lam = 1.0 - 2.0 ** -a.jug_leak
            print(f"     leak       : lambda = 1-2^-{a.jug_leak} = {lam:.4f}  "
                  f"(~{2**a.jug_leak}-fold memory)")
            print(f"                  E_ss = g/(1-lambda) = {2**a.jug_leak:.0f}g  "
                  f"=> fires only if theta < {2**a.jug_leak:.0f}g")
        print(f"     theta      : {a.jug_theta} x the per-fold rms of |E|  (per layer)")

    rng = np.random.default_rng(a.seed)
    best = 0.0
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        A = forward(hw, Wm, Xc_tr, e_fwd_codes(hw, E, THETA))
        W_f, b_f = B.fit_clf(A[-1].astype(np.float64), y_tr)     # DIGITAL readout (boss)
        perm = rng.choice(len(y_tr), size=a.chunk, replace=False)

        for b0 in range(0, a.chunk - a.fold_every + 1, a.fold_every):
            sl = perm[b0:b0 + a.fold_every]
            Ab = forward(hw, Wm, Xc_tr[sl], e_fwd_codes(hw, E, THETA))
            F1, F2, F3 = Ab
            sc = F3.astype(np.float64) @ W_f.T + b_f
            z = sc - sc.max(1, keepdims=True); pr = np.exp(z); pr /= pr.sum(1, keepdims=True)
            S = -pr; S[np.arange(len(sl)), y_tr[sl]] += 1.0      # boss: onehot - softmax

            D3 = hw.channel((S @ W_f) * hw.leaky_d(F3), 'L3')
            for g in range(B.N_L3_CHIPS):
                Dg = D3[:, g*N_ROWS:(g+1)*N_ROWS]
                for p, li in enumerate(B.L3_ROUTING[g]):
                    E[2][g][p] += (Dg.T @ F2[:, li*N_ROWS:(li+1)*N_ROWS]).astype(np.int64)

            G2 = np.zeros((len(sl), B.N_L2_FEATS))
            for g in range(B.N_L3_CHIPS):
                Dg = D3[:, g*N_ROWS:(g+1)*N_ROWS]
                for p, li in enumerate(B.L3_ROUTING[g]):
                    # THE ROUTER-LOCAL W.T. By default this is the SAME 8-bit analog code the
                    # forward pass used — so the backprojection is the TRUE transpose of the
                    # operator that actually computed the forward. (And the router's shadow copy
                    # only needs refreshing when a cell CARRIES: ~2% of synapses per fold.)
                    Ws = hw.wgt(Wm[2][g][p], 'bwd')
                    G2[:, li*N_ROWS:(li+1)*N_ROWS] += Dg @ Ws
            D2 = hw.channel(G2 * hw.leaky_d(F2), 'L2')
            for g in range(B.N_L2_CHIPS):
                Dg = D2[:, g*N_ROWS:(g+1)*N_ROWS]
                st = g * B.N_L1_PER_L2 * N_ROWS
                for p in range(B.N_PAGES):
                    E[1][g][p] += (Dg.T @ F1[:, st+p*N_ROWS:st+(p+1)*N_ROWS]).astype(np.int64)

            G1 = np.zeros((len(sl), B.N_L1_FEATS))
            for g in range(B.N_L2_CHIPS):
                Dg = D2[:, g*N_ROWS:(g+1)*N_ROWS]
                st = g * B.N_L1_PER_L2 * N_ROWS
                for p in range(B.N_PAGES):
                    Ws = hw.wgt(Wm[1][g][p], 'bwd')
                    c = g * B.N_L1_PER_L2 + p
                    G1[:, c*N_ROWS:(c+1)*N_ROWS] = Dg @ Ws
            D1 = hw.channel(G1 * hw.leaky_d(F1), 'L1')
            for c in range(B.N_L1_CHIPS):
                Dg = D1[:, c*N_ROWS:(c+1)*N_ROWS]
                st = c * B.N_FEATS_PER_L1
                for p in range(B.N_PAGES):
                    E[0][c][p] += (Dg.T @ Xc_tr[sl][:, st+p*N_COLS:st+(p+1)*N_COLS]).astype(np.int64)

            if hw.dstats is not None and b0 // a.fold_every >= 19:
                print(f"\n  DELTA CHANNEL DIAGNOSTIC  (20 folds, code={a.delta_code}, "
                      f"agc={a.agc}, {a.delta_bits} bits)\n")
                print(f"  {'layer':<6}{'-> ZERO':>10}{'CLIPPED':>10}{'crest':>9}")
                for L in ('L3', 'L2', 'L1'):
                    z, cl, cr = np.array(hw.dstats[L]).mean(axis=0)
                    print(f"  {L:<6}{z*100:9.1f}%{cl*100:9.1f}%{cr:9.1f}")
                print("\n  A delta quantised to ZERO carries no information. A delta CLIPPED at "
                      "the rail\n  is the informative tail, truncated.\n")
                # ── THE E ACCUMULATOR IS COUPLED TO THE ADC WIDTH ────────────────────────────
                # E += D.T @ F over the fold, so E scales with the ACTIVATION full scale. Its
                # width is therefore NOT free to choose independently of --act_bits: widening the
                # ADC scales F, scales E, and SATURATES the accumulator. (Worst case at 128
                # samples: act8 needs 20 bits, act12 needs 24, act16 needs 28.) Any --act_bits
                # sweep MUST widen --e_acc_bits with it or it is measuring saturation, not the ADC.
                tot = sat = mx = 0
                for L in range(3):
                    for c in range(len(E[L])):
                        for pg in E[L][c]:
                            tot += pg.size
                            sat += int(np.sum(np.abs(pg) >= hw.emax))
                            mx = max(mx, int(np.max(np.abs(pg))))
                print(f"  E ACCUMULATOR  ({a.e_acc_bits}-bit, rail = {hw.emax:,})")
                print(f"    saturated entries : {sat/tot*100:.2f}%")
                print(f"    peak |E| reached  : {mx:,}   ({mx.bit_length()+1} bits to hold it)\n")
                return

            # ── THE LEAKY JUG ────────────────────────────────────────────────────────────
            # E is NOT reset. It leaks, the deltas top it up, and a threshold crossing fires a
            # whole +-1 cell-LSB. `E <- lambda*E + g` IS the momentum recurrence: THE LEAK IS
            # THE MOMENTUM, and E and the velocity register become the SAME REGISTER.
            if a.jug:
                if THETA[0] == 0:      # calibrate theta ONCE, from the first fold's |E| rms
                    for L in range(3):
                        r = np.sqrt(np.mean(np.concatenate(
                            [pg.ravel().astype(np.float64) ** 2
                             for ch in E[L] for pg in ch]))) + 1e-12
                        THETA[L] = max(1, int(a.jug_theta * r))
                    print(f"  jug theta   : {THETA}  (per layer, fixed at calibration)",
                          flush=True)
                for L in range(3):
                    for c in range(len(Wm[L])):
                        for p in range(len(Wm[L][c])):
                            e = E[L][c][p]
                            # DESIGNED leak (a switched-capacitor: a shift, well matched).
                            # sign-magnitude, NEVER an arithmetic shift — that decays positives
                            # and negatives differently and biases every weight.
                            # NB the leak lands on (E_prev + g) rather than E_prev; that is a
                            # constant factor on g, which theta absorbs.
                            if not a.jug_pure:
                                e = np.sign(e) * ((np.abs(e) * ((1 << a.jug_leak) - 1))
                                                  >> a.jug_leak)
                            # PARASITIC leak: a real capacitor really does lose a FRACTION of its
                            # charge, per synapse, at its own rate. Modelled as physics (a float
                            # decay, then back to charge quanta), not as a circuit op.
                            if LAM is not None:
                                e = np.rint(e * LAM[L][c][p]).astype(np.int64)
                            th = THETA[L] if THS is None else np.maximum(
                                1, np.rint(THETA[L] * THS[L][c][p])).astype(np.int64)
                            # ── kT/C THERMAL NOISE ON Ce ────────────────────────────────────
                            # sqrt(kT/C) is REAL and UNAVOIDABLE: 203 uV rms on 100 fF, 64 uV on
                            # 1 pF. It sets the smallest injection Ce can meaningfully hold, and
                            # therefore THE MINIMUM CAPACITOR SIZE. Expressed as a fraction of
                            # theta so it maps straight onto a Ce value.
                            if a.jug_ce_noise > 0:
                                e = e + np.rint(hw.rng.normal(
                                    0.0, a.jug_ce_noise * float(np.mean(th)),
                                    e.shape)).astype(np.int64)
                            np.clip(e, -hw.emax, hw.emax, out=e)
                            # FIRE: a comparator. The cell moves ONE WHOLE LSB — all it can do.
                            # ⚠ THE FOLD-GRID APPROXIMATION. The chip fires ASYNCHRONOUSLY, the
                            # moment a synapse crosses theta. The sim checks the comparator once
                            # per fold (128 samples) and — by default — fires AT MOST ONCE. A
                            # synapse that raced to 5*theta during the fold would fire 5x in the
                            # chip; here it fires once and carries 4*theta forward. Nothing is lost
                            # (the residue is kept) but the WEIGHT LAGS and E climbs — possibly
                            # into the accumulator rail, where the SIGN is destroyed.
                            # --jug_multifire = n = floor(|E|/theta) fires: the CORRECT sigma-delta,
                            # and much closer to what asynchronous silicon actually does.
                            if a.jug_multifire:
                                s = np.sign(e) * (np.abs(e) // th)
                            else:
                                s = np.sign(e) * (np.abs(e) >= th)
                            if a.jug_sign_err > 0:
                                # COMPARATOR NOISE near the zero crossing: the fire goes the WRONG
                                # WAY. NOT covered by the mismatch robustness argument — a threshold
                                # offset only changes HOW OFTEN a synapse fires (a per-synapse lr,
                                # which SGD tolerates); a sign error CORRUPTS the gradient. The
                                # charge pump subtracts in the direction it fired, so a wrong-signed
                                # fire also drives E FURTHER from zero. That is the physical failure.
                                flip = hw.rng.random(s.shape) < a.jug_sign_err
                                s = np.where(flip, -s, s)
                            # THE WRITE. In the chip, Wm is in CODE units and a fire moves it by
                            # exactly ONE code (s = ±1) — all the analog cell can do.
                            # In --float_ref, Wm is a FLOAT WEIGHT, so a fire must move it by the
                            # float EQUIVALENT of one code (--jug_step, default 1/64). Firing ±1.0
                            # there would move the weight by its ENTIRE range in a single step.
                            Wm[L][c][p] = np.clip(Wm[L][c][p] + s * STEP, WMIN, WMAX)
                            # RESIDUE-PRESERVING SUBTRACTION — do NOT zero E. Zeroing would throw
                            # away exactly the sub-threshold charge the jug exists to keep.
                            E[L][c][p] = e - s * th
                            FIRES[L][c][p] += np.abs(s)


        Ate = forward(hw, Wm, Xc_te, e_fwd_codes(hw, E, THETA))
        Atr = forward(hw, Wm, Xc_tr, e_fwd_codes(hw, E, THETA))
        W_f, b_f = B.fit_clf(Atr[-1].astype(np.float64), y_tr, final=True)
        acc = float(np.mean(np.argmax(Ate[-1].astype(np.float64) @ W_f.T + b_f, 1) == y_te))
        best = max(best, acc)
        jug = ''
        if a.jug:
            _e = np.concatenate([np.abs(pg).ravel() for L in E for ch in L for pg in ch])
            esat = float(np.mean(_e >= hw.emax)) * 100.0
            jug = f"  Esat={esat:4.1f}%"
        if a.jug:
            # THE TWO NUMBERS THAT DECIDE THE JUG. A synapse that never fires is FROZEN — the
            # leak drains it faster than the deltas fill it, and it can never reach threshold.
            f = np.concatenate([pg.ravel() for L in FIRES for ch in L for pg in ch])
            nfolds = ep * (a.chunk // a.fold_every)
            jug += (f"  fire={f.mean()/nfolds*100:5.2f}%/fold  FROZEN={np.mean(f == 0)*100:5.1f}%")
        print(f"  ep{ep:2d}  test={acc*100:5.2f}%  (best {best*100:.2f}%){jug}"
              f"  [{time.time()-t0:.0f}s]", flush=True)

    tag = 'FLOAT CEILING' if a.float_ref else (
        f"JUG theta={a.jug_theta} leak={a.jug_leak or 'pure'} act={a.act_bits} "
        f"e_fwd={a.jug_e_fwd} agc={a.agc} code={a.delta_code}")
    print(f"\n  >>> {tag} : BEST = {best*100:.2f}%   (float spec on BIG = 82.50%)\n")


if __name__ == '__main__':
    main()
