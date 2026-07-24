#!/usr/bin/env python3
"""pcn_hw.py — BIT-ACCURATE HARDWARE MODEL of the spec. Can it actually be built?

The float sim (`pcn_bigspec.py`) reaches 82.50% on BIG. But it quietly assumes things the analog
chip may not give us. This model removes every one of those assumptions and re-measures.

EVERYTHING HERE IS AN INTEGER. Declared widths. Only primitives real silicon has.

WHAT THE FLOAT SIM ASSUMED, AND WHAT WE FORCE HERE
--------------------------------------------------
| float sim                          | this model (hardware)                                 |
|------------------------------------|-------------------------------------------------------|
| float activations between layers   | **8-bit ADC codes at EVERY layer** (never tested!)     |
| float weights, updated by +=lr      | **14-bit digital master** (8-bit analog cell + 6      |
|                                    |   SUB-LSB bits). NEEDED: lr=3e-4 is 0.019 LSB, so a   |
|                                    |   single sign-write CANNOT move the analog cell. It    |
|                                    |   changes only every ~52 folds. The analog cell is     |
|                                    |   NOT the accumulator — the boss must hold a finer W.  |
| float delta                        | **6-bit signed delta** on a FIXED range + a per-layer  |
|                                    |   POWER-OF-2 AGC (a shift, not a multiplier)          |
| float E accumulator                | **N-bit integer accumulator with saturation**          |
| leaky alpha = 0.1 (a multiply)     | **alpha = 1/8 (a SHIFT)** — no multiplier in the cell  |
| momentum mu = 0.9 (a multiply)     | **mu = 1 - 1/8 = 0.875 (a SHIFT)** — no multiplier     |
| float momentum velocity            | **N-bit integer velocity**                             |

The readout (softmax + the 26-value error) stays DIGITAL — it is a 256x26 block in the boss,
which is trivially buildable, and softmax is a LUT. (`--err_mode mse` also showed the boss can
avoid exp() entirely, so even that is optional.)

TARGET: reproduce the float spec's 82.50% on BIG. If it holds, the spec is buildable.

Usage:
    python3 pcn_hw.py                       # the proposed hardware spec
    python3 pcn_hw.py --float_ref           # float control (should ~= 82.5%)
    python3 pcn_hw.py --frac_bits 4         # sweep the sub-LSB master width
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'multi_array_level3_BIGspec'))
import pcn_bigspec as B                      # topology, weights, EMNIST L0 (float reference)

N_ROWS, N_COLS = B.N_ROWS, B.N_COLS
CODE_MIN, CODE_MID, CODE_MAX = B.CODE_MIN, B.CODE_MID, B.CODE_MAX   # 71 / 132 / 196
CODE_SCALE = 64                                                     # weight LSB = 1/64
N_CLASSES = B.N_CLASSES


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

    # ---- analog cell: the top 8 bits of the digital master ----
    def cell(self, Wm):
        """The 8-bit analog weight the crossbar actually holds. Signed code, CODE_MID=132.

        The cell IS the master, truncated: cell = round(master / 2^frac). It is not a second,
        separate matrix — it is the master's TOP BITS. The master drifts smoothly at `hw.lr`
        master-LSBs per write; every ~51 writes it CARRIES across a cell-LSB boundary and the
        analog code steps by one. So the analog weight does learn — as a STAIRCASE.

        --dither: STOCHASTIC (dithered) carry instead of round-to-nearest. code = floor(x) +
        Bernoulli(frac(x)) — unbiased, and the classic hardware answer to "my update is smaller
        than my LSB". Tests whether the DETERMINISTIC threshold is what costs us, and whether
        stochastic rounding lets us shrink the digital master (sweep --frac_bits with it on)."""
        x = Wm / (1 << self.frac)
        if self.dither:
            fl = np.floor(x)
            x = fl + (self.rng.random(x.shape) < (x - fl))
        else:
            x = np.rint(x)
        code = np.clip(x + CODE_MID, CODE_MIN, CODE_MAX)
        return (code - CODE_MID).astype(np.int32)

    def master(self, Wm):
        """TEST PROBE, NOT BUILDABLE. The FINE digital master expressed in cell-code units, with
        NO quantisation to the 8-bit analog grid. The analog array cannot hold this. Used by
        --w_mode to ask which of the forward / backward paths the cell quantiser actually hurts."""
        return np.clip(Wm / float(1 << self.frac),
                       CODE_MIN - CODE_MID, CODE_MAX - CODE_MID)

    def wgt(self, Wm, path):
        """The weights the given datapath ('fwd' | 'bwd') sees, under the current --w_mode."""
        if self.float_ref:
            return Wm
        if self.w_mode == 'all_master':
            return self.master(Wm)
        if self.w_mode == 'bwd_master' and path == 'bwd':
            return self.master(Wm)
        return self.cell(Wm)

    def shift(self, acc, s):
        """The per-layer ADC gain (a power-of-2 shift). Integer accumulators use a true `>>`;
        the float-weight TEST probes keep the datapath integer from the ADC onward."""
        if np.issubdtype(np.asarray(acc).dtype, np.integer):
            return acc >> s
        return np.rint(acc / float(1 << s)).astype(np.int64)

    # ---- ADC: every layer's activation is an 8-bit code ----
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
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--act_bits', type=int, default=8, help='ADC width at EVERY layer')
    p.add_argument('--delta_bits', type=int, default=6, help='cross-chip delta broadcast')
    p.add_argument('--frac_bits', type=int, default=10,
                   help='SUB-LSB bits of the digital master W. 10 is the SPEC (an 18-bit master): '
                        'it is the width at which one write lands 20 master-LSB up from the '
                        'bottom, leaving headroom BELOW the update for momentum to accumulate. '
                        'At 6 the write is exactly 1 master-LSB and momentum degenerates.')
    p.add_argument('--e_acc_bits', type=int, default=20, help='E accumulator width (saturating)')
    p.add_argument('--v_bits', type=int, default=12, help='momentum velocity width')
    p.add_argument('--mom_shift', type=int, default=3, help='mu = 1 - 2^-s  (3 => 0.875)')
    p.add_argument('--leaky_shift', type=int, default=3, help='alpha = 2^-s  (3 => 0.125)')
    p.add_argument('--float_ref', action='store_true', help='float control (should ~= 82.5%)')
    p.add_argument('--seed', type=int, default=42)
    # ── PROBES for the OPEN ~6pp gap (hardware ~75% vs float ~82%) ────────────────────────
    p.add_argument('--w_mode', default='cell', choices=['cell', 'bwd_master', 'all_master'],
                   help="which W the datapaths see. 'cell' = the real chip: BOTH the forward and "
                        "the router-local W.T use the 8-bit analog code, so the backprojection is "
                        "the true transpose of the forward operator. 'bwd_master' = forward on the "
                        "cell, W.T on the FINE digital master — isolates the cost of the QUANTISED "
                        "TRANSPOSE. 'all_master' = both on the master — removes ALL analog-cell "
                        "quantisation and so measures the TOTAL cost of the 8-bit cell. The two "
                        "master modes are NOT BUILDABLE; they are attribution probes only.")
    p.add_argument('--dither', action='store_true',
                   help='STOCHASTIC (dithered) carry into the analog cell instead of '
                        'round-to-nearest. Unbiased. Sweep against --frac_bits: if dither at a '
                        'small frac matches rint at frac=10, stochastic rounding buys back '
                        'digital-master bits.')
    p.add_argument('--agc', default='peak', choices=['peak', 'rms'],
                   help='delta-channel AGC reference. The code and HW_BUILD_CHECK.md disagreed '
                        'about which of these won; this flag settles it.')
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
    p.add_argument('--jug', action='store_true',
                   help='LEAKY-JUG fold: E leaks instead of resetting (the leak IS momentum), and '
                        'a threshold crossing fires a whole +-1 cell-LSB. Forces frac_bits=0 — NO '
                        'digital master, NO velocity register, NO per-synapse digital store.')
    p.add_argument('--jug_leak', type=int, default=6,
                   help='leak shift S: lambda = 1 - 2^-S per fold (6 => 0.984, a ~64-fold memory). '
                        'Small S = strong leak = short memory = MORE synapses freeze.')
    p.add_argument('--jug_pure', action='store_true',
                   help='CONTROL: no leak at all (a pure sigma-delta integrator). Separates "the '
                        'charge accumulator works" from "the leak (momentum) helps".')
    # ── PHYSICAL (mismatched) DEVICES — what silicon actually hands you ───────────────────
    # --jug_leak above is a DESIGNED leak (a switched-capacitor: lambda = C_small/C_big, a
    # capacitor RATIO, which is the best-matched thing in CMOS, ~0.1%). These flags model the
    # OTHER case: a PARASITIC leak through a real device. Subthreshold leakage goes
    # EXPONENTIALLY with Vt, and Vt mismatch is Gaussian ⇒ the leak CURRENT is LOG-NORMAL and
    # can vary many-fold across a die and with temperature. tau = C/I_leak inherits that spread.
    #   The question this answers: does the jug need a CONTROL LOOP, or merely a one-sided BOUND
    #   ("tau must exceed the mean inter-fire interval")? At theta=8 the jug fires ~0.91%/fold,
    #   so the mean inter-fire interval is ~110 folds. Sweep tau around that.
    p.add_argument('--jug_tau', type=float, default=0.0,
                   help='PARASITIC leak: median leakage time constant, IN FOLDS. 0 = none (a pure '
                        'integrator — the best config). Compare against the mean inter-fire '
                        'interval (~110 folds at theta=8): if tau << that, the jug drains before '
                        'it can fire.')
    p.add_argument('--jug_tau_spread', type=float, default=1.0,
                   help='per-synapse LOG-NORMAL spread of tau (geometric sigma). 1.0 = perfectly '
                        'matched; 3.0 = a 3x spread, which is realistic for subthreshold leakage.')
    p.add_argument('--jug_theta_spread', type=float, default=1.0,
                   help='per-synapse LOG-NORMAL spread of the FIRE THRESHOLD (comparator offset + '
                        'capacitor mismatch). Equivalent to a random per-synapse learning rate — '
                        'which SGD is normally very tolerant of. 1.0 = perfectly matched.')
    # ── ⚠⚠ THE CHIP-SPECIFIC ONE: E IS LIVE IN THE FORWARD PATH ──────────────────────────
    # The sim's forward uses W ALONE. It could, because in FOLD mode E is a gradient accumulator
    # whose forward contribution is negligible (|E|/|W| ~ 0.004, measured — pcn_bigspec.py:269)
    # and is folded away every 128 samples.
    # ⚠ THE CHIP CANNOT DO THAT. The mac_cell_emx is DUAL-TAIL: I_ntail = I_tail_W + I_tail_E.
    #   Ce is physically wired into the tail, so the MAC computes W + E. It is not optional.
    # ⚠ AND THE JUG MAKES E BIGGER: it accumulates for ~110 folds instead of 1, so at firing
    #   |E| = theta ~ 8x the one-fold rms => |E|/|W| ~ 0.03, roughly 8x the fold rule's.
    # Design point: theta ~ 0.4 weight codes, against an E-cap clamp of +-6.7 codes (E_MAX_V =
    # 55 mV) => ~17x headroom, so the cap will not saturate and the jug CAN fire. But a ~3% live
    # perturbation of the effective forward weight is UNTESTED. It may even HELP (a live E in the
    # MAC is the original predictive-coding intent) — but it must be MEASURED.
    p.add_argument('--jug_e_fwd', type=float, default=0.0,
                   help='WEIGHT CODES that a full-threshold E contributes to the FORWARD MAC. '
                        '0 = E invisible to the forward (what the sim has always done — NOT what '
                        'the chip does). 0.4 = the design point. 6.7 = the E-cap clamp. This is '
                        'the single most chip-specific unvalidated assumption in the design.')
    p.add_argument('--jug_sign_err', type=float, default=0.0,
                   help='probability that a fire has the WRONG SIGN (comparator noise near the '
                        'zero crossing). ⚠ NOT covered by the device-mismatch robustness argument: '
                        'threshold offset only changes HOW OFTEN a synapse fires (a per-synapse '
                        'learning rate, which SGD tolerates); a sign error CORRUPTS the gradient. '
                        'Expected to be the one physical effect that actually hurts.')
    p.add_argument('--jug_theta', type=float, default=32.0,
                   help='fire threshold, as a multiple of the calibrated per-fold rms of |E| '
                        '(per layer, set once at calibration — CONFIG, like the ADC gain). This is '
                        'now the learning rate: effective lr = firing_rate x 1 cell-LSB.')
    p.add_argument('--delta_stats', action='store_true',
                   help='DIAGNOSTIC: report what the delta channel is actually doing — the '
                        'fraction of deltas quantised to ZERO, the fraction CLIPPED at the rail, '
                        'and the crest factor. Then exit.')
    a = p.parse_args()
    if a.jug:
        # THE POINT OF THE JUG: no sub-LSB storage. The weight state IS the analog cell.
        a.frac_bits = 0
    hw = HW(a)

    F_tr, F_te, y_tr, y_te = B.get_emnist_l0_features()
    Xc_tr = B.float_to_act_code(F_tr).astype(np.int64)
    Xc_te = B.float_to_act_code(F_te).astype(np.int64)
    _d = np.concatenate([np.where(y_te == c)[0][:40] for c in range(N_CLASSES)])

    # weights -> integer MASTER (float weight * 64 * 2^frac)
    w1, w2 = B.load_l1_weights(), B.load_l2_weights()
    w3 = B.init_l3_weights_random_ortho(seed=2)
    if a.float_ref:
        Wm = [w1, w2, w3]
    else:
        Wm = [[[np.rint(pg.astype(np.float64) * hw.wscale).astype(np.int64) for pg in ch]
               for ch in L] for L in (w1, w2, w3)]
    WMIN = int(np.rint(B.W_FLOAT_MIN * hw.wscale)); WMAX = int(np.rint(B.W_FLOAT_MAX * hw.wscale))

    E = [[[np.zeros_like(pg, dtype=np.int64) for pg in ch] for ch in L] for L in Wm]
    V = [[[np.zeros_like(pg, dtype=np.int64) for pg in ch] for ch in L] for L in Wm]

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
                            np.clip(e, -hw.emax, hw.emax, out=e)
                            # FIRE: a comparator. The cell moves ONE WHOLE LSB — all it can do.
                            th = THETA[L] if THS is None else np.maximum(
                                1, np.rint(THETA[L] * THS[L][c][p])).astype(np.int64)
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
                            Wm[L][c][p] = np.clip(Wm[L][c][p] + s, WMIN, WMAX)
                            # RESIDUE-PRESERVING SUBTRACTION — do NOT zero E. Zeroing would throw
                            # away exactly the sub-threshold charge the jug exists to keep.
                            E[L][c][p] = e - s * th
                            FIRES[L][c][p] += np.abs(s)
                continue

            # ── FOLD (the current spec): saturate E, take its SIGN, momentum, write ──
            for L in range(3):
                for c in range(len(Wm[L])):
                    for p in range(len(Wm[L][c])):
                        e = E[L][c][p]
                        if not a.float_ref:
                            np.clip(e, -hw.emax, hw.emax, out=e)          # saturating accumulator
                        step = (a.lr if a.float_ref else hw.lr) * np.sign(e)   # 1-bit write x lr
                        if a.float_ref:
                            V[L][c][p] = 0.9 * V[L][c][p] + step
                            Wm[L][c][p] = np.clip(Wm[L][c][p] + V[L][c][p],
                                                  B.W_FLOAT_MIN, B.W_FLOAT_MAX)
                        else:
                            v = V[L][c][p]
                            # SYMMETRIC decay (sign-magnitude). NOT `v -= v >> s` — an arithmetic
                            # shift is asymmetric about zero (v=+1 never decays, v=-1 dies at
                            # once), which biases every weight POSITIVE. Caught the hard way.
                            v = np.sign(v) * ((np.abs(v) * ((1 << hw.mom_sh) - 1)) >> hw.mom_sh)
                            v = v + step.astype(np.int64)
                            np.clip(v, -hw.vmax, hw.vmax, out=v)
                            V[L][c][p] = v
                            Wm[L][c][p] = np.clip(Wm[L][c][p] + v, WMIN, WMAX)
                        E[L][c][p][...] = 0

        Ate = forward(hw, Wm, Xc_te, e_fwd_codes(hw, E, THETA))
        Atr = forward(hw, Wm, Xc_tr, e_fwd_codes(hw, E, THETA))
        W_f, b_f = B.fit_clf(Atr[-1].astype(np.float64), y_tr, final=True)
        acc = float(np.mean(np.argmax(Ate[-1].astype(np.float64) @ W_f.T + b_f, 1) == y_te))
        best = max(best, acc)
        jug = ''
        if a.jug:
            # THE TWO NUMBERS THAT DECIDE THE JUG. A synapse that never fires is FROZEN — the
            # leak drains it faster than the deltas fill it, and it can never reach threshold.
            f = np.concatenate([pg.ravel() for L in FIRES for ch in L for pg in ch])
            nfolds = ep * (a.chunk // a.fold_every)
            jug = (f"  fire={f.mean()/nfolds*100:5.2f}%/fold  FROZEN={np.mean(f == 0)*100:5.1f}%")
        print(f"  ep{ep:2d}  test={acc*100:5.2f}%  (best {best*100:.2f}%){jug}"
              f"  [{time.time()-t0:.0f}s]", flush=True)

    tag = 'FLOAT REF' if a.float_ref else (
        f"HW act={a.act_bits} frac={a.frac_bits} w_mode={a.w_mode} dither={int(a.dither)} "
        f"agc={a.agc} code={a.delta_code}")
    print(f"\n  >>> {tag} : BEST = {best*100:.2f}%   (float spec on BIG = 82.50%)\n")


if __name__ == '__main__':
    main()
