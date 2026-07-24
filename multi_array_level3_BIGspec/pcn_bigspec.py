#!/usr/bin/env python3
"""
pcn_boss_l3_normed_fable.py  —  Option D topology + pooled improvement ideas,
each independently switchable.

Based on pcn_boss_sim_l3_normed_optd.py (8 L3 chips, all L2 pairs, 128-dim).
Design rationale: fable_suggestions_against_norm.md (2026-07-07).

The five switchable ideas
─────────────────────────
1. STAGED GENERATIONAL TRAINING          --staged "1x4,2x4,3x2,1x1"
   Instead of round-robin phases, run stages to completion and freeze:
   Stage "1x4" = 4 epochs training E3 only (E1/E2 frozen), absorbing E3→W3
   each epoch.  When the stage ends the layer is frozen *permanently* (its E
   is never updated again, its W never absorbs again) unless a later stage
   re-opens it.  "2x4" then trains E2 against the now-stable discriminative
   W3.T — recreating the proven 2-layer recipe (single W.T hop from a stable
   trained layer) at each depth.  Phase 5 = all layers simultaneously
   (the optd --combined behaviour).

2. LAYER-LOCAL CLASSIFIER DELTAS         --local_delta --beta 0.5
   Give E2 (and E1) the same CHL-like mechanism E3 has: an auxiliary linear
   classifier fit on that layer's own features supplies a direct correction
   delta_loc = λ_h·Δ·sign(W_aux[y] − W_aux[ŷ_aux]) in the layer's own output
   space.  Blended with the backprojected signal:
       delta_used = β·delta_loc + (1−β)·delta_projected
   β=1 is pure local (diagnostic), β=0 is pure backprojection (off).
   Hardware: each chip stores its 26×16 slice of W_aux; the boss broadcasts
   only (y, ŷ_aux, h) — same class of broadcast as boss_h.

3. NLMS NORMALISATION                    --nlms [--nlms_gain 8.0]
   The _norm unit-RMS f̂ normalises the *learning* but not the forward
   *effect*: ΔE@f = η·δ·N·rms(f), so E1's effective gain is ~100× E3's.
   NLMS divides by ‖f‖² instead:  ΔE = η·gain·outer(δ, f)/‖f‖², giving
   ΔE@f = η·gain·δ exactly, at every layer.  nlms_gain≈8 matches E3's
   empirically-validated effective step under unit-RMS (N·rms(f_l2) ≈ 16·0.5).

4. MID-EPOCH CLASSIFIER REFRESH          --clf_refresh 8000 [--clf_refresh_n 20000]
   Refit W_f (and the aux classifiers) every clf_refresh samples from a
   random subsample, so the boss direction reflects the current W+E state
   rather than the epoch-start state (l3_post_topology_observations.md §2).

5. AVERAGED BACKPROJECTION (Option D)    --avg_bp
   Each L2 chip appears in 4 L3 chips; RMS-preserved contributions are
   currently summed → ~4× oversized deltas → clip-and-cancel.  --avg_bp
   divides by the contributing-chip count (observations §4).

(+) BACKWARD-SHADOW EMA                  --bp_ema 0.2
   Backprojection uses a slow EMA copy of W3/W2 (target-network style)
   instead of the live absorbed weights.  Alternative to staging — normally
   off when --staged is used (with W3 frozen it is a no-op anyway).

Removed relative to optd: --gate_wt, --skip_w2t, --relu_jac, --phases,
--phase_seq, --reverse_phases (all tested and ruled out / superseded).

Default mode (no --staged) reproduces optd --combined: warmup + N combined
(P5) epochs, so the 42.22% baseline config remains runnable from this file:
  python pcn_boss_l3_normed_fable.py --warmup 3x4k --epochs 6 --chunk 32000 \
      --boss_lr 0.003 --diag_l2

All-ideas-on run:
  python pcn_boss_l3_normed_fable.py --staged "1x4,2x4,3x2,1x1" --chunk 32000 \
      --nlms --nlms_gain 8 --local_delta --beta 0.5 \
      --clf_refresh 8000 --avg_bp --diag_l2
"""

import argparse
import os
import sys
import time
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
DIR      = os.path.dirname(os.path.abspath(__file__))
PCN_ROOT = os.path.dirname(DIR)
SHARED   = os.path.join(PCN_ROOT, 'shared', 'sim')
MA_SIM   = os.path.join(PCN_ROOT, 'multi_array_sim')

for _p in [SHARED, MA_SIM]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

L2_WEIGHTS_DIR   = os.path.join(MA_SIM, 'weightsl2')
L0_CACHE_MNIST   = os.path.join(MA_SIM, 'l0_cache')
L0_CACHE_EMNIST  = os.path.join(DIR, 'l0_cache_emnist')
# Reuse the BIG split-sign L0 + per-chip PCA weights already built by bigv2 (287MB).
BIG_WEIGHTS_DIR  = os.path.join(PCN_ROOT, 'multi_array_level3', 'weights_big_emnist')
RESULTS_DIR      = os.path.join(DIR, 'results')
os.makedirs(L0_CACHE_EMNIST, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

from pcn_mnist import load_emnist_letters, preprocess

# ── Architecture constants (Option D) ──────────────────────────────────────────
N_L1_CHIPS     = 24       # BIG: split-sign L0 -> 1152 feats / 48 per chip
N_L2_CHIPS     = 8        # BIG
N_L3_CHIPS     = 16       # BIG: 8 adjacent + 8 skip-2 pairs over the 8 L2 chips
N_L0_COMPS     = 576      # EMNIST-native PCA components -> 1152 split-sign feats
N_L1_PER_L2    = N_L1_CHIPS // N_L2_CHIPS   # 3
N_L2_PER_L3    = 2        # each L3 chip sees 2 L2 chips
N_ROWS         = 16
N_COLS         = 16
N_PAGES        = 3        # L1 and L2: 3 pages
N_PAGES_L3     = 2        # L3: 2 pages
N_FEATS_PER_L1 = N_PAGES   * N_COLS    # 48
N_FEATS_PER_L3 = N_PAGES_L3 * N_ROWS   # 32
N_L1_FEATS     = N_L1_CHIPS * N_ROWS   # 384
N_L2_FEATS     = N_L2_CHIPS * N_ROWS   # 128
N_L3_FEATS     = N_L3_CHIPS * N_ROWS   # 256
N_CLASSES      = 26

# Option D routing: all 6 unique L2 pairs + 2 balanced repeats.
# BIG: every adjacent pair + every skip-2 pair of the 8 L2 chips. Each L2 chip appears in
# exactly 4 L3 chips. OVERLAPPING, not block-diagonal — the L3->L2 backprojection ACCUMULATES
# over the 4 contributing L3 chips (the `+=` in both the per-sample and batched chains already
# does the right thing; --avg_bp averages instead of sums).
L3_ROUTING = ([[k, (k + 1) % N_L2_CHIPS] for k in range(N_L2_CHIPS)] +
              [[k, (k + 2) % N_L2_CHIPS] for k in range(N_L2_CHIPS)])

# Contribution count per L2 chip in the L3→L2 backprojection (for --avg_bp).
L2_BP_COUNT = np.zeros(N_L2_CHIPS, dtype=np.float64)
for _pair in L3_ROUTING:
    for _c in _pair:
        L2_BP_COUNT[_c] += 1.0

# ── L0_A: trainable input transform (optional, --l0a) — see L0A_DESIGN.md ──────
# Inserted between frozen-PCA L0 (576 feats) and L1. Same-size: 576 → 576, so the
# full input is made discriminative before L1 compresses it. Structured like L1 but
# with overlapping input windows so output stays 576-wide.
N_L0_FEATS  = N_L1_CHIPS * N_FEATS_PER_L1     # 1152 (what L1 consumes)
N_L0A_CHIPS = N_L0_FEATS // N_ROWS            # 36 chips (36*16 = 576 out)
# chip j -> output block j (feats [16j:16j+16)); reads L0 window [16j:16j+48) mod 576
# = the 3 L0 feature-blocks [j, j+1, j+2] (mod 36). Each L0 block read by 3 chips.
L0A_ROUTING = [[(j + p) % N_L0A_CHIPS for p in range(N_PAGES)]
               for j in range(N_L0A_CHIPS)]
# L1 reads f_l0a in contiguous non-overlapping 48-windows (12 chips tile 576), so each
# L0_A OUTPUT block b is fed back by exactly one L1 chip (c=b//3, page b%3) -> count 1.
# (Kept explicit for symmetry with the other BP_COUNT tables / future avg_bp use.)
L0A_BP_COUNT = np.ones(N_L0A_CHIPS, dtype=np.float64)

# ── Hardware constants (match Run 6 sim exactly) ───────────────────────────────
ADC_V_MAX  = 1.8
CODE_MIN   = 71
CODE_MID   = 132
CODE_MAX   = 192
CODE_SCALE = 64.0
HW_SPAN    = CODE_MAX - CODE_MIN
E_MAX_V    = 0.055
E_CLIP_HW  = E_MAX_V * HW_SPAN / CODE_SCALE   # ≈ 0.1041
W_FLOAT_MIN = -(CODE_MID - CODE_MIN) / CODE_SCALE  # ≈ -0.8906
W_FLOAT_MAX =  (CODE_MAX - CODE_MID) / CODE_SCALE  # 1.0

# ── boss_h parameters ─────────────────────────────────────────────────────────
BOSS_LR    = 0.01
BOSS_H_MAX = 7
DELTA_BITS   = 0      # --delta_bits N: quantise the TRANSPORTED delta to N bits (0=off).
                     # THE silicon question: the delta is what the boss broadcasts across
                     # chips, and cross-chip DIGITAL comms is the expensive thing (local
                     # analog real-values are nearly free). How few bits can it be?
DELTA_SIGN   = False # --delta_sign: transport sign(d) only. The WRITE is already 1-bit
                     # (--sign_at_fold); can the TRANSPORT be too?
DELTA_RENORM = False # --delta_renorm: RMS-renorm the delta after the sigma' gate (the old
                     # rule's _gate_renorm). Does it cost or help?
BH_GATE      = False # --bh_gate: apply the old rule's 3-bit boss_h severity to the CE error.
BH_LEAK      = 0.0   # --bh_leak L: when bh quantises to 0, apply a SMALL step L*0.25 instead
                     # of SKIPPING the sample. L=0 = the current hard skip.
                     # WHY THIS MATTERS: bh=0 means a LOW MARGIN, i.e. the sample sits right
                     # ON THE DECISION BOUNDARY -- the most informative region there is. The
                     # hard skip throws away the hardest examples and keeps the easy,
                     # decisively-wrong ones. Measured live damage: logistic (a BETTER
                     # classifier) is more confident -> margins shrink -> 40.2%% of wrong
                     # samples hit bh=0 and were SKIPPED (vs 15.7%%), costing ~10pp until
                     # --bh_gain restored the operating point. The gate is a liability.
E_BITS       = 0      # --e_bits N: quantise the E accumulator to N bits (0=off).
DELTA_AGC    = 0.0   # --delta_agc TAU: SLOW per-layer automatic gain control on the delta.
                     # WHY: the delta ATTENUATES 0.538x PER LAYER through the W.T chain
                     # (measured: rms 6.66 -> 3.66 -> 1.92 at L3/L2/L1) = ~0.9 BITS LOST PER
                     # LAYER. Our 3-layer chip only loses ~2.7 bits, which a 6-bit channel
                     # absorbs -- which is WHY the ladder found --delta_renorm to be pure cost
                     # here. At depth 6 a FIXED-RANGE 6-bit quantiser has NOTHING left (5.4
                     # bits gone), and the crest factor grows with depth too (3.2 -> 5.7), so
                     # peaks eat further headroom. Gain control is LOAD-BEARING AT DEPTH.
                     # BUT --delta_renorm does gain control the WRONG WAY: d/rms(d) per SAMPLE
                     # also flattens ACROSS SAMPLES, so a confidently-wrong and a marginally-
                     # wrong sample produce equal-magnitude updates. That is the 8.6pp it costs.
                     # AGC separates the two jobs: track the running rms with a SLOW EMA (TAU)
                     # and apply a per-layer gain. Slow => per-sample magnitude survives; the
                     # per-layer scale is still held constant. Cheap in analog (a slow feedback
                     # loop). TAU ~ 0.01. 0 = off.
DELTA_FIXED  = 0.0   # --delta_fixed R: quantise --delta_bits against a FIXED range R instead of
                     # the per-vector max|d|. The per-vector max is an IDEAL instantaneous AGC and
                     # is what the ladder implicitly assumed -- real hardware has a FIXED DAC
                     # range. Use with --delta_agc to keep the signal inside it. This is the
                     # honest hardware model, and the one that actually tests depth-robustness.
_AGC_STATE   = {}    # layer -> running rms (EMA)
AGC_TARGET   = 4.0   # nominal delta rms each layer is driven to by the AGC
BH_GAIN    = 1.0    # --bh_gain: scales the 7.0 in bh = int(7*GAIN*margin/denom).
                    # The boss_h gate's operating point is calibrated to the SCORE
                    # DISTRIBUTION of the teaching classifier. lstsq (score rms 0.05)
                    # gates 15.7% of wrong samples and uses 63.6%, mean bh 2.57.
                    # logistic (rms 2.12) is MORE CONFIDENT -> margins shrink relative to
                    # the span -> the int() floor sends 40.2% to bh=0 (SKIPPED) and only
                    # 42.4% are used, mean bh 1.80 => ~2.1x less learning per epoch.
                    # bh_gain restores the operating point when swapping classifiers.
AUTO_FRAC  = 5e-5   # auto_lr: per-settle-step output move as fraction of layer rms
LR_MULT    = (1.0, 1.0, 1.0)   # per-layer (L3,L2,L1) auto_lr multiplier (--lr_mult)
ERR_MODE = 'softmax'  # --err_mode: how the boss turns readout scores into the error s.
                    #  'softmax' : s = onehot - softmax(scores).  CORRECT ONLY FOR LOGITS.
                    #  'mse'     : s = onehot - scores.  THE RIGHT ONE FOR lstsq, and free.
                    # WHY: the fast fit_clf regresses activations onto ONE-HOT targets, so
                    # its solution is the linear MMSE estimator of the one-hot vector --
                    # i.e. its outputs already approximate P(class|a). They are PROBABILITIES,
                    # not logits. Softmaxing them squashes an already-squashed quantity into
                    # near-uniform mush (rms 0.05 -> p ~ 1/26). The residual onehot - p_hat is
                    # exactly what the least-squares fit is already minimising (Widrow-Hoff).
                    # HARDWARE: 'mse' needs NO exp() -- the boss error is a plain subtraction
                    # (target - readout), i.e. a difference amplifier. softmax on-chip would
                    # have been painful to justify; this removes the need entirely.
CLF_LOGISTIC = True   # DEFAULT since 2026-07-13 (--clf_lstsq opts out): use LOGISTIC REGRESSION (real logits) for the
                    # per-epoch TEACHING classifier, not just the final report. The fast
                    # lstsq path regresses onto ONE-HOT targets, so its scores have rms
                    # ~0.05 -- softmax over them is near-uniform and the CE gradient
                    # degenerates to a constant class template. Logistic gives rms ~2.1
                    # (proper logits) AND is a 5pp better classifier (24.2% -> 29.2%),
                    # which is exactly the ~9pp 'fast_final' eval artifact at its source.
                    # Costs 3.7s vs 0.2s per fit -- negligible against 60s+ epochs.
LOGIT_SCALE = 1.0   # --logit_scale: multiply classifier scores before the softmax.
                    # fit_clf's training-time path is LEAST-SQUARES, so its scores are
                    # regression outputs with rms ~0.05 -- NOT logits. softmax(0.05-rms)
                    # is essentially UNIFORM (p ~ 1/26 for every class), so the error
                    # s = onehot - softmax collapses to a near-CONSTANT class template
                    # that barely depends on what the net predicted. That is a degenerate
                    # teaching signal: a bigger batch just estimates the same constant more
                    # precisely (= the 'bias not noise' signature the cosine probe found).
                    # Rescaling to rms ~2-4 restores a real cross-entropy gradient.
SEED_INIT  = 7      # --seed_init: --rand_init weight draw  (seed axis of the robustness map)
SEED_DATA  = 42     # --seed_data: training sample order     (   "        "        "        )
DELTA      = 0.3
LEAKY_ALPHA = 0.1

# boss_h improvement switches (pcn_l3_mathematics_norm §9 q5; set from CLI)
BH_DEADZONE  = 0.0    # zero δ components with |W_f[y]−W_f[ŷ]| < T×rms(diff)
BH_MAGLEVELS = False  # quantise |diff| to {0, ±0.5, ±1} instead of pure sign
BH_STD       = False  # normalise h by 4·std(scores) instead of score span

# Leaky-ReLU α re-application under the new scaling (set from CLI; exclusive)
RELU_JAC_DIR = False  # δ ×= σ'(f) then RMS-renormalise: purely DIRECTIONAL
                      # gate (old --relu_jac failed because it attenuated
                      # globally without renorm, compounding with /N_ROWS)
RELU_JAC_INV = False  # outer-product δ ×= min(1/σ', 3): delivery compensation
                      # (old ×1/α failed on e_clip saturation; under NLMS the
                      # weight steps are far below clip). Applied only to what
                      # is WRITTEN to E, not the backprojected chain.


# ── MAC functions ──────────────────────────────────────────────────────────────

def float_to_act_code(x):
    return np.clip(np.round(np.clip(x, 0.0, ADC_V_MAX) / ADC_V_MAX * 255),
                   0, 255).astype(np.uint8)


# ── PERF: quantised-weight cache (--frozen_fwd) ───────────────────────────────
# Profiling showed 66% of the training loop was w_float_to_signed + clip + round + astype:
# we re-quantised EVERY weight on EVERY sample (312k calls / 3k samples on smallBIG; ~128
# quantisations per sample on BIG, which is most of why BIG is 30x slower).
#
# But in FOLD mode the weights only change at the FOLD (every --fold_every samples). The only
# reason we could not cache was that the forward used W+E and E changes each sample. E in fold
# mode is a GRADIENT ACCUMULATOR, not a residual -- and we MEASURED that its effect on the
# forward is negligible: |E|/|W| ~ 0.004, and the cosine probe found the gradient WITH drift
# (0.980) indistinguishable from WITHOUT (0.973). torch holds weights fixed across a batch too.
#
# So: drop E from the forward during accumulation, quantise W once per fold, reuse it for the
# whole batch. ~128x less quantisation work.
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PERFORMANCE FLAGS — read PERFORMANCE.md before using any of these.           ║
# ║                                                                              ║
# ║  ✅ --fast_batch  : 9.1x faster AND ACCURACY-NEUTRAL. This is the fast path.  ║
# ║                     8-epoch A/B (spec, chunk 32k):                            ║
# ║                       accurate 71.47% / 437s   fast_batch 72.24% / 48s        ║
# ║                     Vectorises the inner loop over the fold batch. Safe.      ║
# ║  ⚠ --clf_sub N   : fits the readout on N rows, not all 124,800. The logistic ║
# ║                     fit is a FIXED ~11s tax (BIG: ~113s) that does NOT shrink ║
# ║                     with --chunk, and runs TWICE per epoch.                    ║
# ║                     ⚠ DOES NOT TRANSFER — SCALE N WITH THE READOUT WIDTH:      ║
# ║                        smallBIG readout  32->26  (  858 params): 20000 = FREE  ║
# ║                        BIG      readout 256->26  (6,682 params): 20000 = -5pp! ║
# ║                     ⇒ smallBIG: --fast_batch --clf_sub 20000                   ║
# ║                     ⇒ BIG:      --fast_batch  ALONE  (no --clf_sub)            ║
# ║                     On BIG the logistic fit is now THE bottleneck (~226s of a  ║
# ║                     266s epoch). The right fix there is to fit it LESS OFTEN   ║
# ║                     or warm-start it — NOT to subsample it.                    ║
# ║                                                                              ║
# ║  ❌ --frozen_fwd  : DEPRECATED / BUGGY. Scores ~3pp LOW (55.38 vs 58.46 on a  ║
# ║                     3-epoch A/B). It was only a stepping stone to --fast_batch,║
# ║                     which uses the SAME frozen-forward semantics and shows NO  ║
# ║                     accuracy cost — so the deficit is a defect in THIS path,   ║
# ║                     not a cost of freezing the forward. DO NOT USE ALONE.      ║
# ║                     (--fast_batch sets FROZEN_FWD itself; that is fine.)       ║
# ║                                                                              ║
# ║  METHOD RULE (learned twice, the hard way): EVERY performance change gets an  ║
# ║  accuracy A/B against the un-optimised path before it is trusted. Two silent  ║
# ║  corruptions were caught this way — a loop-counter/sample-index mixup (54.4 → ║
# ║  28.9%) and a fold-order swap that disabled --e_bits (+3pp of fake gain).      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
FROZEN_FWD = False
_WQ1 = _WQ2 = _WQ3 = None


def refresh_wq(l1_weights, l2_weights, l3_weights):
    """Re-quantise the weights. Called once per FOLD, not once per sample."""
    global _WQ1, _WQ2, _WQ3
    _WQ1 = [[w_float_to_signed(p) for p in ch] for ch in l1_weights]
    _WQ2 = [[w_float_to_signed(p) for p in ch] for ch in l2_weights]
    _WQ3 = [[w_float_to_signed(p) for p in ch] for ch in l3_weights]


def w_float_to_signed(W_float):
    codes = np.clip(np.round(W_float * CODE_SCALE + CODE_MID), CODE_MIN, CODE_MAX)
    return codes.astype(np.int64) - CODE_MID


def chip_mac_leaky(W0s, W1s, W2s, F, alpha):
    """3-page leaky ReLU MAC: (N,48) → (N,16) float32."""
    y_raw = (F[:,  0:16].astype(np.float64) @ W0s.T +
             F[:, 16:32].astype(np.float64) @ W1s.T +
             F[:, 32:48].astype(np.float64) @ W2s.T)
    y = y_raw / 64.0
    return np.where(y >= 0, y, alpha * y).astype(np.float32)


def chip_mac_leaky_one(W0s, W1s, W2s, f, alpha):
    """3-page single-sample leaky ReLU MAC: (48,) → (16,) float32."""
    y_raw = (W0s @ f[:16].astype(np.float64) +
             W1s @ f[16:32].astype(np.float64) +
             W2s @ f[32:].astype(np.float64))
    y = y_raw / 64.0
    return np.where(y >= 0, y, alpha * y).astype(np.float32)


def chip_mac_l3(W0s, W1s, F, alpha):
    """2-page leaky ReLU MAC for L3: (N,32) → (N,16) float32."""
    y_raw = (F[:,  0:16].astype(np.float64) @ W0s.T +
             F[:, 16:32].astype(np.float64) @ W1s.T)
    y = y_raw / 64.0
    return np.where(y >= 0, y, alpha * y).astype(np.float32)


def chip_mac_l3_one(W0s, W1s, f, alpha):
    """2-page single-sample leaky ReLU MAC for L3: (32,) → (16,) float32."""
    y_raw = (W0s @ f[:16].astype(np.float64) +
             W1s @ f[16:32].astype(np.float64))
    y = y_raw / 64.0
    return np.where(y >= 0, y, alpha * y).astype(np.float32)


# ── Forward passes ─────────────────────────────────────────────────────────────

# ── L0_A trainable input transform (globals; set by run_sim when --l0a) ─────────
# When _L0A_ON, forward_l1 first maps f_l0 (576) → f_l0a (576) through L0_A, so the
# input is re-transformed (and, when E_L0A is trained, made discriminative) before
# L1 compresses it. _W_L0A/_E_L0A are lists [36][3] of 16×16 float weight pages.
_L0A_ON = False
_W_L0A  = None
_E_L0A  = None

def _l0a_cols(j):
    """L0-feature column indices read by L0_A chip j (its 3 blocks, 48 feats)."""
    return np.concatenate([np.arange(b * N_ROWS, (b + 1) * N_ROWS) for b in L0A_ROUTING[j]])

def forward_l0a_batch(F_l0):
    """(N,576) float → (N,576) float32 via 36 L0_A chips (overlapping windows)."""
    F_u8 = float_to_act_code(F_l0)
    parts = []
    for j in range(N_L0A_CHIPS):
        if _E_L0A is not None:
            W0 = _W_L0A[j][0] + _E_L0A[j][0]
            W1 = _W_L0A[j][1] + _E_L0A[j][1]
            W2 = _W_L0A[j][2] + _E_L0A[j][2]
        else:
            W0, W1, W2 = _W_L0A[j][0], _W_L0A[j][1], _W_L0A[j][2]
        F_g = F_u8[:, _l0a_cols(j)]
        parts.append(chip_mac_leaky(w_float_to_signed(W0), w_float_to_signed(W1),
                                    w_float_to_signed(W2), F_g, LEAKY_ALPHA))
    return np.concatenate(parts, axis=1)   # (N, 576) float32

def forward_l0a_one(f_l0_float):
    """Single-sample: (576,) float → (576,) float32."""
    f_u8 = float_to_act_code(f_l0_float.reshape(1, -1))[0]
    parts = []
    for j in range(N_L0A_CHIPS):
        if _E_L0A is not None:
            W0 = _W_L0A[j][0] + _E_L0A[j][0]
            W1 = _W_L0A[j][1] + _E_L0A[j][1]
            W2 = _W_L0A[j][2] + _E_L0A[j][2]
        else:
            W0, W1, W2 = _W_L0A[j][0], _W_L0A[j][1], _W_L0A[j][2]
        f_g = f_u8[_l0a_cols(j)]
        parts.append(chip_mac_leaky_one(w_float_to_signed(W0), w_float_to_signed(W1),
                                        w_float_to_signed(W2), f_g, LEAKY_ALPHA))
    return np.concatenate(parts)   # (576,) float32


def forward_l1_batch(l1_weights, F_l0, e1_weights=None):
    """(N,576) float → (N,192) float32 via 12 L1 chips (leaky ReLU)."""
    if _L0A_ON:
        F_l0 = forward_l0a_batch(F_l0)
    F_u8 = float_to_act_code(F_l0)
    parts = []
    for c in range(N_L1_CHIPS):
        if e1_weights is not None:
            W0 = l1_weights[c][0] + e1_weights[c][0]
            W1 = l1_weights[c][1] + e1_weights[c][1]
            W2 = l1_weights[c][2] + e1_weights[c][2]
        else:
            W0, W1, W2 = l1_weights[c][0], l1_weights[c][1], l1_weights[c][2]
        F_c = F_u8[:, c * N_FEATS_PER_L1:(c + 1) * N_FEATS_PER_L1]
        if FROZEN_FWD and _WQ1 is not None:
            W0s, W1s, W2s = _WQ1[c]
        else:
            W0s, W1s, W2s = (w_float_to_signed(W0), w_float_to_signed(W1),
                             w_float_to_signed(W2))
        parts.append(chip_mac_leaky(W0s, W1s, W2s, F_c, LEAKY_ALPHA))
    return np.concatenate(parts, axis=1)   # (N, 192) float32


def forward_l1_one(l1_weights, f_l0_float, e1_weights=None, f_l0_u8=None):
    """Single-sample: (576,) float → (192,) float32.

    f_l0_u8: precomputed act codes (the training set's codes are FIXED — computing them per
    sample was 3.4s/3k samples). FROZEN_FWD: use the per-fold quantised-weight cache."""
    if _L0A_ON:
        f_l0_float = forward_l0a_one(f_l0_float)
        f_l0_u8 = None
    f_u8 = float_to_act_code(f_l0_float.reshape(1, -1))[0] if f_l0_u8 is None else f_l0_u8
    parts = []
    for c in range(N_L1_CHIPS):
        f_c = f_u8[c * N_FEATS_PER_L1:(c + 1) * N_FEATS_PER_L1]
        if FROZEN_FWD and _WQ1 is not None:
            W0s, W1s, W2s = _WQ1[c]
        else:
            if e1_weights is not None:
                W0 = l1_weights[c][0] + e1_weights[c][0]
                W1 = l1_weights[c][1] + e1_weights[c][1]
                W2 = l1_weights[c][2] + e1_weights[c][2]
            else:
                W0, W1, W2 = l1_weights[c][0], l1_weights[c][1], l1_weights[c][2]
            W0s, W1s, W2s = (w_float_to_signed(W0), w_float_to_signed(W1),
                             w_float_to_signed(W2))
        parts.append(chip_mac_leaky_one(W0s, W1s, W2s, f_c, LEAKY_ALPHA))
    return np.concatenate(parts)   # (192,) float32


def forward_l2_batch(l2_weights, F_l1, e2_weights=None):
    """(N,192) float → (N,64) float [ADC voltage] via 4 L2 chips."""
    parts = []
    for g in range(N_L2_CHIPS):
        if e2_weights is not None:
            W0 = l2_weights[g][0] + e2_weights[g][0]
            W1 = l2_weights[g][1] + e2_weights[g][1]
            W2 = l2_weights[g][2] + e2_weights[g][2]
        else:
            W0, W1, W2 = l2_weights[g][0], l2_weights[g][1], l2_weights[g][2]
        start = g * N_L1_PER_L2 * N_ROWS
        F_g = F_l1[:, start:start + N_L1_PER_L2 * N_ROWS]
        if FROZEN_FWD and _WQ2 is not None:
            W0s, W1s, W2s = _WQ2[g]
        else:
            W0s, W1s, W2s = (w_float_to_signed(W0), w_float_to_signed(W1),
                             w_float_to_signed(W2))
        out = chip_mac_leaky(W0s, W1s, W2s, F_g, LEAKY_ALPHA)
        parts.append(out / 255.0 * ADC_V_MAX)
    return np.concatenate(parts, axis=1)   # (N, 64) float


def forward_l2_one(l2_weights, f_l1, e2_weights=None):
    """Single-sample: (192,) float → (64,) float [ADC voltage]."""
    parts = []
    for g in range(N_L2_CHIPS):
        if e2_weights is not None:
            W0 = l2_weights[g][0] + e2_weights[g][0]
            W1 = l2_weights[g][1] + e2_weights[g][1]
            W2 = l2_weights[g][2] + e2_weights[g][2]
        else:
            W0, W1, W2 = l2_weights[g][0], l2_weights[g][1], l2_weights[g][2]
        start = g * N_L1_PER_L2 * N_ROWS
        f_g = f_l1[start:start + N_L1_PER_L2 * N_ROWS]
        if FROZEN_FWD and _WQ2 is not None:
            W0s, W1s, W2s = _WQ2[g]
        else:
            W0s, W1s, W2s = (w_float_to_signed(W0), w_float_to_signed(W1),
                             w_float_to_signed(W2))
        y = chip_mac_leaky_one(W0s, W1s, W2s, f_g, LEAKY_ALPHA)
        parts.append(y / 255.0 * ADC_V_MAX)
    return np.concatenate(parts)   # (64,) float


def forward_l3_batch(l3_weights, e3_weights, F_l2):
    """(N,64) float → (N,128) float via 8 L3 chips (2-page, routed, leaky ReLU)."""
    parts = []
    for g in range(N_L3_CHIPS):
        if e3_weights is not None:
            W0 = l3_weights[g][0] + e3_weights[g][0]
            W1 = l3_weights[g][1] + e3_weights[g][1]
        else:
            W0, W1 = l3_weights[g][0], l3_weights[g][1]
        l2a, l2b = L3_ROUTING[g]
        F_g = np.concatenate([F_l2[:, l2a*N_ROWS:(l2a+1)*N_ROWS],
                               F_l2[:, l2b*N_ROWS:(l2b+1)*N_ROWS]], axis=1)  # (N, 32)
        if FROZEN_FWD and _WQ3 is not None:
            W0s, W1s = _WQ3[g]
        else:
            W0s, W1s = w_float_to_signed(W0), w_float_to_signed(W1)
        parts.append(chip_mac_l3(W0s, W1s, F_g, LEAKY_ALPHA))
    return np.concatenate(parts, axis=1)   # (N, 128) float32


def forward_l3_one(l3_weights, e3_weights, f_l2):
    """Single-sample: (64,) float → (128,) float (raw, no ADC rescaling)."""
    parts = []
    for g in range(N_L3_CHIPS):
        if e3_weights is not None:
            W0 = l3_weights[g][0] + e3_weights[g][0]
            W1 = l3_weights[g][1] + e3_weights[g][1]
        else:
            W0, W1 = l3_weights[g][0], l3_weights[g][1]
        l2a, l2b = L3_ROUTING[g]
        f_g = np.concatenate([f_l2[l2a*N_ROWS:(l2a+1)*N_ROWS],
                               f_l2[l2b*N_ROWS:(l2b+1)*N_ROWS]])  # (32,)
        if FROZEN_FWD and _WQ3 is not None:
            W0s, W1s = _WQ3[g]
        else:
            W0s, W1s = w_float_to_signed(W0), w_float_to_signed(W1)
        parts.append(chip_mac_l3_one(W0s, W1s, f_g, LEAKY_ALPHA))
    return np.concatenate(parts)   # (128,) float32


# ── EMNIST L0 feature cache ────────────────────────────────────────────────────

def _top_pca_rows(X, n_comp):
    """Top-n uncentered-PCA rows of X — the converged ideal of the chip's native GHA rule."""
    C = (X.T @ X) / max(len(X), 1)
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1][:n_comp]
    W = vecs[:, order].T.astype(np.float64)
    W[((X @ W.T).mean(axis=0) < 0)] *= -1.0
    return W


def ensure_big_weights():
    """BIG: EMNIST-native L0 PCA (split-sign, 1152 feats) + per-chip PCA weights for the 24 L1
    and 8 L2 chips.  Fully regenerable from data + fixed seed (no hex-file dependency), but
    already cached by bigv2 — we reuse that, so this is a no-op in practice."""
    paths = {k: os.path.join(BIG_WEIGHTS_DIR, f'{k}.npy') for k in
             ['l0_W', 'F_l0_train', 'F_l0_test', 'y_train', 'y_test', 'w1', 'w2']}
    if all(os.path.exists(pp) for pp in paths.values()):
        return paths

    os.makedirs(BIG_WEIGHTS_DIR, exist_ok=True)
    print("Building EMNIST-native BIG L0 + per-chip PCA weights (one-off) ...", flush=True)
    X_train, y_train, X_test, y_test = load_emnist_letters()
    X_tr_pp, X_te_pp, _ = preprocess(X_train, X_test)
    rng = np.random.default_rng(7)
    sub = rng.choice(len(X_tr_pp), size=min(30000, len(X_tr_pp)), replace=False)

    l0_W = _top_pca_rows(X_tr_pp[sub], N_L0_COMPS)                  # (576, 784)

    def _split_sign(F):
        out = np.empty((F.shape[0], 2 * F.shape[1]), dtype=np.float32)
        out[:, 0::2] = np.maximum(F, 0.0)
        out[:, 1::2] = np.maximum(-F, 0.0)
        return out

    F_l0_train = _split_sign((X_tr_pp @ l0_W.T).astype(np.float32))  # (N, 1152)
    F_l0_test  = _split_sign((X_te_pp @ l0_W.T).astype(np.float32))

    F_codes = float_to_act_code(F_l0_train[sub]).astype(np.float64)
    w1 = np.zeros((N_L1_CHIPS, N_PAGES, N_ROWS, N_COLS), dtype=np.float32)
    for c in range(N_L1_CHIPS):
        Wc = _top_pca_rows(F_codes[:, c * N_FEATS_PER_L1:(c + 1) * N_FEATS_PER_L1], N_ROWS)
        for pg in range(N_PAGES):
            w1[c, pg] = Wc[:, pg * N_COLS:(pg + 1) * N_COLS]
    l1_w = [[w1[c, pg].copy() for pg in range(N_PAGES)] for c in range(N_L1_CHIPS)]

    F_l1_sub = forward_l1_batch(l1_w, F_l0_train[sub]).astype(np.float64)
    w2 = np.zeros((N_L2_CHIPS, N_PAGES, N_ROWS, N_COLS), dtype=np.float32)
    for g in range(N_L2_CHIPS):
        st = g * N_L1_PER_L2 * N_ROWS
        Wg = _top_pca_rows(F_l1_sub[:, st:st + N_L1_PER_L2 * N_ROWS], N_ROWS)
        for pg in range(N_PAGES):
            w2[g, pg] = Wg[:, pg * N_COLS:(pg + 1) * N_COLS]

    np.save(paths['l0_W'], l0_W.astype(np.float32))
    np.save(paths['F_l0_train'], F_l0_train.astype(np.float16))
    np.save(paths['F_l0_test'],  F_l0_test.astype(np.float16))
    np.save(paths['y_train'], y_train); np.save(paths['y_test'], y_test)
    np.save(paths['w1'], w1); np.save(paths['w2'], w2)
    return paths


def get_emnist_l0_features():
    paths = ensure_big_weights()
    print("Loading EMNIST split-sign BIG L0 features ...", flush=True)
    F_tr = np.load(paths['F_l0_train']).astype(np.float32)
    F_te = np.load(paths['F_l0_test']).astype(np.float32)
    y_tr = np.load(paths['y_train']).astype(int)
    y_te = np.load(paths['y_test']).astype(int)
    print(f"  {len(y_tr):,} train / {len(y_te):,} test  "
          f"(26 classes, {F_tr.shape[1]} split-sign feats)", flush=True)
    return F_tr, F_te, y_tr, y_te


def load_l1_weights():
    w1 = np.load(ensure_big_weights()['w1'])
    print(f"  Loaded L1 weights ({N_L1_CHIPS} chips x {N_PAGES} pages, EMNIST per-chip PCA)",
          flush=True)
    return [[w1[c, pg].copy() for pg in range(N_PAGES)] for c in range(N_L1_CHIPS)]


def load_l2_weights():
    w2 = np.load(ensure_big_weights()['w2'])
    print(f"  Loaded L2 weights ({N_L2_CHIPS} chips x {N_PAGES} pages, EMNIST per-chip PCA)",
          flush=True)
    return [[w2[g, pg].copy() for pg in range(N_PAGES)] for g in range(N_L2_CHIPS)]


def init_l3_weights_random_ortho(seed=2):
    """Initialise 8 L3 chips × 2 pages as random orthonormal (16×16) matrices."""
    rng = np.random.default_rng(seed)
    weights = []
    for _ in range(N_L3_CHIPS):
        pages = []
        for _ in range(N_PAGES_L3):
            Q, _ = np.linalg.qr(rng.standard_normal((N_ROWS, N_ROWS)))
            pages.append(Q.astype(np.float32))
        weights.append(pages)
    return weights


def init_e_weights(weights):
    return [[np.zeros_like(page) for page in chip] for chip in weights]


# W_MASTER_CLIP: clip the FLOAT MASTER weight to the hardware rail [-0.89, 1.0].
# The forward is clamped regardless (w_float_to_signed clips codes to CODE_MIN/CODE_MAX),
# so this only decides whether the master may hold "pressure" beyond the rail — which is
# exactly what torch's straight-through --quant does (it clamps the forward copy q and
# leaves the float master self.w free). Clipping the master DESTROYS that pressure: a weight
# pinned at the rail forgets how hard it was being pushed there. Buildable either way — a
# free master is just a few extra bits on the boss's digital copy.  --w_master_free turns
# the clip OFF (torch semantics).
W_MASTER_CLIP = True


def _wclip(a):
    return np.clip(a, W_FLOAT_MIN, W_FLOAT_MAX) if W_MASTER_CLIP else a


# ── Absorb functions ───────────────────────────────────────────────────────────

def absorb_e1_only(l1_weights, e1_weights, lr_slow, decay_e=1.0):
    for c in range(N_L1_CHIPS):
        for p in range(N_PAGES):
            l1_weights[c][p] += lr_slow * e1_weights[c][p]
            l1_weights[c][p]  = _wclip(l1_weights[c][p])
            e1_weights[c][p] *= (1.0 - decay_e)


def absorb_e2_only(l2_weights, e2_weights, lr_slow, decay_e=1.0):
    for g in range(N_L2_CHIPS):
        for p in range(N_PAGES):
            l2_weights[g][p] += lr_slow * e2_weights[g][p]
            l2_weights[g][p]  = _wclip(l2_weights[g][p])
            e2_weights[g][p] *= (1.0 - decay_e)


def absorb_e3_only(l3_weights, e3_weights, lr_slow, decay_e=1.0):
    for g in range(N_L3_CHIPS):
        for p in range(N_PAGES_L3):
            l3_weights[g][p] += lr_slow * e3_weights[g][p]
            l3_weights[g][p]  = _wclip(l3_weights[g][p])
            e3_weights[g][p] *= (1.0 - decay_e)


def absorb_l0a_only(lr_slow, decay_e=1.0):
    """Absorb E_L0A into W_L0A (module globals), same cycle as E1/E2/E3."""
    global _W_L0A, _E_L0A
    for j in range(N_L0A_CHIPS):
        for p in range(N_PAGES):
            _W_L0A[j][p] += lr_slow * _E_L0A[j][p]
            _W_L0A[j][p]  = np.clip(_W_L0A[j][p], W_FLOAT_MIN, W_FLOAT_MAX)
            _E_L0A[j][p] *= (1.0 - decay_e)


def build_l0a_weights(F_l0_sub, l1_weights):
    """PCA-of-windows init for L0_A (identity impossible under 8-bit quantisation —
    see L0A_DESIGN.md).  Each chip = top-16 PCA of its 48-feat L0 window, scaled so the
    weights match L1's GHA weight magnitude (the known-good, quantisation-visible scale;
    a same-OUTPUT-scale calibration is impossible — the chip MAC's ~2.2× per-tap gain
    forces sub-quantum weights that round to 0).  Returns a [36][3] list of float pages."""
    Fc = float_to_act_code(F_l0_sub).astype(np.float64)
    # target weight scale = RMS of L1's GHA weight entries
    tgt = float(np.sqrt(np.mean([np.mean(p ** 2) for ch in l1_weights for p in ch])))
    w = []
    for j in range(N_L0A_CHIPS):
        Xj = Fc[:, _l0a_cols(j)]                       # (N,48) act codes
        _, _, Vt = np.linalg.svd(Xj - Xj.mean(0), full_matrices=False)
        comps = Vt[:N_ROWS]                            # (16,48) unit-norm rows
        rms_c = float(np.sqrt(np.mean(comps ** 2)) + 1e-9)
        comps = comps * (tgt / rms_c)                  # match L1 weight magnitude
        w.append([np.clip(comps[:, p*N_COLS:(p+1)*N_COLS], W_FLOAT_MIN, W_FLOAT_MAX
                          ).astype(np.float32) for p in range(N_PAGES)])
    print(f"  Built L0_A weights ({N_L0A_CHIPS} chips × {N_PAGES} pages, PCA-of-windows, "
          f"scaled to L1 weight RMS={tgt:.3f})", flush=True)
    return w


# ── Classifier fitting / evaluation ───────────────────────────────────────────

CLF_SUB = 0    # --clf_sub N: fit the per-epoch READOUT on N rows instead of all 124,800.
               # The readout is refit EVERY EPOCH, TWICE — once as the TEACHING classifier
               # (its softmax residual s = onehot - softmax(W_f.f) IS the error signal that
               # drives the whole rule) and once for eval. Each fit is a FIXED ~11s tax that
               # does NOT shrink with --chunk, so it dominates short epochs.
               # ⚠ N MUST SCALE WITH THE READOUT WIDTH — it does not transfer between rigs:
               #     smallBIG  32->26 readout (  858 params): N=20000 is FREE (0.1pp)
               #     BIG      256->26 readout (6,682 params): N=20000 COSTS ~5pp
               # On BIG, prefer --fast_batch alone; the fit there is now THE bottleneck
               # (~226s of a 266s epoch), and the right fix is to fit it LESS OFTEN or
               # warm-start it, not to thin the data.


def fit_clf(acts, labels, final=False):
    """Fit a linear classifier on activations.

    fast (default): least-squares multiclass (numpy lstsq, ~0.1s).
    final=True: sklearn lbfgs logistic regression for the final report only.
    """
    try:
        if CLF_SUB and not final and len(labels) > CLF_SUB:
            _s = np.random.default_rng(0).choice(len(labels), size=CLF_SUB, replace=False)
            acts, labels = acts[_s], labels[_s]
        A = acts.astype(np.float64)
        mean  = A.mean(axis=0)
        scale = np.maximum(A.std(axis=0), 1e-4)
        A_sc  = (A - mean) / scale

        if final or CLF_LOGISTIC:
            import warnings
            from sklearn.linear_model import LogisticRegression
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                clf = LogisticRegression(solver='lbfgs', C=10.0, max_iter=2000,
                                         random_state=0)
                clf.fit(A_sc, labels)
            W_f = (clf.coef_ / scale).astype(np.float64)
            b_f = (clf.intercept_ - clf.coef_ @ (mean / scale)).astype(np.float64)
        else:
            n_cls = N_CLASSES
            Y_oh  = (labels[:, None] == np.arange(n_cls)).astype(np.float64)
            A_aug = np.c_[A_sc, np.ones(len(A_sc))]
            Wb, _, _, _ = np.linalg.lstsq(A_aug, Y_oh, rcond=None)
            W_raw = Wb[:-1]   # (n_feats, n_cls)
            b_raw = Wb[-1]    # (n_cls,)
            W_f = (W_raw / scale[:, None]).T.astype(np.float64)  # (n_cls, n_feats)
            b_f = (b_raw - W_raw.T @ (mean / scale)).astype(np.float64)

        return W_f, b_f
    except Exception as exc:
        print(f"  [clf fit failed: {exc}]", flush=True)
        return None, None


def eval_acc_l3(l1_weights, e1_weights, l2_weights, e2_weights,
                l3_weights, e3_weights, F_l0_eval, y_eval, W_f, b_f):
    if W_f is None:
        return 0.0
    F_l1 = forward_l1_batch(l1_weights, F_l0_eval, e1_weights=e1_weights)
    F_l2 = forward_l2_batch(l2_weights, F_l1, e2_weights=e2_weights)
    F_l3 = forward_l3_batch(l3_weights, e3_weights, F_l2)
    preds = np.argmax(F_l3.astype(np.float64) @ W_f.T + b_f, axis=1)
    return float(np.mean(preds == y_eval))


def eval_acc_l2(l1_weights, e1_weights, l2_weights, e2_weights,
               F_l0_eval, y_eval, W_f_l2, b_f_l2):
    """Evaluate classifier accuracy using L2 features directly (64-dim, no L3)."""
    if W_f_l2 is None:
        return 0.0
    F_l1 = forward_l1_batch(l1_weights, F_l0_eval, e1_weights=e1_weights)
    F_l2 = forward_l2_batch(l2_weights, F_l1, e2_weights=e2_weights)
    preds = np.argmax(F_l2.astype(np.float64) @ W_f_l2.T + b_f_l2, axis=1)
    return float(np.mean(preds == y_eval))


# ── Normalisation / signal helpers ─────────────────────────────────────────────

def _rms_normalize(vec, eps=1e-4):
    """Scale vec to unit per-element RMS.  Safe for near-zero vectors."""
    rms = np.linalg.norm(vec) / np.sqrt(vec.size)
    return vec / rms if rms > eps else vec


def _f_factor(f_p, nlms, nlms_gain):
    """Input factor for the outer-product E update.

    unit-RMS (default):  f̂ = f/rms(f)          → ΔE@f = η·δ·N·rms(f)
    NLMS (--nlms):       f·gain/‖f‖²            → ΔE@f = η·gain·δ  exactly
    """
    if nlms:
        ss = float(f_p @ f_p)
        return f_p * (nlms_gain / ss) if ss > 1e-8 else np.zeros_like(f_p)
    return _rms_normalize(f_p)


def _rms_preserving_proj(W_T, delta_src):
    """Apply W_T @ delta_src and rescale output to preserve per-element RMS."""
    d_out = W_T @ delta_src
    rms_in  = np.linalg.norm(delta_src) / np.sqrt(delta_src.size)
    rms_out = np.linalg.norm(d_out)     / np.sqrt(d_out.size)
    if rms_out > 1e-8:
        d_out *= rms_in / rms_out
    return d_out


def _bh_step_scale(scores, pred, label, boss_h_fixed):
    """Boss_h severity scalar λ_h.  Returns None when the update should be
    skipped (margin ≤ 0 or h quantises to 0 — the beneficial noise gate,
    which all variants preserve).

    Default h denominator: full score span (max−min).  --bh_std replaces it
    with 4·std(scores): the span is inflated by any single outlier class
    score, deflating h for a decisively-wrong sample; std is robust to one
    outlier.  The factor 4 calibrates std→span for ~26 scores (range ≈ 3.8σ)
    so the h distribution keeps roughly the same operating point."""
    if boss_h_fixed is not None:
        return (boss_h_fixed + 1) / 4.0
    margin = float(scores[pred]) - float(scores[label])
    if margin <= 0:
        return None
    if BH_STD:
        denom = max(4.0 * float(np.std(scores)), 1e-6)
    else:
        denom = max(float(scores.max() - scores.min()), 1e-6)
    bh = min(int(7.0 * BH_GAIN * margin / denom), BOSS_H_MAX)
    if bh == 0:
        return None
    return (bh + 1) / 4.0


def _bh_direction(row_diff):
    """Per-dimension direction from the classifier row difference
    W_f[y] − W_f[ŷ].

    Default: sign() — every dimension pushed at full amplitude, including
    ones where the two rows nearly agree and the sign is noise.
    --bh_deadzone T: zero components with |diff| < T×rms(diff) — only
    dimensions the classifier meaningfully distinguishes get pushed.
    --bh_maglevels: 3-level magnitude {0, ±0.5, ±1} at thresholds
    (T_lo, 1.0)×rms(diff), T_lo = bh_deadzone or 0.3 — a quantised step
    toward backprop's per-dimension magnitudes without losing the gate."""
    d = np.sign(row_diff)
    if BH_MAGLEVELS or BH_DEADZONE > 0.0:
        a    = np.abs(row_diff)
        rms_ = float(np.sqrt(np.mean(a * a)))
        if rms_ > 1e-12:
            if BH_MAGLEVELS:
                lo = (BH_DEADZONE if BH_DEADZONE > 0.0 else 0.3) * rms_
                d  = np.where(a < lo, 0.0, np.where(a < rms_, 0.5 * d, d))
            else:
                d  = np.where(a < BH_DEADZONE * rms_, 0.0, d)
    return d


def _gate_renorm(delta, f_cur):
    """Directional leaky-ReLU gate: δ ×= σ'(f), then rescale back to the
    pre-gate RMS.  Redistributes the correction toward units whose output can
    actually respond, with zero net attenuation — safe under RMS-preserving
    scaling where the old ungated --relu_jac collapsed the signal."""
    g = delta * np.where(f_cur > 0, 1.0, LEAKY_ALPHA)
    n_in  = np.linalg.norm(delta)
    n_out = np.linalg.norm(g)
    if n_out > 1e-12:
        g *= n_in / n_out
    return g


def _inv_boost(delta, f_cur):
    """Delivery compensation: pre-activation request δ/σ' so the
    post-activation shift matches the requested δ.  Capped at ×3 (full ×1/α
    = ×10 amplifies noise on dead units).  No renormalisation — the boost IS
    the point.  Only used for the outer-product write, not the chain."""
    return delta * np.where(f_cur > 0, 1.0, min(1.0 / max(LEAKY_ALPHA, 1e-3), 3.0))


def _local_delta(W_aux, b_aux, f_cur, label, boss_h_fixed):
    """Layer-local CHL-like correction from an aux classifier on this layer's
    own features: λ_h·Δ·direction(W_aux[y] − W_aux[ŷ]).  Zeros when the layer
    is locally correct (or margin below the boss_h quantisation gate)."""
    scores = W_aux @ f_cur + b_aux
    pred   = int(np.argmax(scores))
    if pred == label:
        return np.zeros(f_cur.size, dtype=np.float64)
    step_scale = _bh_step_scale(scores, pred, label, boss_h_fixed)
    if step_scale is None:
        return np.zeros(f_cur.size, dtype=np.float64)
    return step_scale * DELTA * _bh_direction(W_aux[label] - W_aux[pred])


# ── Core per-sample update ─────────────────────────────────────────────────────

def settle_and_update(
    l1_weights, e1_weights,
    l2_weights, e2_weights,
    l3_weights, e3_weights,
    f_l0_float, f_l1, f_l2,
    label, W_f, b_f,
    boss_h_fixed, n_settle, e_clip,
    freeze_e1=True, freeze_e2=True, freeze_e3=False,
    lr_scale=1.0, lr_layers=None,
    nlms=False, nlms_gain=1.0,
    aux2=None, aux1=None, beta=0.5,
    avg_bp=False,
    bp_l3=None, bp_l2=None,
    freeze_e0=True, aux0=None,
):
    _lr3 = lr_layers[0] if lr_layers else BOSS_LR * lr_scale   # per-layer auto η
    _lr2 = lr_layers[1] if lr_layers else BOSS_LR * lr_scale
    _lr1 = lr_layers[2] if lr_layers else BOSS_LR * lr_scale
    """
    E-settle: update E iteratively with forward-pass recomputation each step.
    freeze_e0=False (phase P0): train E_L0A (module globals) from its own local aux
    classifier aux0 on f_l0a — the input-transform layer.  local-delta driven (v1):
    bypasses the deepest backprojection, per L0A_DESIGN.md.

    aux2/aux1: (W_aux, b_aux) tuples for layer-local classifier deltas at
    L2/L1, blended as β·local + (1−β)·backprojected.  None = pure
    backprojection (optd behaviour).

    bp_l3/bp_l2: weight matrices used for the backprojection hops (either the
    live absorbed W or an EMA shadow).  None = live W.
    """
    if W_f is None:
        return 0
    if bp_l3 is None:
        bp_l3 = l3_weights
    if bp_l2 is None:
        bp_l2 = l2_weights

    # Pre-compute f_l0_u8 once — f_l0 never changes within the settle loop
    f_l0_u8 = float_to_act_code(f_l0_float.reshape(1, -1))[0]

    # ── Initial forward state ──────────────────────────────────────────────────
    # When E_L0A trains (P0), the input transform changes, so recompute from f_l0
    # through the whole stack (forward_l1_one internally re-applies L0_A).
    f_l0a_cur = forward_l0a_one(f_l0_float) if not freeze_e0 else None
    if not freeze_e1 or not freeze_e0:
        f_l1_cur = forward_l1_one(l1_weights, f_l0_float, e1_weights=e1_weights)
        f_l2_cur = forward_l2_one(l2_weights, f_l1_cur, e2_weights=e2_weights)
        f_l3_cur = forward_l3_one(l3_weights, e3_weights, f_l2_cur)
    elif not freeze_e2:
        f_l1_cur = f_l1
        f_l2_cur = forward_l2_one(l2_weights, f_l1_cur, e2_weights=e2_weights)
        f_l3_cur = forward_l3_one(l3_weights, e3_weights, f_l2_cur)
    else:
        f_l1_cur = f_l1
        f_l2_cur = f_l2
        f_l3_cur = forward_l3_one(l3_weights, e3_weights, f_l2_cur)

    n_moves = 0

    for _ in range(n_settle):
        # ── Classification check ───────────────────────────────────────────────
        scores = W_f @ f_l3_cur + b_f
        pred   = int(np.argmax(scores))
        if pred == label:
            break

        # ── Boss_h step scale ──────────────────────────────────────────────────
        step_scale = _bh_step_scale(scores, pred, label, boss_h_fixed)
        if step_scale is None:
            break

        # ── Direction in L3 output space ───────────────────────────────────────
        dirn     = _bh_direction(W_f[label] - W_f[pred])
        delta_l3 = step_scale * DELTA * dirn
        if RELU_JAC_DIR:
            delta_l3 = _gate_renorm(delta_l3, f_l3_cur)
        delta_l3_w = _inv_boost(delta_l3, f_l3_cur) if RELU_JAC_INV else delta_l3

        # ── E3 update ──────────────────────────────────────────────────────────
        if not freeze_e3:
            for g in range(N_L3_CHIPS):
                delta_g = delta_l3_w[g * N_ROWS:(g + 1) * N_ROWS]
                if not np.any(delta_g != 0):
                    continue
                l2a, l2b = L3_ROUTING[g]
                f_l2_g = np.concatenate([f_l2_cur[l2a*N_ROWS:(l2a+1)*N_ROWS],
                                         f_l2_cur[l2b*N_ROWS:(l2b+1)*N_ROWS]])
                for p in range(N_PAGES_L3):
                    f_p   = f_l2_g[p * N_ROWS:(p + 1) * N_ROWS].astype(np.float64)
                    f_use = _f_factor(f_p, nlms, nlms_gain)
                    e3_weights[g][p] += _lr3 * np.outer(delta_g, f_use)
                    e3_weights[g][p]  = np.clip(e3_weights[g][p], -e_clip, e_clip)

        # ── Backprojection + local blend → delta_l2 (needed for E2 or E1) ─────
        if not (freeze_e2 and freeze_e1):
            delta_l2 = np.zeros(N_L2_FEATS, dtype=np.float64)
            for g in range(N_L3_CHIPS):
                delta_g = delta_l3[g * N_ROWS:(g + 1) * N_ROWS]
                for p, l2_chip_idx in enumerate(L3_ROUTING[g]):
                    delta_l2[l2_chip_idx * N_ROWS:(l2_chip_idx + 1) * N_ROWS] += \
                        _rms_preserving_proj(bp_l3[g][p].T, delta_g)
            if avg_bp:
                for g2 in range(N_L2_CHIPS):
                    delta_l2[g2 * N_ROWS:(g2 + 1) * N_ROWS] /= max(L2_BP_COUNT[g2], 1.0)

            delta_l2_used = delta_l2
            if aux2 is not None:
                loc2 = _local_delta(aux2[0], aux2[1], f_l2_cur.astype(np.float64),
                                    label, boss_h_fixed)
                delta_l2_used = beta * loc2 + (1.0 - beta) * delta_l2
            if RELU_JAC_DIR:
                delta_l2_used = _gate_renorm(delta_l2_used, f_l2_cur)
            delta_l2_w = (_inv_boost(delta_l2_used, f_l2_cur)
                          if RELU_JAC_INV else delta_l2_used)

            # ── E2 update ──────────────────────────────────────────────────────
            if not freeze_e2:
                for g2 in range(N_L2_CHIPS):
                    delta_g2 = delta_l2_w[g2 * N_ROWS:(g2 + 1) * N_ROWS]
                    if not np.any(delta_g2 != 0):
                        continue
                    start   = g2 * N_L1_PER_L2 * N_ROWS
                    f_l1_g2 = f_l1_cur[start:start + N_L1_PER_L2 * N_ROWS].astype(np.float64)
                    for p in range(N_PAGES):
                        f_p   = f_l1_g2[p * N_ROWS:(p + 1) * N_ROWS]
                        f_use = _f_factor(f_p, nlms, nlms_gain)
                        e2_weights[g2][p] += _lr2 * np.outer(delta_g2, f_use)
                        e2_weights[g2][p]  = np.clip(e2_weights[g2][p], -e_clip, e_clip)

            # ── W2.T backprojection + local blend → delta_l1 ───────────────────
            if not freeze_e1:
                delta_l1 = np.zeros(N_L1_FEATS, dtype=np.float64)
                for g2 in range(N_L2_CHIPS):
                    delta_g2 = delta_l2_used[g2 * N_ROWS:(g2 + 1) * N_ROWS]
                    for p in range(N_PAGES):
                        c = g2 * N_L1_PER_L2 + p
                        delta_l1[c * N_ROWS:(c + 1) * N_ROWS] = \
                            _rms_preserving_proj(bp_l2[g2][p].T, delta_g2)

                delta_l1_used = delta_l1
                if aux1 is not None:
                    loc1 = _local_delta(aux1[0], aux1[1], f_l1_cur.astype(np.float64),
                                        label, boss_h_fixed)
                    delta_l1_used = beta * loc1 + (1.0 - beta) * delta_l1
                if RELU_JAC_DIR:
                    delta_l1_used = _gate_renorm(delta_l1_used, f_l1_cur)
                delta_l1_w = (_inv_boost(delta_l1_used, f_l1_cur)
                              if RELU_JAC_INV else delta_l1_used)

                # ── E1 update ──────────────────────────────────────────────────
                for c in range(N_L1_CHIPS):
                    delta_c = delta_l1_w[c * N_ROWS:(c + 1) * N_ROWS]
                    if not np.any(delta_c != 0):
                        continue
                    f_l0_c = f_l0_u8[c * N_FEATS_PER_L1:(c + 1) * N_FEATS_PER_L1]
                    for p in range(N_PAGES):
                        f_p   = f_l0_c[p * N_ROWS:(p + 1) * N_ROWS].astype(np.float64)
                        f_use = _f_factor(f_p, nlms, nlms_gain)
                        e1_weights[c][p] += _lr1 * np.outer(delta_c, f_use)
                        e1_weights[c][p]  = np.clip(e1_weights[c][p], -e_clip, e_clip)

        # ── E_L0A update (phase P0): local-aux-driven input-transform learning ──
        if not freeze_e0 and aux0 is not None:
            delta_l0a = _local_delta(aux0[0], aux0[1], f_l0a_cur.astype(np.float64),
                                     label, boss_h_fixed)
            if RELU_JAC_DIR:
                delta_l0a = _gate_renorm(delta_l0a, f_l0a_cur)
            for j in range(N_L0A_CHIPS):
                delta_j = delta_l0a[j * N_ROWS:(j + 1) * N_ROWS]
                if not np.any(delta_j != 0):
                    continue
                f_in = f_l0_u8[_l0a_cols(j)].astype(np.float64)   # this chip's L0 window
                for p in range(N_PAGES):
                    f_p   = f_in[p * N_ROWS:(p + 1) * N_ROWS]
                    f_use = _f_factor(f_p, nlms, nlms_gain)
                    _E_L0A[j][p] += BOSS_LR * lr_scale * np.outer(delta_j, f_use)
                    _E_L0A[j][p]  = np.clip(_E_L0A[j][p], -e_clip, e_clip)

        n_moves += 1

        # ── Recompute forward pass with updated E matrices ─────────────────────
        if not freeze_e1 or not freeze_e0:
            if not freeze_e0:
                f_l0a_cur = forward_l0a_one(f_l0_float)
            f_l1_cur = forward_l1_one(l1_weights, f_l0_float, e1_weights=e1_weights)
            f_l2_cur = forward_l2_one(l2_weights, f_l1_cur, e2_weights=e2_weights)
            f_l3_cur = forward_l3_one(l3_weights, e3_weights, f_l2_cur)
        elif not freeze_e2:
            f_l2_cur = forward_l2_one(l2_weights, f_l1_cur, e2_weights=e2_weights)
            f_l3_cur = forward_l3_one(l3_weights, e3_weights, f_l2_cur)
        else:
            f_l3_cur = forward_l3_one(l3_weights, e3_weights, f_l2_cur)

    return n_moves


# ── DFA (Direct Feedback Alignment) — coordinated forwards-only rule ─────────────
# Broadcast the OUTPUT error e (26-vec) to every layer through a FIXED RANDOM matrix
# B_ℓ (n_ℓ × 26); δ_ℓ = (B_ℓ·e) ⊙ σ'(f_ℓ); ΔW_ℓ ∝ outer(δ_ℓ, input).  No backward
# chain, no W.T, no per-layer classifiers — all layers serve the SAME global error,
# so they cooperate instead of fighting.  Forwards-only + broadcast-only.
def _constrain_delta(d, layer=None):
    """Apply the hardware-constraint ladder to a transported delta vector.

    Order matters and mirrors what silicon would actually do: gate (sigma'), then renorm,
    then reduce precision. Each knob is separable so its cost can be priced on its own.
    """
    if DELTA_AGC > 0.0 and layer is not None:
        # SLOW per-layer AGC: track rms with an EMA, apply a gain to hold the layer's scale.
        # Slow => within-layer AND across-sample magnitude variation both survive; only the
        # slowly-drifting per-layer scale is corrected. This is the depth fix.
        r = float(np.sqrt(np.mean(d * d)))
        prev = _AGC_STATE.get(layer)
        _AGC_STATE[layer] = r if prev is None else (1.0 - DELTA_AGC) * prev + DELTA_AGC * r
        ref = max(_AGC_STATE[layer], 1e-12)
        d = d * (AGC_TARGET / ref)
    if DELTA_RENORM:
        # per-SAMPLE renorm (the old rule). Also flattens across samples -> costs ~8.6pp.
        r = float(np.sqrt(np.mean(d * d)))
        if r > 1e-12:
            d = d / r
    if DELTA_SIGN:
        return np.sign(d)
    if DELTA_BITS > 0:
        # FIXED range = the honest hardware model (a real DAC has fixed rails).
        # per-vector max|d| = an IDEAL instantaneous AGC (what the ladder implicitly assumed).
        scale = DELTA_FIXED if DELTA_FIXED > 0.0 else float(np.max(np.abs(d)))
        if scale <= 1e-12:
            return d
        if DELTA_BITS == 1:
            return np.sign(d) * scale
        L = 2 ** (DELTA_BITS - 1) - 1          # signed levels either side of zero
        q = np.round(d / scale * L) / L * scale
        return np.clip(q, -scale, scale) if DELTA_FIXED > 0.0 else q
    return d


def _leaky_deriv(f):
    return np.where(f >= 0.0, 1.0, LEAKY_ALPHA)

def settle_and_update_dfa(l1_weights, e1_weights, l2_weights, e2_weights,
                          l3_weights, e3_weights, f_l0_float, label, W_f, b_f,
                          B1, B2, B3, n_settle, e_clip, nlms, nlms_gain,
                          freeze_e1, freeze_e2, freeze_e3, hard=False, lr_layers=None):
    _lr3 = lr_layers[0] if lr_layers else BOSS_LR
    _lr2 = lr_layers[1] if lr_layers else BOSS_LR
    _lr1 = lr_layers[2] if lr_layers else BOSS_LR
    f_l0_u8 = float_to_act_code(f_l0_float.reshape(1, -1))[0]
    f_l1 = forward_l1_one(l1_weights, f_l0_float, e1_weights=e1_weights)
    f_l2 = forward_l2_one(l2_weights, f_l1, e2_weights=e2_weights)
    f_l3 = forward_l3_one(l3_weights, e3_weights, f_l2)
    n_moves = 0
    for _ in range(n_settle):
        scores = W_f @ f_l3 + b_f
        if int(np.argmax(scores)) == label:
            break
        pred = int(np.argmax(scores))
        if hard:
            # CONTRASTIVE broadcast: only the two classes in play. onehot(y)−onehot(ŷ):
            # toward true, away from predicted-wrong. No softmax noise from the other 24
            # classes; depends on W_f only via which class won (robust to the epoch-refit).
            neg_e = np.zeros(N_CLASSES, dtype=np.float64)
            neg_e[label] = 1.0
            neg_e[pred] = neg_e[pred] - 1.0
        else:
            z = scores - scores.max(); pr = np.exp(z); pr /= pr.sum()
            e = pr.copy(); e[label] -= 1.0             # ∂CE/∂logits = p − onehot
            neg_e = (-e).astype(np.float64)            # so E += lr·outer(δ,x) reduces loss

        if not freeze_e3:
            d3 = (B3 @ neg_e) * _leaky_deriv(f_l3)
            for g in range(N_L3_CHIPS):
                dg = d3[g*N_ROWS:(g+1)*N_ROWS]
                l2a, l2b = L3_ROUTING[g]
                f_l2_g = np.concatenate([f_l2[l2a*N_ROWS:(l2a+1)*N_ROWS],
                                         f_l2[l2b*N_ROWS:(l2b+1)*N_ROWS]])
                for p in range(N_PAGES_L3):
                    f_use = _f_factor(f_l2_g[p*N_ROWS:(p+1)*N_ROWS].astype(np.float64), nlms, nlms_gain)
                    e3_weights[g][p] += _lr3 * np.outer(dg, f_use)
                    e3_weights[g][p]  = np.clip(e3_weights[g][p], -e_clip, e_clip)
        if not freeze_e2:
            d2 = (B2 @ neg_e) * _leaky_deriv(f_l2)
            for g2 in range(N_L2_CHIPS):
                dg2 = d2[g2*N_ROWS:(g2+1)*N_ROWS]
                start = g2 * N_L1_PER_L2 * N_ROWS
                f_l1_g2 = f_l1[start:start + N_L1_PER_L2 * N_ROWS].astype(np.float64)
                for p in range(N_PAGES):
                    f_use = _f_factor(f_l1_g2[p*N_ROWS:(p+1)*N_ROWS], nlms, nlms_gain)
                    e2_weights[g2][p] += _lr2 * np.outer(dg2, f_use)
                    e2_weights[g2][p]  = np.clip(e2_weights[g2][p], -e_clip, e_clip)
        if not freeze_e1:
            d1 = (B1 @ neg_e) * _leaky_deriv(f_l1)
            for c in range(N_L1_CHIPS):
                dc = d1[c*N_ROWS:(c+1)*N_ROWS]
                f_l0_c = f_l0_u8[c*N_FEATS_PER_L1:(c+1)*N_FEATS_PER_L1].astype(np.float64)
                for p in range(N_PAGES):
                    f_use = _f_factor(f_l0_c[p*N_ROWS:(p+1)*N_ROWS], nlms, nlms_gain)
                    e1_weights[c][p] += _lr1 * np.outer(dc, f_use)
                    e1_weights[c][p]  = np.clip(e1_weights[c][p], -e_clip, e_clip)
        n_moves += 1
        if not freeze_e1:
            f_l1 = forward_l1_one(l1_weights, f_l0_float, e1_weights=e1_weights)
        if not (freeze_e1 and freeze_e2):
            f_l2 = forward_l2_one(l2_weights, f_l1, e2_weights=e2_weights)
        f_l3 = forward_l3_one(l3_weights, e3_weights, f_l2)
    return n_moves


# ── PURE BACKPROP through the chip-factored net — the capability oracle ──────────
# Same forward pass and chip factoring as every other rule, but the backward signal
# is the TRUE gradient: dense softmax residual at the readout, transported through the
# ACTUAL W.T chain (current W via --fold), with REAL magnitudes and the real leaky-σ'
# Jacobian — NO sign(), NO rms-renorm, NO gate-renorm.  Deliberately breaks the few-
# bit-boss hardware constraint; its job is to prove the fold pipeline CAN reach the
# 82% backprop ceiling.  We then ablate ingredients back toward the cheap boss.
def settle_and_update_backprop(l1_weights, e1_weights, l2_weights, e2_weights,
                               l3_weights, e3_weights, f_l0_float, label, W_f, b_f,
                               n_settle, e_clip, nlms, nlms_gain,
                               freeze_e1, freeze_e2, freeze_e3, hard=False,
                               lr_layers=None, stop_on_correct=True, sign_grad=False,
                               clf_lr=0.0, f_l0_u8=None):
    _lr3 = lr_layers[0] if lr_layers else BOSS_LR
    _lr2 = lr_layers[1] if lr_layers else BOSS_LR
    _lr1 = lr_layers[2] if lr_layers else BOSS_LR
    # sign_grad → signSGD: unit-magnitude per-weight update (scale-invariant, so a single
    # LR works across all layers; = the torch --signsgd proof, and the cheap sign-delta the
    # boss broadcasts). Momentum then rebuilds effective magnitude from the sign stream.
    _outer = (lambda a, b: np.sign(np.outer(a, b))) if sign_grad else np.outer
    if f_l0_u8 is None:
        f_l0_u8 = float_to_act_code(f_l0_float.reshape(1, -1))[0]
    f_l1 = forward_l1_one(l1_weights, f_l0_float, e1_weights=e1_weights, f_l0_u8=f_l0_u8)
    f_l2 = forward_l2_one(l2_weights, f_l1, e2_weights=e2_weights)
    f_l3 = forward_l3_one(l3_weights, e3_weights, f_l2)
    n_moves = 0
    for _it in range(n_settle):
        scores = W_f @ f_l3 + b_f
        pred   = int(np.argmax(scores))
        # stop_on_correct=True  → perceptron-style: update only on misclassification.
        # stop_on_correct=False → literal CE backprop: keep pushing margin on correct
        # samples too (p − onehot ≠ 0), which is what the torch rig does.
        if pred == label and stop_on_correct:
            break
        # ── Output error s = onehot(y) − p  (ascent toward correct; E += lr·δ⊗x) ──
        if hard:
            s = np.zeros(N_CLASSES, dtype=np.float64)
            s[label] = 1.0; s[pred] -= 1.0
        elif ERR_MODE == 'mse':
            # lstsq scores ARE p_hat -> the residual IS the error. No exp, no temperature.
            s  = -scores.astype(np.float64); s[label] += 1.0    # onehot(y) − p_hat
        else:
            zs = scores * LOGIT_SCALE
            z  = zs - zs.max(); pr = np.exp(z); pr /= pr.sum()
            s  = -pr.astype(np.float64); s[label] += 1.0        # onehot(y) − softmax

        # ── Rung 4: boss_h severity gate ──────────────────────────────────────
        # The old rule can only broadcast a 3-bit severity scalar, and SKIPS the sample when
        # it quantises to 0. bh=0 means a LOW MARGIN => the sample is ON THE DECISION BOUNDARY,
        # i.e. the most informative one we have. --bh_leak L applies a small step L*0.25 there
        # instead of discarding it.
        if BH_GATE and pred != label:
            _mg = float(scores[pred]) - float(scores[label])
            _dn = max(4.0 * float(np.std(scores)), 1e-6) if BH_STD else \
                  max(float(scores.max() - scores.min()), 1e-6)
            _bh = min(int(7.0 * BH_GAIN * _mg / _dn), BOSS_H_MAX)
            if _bh > 0:
                _sev = (_bh + 1) / 4.0
            elif BH_LEAK > 0.0:
                _sev = BH_LEAK * 0.25          # leaky gate — keep the boundary samples
            else:
                break                          # hard skip (current behaviour)
            s = s * _sev

        # ── Readout co-adaptation (--clf_sgd) ─────────────────────────────────
        # Our readout is refit once per EPOCH and then frozen while the features it scores
        # keep training — so the teaching signal goes stale within the epoch, and the longer
        # the epoch the worse it rots (full 124,800-sample epochs DECLINE: 53.9 → 42.6).
        # torch trains its classifier JOINTLY, every step, and never sees this. Let the
        # readout follow the features online. It is 26x32 = 832 weights in the digital
        # readout — trivially buildable; this is not a hardware concession.
        if clf_lr > 0.0:
            W_f += clf_lr * np.outer(s, f_l3)
            b_f += clf_lr * s

        # ── L3: grad wrt f_l3 → pre-activation grad d3 ─────────────────────────
        d3 = _constrain_delta((W_f.T @ s) * _leaky_deriv(f_l3), 'L3')   # (128,)
        if not freeze_e3:
            for g in range(N_L3_CHIPS):
                dg = d3[g*N_ROWS:(g+1)*N_ROWS]
                l2a, l2b = L3_ROUTING[g]
                f_l2_g = np.concatenate([f_l2[l2a*N_ROWS:(l2a+1)*N_ROWS],
                                         f_l2[l2b*N_ROWS:(l2b+1)*N_ROWS]])
                for p in range(N_PAGES_L3):
                    f_use = _f_factor(f_l2_g[p*N_ROWS:(p+1)*N_ROWS].astype(np.float64),
                                      nlms, nlms_gain)
                    e3_weights[g][p] += _lr3 * _outer(dg, f_use)
                    e3_weights[g][p]  = np.clip(e3_weights[g][p], -e_clip, e_clip)

        # ── Transport d3 through the ACTUAL W3.T chain → grad wrt f_l2 → d2 ─────
        g2 = np.zeros(N_L2_FEATS, dtype=np.float64)
        for g in range(N_L3_CHIPS):
            dg = d3[g*N_ROWS:(g+1)*N_ROWS]
            for p, l2_idx in enumerate(L3_ROUTING[g]):
                g2[l2_idx*N_ROWS:(l2_idx+1)*N_ROWS] += l3_weights[g][p].T @ dg
        d2 = _constrain_delta(g2 * _leaky_deriv(f_l2), 'L2')            # (64,)
        if not freeze_e2:
            for g2i in range(N_L2_CHIPS):
                dg2   = d2[g2i*N_ROWS:(g2i+1)*N_ROWS]
                start = g2i * N_L1_PER_L2 * N_ROWS
                f_l1_g2 = f_l1[start:start + N_L1_PER_L2 * N_ROWS].astype(np.float64)
                for p in range(N_PAGES):
                    f_use = _f_factor(f_l1_g2[p*N_ROWS:(p+1)*N_ROWS], nlms, nlms_gain)
                    e2_weights[g2i][p] += _lr2 * _outer(dg2, f_use)
                    e2_weights[g2i][p]  = np.clip(e2_weights[g2i][p], -e_clip, e_clip)

        # ── Transport d2 through the ACTUAL W2.T chain → grad wrt f_l1 → d1 ─────
        g1 = np.zeros(N_L1_FEATS, dtype=np.float64)
        for g2i in range(N_L2_CHIPS):
            dg2 = d2[g2i*N_ROWS:(g2i+1)*N_ROWS]
            for p in range(N_PAGES):
                c = g2i * N_L1_PER_L2 + p
                g1[c*N_ROWS:(c+1)*N_ROWS] = l2_weights[g2i][p].T @ dg2
        d1 = _constrain_delta(g1 * _leaky_deriv(f_l1), 'L1')            # (192,)
        if not freeze_e1:
            for c in range(N_L1_CHIPS):
                dc     = d1[c*N_ROWS:(c+1)*N_ROWS]
                f_l0_c = f_l0_u8[c*N_FEATS_PER_L1:(c+1)*N_FEATS_PER_L1].astype(np.float64)
                for p in range(N_PAGES):
                    f_use = _f_factor(f_l0_c[p*N_ROWS:(p+1)*N_ROWS], nlms, nlms_gain)
                    e1_weights[c][p] += _lr1 * _outer(dc, f_use)
                    e1_weights[c][p]  = np.clip(e1_weights[c][p], -e_clip, e_clip)

        n_moves += 1
        # PERF: on the LAST settle step the recomputed forwards are never used — skip them.
        # (With the common --n_settle 1 this was ~half of all forward work.)
        if _it == n_settle - 1:
            break
        if not freeze_e1:
            f_l1 = forward_l1_one(l1_weights, f_l0_float, e1_weights=e1_weights,
                                  f_l0_u8=f_l0_u8)
        if not (freeze_e1 and freeze_e2):
            f_l2 = forward_l2_one(l2_weights, f_l1, e2_weights=e2_weights)
        f_l3 = forward_l3_one(l3_weights, e3_weights, f_l2)
    return n_moves


def settle_and_update_backprop_batch(l1_weights, e1_weights, l2_weights, e2_weights,
                                    l3_weights, e3_weights, F_l0, F_l0_u8, labels,
                                    W_f, b_f, lr3, lr2, lr1,
                                    freeze_e1, freeze_e2, freeze_e3,
                                    stop_on_correct, bh_gate, bh_leak):
    """VECTORISED fold-batch update — the same rule, one batch instead of B python passes.

    Profiling showed the per-sample loop was ~95% NumPy call overhead on tiny 16x16 ops (the
    actual MACs were 5% of runtime). With the weights FIXED for the whole fold (which the fold
    already assumes), all B samples can go through as batched matmuls:
        E[c][p] += (lr/B) * (D_c.T @ F_page)      <- ONE (16,B)@(B,16) matmul
    instead of B separate np.outer calls. Same arithmetic, ~B fewer python/NumPy dispatches.

    ⚠ REQUIRES frozen-forward semantics (E cannot feed the forward if all B samples are computed
    at once), so THE BATCHED PATH INHERENTLY CARRIES THE ~1pp OF --frozen_fwd. See PERFORMANCE.md.
    n_settle must be 1 (a multi-step settle is inherently sequential).
    """
    B = len(labels)
    F1 = forward_l1_batch(l1_weights, F_l0)
    F2 = forward_l2_batch(l2_weights, F1)
    F3 = forward_l3_batch(l3_weights, None, F2)

    scores = F3.astype(np.float64) @ W_f.T + b_f          # (B, 26)
    pred = np.argmax(scores, axis=1)
    z = scores - scores.max(axis=1, keepdims=True)
    pr = np.exp(z); pr /= pr.sum(axis=1, keepdims=True)
    S = -pr; S[np.arange(B), labels] += 1.0               # onehot − softmax
    wrong = pred != labels

    if stop_on_correct:
        S[~wrong] = 0.0                                   # perceptron gate, vectorised
    if bh_gate:
        mg = scores[np.arange(B), pred] - scores[np.arange(B), labels]
        dn = np.maximum(scores.max(1) - scores.min(1), 1e-6)
        bh = np.minimum((7.0 * BH_GAIN * mg / dn).astype(int), BOSS_H_MAX)
        sev = np.where(bh > 0, (bh + 1) / 4.0, bh_leak * 0.25)
        S *= np.where(wrong, sev, 1.0)[:, None]

    n_moves = int(wrong.sum()) if stop_on_correct else B

    D3 = _constrain_delta_batch((S @ W_f) * _leaky_deriv(F3), 'L3')      # (B, 128)
    if not freeze_e3:
        for g in range(N_L3_CHIPS):
            Dg = D3[:, g*N_ROWS:(g+1)*N_ROWS]
            l2a, l2b = L3_ROUTING[g]
            for p, li in enumerate((l2a, l2b)):
                Fp = F2[:, li*N_ROWS:(li+1)*N_ROWS].astype(np.float64)
                e3_weights[g][p] += lr3 * (Dg.T @ Fp)

    G2 = np.zeros((B, N_L2_FEATS))
    for g in range(N_L3_CHIPS):
        Dg = D3[:, g*N_ROWS:(g+1)*N_ROWS]
        for p, li in enumerate(L3_ROUTING[g]):
            G2[:, li*N_ROWS:(li+1)*N_ROWS] += Dg @ l3_weights[g][p]
    D2 = _constrain_delta_batch(G2 * _leaky_deriv(F2), 'L2')             # (B, 64)
    if not freeze_e2:
        for g in range(N_L2_CHIPS):
            Dg = D2[:, g*N_ROWS:(g+1)*N_ROWS]
            st = g * N_L1_PER_L2 * N_ROWS
            for p in range(N_PAGES):
                Fp = F1[:, st + p*N_ROWS: st + (p+1)*N_ROWS].astype(np.float64)
                e2_weights[g][p] += lr2 * (Dg.T @ Fp)

    G1 = np.zeros((B, N_L1_FEATS))
    for g in range(N_L2_CHIPS):
        Dg = D2[:, g*N_ROWS:(g+1)*N_ROWS]
        for p in range(N_PAGES):
            c = g * N_L1_PER_L2 + p
            G1[:, c*N_ROWS:(c+1)*N_ROWS] = Dg @ l2_weights[g][p]
    D1 = _constrain_delta_batch(G1 * _leaky_deriv(F1), 'L1')             # (B, 192)
    if not freeze_e1:
        Fu = F_l0_u8.astype(np.float64)
        for c in range(N_L1_CHIPS):
            Dg = D1[:, c*N_ROWS:(c+1)*N_ROWS]
            st = c * N_FEATS_PER_L1
            for p in range(N_PAGES):
                Fp = Fu[:, st + p*N_COLS: st + (p+1)*N_COLS]
                e1_weights[c][p] += lr1 * (Dg.T @ Fp)
    return n_moves


def _constrain_delta_batch(D, layer):
    """Row-wise version of _constrain_delta — each SAMPLE gets its own quantiser scale/severity,
    exactly as the per-sample path does. The AGC uses the batch rms (it is slow by design)."""
    if DELTA_AGC > 0.0:
        r = float(np.sqrt(np.mean(D * D)))
        prev = _AGC_STATE.get(layer)
        _AGC_STATE[layer] = r if prev is None else (1.0 - DELTA_AGC) * prev + DELTA_AGC * r
        D = D * (AGC_TARGET / max(_AGC_STATE[layer], 1e-12))
    if DELTA_RENORM:
        r = np.sqrt(np.mean(D * D, axis=1, keepdims=True))
        D = D / np.maximum(r, 1e-12)
    if DELTA_SIGN:
        return np.sign(D)
    if DELTA_BITS > 0:
        if DELTA_FIXED > 0.0:
            scale = np.full((len(D), 1), DELTA_FIXED)
        else:
            scale = np.maximum(np.abs(D).max(axis=1, keepdims=True), 1e-12)
        if DELTA_BITS == 1:
            return np.sign(D) * scale
        L = 2 ** (DELTA_BITS - 1) - 1
        Q = np.round(D / scale * L) / L * scale
        return np.clip(Q, -scale, scale) if DELTA_FIXED > 0.0 else Q
    return D


# ── W vs E analysis ────────────────────────────────────────────────────────────

def analyze_we(l1_weights, e1_weights, l2_weights, e2_weights,
               l3_weights, e3_weights, label):
    def layer_stats(weights, e_weights, n_chips, n_pages):
        ratios, cosines = [], []
        for c in range(n_chips):
            for p in range(n_pages):
                w = weights[c][p].flatten()
                e = e_weights[c][p].flatten()
                wn = np.linalg.norm(w)
                en = np.linalg.norm(e)
                ratios.append(en / (wn + 1e-10))
                cosines.append(np.dot(w, e) / (wn * en + 1e-10))
        return float(np.mean(ratios)), float(np.mean(cosines))

    r1, c1 = layer_stats(l1_weights, e1_weights, N_L1_CHIPS, N_PAGES)
    r2, c2 = layer_stats(l2_weights, e2_weights, N_L2_CHIPS, N_PAGES)
    r3, c3 = layer_stats(l3_weights, e3_weights, N_L3_CHIPS, N_PAGES_L3)
    print(f"  E vs W ({label}, pre-absorb):"
          f"  L1 ||E||/||W||={r1:.3f} cos={c1:+.3f}"
          f"  |  L2 ||E||/||W||={r2:.3f} cos={c2:+.3f}"
          f"  |  L3 ||E||/||W||={r3:.3f} cos={c3:+.3f}", flush=True)


# ── Parse helpers ──────────────────────────────────────────────────────────────

def parse_warmup(s):
    """Parse '12x2k,4x4k' → [(12, 2000), (4, 4000)]."""
    stages = []
    for part in s.split(','):
        nx, c = part.strip().split('x')
        c = c.strip()
        c = int(float(c[:-1]) * 1000) if c.endswith('k') else int(c)
        stages.append((int(nx), c))
    return stages


def parse_staged(s):
    """Parse '1x4,2x4,3x2,1x1' → [(1,4),(2,4),(3,2),(1,1)]  (phase, n_epochs)."""
    stages = []
    for part in s.split(','):
        ph, n = part.strip().split('x')
        ph, n = int(ph), int(n)
        assert ph in (0, 1, 2, 3, 5), f"stage phase must be 0, 1, 2, 3 or 5 (got {ph})"
        assert n >= 1
        stages.append((ph, n))
    return stages


PHASE_FREEZE = {   # phase → (freeze_e1, freeze_e2, freeze_e3); phase 0 also freezes these
    0: (True,  True,  True),   # P0: train E_L0A only (freeze_e0=False handled separately)
    1: (True,  True,  False),
    2: (True,  False, True),
    3: (False, True,  True),
    5: (False, False, False),
}
PHASE_NAME = {0: 'E_L0A', 1: 'E3', 2: 'E2', 3: 'E1', 5: 'E3+E2+E1'}


# ── Main simulation ────────────────────────────────────────────────────────────

def run_sim(stage_list, n_settle, boss_h_fixed, lr_slow, e_clip, report_secs,
            diag_l2=False,
            nlms=False, nlms_gain=1.0,
            local_delta=False, beta=0.5,
            clf_refresh=0, clf_refresh_n=20000,
            avg_bp=False, bp_ema=0.0,
            stage_patience=0, stage_rollback=False, l0a=False, rand_init=False, dfa=False,
            dfa_hard=False, fix_readout=False, auto_lr=False, fold=False,
            backprop=False, bp_hard=False, bp_full=False, fold_momentum=0.0,
            sign_grad=False, fold_every=1, fold_mean=True, sign_at_fold=False,
            clf_lr=0.0, fast_batch=False):
    """stage_list: [(phase, n_epochs, chunk), ...] executed sequentially.
    Layers not named by a stage's phase are fully frozen during that stage
    (E never updated, W never absorbed) — generational semantics.

    stage_rollback: snapshot all weights at each stage's best decision-eval
    epoch and restore that snapshot when the stage ends, so the layer freezes
    at its peak rather than 1-2 epochs past it.  Stage decisions (best /
    patience / rollback) use a 1,040-sample stratified eval (40 per class);
    the historical 208-sample eval is still printed for comparability."""
    print(f"\n{'='*72}")
    print(f"PCN 3-layer EMNIST — BIG topology + THE SPEC  [pcn_bigspec]")
    print(f"  Architecture: {N_L0_FEATS}→{N_L1_FEATS}→{N_L2_FEATS}→{N_L3_FEATS}"
          f" ({N_L1_CHIPS}/{N_L2_CHIPS}/{N_L3_CHIPS} chips, OVERLAPPING L3 routing)→clf({N_CLASSES})")
    print(f"  Stages: " + ' → '.join(f"P{ph}[{PHASE_NAME[ph]}]×{n}(c{c//1000}k)"
                                     for ph, n, c in stage_list))
    print(f"  n_settle: {n_settle}  boss_h: "
          f"{'variable' if boss_h_fixed is None else boss_h_fixed}"
          f"  lr_slow: {lr_slow}  e_clip: {e_clip:.4f}  leaky_alpha: {LEAKY_ALPHA}"
          f"  BOSS_LR: {BOSS_LR}")
    print(f"  Ideas:  nlms={'ON gain=%.1f' % nlms_gain if nlms else 'off'}"
          f"  local_delta={'ON beta=%.2f' % beta if local_delta else 'off'}"
          f"  clf_refresh={'every %d (n=%d)' % (clf_refresh, clf_refresh_n) if clf_refresh else 'off'}"
          f"  avg_bp={'ON' if avg_bp else 'off'}"
          f"  bp_ema={bp_ema if bp_ema else 'off'}"
          f"  stage_patience={stage_patience if stage_patience else 'off'}"
          f"  stage_rollback={'ON' if stage_rollback else 'off'}")
    if BH_DEADZONE or BH_MAGLEVELS or BH_STD or RELU_JAC_DIR or RELU_JAC_INV:
        print(f"  boss_h: deadzone={BH_DEADZONE if BH_DEADZONE else 'off'}"
              f"  maglevels={'ON' if BH_MAGLEVELS else 'off'}"
              f"  std_norm={'ON' if BH_STD else 'off'}"
              f"  relu_jac={'DIR' if RELU_JAC_DIR else ('INV' if RELU_JAC_INV else 'off')}")
    print(f"{'='*72}\n", flush=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    F_l0_train, F_l0_test, y_train, y_test = get_emnist_l0_features()
    # PERF: the training set's act codes are FIXED — computing them per sample cost 3.4s/3k.
    F_l0_codes_train = float_to_act_code(F_l0_train)
    N_TRAIN = len(y_train)
    N_TEST  = len(y_test)

    _eval_idx = np.concatenate([np.where(y_test == c)[0][:8] for c in range(N_CLASSES)])
    F_l0_eval  = F_l0_test[_eval_idx]
    y_eval     = y_test[_eval_idx]
    # Larger stratified set for stage decisions (best / patience / rollback):
    # 40 per class = 1,040 samples, ~±1.5pp noise vs ~±3pp on the 208 set.
    _dec_idx  = np.concatenate([np.where(y_test == c)[0][:40] for c in range(N_CLASSES)])
    F_l0_dec  = F_l0_test[_dec_idx]
    y_dec     = y_test[_dec_idx]
    print(f"  Eval sets: {len(y_eval)} (8/class, reporting) + {len(y_dec)} "
          f"(40/class, stage decisions)", flush=True)

    # ── Weights ───────────────────────────────────────────────────────────────
    if rand_init:
        # RANDOM init of L1/L2 (L3 is already random-ortho) — the acid test: can the
        # rule learn from scratch, with NO GHA/PCA starting basis?  Random-ortho pages
        # at the GHA weight scale (RMS≈0.25) so the forward is well-conditioned.
        _rng = np.random.default_rng(SEED_INIT)
        def _rand_chip(n_chips, n_pages):
            w = []
            for _ in range(n_chips):
                pages = []
                for _ in range(n_pages):
                    Q, _r = np.linalg.qr(_rng.standard_normal((N_ROWS, N_COLS)))
                    pages.append((Q * 0.25).astype(np.float32))
                w.append(pages)
            return w
        print("RANDOM init (--rand_init): L1/L2/L3 all random, NO GHA basis", flush=True)
        l1_weights = _rand_chip(N_L1_CHIPS, N_PAGES)
        l2_weights = _rand_chip(N_L2_CHIPS, N_PAGES)
        l3_weights = init_l3_weights_random_ortho(seed=2)
    else:
        print("Loading weights ...", flush=True)
        l1_weights = load_l1_weights()
        l2_weights = load_l2_weights()
        l3_weights = init_l3_weights_random_ortho(seed=2)
    print(f"  L3 weights: {N_L3_CHIPS} chips × {N_PAGES_L3} pages (random ortho, seed=2)",
          flush=True)

    e1_weights = init_e_weights(l1_weights)
    e2_weights = init_e_weights(l2_weights)
    e3_weights = init_e_weights(l3_weights)

    # ── DFA fixed random feedback matrices B_ℓ (n_ℓ × N_CLASSES), unit-norm rows ──
    B1 = B2 = B3 = None
    if dfa:
        _rb = np.random.default_rng(11)
        def _mkB(n):
            M = _rb.standard_normal((n, N_CLASSES))
            M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)   # unit-norm per neuron
            return M
        B1, B2, B3 = _mkB(N_L1_FEATS), _mkB(N_L2_FEATS), _mkB(N_L3_FEATS)
        print(f"DFA: fixed random feedback B1{B1.shape} B2{B2.shape} B3{B3.shape} "
              f"(broadcast output error to all layers)", flush=True)

    # ── L0_A trainable input transform (optional) ───────────────────────────────
    global _L0A_ON, _W_L0A, _E_L0A
    if l0a:
        _sub = np.random.default_rng(0).choice(N_TRAIN, size=min(20000, N_TRAIN),
                                                replace=False)
        _W_L0A  = build_l0a_weights(F_l0_train[_sub], l1_weights)
        _E_L0A  = init_e_weights(_W_L0A)
        _L0A_ON = True
    else:
        _L0A_ON, _W_L0A, _E_L0A = False, None, None

    # Backprojection shadows (target-network style).  bp_ema=0 → use live W.
    if bp_ema > 0.0:
        w3_bp = [[pg.copy() for pg in ch] for ch in l3_weights]
        w2_bp = [[pg.copy() for pg in ch] for ch in l2_weights]
    else:
        w3_bp = w2_bp = None

    _fold_note = []        # one-shot guard for the fold_mean diagnostic print

    # sign_at_fold: E accumulates the RAW gradient over the minibatch, and the SIGN is
    # taken once, at the fold — i.e. sign(mean grad), which is what torch's signSGD does
    # and what reached 81%.  Contrast --sign_grad, which signs every per-sample delta and
    # then averages: sign-inconsistent weights average toward ZERO and are under-stepped,
    # precisely the weights momentum exists to rescue.  Both are hardware-buildable — this
    # one is "accumulate in E, threshold on the W write" (a comparator), arguably the more
    # natural analog primitive than a per-sample few-bit broadcast.
    def _quant_e(e_weights, n_chips, n_pages, bits):
        """Rung 5: the E integrator is a real capacitor/register with finite resolution."""
        L = 2 ** (bits - 1) - 1
        for c in range(n_chips):
            for p in range(n_pages):
                E = e_weights[c][p]
                sc = float(np.max(np.abs(E)))
                if sc <= 1e-12:
                    continue
                if bits == 1:
                    E[...] = np.sign(E) * sc
                else:
                    E[...] = np.round(E / sc * L) / L * sc

    def _sign_e(e_weights, n_chips, n_pages, step):
        for c in range(n_chips):
            for p in range(n_pages):
                np.sign(e_weights[c][p], out=e_weights[c][p])
                e_weights[c][p] *= step

    # Momentum velocity buffers for the fold (lever 3): v = μ·v + E; W += v; E = 0.
    vel1 = init_e_weights(l1_weights)
    vel2 = init_e_weights(l2_weights)
    vel3 = init_e_weights(l3_weights)

    def _fold_momentum(weights, e_weights, vel, n_chips, n_pages, mu):
        for c in range(n_chips):
            for p in range(n_pages):
                vel[c][p] = mu * vel[c][p] + e_weights[c][p]
                weights[c][p] += vel[c][p]
                weights[c][p]  = _wclip(weights[c][p])
                e_weights[c][p][...] = 0.0

    def _update_shadows():
        if bp_ema <= 0.0:
            return
        for g in range(N_L3_CHIPS):
            for p in range(N_PAGES_L3):
                w3_bp[g][p] = (1.0 - bp_ema) * w3_bp[g][p] + bp_ema * l3_weights[g][p]
        for g in range(N_L2_CHIPS):
            for p in range(N_PAGES):
                w2_bp[g][p] = (1.0 - bp_ema) * w2_bp[g][p] + bp_ema * l2_weights[g][p]

    _all_mats = (l1_weights, l2_weights, l3_weights,
                 e1_weights, e2_weights, e3_weights)

    def _snap_weights():
        return [[[pg.copy() for pg in ch] for ch in mats] for mats in _all_mats]

    def _restore_weights(snap):
        """In-place restore so existing references (incl. bp shadows' sources)
        stay valid."""
        for mats, saved in zip(_all_mats, snap):
            for ci in range(len(mats)):
                for pi in range(len(mats[ci])):
                    mats[ci][pi][...] = saved[ci][pi]

    # ── GHA baseline (W-only) ─────────────────────────────────────────────────
    print("\nComputing GHA baseline (W-only, no E updates) ...", flush=True)
    F_l1_base = forward_l1_batch(l1_weights, F_l0_train)
    F_l2_base = forward_l2_batch(l2_weights, F_l1_base)
    F_l3_base = forward_l3_batch(l3_weights, None, F_l2_base)
    W_f_base, b_f_base = fit_clf(F_l3_base, y_train)
    F_l1_te = forward_l1_batch(l1_weights, F_l0_eval)
    F_l2_te = forward_l2_batch(l2_weights, F_l1_te)
    F_l3_te = forward_l3_batch(l3_weights, None, F_l2_te)
    acc_gha = (float(np.mean(np.argmax(
        F_l3_te.astype(np.float64) @ W_f_base.T + b_f_base, axis=1) == y_eval))
        if W_f_base is not None else 0.0)
    print(f"  GHA baseline (208-eval): {acc_gha*100:.2f}%  [emnist gha-only ref: 64.03%]",
          flush=True)
    if diag_l2:
        W_f_l2_base, b_f_l2_base = fit_clf(F_l2_base, y_train)
        acc_gha_l2 = (float(np.mean(np.argmax(
            F_l2_te.astype(np.float64) @ W_f_l2_base.T + b_f_l2_base, axis=1) == y_eval))
            if W_f_l2_base is not None else 0.0)
        print(f"  GHA baseline L2-direct (208-eval): {acc_gha_l2*100:.2f}%", flush=True)

    acc_log   = []
    t_start   = time.time()
    t_report  = t_start
    rng       = np.random.default_rng(SEED_DATA)
    total_samples = sum(n * c for _, n, c in stage_list)
    done_prev = 0

    # Mutable classifier state (rebound by _mid_refresh)
    W_f = b_f = None
    aux2 = aux1 = None
    phase = ep_chunk = None
    F_l1_tr = F_l2_tr = None

    def _mid_refresh():
        """Refit W_f (and aux classifiers) from a random subsample, using the
        current live W+E state for whichever layers are unfrozen."""
        nonlocal W_f, b_f, aux2, aux1
        sub = rng.choice(N_TRAIN, size=min(clf_refresh_n, N_TRAIN), replace=False)
        if phase == 1:
            F1s, F2s = F_l1_tr[sub], F_l2_tr[sub]
        elif phase == 2:
            F1s = F_l1_tr[sub]
            F2s = forward_l2_batch(l2_weights, F1s, e2_weights=e2_weights)
        else:   # phase 3 or 5: E1 live — recompute from L0
            F1s = forward_l1_batch(l1_weights, F_l0_train[sub], e1_weights=e1_weights)
            F2s = forward_l2_batch(l2_weights, F1s, e2_weights=e2_weights)
        F3s = forward_l3_batch(l3_weights, e3_weights, F2s)
        W_f, b_f = fit_clf(F3s, y_train[sub])
        if aux2 is not None:
            aux2 = fit_clf(F2s, y_train[sub])
        if aux1 is not None:
            aux1 = fit_clf(F1s, y_train[sub])

    # ── Stage loop ────────────────────────────────────────────────────────────
    prev_stage_best = 0.0   # entry baseline for rollback: a stage that never
                            # beats its incoming state is declined entirely
    W_f = b_f = None        # readout; if --fix_readout, fit once then reuse
    for st_idx, (st_phase, st_eps, st_chunk) in enumerate(stage_list):
        phase = st_phase
        fz_e1, fz_e2, fz_e3 = PHASE_FREEZE[phase]
        fz_e0 = (phase != 0)   # P0 trains E_L0A; all other phases freeze it
        lr_base = None         # auto_lr: per-layer η, calibrated at stage start
        stage_lbl  = f"S{st_idx+1}"
        stage_bad  = 0
        best_ep    = 0
        if stage_rollback:
            stage_best = prev_stage_best
            best_snap  = _snap_weights()   # entry state as candidate 0
        else:
            stage_best = -1.0
            best_snap  = None
        print(f"\n{'█'*72}")
        print(f"  STAGE {st_idx+1}/{len(stage_list)}  —  phase P{phase} "
              f"(train {PHASE_NAME[phase]}; "
              f"frozen: {', '.join(n for n, fz in zip(['E1','E2','E3'], [fz_e1, fz_e2, fz_e3]) if fz) or 'none'})"
              f"  ×{st_eps} epochs, chunk {st_chunk:,}", flush=True)

        for ep in range(1, st_eps + 1):
            ep_chunk = st_chunk
            print(f"\n{'─'*72}")
            print(f"  {stage_lbl} P{phase} epoch {ep}/{st_eps}", flush=True)

            # Recompute caches (frozen-layer features are stable within a stage,
            # but W may have absorbed at the previous epoch end)
            F_l1_tr = forward_l1_batch(l1_weights, F_l0_train, e1_weights=e1_weights)
            F_l2_tr = forward_l2_batch(l2_weights, F_l1_tr,    e2_weights=e2_weights)

            print(f"  Fitting classifier ({N_TRAIN:,} samples) ...", end=' ', flush=True)
            F_l3_tr  = forward_l3_batch(l3_weights, e3_weights, F_l2_tr)
            if fix_readout and W_f is not None:
                pass   # keep the readout fixed after the first fit (stable target)
            else:
                W_f, b_f = fit_clf(F_l3_tr, y_train)
                W_f = np.ascontiguousarray(W_f, dtype=np.float64)
                b_f = np.ascontiguousarray(b_f, dtype=np.float64)
            acc_tr   = float(np.mean(
                np.argmax(F_l3_tr.astype(np.float64) @ W_f.T + b_f, axis=1) == y_train))
            print(f"acc(tr)={acc_tr*100:.1f}%", flush=True)

            # ── auto_lr: per-layer η from measured rms (one settle step moves each
            #    layer's output by AUTO_FRAC×rms). Calibrated once per stage. ──────
            if auto_lr and lr_base is None:
                cal = rng.choice(N_TRAIN, size=min(2000, N_TRAIN), replace=False)
                F3c = forward_l3_batch(l3_weights, e3_weights, F_l2_tr[cal])
                scc = F3c.astype(np.float64) @ W_f.T + b_f
                prc = np.argmax(scc, axis=1); wr = prc != y_train[cal]
                if np.any(wr):
                    mg = scc[wr, prc[wr]] - scc[wr, y_train[cal][wr]]
                    sp = np.maximum(scc[wr].max(1) - scc[wr].min(1), 1e-6)
                    bh = np.clip((7.0 * mg / sp).astype(int), 0, BOSS_H_MAX)
                    st_ = ((bh + 1) / 4.0)[bh > 0]
                    mean_step = float(st_.mean()) if len(st_) else 1.0
                else:
                    mean_step = 1.0
                rms_delta = mean_step * DELTA
                _rms = lambda A: float(np.sqrt(np.mean(np.asarray(A, np.float64) ** 2)))
                rms_f0 = _rms(float_to_act_code(F_l0_train[cal]))
                rms_f1 = _rms(F_l1_tr[cal]); rms_f2 = _rms(F_l2_tr[cal]); rms_r3 = _rms(F3c)
                eta3 = AUTO_FRAC * rms_r3 / max(rms_delta*N_COLS*rms_f2*N_PAGES_L3/64.0, 1e-12)
                eta2 = AUTO_FRAC * rms_f2 / max(rms_delta*N_COLS*rms_f1*N_PAGES   /64.0, 1e-12)
                eta1 = AUTO_FRAC * rms_f1 / max(rms_delta*N_COLS*rms_f0*N_PAGES   /64.0, 1e-12)
                eta3 *= LR_MULT[0]; eta2 *= LR_MULT[1]; eta1 *= LR_MULT[2]
                lr_base = (eta3, eta2, eta1)
                print(f"  auto_lr η(L3/L2/L1)={eta3:.2e}/{eta2:.2e}/{eta1:.2e}"
                      f"  (lr_mult {LR_MULT})", flush=True)
            if FROZEN_FWD:
                refresh_wq(l1_weights, l2_weights, l3_weights)
            lr_layers_eff = lr_base if (auto_lr and lr_base) else None

            # ── fold_mean: E accumulated over a fold_every minibatch is a SUM of the
            #    per-sample deltas, so its magnitude scales with the batch — coupling the
            #    step size to the batch size (every (lr, fold_every) pair then needs its own
            #    retune).  Scaling the per-sample lr by 1/fold_every makes E the MEAN of the
            #    deltas, exactly as torch's batch-mean gradient, so lr and batch decouple.
            if fold and fold_mean and fold_every > 1:
                _b = lr_layers_eff if lr_layers_eff else (BOSS_LR, BOSS_LR, BOSS_LR)
                lr_layers_eff = tuple(x / float(fold_every) for x in _b)
                if not _fold_note:
                    _fold_note.append(1)
                    # With sign deltas each per-sample |ΔE| is exactly lr/B, so |E| at the
                    # fold is at most lr — and the momentum buffer then runs it up to
                    # lr/(1−μ).  e_clip is applied to E *inside* the sample loop, so if it
                    # is smaller than lr it silently clips the gradient and destroys the very
                    # magnitudes momentum exists to rebuild.  Say so loudly.
                    _lr_max = max(lr_layers_eff) * fold_every
                    _eff    = _lr_max / max(1.0 - fold_momentum, 1e-9)
                    print(f"  fold_mean: lr/{fold_every} per sample → |E|≤{_lr_max:.2e} "
                          f"per fold; momentum μ={fold_momentum} → effective step "
                          f"≈{_eff:.2e}", flush=True)
                    if _lr_max > e_clip:
                        print(f"  !! e_clip={e_clip:.3g} < per-fold |E|={_lr_max:.2e}: "
                              f"e_clip is acting as a GRADIENT clip. Raise --e_clip.",
                              flush=True)

            aux2 = aux1 = aux0 = None
            if phase == 0 and l0a:
                F_l0a_tr = forward_l0a_batch(F_l0_train)
                aux0 = fit_clf(F_l0a_tr, y_train)
                acc_a0 = float(np.mean(np.argmax(
                    F_l0a_tr.astype(np.float64) @ aux0[0].T + aux0[1], axis=1) == y_train))
                print(f"  Aux L0_A classifier: acc(tr)={acc_a0*100:.1f}%", flush=True)
            if local_delta and phase in (2, 5):
                aux2 = fit_clf(F_l2_tr, y_train)
                acc_a2 = float(np.mean(np.argmax(
                    F_l2_tr.astype(np.float64) @ aux2[0].T + aux2[1], axis=1) == y_train))
                print(f"  Aux L2 classifier: acc(tr)={acc_a2*100:.1f}%", flush=True)
            if local_delta and phase in (3, 5):
                aux1 = fit_clf(F_l1_tr, y_train)
                acc_a1 = float(np.mean(np.argmax(
                    F_l1_tr.astype(np.float64) @ aux1[0].T + aux1[1], axis=1) == y_train))
                print(f"  Aux L1 classifier: acc(tr)={acc_a1*100:.1f}%", flush=True)

            perm         = rng.permutation(N_TRAIN)[:ep_chunk]
            n_moves_pass = 0
            # ── FAST BATCHED PATH (--fast_batch) ──────────────────────────────
            # One fold-batch at a time, all matmuls. Same rule, ~B fewer NumPy dispatches.
            # ⚠ Inherently carries the ~1pp of --frozen_fwd (E cannot feed the forward when all
            # B samples are computed at once). SWEEPS ONLY — see PERFORMANCE.md.
            if fast_batch:
                _lr3, _lr2, _lr1 = (lr_layers_eff if lr_layers_eff
                                    else (BOSS_LR, BOSS_LR, BOSS_LR))
                for _b0 in range(0, len(perm) - fold_every + 1, fold_every):
                    sl = perm[_b0:_b0 + fold_every]
                    n_moves_pass += settle_and_update_backprop_batch(
                        l1_weights, e1_weights, l2_weights, e2_weights,
                        l3_weights, e3_weights,
                        F_l0_train[sl], F_l0_codes_train[sl], y_train[sl],
                        W_f, b_f, _lr3, _lr2, _lr1,
                        fz_e1, fz_e2, fz_e3,
                        stop_on_correct=not bp_full,
                        bh_gate=BH_GATE, bh_leak=BH_LEAK)
                    # ORDER MATTERS and must match the per-sample path: QUANTISE E, THEN sign.
                    # Signing first makes every entry ±BOSS_LR, which turns the E quantiser into a
                    # NO-OP — and, worse, stops it zeroing the small E entries the real rule drops.
                    # (Getting this backwards made the batched path score 3pp HIGHER than the
                    # per-sample path it is supposed to reproduce. Caught by the A/B.)
                    if E_BITS > 0:
                        if not fz_e3: _quant_e(e3_weights, N_L3_CHIPS, N_PAGES_L3, E_BITS)
                        if not fz_e2: _quant_e(e2_weights, N_L2_CHIPS, N_PAGES,    E_BITS)
                        if not fz_e1: _quant_e(e1_weights, N_L1_CHIPS, N_PAGES,    E_BITS)
                    if sign_at_fold:
                        if not fz_e3: _sign_e(e3_weights, N_L3_CHIPS, N_PAGES_L3, BOSS_LR)
                        if not fz_e2: _sign_e(e2_weights, N_L2_CHIPS, N_PAGES,    BOSS_LR)
                        if not fz_e1: _sign_e(e1_weights, N_L1_CHIPS, N_PAGES,    BOSS_LR)
                    if fold_momentum > 0.0:
                        if not fz_e3: _fold_momentum(l3_weights, e3_weights, vel3, N_L3_CHIPS, N_PAGES_L3, fold_momentum)
                        if not fz_e2: _fold_momentum(l2_weights, e2_weights, vel2, N_L2_CHIPS, N_PAGES,    fold_momentum)
                        if not fz_e1: _fold_momentum(l1_weights, e1_weights, vel1, N_L1_CHIPS, N_PAGES,    fold_momentum)
                    else:
                        if not fz_e3: absorb_e3_only(l3_weights, e3_weights, 1.0, decay_e=1.0)
                        if not fz_e2: absorb_e2_only(l2_weights, e2_weights, 1.0, decay_e=1.0)
                        if not fz_e1: absorb_e1_only(l1_weights, e1_weights, 1.0, decay_e=1.0)
                    refresh_wq(l1_weights, l2_weights, l3_weights)
                perm = perm[:0]     # skip the per-sample loop below

            for i, s in enumerate(perm):
                if backprop:
                    n_moves_pass += settle_and_update_backprop(
                        l1_weights, e1_weights, l2_weights, e2_weights,
                        l3_weights, e3_weights, F_l0_train[s],
                        int(y_train[s]), W_f, b_f,
                        n_settle, e_clip, nlms, nlms_gain,
                        freeze_e1=fz_e1, freeze_e2=fz_e2, freeze_e3=fz_e3,
                        hard=bp_hard, lr_layers=lr_layers_eff,
                        stop_on_correct=not bp_full, sign_grad=sign_grad,
                        clf_lr=clf_lr, f_l0_u8=F_l0_codes_train[s])
                elif dfa:
                    n_moves_pass += settle_and_update_dfa(
                        l1_weights, e1_weights, l2_weights, e2_weights,
                        l3_weights, e3_weights, F_l0_train[s],
                        int(y_train[s]), W_f, b_f, B1, B2, B3,
                        n_settle, e_clip, nlms, nlms_gain,
                        freeze_e1=fz_e1, freeze_e2=fz_e2, freeze_e3=fz_e3,
                        hard=dfa_hard, lr_layers=lr_layers_eff)
                else:
                    n_moves_pass += settle_and_update(
                        l1_weights, e1_weights, l2_weights, e2_weights,
                        l3_weights, e3_weights,
                        F_l0_train[s], F_l1_tr[s], F_l2_tr[s],
                        int(y_train[s]), W_f, b_f,
                        boss_h_fixed, n_settle, e_clip,
                        freeze_e1=fz_e1, freeze_e2=fz_e2, freeze_e3=fz_e3,
                        freeze_e0=fz_e0, aux0=aux0,
                        lr_scale=1.0, lr_layers=lr_layers_eff,
                        nlms=nlms, nlms_gain=nlms_gain,
                        aux2=aux2, aux1=aux1, beta=beta,
                        avg_bp=avg_bp,
                        bp_l3=w3_bp, bp_l2=w2_bp)

                # fold_every>1: accumulate E over a MINIBATCH of samples, fold once —
                # averages the (noisy, batch-1) sign updates before stepping, as torch's
                # batch-256 signSGD does. The boss naturally accumulates a stream of deltas.
                do_fold = fold and ((i + 1) % fold_every == 0)
                if do_fold:
                    if E_BITS > 0:
                        if not fz_e3: _quant_e(e3_weights, N_L3_CHIPS, N_PAGES_L3, E_BITS)
                        if not fz_e2: _quant_e(e2_weights, N_L2_CHIPS, N_PAGES,    E_BITS)
                        if not fz_e1: _quant_e(e1_weights, N_L1_CHIPS, N_PAGES,    E_BITS)
                    # sign(mean grad) — unit step per weight, scale-free (so one LR serves
                    # every layer). Applied before momentum, so the velocity integrates
                    # unit signs exactly as torch's signSGD+momentum does.
                    if sign_at_fold:
                        if not fz_e3: _sign_e(e3_weights, N_L3_CHIPS, N_PAGES_L3, BOSS_LR)
                        if not fz_e2: _sign_e(e2_weights, N_L2_CHIPS, N_PAGES,    BOSS_LR)
                        if not fz_e1: _sign_e(e1_weights, N_L1_CHIPS, N_PAGES,    BOSS_LR)
                    # Immediate fold: W += E, E = 0, so the backward operator (bp aliases
                    # live W when bp_ema=0) is the true transpose of the current forward map
                    # — removes E/W staleness (and the e_clip ceiling). = (mini)batch SGD.
                    # With fold_momentum>0: v=μ·v+E; W+=v (lever 3 — the momentum that lets
                    # high-LR SGD escape the dead init and reach the backprop ceiling).
                    if fold_momentum > 0.0:
                        if not fz_e3: _fold_momentum(l3_weights, e3_weights, vel3, N_L3_CHIPS, N_PAGES_L3, fold_momentum)
                        if not fz_e2: _fold_momentum(l2_weights, e2_weights, vel2, N_L2_CHIPS, N_PAGES,    fold_momentum)
                        if not fz_e1: _fold_momentum(l1_weights, e1_weights, vel1, N_L1_CHIPS, N_PAGES,    fold_momentum)
                    else:
                        if not fz_e3:
                            absorb_e3_only(l3_weights, e3_weights, 1.0, decay_e=1.0)
                        if not fz_e2:
                            absorb_e2_only(l2_weights, e2_weights, 1.0, decay_e=1.0)
                        if not fz_e1:
                            absorb_e1_only(l1_weights, e1_weights, 1.0, decay_e=1.0)
                    if not fz_e0 and l0a:
                        absorb_l0a_only(1.0, decay_e=1.0)

                    if FROZEN_FWD:
                        refresh_wq(l1_weights, l2_weights, l3_weights)

                if clf_refresh and (i + 1) % clf_refresh == 0 and (i + 1) < len(perm):
                    _mid_refresh()

                t_now = time.time()
                if t_now - t_report >= report_secs:
                    elapsed = t_now - t_start
                    done    = done_prev + i + 1
                    eta     = (total_samples - done) / max(done / max(elapsed, 1e-3), 1e-3)
                    acc_q   = eval_acc_l3(l1_weights, e1_weights, l2_weights, e2_weights,
                                          l3_weights, e3_weights, F_l0_eval, y_eval, W_f, b_f)
                    print(f"  [{elapsed:5.0f}s] {stage_lbl} P{phase} ep{ep}"
                          f"  s {i+1:6d}/{ep_chunk}"
                          f"  acc(eval)={acc_q*100:.1f}%  moves={n_moves_pass}"
                          f"  ETA {eta:.0f}s", flush=True)
                    t_report = t_now

            done_prev += ep_chunk
            analyze_we(l1_weights, e1_weights, l2_weights, e2_weights,
                       l3_weights, e3_weights, f"{stage_lbl} P{phase} ep{ep}")

            # Absorb the E matrices trained in this phase; frozen layers untouched
            if phase == 0 and l0a:
                _e = np.mean([np.linalg.norm(_E_L0A[j][p])
                              for j in range(N_L0A_CHIPS) for p in range(N_PAGES)])
                _w = np.mean([np.linalg.norm(_W_L0A[j][p])
                              for j in range(N_L0A_CHIPS) for p in range(N_PAGES)])
                print(f"  E_L0A ‖E‖/‖W‖ (pre-absorb) = {_e/max(_w,1e-9):.3f}", flush=True)
                absorb_l0a_only(lr_slow)
            if phase in (1, 5):
                absorb_e3_only(l3_weights, e3_weights, lr_slow)
            if phase in (2, 5):
                absorb_e2_only(l2_weights, e2_weights, lr_slow)
            if phase in (3, 5):
                absorb_e1_only(l1_weights, e1_weights, lr_slow)
            _update_shadows()

            # Epoch summary
            F_l1_tr = forward_l1_batch(l1_weights, F_l0_train, e1_weights=e1_weights)
            F_l2_tr = forward_l2_batch(l2_weights, F_l1_tr,    e2_weights=e2_weights)
            print(f"  Fitting epoch classifier ...", end=' ', flush=True)
            F_l3_c   = forward_l3_batch(l3_weights, e3_weights, F_l2_tr)
            W_f, b_f = fit_clf(F_l3_c, y_train)
            acc_ep   = eval_acc_l3(l1_weights, e1_weights, l2_weights, e2_weights,
                                   l3_weights, e3_weights, F_l0_eval, y_eval, W_f, b_f)
            acc_dec  = eval_acc_l3(l1_weights, e1_weights, l2_weights, e2_weights,
                                   l3_weights, e3_weights, F_l0_dec, y_dec, W_f, b_f)
            l2_sfx = ''
            if diag_l2:
                W_f_l2_ep, b_f_l2_ep = fit_clf(F_l2_tr, y_train)
                acc_l2_ep = eval_acc_l2(l1_weights, e1_weights, l2_weights, e2_weights,
                                        F_l0_dec, y_dec, W_f_l2_ep, b_f_l2_ep)
                l2_sfx = f'  L2-direct(1040)={acc_l2_ep*100:.2f}%'
            elapsed = time.time() - t_start
            print(f"acc(208)={acc_ep*100:.2f}%  acc(1040)={acc_dec*100:.2f}%{l2_sfx}", flush=True)
            print(f"  {stage_lbl} P{phase} ep{ep} complete:"
                  f"  acc(208)={acc_ep*100:.2f}%  acc(1040)={acc_dec*100:.2f}%{l2_sfx}"
                  f"  moves={n_moves_pass}  [{elapsed:.1f}s]", flush=True)
            acc_log.append((elapsed, done_prev, f's{st_idx+1}p{phase}', acc_dec))

            # Stage-level best tracking / snapshot / early stop  (on 1040-eval)
            if acc_dec > stage_best:
                stage_best = acc_dec
                stage_bad  = 0
                best_ep    = ep
                if stage_rollback:
                    best_snap = _snap_weights()
            else:
                stage_bad += 1
                if stage_patience and stage_bad >= stage_patience:
                    print(f"  {stage_lbl}: early stop (1040-eval declined "
                          f"{stage_bad} epoch(s) from best {stage_best*100:.2f}%)", flush=True)
                    done_prev += (st_eps - ep) * st_chunk
                    break

        if stage_rollback and best_snap is not None and best_ep != ep:
            _restore_weights(best_snap)
            print(f"  {stage_lbl}: ROLLBACK to "
                  f"{'ENTRY state (stage declined)' if best_ep == 0 else f'ep{best_ep} weights'} "
                  f"(1040-eval {stage_best*100:.2f}%)", flush=True)
        best_snap = None
        prev_stage_best = stage_best
        print(f"\n  STAGE {st_idx+1} (P{phase}) done — best 1040-eval "
              f"{stage_best*100:.2f}% ({'entry' if best_ep == 0 else f'ep{best_ep}'}"
              f"{', rolled back' if stage_rollback and best_ep != ep else ''})"
              f"  — {PHASE_NAME[phase]} now frozen "
              f"(unless re-opened by a later stage)", flush=True)

    # ── Final full-test evaluation ─────────────────────────────────────────────
    print(f"\n  Final classifier ({N_TRAIN:,} train, {N_TEST:,} test) ...",
          flush=True)
    F_l1_fin  = forward_l1_batch(l1_weights, F_l0_train, e1_weights=e1_weights)
    F_l2_fin  = forward_l2_batch(l2_weights, F_l1_fin,   e2_weights=e2_weights)
    F_l3_fin  = forward_l3_batch(l3_weights, e3_weights,  F_l2_fin)
    W_f_f, b_f_f = fit_clf(F_l3_fin, y_train, final=True)
    acc_tr_f  = float(np.mean(np.argmax(
        F_l3_fin.astype(np.float64) @ W_f_f.T + b_f_f, axis=1) == y_train))
    acc_final = eval_acc_l3(l1_weights, e1_weights, l2_weights, e2_weights,
                            l3_weights, e3_weights, F_l0_test, y_test, W_f_f, b_f_f)
    acc_final_l2, acc_gha_l2_full = 0.0, 0.0
    if diag_l2:
        W_f_l2_fin, b_f_l2_fin = fit_clf(F_l2_fin, y_train, final=True)
        acc_final_l2 = eval_acc_l2(l1_weights, e1_weights, l2_weights, e2_weights,
                                   F_l0_test, y_test, W_f_l2_fin, b_f_l2_fin)
        # L2 GHA reference on the full test set (original W, no E)
        l1_gha = load_l1_weights()
        l2_gha = load_l2_weights()
        F_l2_gha_tr = forward_l2_batch(l2_gha, forward_l1_batch(l1_gha, F_l0_train))
        F_l2_gha_te = forward_l2_batch(l2_gha, forward_l1_batch(l1_gha, F_l0_test))
        W_f_l2_gha, b_f_l2_gha = fit_clf(F_l2_gha_tr, y_train, final=True)
        acc_gha_l2_full = float(np.mean(np.argmax(
            F_l2_gha_te.astype(np.float64) @ W_f_l2_gha.T + b_f_l2_gha, axis=1) == y_test))
    elapsed = time.time() - t_start

    print(f"\n{'='*72}")
    stg_desc = ' → '.join(f"P{ph}×{n}(c{c//1000}k)" for ph, n, c in stage_list)
    print(f"RESULTS: FABLE pooled-ideas sim  ({stg_desc})")
    print(f"  Ideas:  nlms={'ON g%.1f' % nlms_gain if nlms else 'off'}"
          f"  local_delta={'ON b%.2f' % beta if local_delta else 'off'}"
          f"  clf_refresh={clf_refresh or 'off'}  avg_bp={'ON' if avg_bp else 'off'}"
          f"  bp_ema={bp_ema or 'off'}")
    print(f"  EMNIST GHA-only baseline (ref)  : 64.03%")
    print(f"  GHA baseline this run (208)     : {acc_gha*100:.2f}%")
    print(f"  After training (full test)      : {acc_final*100:.2f}%")
    print(f"  Train acc ({N_TRAIN:,} samples)   : {acc_tr_f*100:.2f}%")
    print(f"  Train–Test gap                  : {(acc_tr_f-acc_final)*100:+.2f} pp")
    print(f"  Change from GHA baseline        : {(acc_final-acc_gha)*100:+.2f} pp")
    if diag_l2:
        print(f"  --- L2-direct diagnostic (KEY: does E2/E1 training move L2?) ---")
        print(f"  L2 GHA-only (full test)         : {acc_gha_l2_full*100:.2f}%")
        print(f"  L2-direct after training        : {acc_final_l2*100:.2f}%")
        print(f"  L2-direct change                : {(acc_final_l2-acc_gha_l2_full)*100:+.2f} pp")
        print(f"  L3 vs L2 gap                    : {(acc_final-acc_final_l2)*100:+.2f} pp")
    print(f"  Wall time                       : {elapsed:.1f}s")
    print(f"{'='*72}\n", flush=True)

    stg_sfx  = '_stg' + '-'.join(f'{ph}x{n}' for ph, n, _ in stage_list)
    idea_sfx = ((f'_nlms{nlms_gain:g}' if nlms else '')
                + (f'_ld{beta:g}' if local_delta else '')
                + (f'_cr{clf_refresh//1000}k' if clf_refresh else '')
                + ('_avgbp' if avg_bp else '')
                + (f'_ema{bp_ema:g}' if bp_ema else '')
                + ('_rb' if stage_rollback else '')
                + (f'_dz{BH_DEADZONE:g}' if BH_DEADZONE else '')
                + ('_ml' if BH_MAGLEVELS else '')
                + ('_bhstd' if BH_STD else '')
                + ('_rjd' if RELU_JAC_DIR else '')
                + ('_rji' if RELU_JAC_INV else ''))
    tag = (f'l3nrm_fable_sm{stg_sfx}_ns{n_settle}'
           f'_bh{"var" if boss_h_fixed is None else boss_h_fixed}'
           f'_lr{lr_slow}{idea_sfx}')
    log_path = os.path.join(RESULTS_DIR, f'{tag}_acc_log.txt')
    with open(log_path, 'w') as f:
        f.write(f"# FABLE pooled-ideas sim  stages={stg_desc}\n")
        f.write(f"# nlms={nlms}({nlms_gain})  local_delta={local_delta}({beta})  "
                f"clf_refresh={clf_refresh}  avg_bp={avg_bp}  bp_ema={bp_ema}\n")
        f.write(f"# gha_baseline(208)={acc_gha*100:.2f}%  final(full)={acc_final*100:.2f}%\n")
        if diag_l2:
            f.write(f"# l2_gha(full)={acc_gha_l2_full*100:.2f}%  "
                    f"l2_direct(full)={acc_final_l2*100:.2f}%\n")
        f.write(f"# elapsed_s  samples  stage_phase  acc\n")
        for row in acc_log:
            f.write(f"{row[0]:.1f}  {row[1]}  {row[2]}  {row[3]:.4f}\n")
    print(f"Log → {log_path}", flush=True)
    return acc_final


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    p = argparse.ArgumentParser(description='PCN 3-layer EMNIST — pooled improvement ideas (fable)')
    # Schedule
    p.add_argument('--staged',      type=str,   default=None,
                   help='Generational stage list as PHxN pairs, e.g. "1x4,2x4,3x2,1x1". '
                        'Phases: 1=E3, 2=E2, 3=E1, 5=all-combined. Layers outside the '
                        'current stage are fully frozen. Overrides --epochs/--warmup.')
    p.add_argument('--epochs',      type=int,   default=6,
                   help='Non-staged mode: number of combined (P5) epochs (default 6)')
    p.add_argument('--warmup',      type=str,   default=None,
                   help='Non-staged mode: warmup stages as NxC[k] pairs, e.g. "3x4k"')
    p.add_argument('--chunk',       type=int,   default=32000,
                   help='Samples per epoch (default 32000)')
    p.add_argument('--stage_patience', type=int, default=0,
                   help='Early-stop a stage after N consecutive epochs without a new '
                        'best 1040-eval (default 0 = off)')
    p.add_argument('--stage_rollback', action='store_true',
                   help='Snapshot weights at each stage\'s best 1040-eval epoch and '
                        'restore them when the stage ends (freeze at peak, not past it)')
    # Idea switches
    p.add_argument('--nlms',        action='store_true',
                   help='NLMS update normalisation: ΔE = lr·gain·outer(δ,f)/||f||² '
                        '(layer-neutral forward EFFECT, not just layer-neutral learning)')
    p.add_argument('--nlms_gain',   type=float, default=8.0,
                   help='NLMS gain (default 8.0 ≈ N·rms(f_l2), matching E3\'s proven '
                        'effective step under unit-RMS normalisation)')
    p.add_argument('--local_delta', action='store_true',
                   help='Layer-local aux-classifier deltas for E2/E1 (deep supervision)')
    p.add_argument('--beta',        type=float, default=0.5,
                   help='Blend weight for local delta: β·local + (1−β)·backprojected '
                        '(default 0.5; β=1 pure local)')
    p.add_argument('--clf_refresh', type=int,   default=0,
                   help='Refit classifier(s) every N samples mid-epoch (default 0 = off)')
    p.add_argument('--clf_refresh_n', type=int, default=20000,
                   help='Subsample size for mid-epoch refits (default 20000)')
    p.add_argument('--avg_bp',      action='store_true',
                   help='Average (not sum) multi-chip L3→L2 backprojection contributions')
    p.add_argument('--l0a',         action='store_true',
                   help='Insert a trainable input transform L0_A (576→576) between frozen '
                        'L0 and L1; train E_L0A via phase P0. See L0A_DESIGN.md.')
    p.add_argument('--rand_init',   action='store_true',
                   help='Random init of L1/L2/L3 (NO GHA/PCA basis) — the acid test that '
                        'the learning rule works from scratch.')
    p.add_argument('--local_only',  action='store_true',
                   help='Local-delta as the SOLE error signal (sets --local_delta, beta=1: '
                        'no W.T backprojection). Each layer trained by its own aux target.')
    p.add_argument('--dfa',         action='store_true',
                   help='Direct Feedback Alignment: broadcast the OUTPUT error to every layer '
                        'via fixed random B_ℓ (coordinated forwards-only rule). Use with '
                        '--rand_init and a phase-5 schedule.')
    p.add_argument('--dfa_hard',    action='store_true',
                   help='Contrastive DFA: broadcast onehot(y)−onehot(ŷ) (2-class, no softmax '
                        'noise, classifier-robust) instead of the soft error. Implies --dfa.')
    p.add_argument('--fix_readout', action='store_true',
                   help='Fit the readout ONCE (epoch 1) then keep it fixed — a STABLE target '
                        'for the layers (tests whether epoch-refit readout-chasing co-limits).')
    p.add_argument('--auto_lr',     action='store_true',
                   help='Per-layer learning rate auto-calibrated from measured rms (one settle '
                        'step moves each layer output by AUTO_FRAC×rms). Fixes one-LR-for-all.')
    p.add_argument('--backprop',    action='store_true',
                   help='CAPABILITY ORACLE: true backprop through the chip net — dense '
                        'softmax residual, real magnitudes, actual W.T chain, real leaky-σ\' '
                        'Jacobian (no sign/renorm). Breaks the boss hardware constraint; '
                        'proves the fold pipeline can reach the ~82%% backprop ceiling. '
                        'Use with --fold --rand_init --auto_lr and a phase-5 schedule.')
    p.add_argument('--sign_grad',   action='store_true',
                   help='signSGD: sign the per-weight update (unit magnitude, scale-invariant '
                        '→ a single LR works for all layers; the cheap sign-delta the boss '
                        'broadcasts). Pair with --fold_momentum (momentum rebuilds magnitude '
                        'from the sign stream). This is the hardware-relevant momentum proof.')
    p.add_argument('--bp_full',     action='store_true',
                   help='Backprop lever 1: do NOT stop on correct — apply the CE gradient '
                        'on every sample incl. correctly-classified ones (margin pushing), '
                        'as the torch rig does. Tests whether "update only on error" caps '
                        'the online rule at ~65%%.')
    p.add_argument('--bp_hard',     action='store_true',
                   help='Backprop with a CONTRASTIVE output error onehot(y)−onehot(ŷ) '
                        'instead of the dense softmax residual (ablation: how much does the '
                        'dense residual matter vs the 2-class contrast?).')
    p.add_argument('--lr_mult',     type=str, default=None,
                   help='Per-layer auto_lr multiplier "m3,m2,m1" (L3,L2,L1). Backprop '
                        'lever 2: un-starve the middle layer, e.g. "1,20,1" boosts L2 η 20x.')
    p.add_argument('--auto_frac',   type=float, default=None,
                   help='Override AUTO_FRAC (per-settle-step output move as fraction of '
                        'layer rms). Default 5e-5; for --fold backprop try 1e-2 … 5e-2.')
    p.add_argument('--fold_every',  type=int, default=1,
                   help='Fold (and momentum-step) every N samples — minibatch the update. '
                        'Averages noisy batch-1 sign deltas before stepping (torch signSGD '
                        'used batch-256). Try 32–64 with --sign_grad --fold_momentum.')
    p.add_argument('--fold_momentum', type=float, default=0.0,
                   help='Backprop lever 3: momentum μ on the fold (v=μ·v+E; W+=v). '
                        'The ingredient that lets high-LR SGD escape the dead init and '
                        'approach the backprop ceiling (torch needed mom 0.9). Try 0.9.')
    p.add_argument('--err_mode', choices=('softmax', 'mse'), default='softmax',
                   help="How the boss forms the error s from the readout scores. "
                        "'softmax' (default) assumes the scores are LOGITS -- true only for "
                        "--clf_logistic. 'mse' uses s = onehot - scores, which is CORRECT for "
                        "the fast lstsq fit (it regresses onto one-hot targets, so its output "
                        "already approximates P(class|a) -- a probability, NOT a logit). "
                        "'mse' is free (no exp, no logistic fit) and needs no exponential in "
                        "the boss -- the error is a plain subtraction.")
    p.add_argument('--clf_lstsq', action='store_true',
                   help='Opt OUT of the logistic teaching/eval classifier and use the old fast \n'
                        'least-squares fit. NOT RECOMMENDED. lstsq regresses onto ONE-HOT targets, \n'
                        'so its outputs are PROBABILITIES, not logits -- pairing it with the \n'
                        'default softmax error is the 2026-07-13 degenerate-softmax BUG (the \n'
                        'informative gradient ends up 26x smaller than a constant class template). \n'
                        'If you use this, you MUST also pass --err_mode mse. Logistic is both a \n'
                        'stronger estimator (+5pp) and only ~6%% of epoch wall-clock.')
    p.add_argument('--logit_scale', type=float, default=1.0,
                   help='Multiply classifier scores by this before the softmax. The '
                        'training-time fit_clf is LEAST-SQUARES, so its scores have rms '
                        '~0.05 and softmax over them is near-UNIFORM -- the error signal '
                        'degenerates to a near-constant class template (no real CE). '
                        'Scale to get logit rms ~2-4. Try 40-80.')
    p.add_argument('--clf_sgd', type=float, default=0.0, metavar='LR',
                   help='Let the READOUT co-adapt online (SGD at this lr on W_f/b_f every '
                        'sample), instead of refitting it once per epoch and freezing it '
                        'while the features drift underneath. Our full-epoch runs DECLINE '
                        '(53.9 -> 42.6) from exactly this staleness; torch trains its '
                        'classifier jointly, every step. 832 weights — buildable. Try 1e-4.')
    p.add_argument('--w_master_free', action='store_true',
                   help='Do NOT clip the float MASTER weight to the hardware rail '
                        '[-0.89, 1.0]. The forward stays clamped either way '
                        '(w_float_to_signed clips the codes), so this only lets the master '
                        'hold "pressure" beyond the rail — exactly what torch\'s '
                        'straight-through --quant does (it clamps the forward copy and '
                        'leaves the float master free). Clipping the master DESTROYS that '
                        'pressure: a weight pinned at the rail forgets how hard it was '
                        'pushed there. Buildable: a few extra bits on the boss digital copy.')
    p.add_argument('--sign_at_fold', action='store_true',
                   help='Take the sign ONCE, at the fold: E accumulates the raw gradient '
                        'over the --fold_every minibatch, then W += lr·sign(E). This is '
                        'sign(mean grad) = what torch signSGD does (81%%). Contrast '
                        '--sign_grad = mean(sign of each sample), which averages '
                        'sign-inconsistent weights toward zero and under-steps them. '
                        'Hardware: "accumulate in E, threshold on the W write".')
    p.add_argument('--seed_init',   type=int, default=7,
                   help='Seed for the --rand_init weight draw. Seed axis of the '
                        'robustness map (does the result survive a different init?).')
    p.add_argument('--seed_data',   type=int, default=42,
                   help='Seed for the training sample order. Seed axis of the robustness '
                        'map (does the result survive a different sample stream?).')
    p.add_argument('--fold_sum',    action='store_true',
                   help='Accumulate E over the --fold_every minibatch as a SUM of the '
                        'per-sample deltas (the pre-2026-07-13 behaviour). The sum scales '
                        'with the batch, COUPLING step size to batch size — which is why '
                        'every (lr, fold_every) pair needed its own retune. Default is now '
                        'the MEAN (per-sample lr scaled by 1/fold_every), matching torch. '
                        'Only for reproducing the old runs.')
    p.add_argument('--fold',        action='store_true',
                   help='Immediate fold: absorb E into W every sample (W += E, E = 0) so '
                        'the W.T backward operator is always current (kills E/W staleness '
                        'and the e_clip ceiling) = online SGD through (W+E). Diagnostic for '
                        'whether the backprojection plateau is staleness vs the sign/'
                        'normalise approximations. Use with the backprojection chain '
                        '(NOT --local_only) and a phase-5 schedule.')
    p.add_argument('--bp_ema',      type=float, default=0.0,
                   help='Backprojection shadow EMA rate toward live W per absorption '
                        '(default 0 = use live W; e.g. 0.2)')
    p.add_argument('--bh_deadzone', type=float, default=0.0,
                   help='Boss_h direction dead-zone: zero δ components where '
                        '|W_f[y]−W_f[ŷ]| < T×rms(diff) (default 0 = off; try 0.3). '
                        'Applies to top delta AND layer-local deltas.')
    p.add_argument('--bh_maglevels', action='store_true',
                   help='Boss_h 3-level per-dimension magnitude {0, ±0.5, ±1} '
                        'instead of pure sign (thresholds (0.3, 1.0)×rms(diff); '
                        'lower threshold overridable via --bh_deadzone)')
    p.add_argument('--fast_batch', action='store_true',
                   help='✅ THE FAST PATH: 9.1x faster and ACCURACY-NEUTRAL. Vectorises the inner '
                        'loop over the fold batch (one batched matmul instead of B per-sample '
                        'passes). 8-epoch A/B: accurate 71.47%%/437s vs fast_batch 72.24%%/48s. '
                        'Requires --backprop --fold --n_settle 1. On BIG: 7x, also neutral '
                        '(ep1 73.75%% vs slow-path 72.21%%; 266s vs ~31 min). Pair with '
                        '--clf_sub 20000 on smallBIG ONLY — on BIG use --fast_batch ALONE '
                        '(--clf_sub costs 5pp there). See PERFORMANCE.md.')
    p.add_argument('--frozen_fwd', action='store_true',
                   help='❌ DEPRECATED / BUGGY — scores ~3pp LOW on its own. Use --fast_batch '
                        'instead (same frozen-forward semantics, 9.1x faster, NO accuracy cost). '
                        'Kept only because --fast_batch sets it. Mechanism: in fold mode, drop E '
                        'from the forward during accumulation '
                        '(it is a gradient accumulator, not a residual: measured |E|/|W| ~ '
                        '0.004, and the cosine probe found the gradient with drift 0.980 vs '
                        '0.973 without) and CACHE the quantised weights, rebuilding them once '
                        'per FOLD instead of once per SAMPLE. Profiling: 66%% of the training '
                        'loop was w_float_to_signed+clip+round. ~128x less quantisation work. '
                        'Matches torch (weights fixed across a batch).')
    p.add_argument('--clf_sub', type=int, default=0, metavar='N',
                   help='PERF ⚠ SCALE N WITH THE READOUT WIDTH — this flag DOES NOT TRANSFER. '
                        'Fits the per-epoch readout on N rows instead of all 124,800 (the logistic '
                        'fit is a FIXED ~11s tax — BIG ~113s — that does NOT shrink with --chunk, '
                        'and runs TWICE: once as the TEACHING classifier whose softmax residual IS '
                        'the error signal, once for eval). smallBIG readout is 32->26 (858 params) '
                        'so N=20000 is FREE (0.1pp). BIG readout is 256->26 (6,682 params) and '
                        'N=20000 COSTS ~5pp (ep1 68.56%% vs 73.75%%) — 20k rows is too thin. '
                        '>>> THIS IS THE BIG RIG: DO NOT USE --clf_sub HERE. <<< '
                        'Use --fast_batch alone. See PERFORMANCE.md.')
    p.add_argument('--delta_bits', type=int, default=0, metavar='N',
                   help='LADDER rung 1: quantise the TRANSPORTED delta to N bits (0=off). '
                        'The delta is what the boss broadcasts ACROSS CHIPS, and cross-chip '
                        'digital comms is the expensive thing (local analog real-values are '
                        'nearly free). How few bits can it be? Try 8/6/4/2/1.')
    p.add_argument('--delta_sign', action='store_true',
                   help='LADDER rung 2: transport sign(delta) only. The WRITE is already '
                        '1-bit (--sign_at_fold); can the TRANSPORT be too?')
    p.add_argument('--delta_renorm', action='store_true',
                   help="LADDER rung 3: RMS-renorm the delta after the sigma' gate (the old "
                        'rule does this). Does it cost or help?')
    p.add_argument('--bh_gate', action='store_true',
                   help='LADDER rung 4: apply the old rule 3-bit boss_h severity to the CE '
                        'error. Without --bh_leak it also SKIPS samples whose margin '
                        'quantises to bh=0.')
    p.add_argument('--bh_leak', type=float, default=0.0, metavar='L',
                   help='LEAKY GATE: when bh quantises to 0, apply a small step L*0.25 '
                        'instead of SKIPPING the sample. bh=0 means a LOW MARGIN, i.e. the '
                        'sample is ON THE DECISION BOUNDARY -- the most informative region. '
                        'The hard skip discards the hardest examples and keeps the easy, '
                        'decisively-wrong ones. L=0 = current hard skip. Try 0.25/0.5/1.0.')
    p.add_argument('--e_bits', type=int, default=0, metavar='N',
                   help='LADDER rung 5: quantise the E accumulator to N bits at the fold '
                        '(0=off). How many bits does the integrator actually need?')
    p.add_argument('--delta_agc', type=float, default=0.0, metavar='TAU',
                   help='SLOW per-layer AGC on the delta (EMA rate TAU, try 0.01). The delta '
                        'ATTENUATES 0.538x PER LAYER (~0.9 bits/layer); at depth 6 a fixed-range '
                        '6-bit channel has NOTHING left. Gain control is load-bearing AT DEPTH. '
                        'Unlike --delta_renorm (which is per-SAMPLE and so also flattens across '
                        'samples, costing ~8.6pp), a SLOW AGC corrects only the drifting per-layer '
                        'scale and leaves per-sample magnitude intact.')
    p.add_argument('--delta_fixed', type=float, default=0.0, metavar='R',
                   help='Quantise --delta_bits against a FIXED range R (a real DAC has fixed '
                        'rails) instead of the per-vector max|d| (an IDEAL instantaneous AGC). '
                        'Use with --delta_agc. This is the honest hardware model and the one '
                        'that actually tests depth-robustness.')
    p.add_argument('--bh_gain', type=float, default=1.0,
                   help='Scale the boss_h gate: bh = int(7*GAIN*margin/denom). The gate is '
                        'calibrated to the teaching classifier score distribution; logistic '
                        'is more confident than lstsq and over-gates (40%% of wrong samples '
                        'skipped vs 15.7%%). Raise GAIN to restore the operating point.')
    p.add_argument('--bh_std',      action='store_true',
                   help='Boss_h severity denominator: 4·std(scores) instead of '
                        'score span (robust to single outlier class scores)')
    p.add_argument('--relu_jac_dir', action='store_true',
                   help='Directional leaky-ReLU gate: δ ×= σ\'(f) then RMS-renorm '
                        '(pure redistribution toward responsive units; the untested '
                        'renormalised form of the old --relu_jac)')
    p.add_argument('--relu_jac_inv', action='store_true',
                   help='Delivery compensation: outer-product δ ×= min(1/σ\', 3) so '
                        'the realised post-activation shift matches the request. '
                        'Applied to E writes only; intended for NLMS mode where '
                        'weight steps sit far below e_clip')
    # Standard knobs
    p.add_argument('--n_settle',    type=int,   default=8)
    p.add_argument('--boss_h',      type=int,   default=None,
                   help='Fixed boss_h level 0-7; omit for variable')
    p.add_argument('--lr_slow',     type=float, default=1.0)
    p.add_argument('--boss_lr',     type=float, default=None,
                   help='Override BOSS_LR (default 0.01; optd combined baseline used 0.003)')
    p.add_argument('--e_clip',      type=float, default=E_CLIP_HW)
    p.add_argument('--leaky_alpha', type=float, default=0.1)
    p.add_argument('--report',      type=float, default=30.0)
    p.add_argument('--diag_l2',     action='store_true',
                   help='L2-direct diagnostic each epoch and in the final report')
    args = p.parse_args()

    if args.local_only:            # local-delta as the sole signal (no backprojection)
        args.local_delta = True
        args.beta = 1.0
    if args.dfa_hard:              # contrastive DFA implies DFA
        args.dfa = True

    LEAKY_ALPHA = args.leaky_alpha
    if args.auto_frac is not None:
        AUTO_FRAC = args.auto_frac
    if args.lr_mult is not None:
        LR_MULT = tuple(float(x) for x in args.lr_mult.split(','))
        assert len(LR_MULT) == 3, '--lr_mult needs "m3,m2,m1"'
    if args.boss_lr is not None:
        BOSS_LR = args.boss_lr
    SEED_INIT    = args.seed_init
    SEED_DATA    = args.seed_data
    W_MASTER_CLIP = not args.w_master_free
    LOGIT_SCALE  = args.logit_scale
    CLF_LOGISTIC = not args.clf_lstsq
    ERR_MODE     = args.err_mode
    # ── Guard the 2026-07-13 degenerate-softmax bug ───────────────────────────────
    # Each error rule is valid for exactly ONE fit. Measured (b32, 20ep, acc(1040)):
    #     lstsq    + softmax = 46.1  <- THE BUG (mismatched: p_hat is not a logit)
    #     lstsq    + mse     = 53.8     correct pairing
    #     logistic + softmax = 68.7     correct pairing  <- best
    #     logistic + mse     = ~10      collapses (mismatched: a logit is not a p_hat)
    if not CLF_LOGISTIC and ERR_MODE == 'softmax' and LOGIT_SCALE == 1.0:
        raise SystemExit(
            '\nREFUSING TO RUN: --clf_lstsq with --err_mode softmax is the degenerate-softmax\n'
            'bug. lstsq outputs are PROBABILITIES (score rms ~0.05), not logits; softmaxing\n'
            'them leaves a near-constant class template and the informative gradient is 26x\n'
            'smaller. Use --err_mode mse with --clf_lstsq, or just drop --clf_lstsq (logistic\n'
            'is the default and is both stronger and cheap). See THE_SOFTMAX_BUG.md.\n')
    if CLF_LOGISTIC and ERR_MODE == 'mse':
        raise SystemExit(
            '\nREFUSING TO RUN: logistic outputs are LOGITS; --err_mode mse subtracts them from\n'
            'a one-hot, which is meaningless (measured: collapses to ~10%). Use the default\n'
            'softmax error with the logistic fit. See THE_SOFTMAX_BUG.md.\n')
    BH_GAIN      = args.bh_gain
    FROZEN_FWD   = args.frozen_fwd or args.fast_batch   # batching REQUIRES a frozen forward
    CLF_SUB      = args.clf_sub
    DELTA_BITS   = args.delta_bits
    DELTA_SIGN   = args.delta_sign
    DELTA_RENORM = args.delta_renorm
    BH_GATE      = args.bh_gate or args.bh_leak > 0.0
    BH_LEAK      = args.bh_leak
    E_BITS       = args.e_bits
    DELTA_AGC    = args.delta_agc
    DELTA_FIXED  = args.delta_fixed
    BH_DEADZONE  = args.bh_deadzone
    BH_MAGLEVELS = args.bh_maglevels
    BH_STD       = args.bh_std
    assert not (args.relu_jac_dir and args.relu_jac_inv), \
        "--relu_jac_dir and --relu_jac_inv are opposite reweightings; pick one"
    RELU_JAC_DIR = args.relu_jac_dir
    RELU_JAC_INV = args.relu_jac_inv

    if args.staged:
        stage_list = [(ph, n, args.chunk) for ph, n in parse_staged(args.staged)]
    else:
        stage_list = []
        if args.warmup:
            for n, c in parse_warmup(args.warmup):
                stage_list.append((5, n, c))
        stage_list.append((5, args.epochs, args.chunk))

    run_sim(
        stage_list     = stage_list,
        n_settle       = args.n_settle,
        boss_h_fixed   = args.boss_h,
        lr_slow        = args.lr_slow,
        e_clip         = args.e_clip,
        report_secs    = args.report,
        diag_l2        = args.diag_l2,
        nlms           = args.nlms,
        nlms_gain      = args.nlms_gain,
        local_delta    = args.local_delta,
        beta           = args.beta,
        clf_refresh    = args.clf_refresh,
        clf_refresh_n  = args.clf_refresh_n,
        avg_bp         = args.avg_bp,
        bp_ema         = args.bp_ema,
        stage_patience = args.stage_patience,
        stage_rollback = args.stage_rollback,
        l0a            = args.l0a,
        rand_init      = args.rand_init,
        dfa            = args.dfa,
        dfa_hard       = args.dfa_hard,
        fix_readout    = args.fix_readout,
        auto_lr        = args.auto_lr,
        fold           = args.fold,
        backprop       = args.backprop,
        bp_hard        = args.bp_hard,
        bp_full        = args.bp_full,
        fold_momentum  = args.fold_momentum,
        sign_grad      = args.sign_grad,
        fold_every     = args.fold_every,
        fold_mean      = not args.fold_sum,
        sign_at_fold   = args.sign_at_fold,
        clf_lr         = args.clf_sgd,
        fast_batch     = args.fast_batch,
    )
