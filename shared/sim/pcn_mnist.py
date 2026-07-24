"""
pcn_mnist.py — MNIST digit / EMNIST letter classification using hardware-faithful
PCN simulation (DATASET=mnist | emnist_letters).

Demonstrates that the PCN multi-chip architecture, scaled to real image data,
learns meaningful visual features via unsupervised Hebbian GHA and achieves
competitive MNIST accuracy using only hardware-faithful operations:
  - GHA (Sanger 1989) for unsupervised feature learning (the chip's learning rule)
  - 8-bit weight quantisation (CMOS TG range, codes 71–192)
  - KCL current summation for matrix-vector multiply (modelled as W @ x)
  - ReLU activation: PMOS clamp clips negative outputs to 0 (V1 hardware)

Multi-chip topology (Sky130A, 16×16 MAC cells per chip):

  Layer 0 — pixel projection  (784 → N_L0=64)
    Column tiles : ⌈784/16⌉ = 49 chips  (each handles 16 of the 784 pixel inputs)
    Row tiles    : ⌈64/16⌉  = 4  chips  (each contributes 16 output feature rows)
    Chips (L0)   : 49 × 4 = 196
    KCL bus      : chips in the same row band sum partial currents on a shared bus;
                   the result is identical to a monolithic 64×784 matrix-vector product.

  Layer 1 — feature abstraction  (N_L0=64 → N_L1=16)
    Column tiles : ⌈64/16⌉  = 4 chips
    Row tiles    : ⌈16/16⌉  = 1 chip
    Chips (L1)   : 4 × 1 = 4

  Total          : 200 chips, 51 200 weight cells, 10.24 nF on-chip weight storage
  Off-chip weight bandwidth : 0 bit/s  (weights never leave the chip)

  Why KCL tiling is exact:
    Each chip_k in a row band computes  Σ_{j ∈ cols_k} W_ij × x_j  (partial product).
    All chips in the row band share the output current bus; KCL gives:
      I_out_i = Σ_k Σ_{j ∈ cols_k} W_ij × x_j = Σ_j W_ij × x_j  ✓

Training:
  L0 and L1 trained unsupervised with GHA (float weights, cosine LR decay).
  A thin supervised classifier (least-squares or logistic) runs on the host
  processor using the learned L0 features — the chip itself is never supervised.

Optional 8-bit inference:
  After GHA, weights are quantised to the 8-bit hardware DAC range and
  inference accuracy is compared against float weights.  For 784-input arrays,
  unit-norm rows have element magnitudes ≈ 1/√784 ≈ 0.036, which occupies
  about 2–3 quantisation steps (step ≈ 0.0156); accuracy impact is quantified.

Outputs  (written to sim/results/, filenames prefixed by dataset — 'mnist_*' or
'emnist_letters_*'):
  {tag}_topology.txt      — hardware topology summary (copy of console header)
  {tag}_filters_l0.png    — N_L0 learned L0 filters as a grid of 28×28 patches
  {tag}_filters_l1.png    — N_L1 codes projected back to pixel space
  {tag}_training.png      — GHA reconstruction error + LR schedule
  {tag}_confusion.png     — N_CLASSES × N_CLASSES confusion matrix on the test set

Configuration:
  Edit the constants below, or override via environment variables:
    DATASET=emnist_letters python pcn_mnist.py   # 26-class a-z instead of MNIST digits
    CLASSIFIER=logistic python pcn_mnist.py      # sklearn LogisticRegression
    QUANTISE=0 python pcn_mnist.py               # skip 8-bit comparison

Data:
  MNIST    — requires sklearn (pip install scikit-learn) or torchvision.
             sklearn.datasets.fetch_openml downloads and caches MNIST (~55 MB).
  EMNIST   — requires torchvision (pip install torchvision).  The 'letters'
             split (26 balanced classes, a-z, merged case) is loaded; the full
             EMNIST archive (~560 MB) is downloaded once and cached under
             ~/.cache/emnist/.  EMNIST images ship transposed relative to the
             MNIST pixel convention — corrected on load so filter plots render
             upright letters.
"""

import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from pcn_core import _quantise, CODE_MIN, CODE_MID, CODE_MAX, CODE_SCALE

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Configuration ──────────────────────────────────────────────────────────────
#
# Parameters are grouped by what changes on the hardware vs what is purely
# a software / training choice.
#
# ADJUSTABLE — hardware-faithful (maps to a physical chip or operating parameter):
#
#   N_L0        Number of L0 output features.  Each additional 16 costs 49 more
#               L0 chips (one extra row band across all 49 column tiles).
#               64 → 196 chips;  128 → 392 chips;  256 → 784 chips.
#
#   N_L1        Number of L1 output codes.  Each additional 16 costs 4 more L1
#               chips.  16 → 4 chips;  32 → 8 chips.
#
#   N_EPOCHS    Training duration.  More epochs = more Hebbian pulse cycles per
#               cell.  Hardware equivalent: longer unsupervised exposure phase.
#               Cost is linear in wall-clock time; no chip area or power change.
#
#   LR / LR_MIN Learning rate.  Maps to I_hebb × t_pulse / Cw on the chip:
#               higher LR = longer Hebbian pulse or larger I_hebb current.
#               Cosine schedule: LR decays smoothly from LR_MAX → LR_MIN.
#               Smaller LR_MIN = more fine-tuning near convergence.
#
# FIXED — determined by Sky130A tape-out, cannot be changed without respinning:
#
#   CHIP_ROWS / CHIP_COLS = 16   Physical MAC array dimensions.
#   8-bit weight DAC              Codes 71–192 (Vw = 0.50–1.35 V, step = 1/64).
#   ReLU activation               V1 PMOS clamp: MAC outputs clipped to ≥ 0.
#                                 (A V2 Gilbert-cell upgrade would remove this,
#                                 giving signed outputs and better accuracy.)
#
# OFF-CHIP ONLY — runs on the host processor, not on the PCN chip:
#
#   CLASSIFIER   'logistic' uses sklearn LogisticRegression (better accuracy).
#                'lstsq'    uses numpy least-squares (no sklearn dependency).
#                Neither runs on the chip; both represent a thin host-side head.

DATASET = os.environ.get('DATASET', 'mnist')   # 'mnist' (digits) | 'emnist_letters' (a-z)

if DATASET == 'mnist':
    N_CLASSES   = 10
    CLASS_NAMES = [str(d) for d in range(10)]
    RESULTS_TAG = 'mnist'
elif DATASET == 'emnist_letters':
    N_CLASSES   = 26
    CLASS_NAMES = list('abcdefghijklmnopqrstuvwxyz')
    RESULTS_TAG = 'emnist_letters'
else:
    raise ValueError(f"Unknown DATASET={DATASET!r}; expected 'mnist' or 'emnist_letters'")

N_IN        = 784    # 28 × 28 pixels (fixed by image size — same for MNIST and EMNIST)
N_L0        = int(os.environ.get('N_L0', 64 if DATASET == 'mnist' else 96))
N_L1        = int(os.environ.get('N_L1', 16 if DATASET == 'mnist' else 32))
            # L0/L1 sizing  — 196 L0 chips @ 64, 392 @ 128;  4 L1 chips @ 16, 8 @ 32.
            # EMNIST letters defaults to a wider L0/L1 (26-class task needs more
            # discriminative capacity than 10-digit MNIST).

CHIP_ROWS   = 16     # Fixed by Sky130A tape-out
CHIP_COLS   = 16

LR_L0       = 0.01   # L0 peak LR  (hardware: I_hebb × t_pulse / Cw)
LR_L0_MIN   = 0.0005 # L0 floor    — steeper final decay than first run
LR_L1       = 0.02   # L1 peak LR
LR_L1_MIN   = 0.001  # L1 floor

N_EPOCHS_L0 = 12     # L0 training epochs (was 5; recon_mse still declining at epoch 5)
N_EPOCHS_L1 = 6      # L1 training epochs (was 3)

EVAL_EVERY  = 5000   # Steps between reconstruction-MSE log lines (logging only)
EVAL_N      = 1000   # Samples used for mid-training eval (logging only)

W_INIT_STD  = 0.01   # Initial weight scatter; rows normalised to unit norm at init

# Switchable at runtime via env vars
CLASSIFIER   = os.environ.get('CLASSIFIER', 'logistic')   # 'logistic' | 'lstsq'
QUANTISE_INF = os.environ.get('QUANTISE',   '1') == '1'   # 8-bit inference comparison


# ── Hardware topology ──────────────────────────────────────────────────────────

def _chip_tiles(n_rows, n_cols):
    col_tiles = math.ceil(n_cols / CHIP_COLS)
    row_tiles = math.ceil(n_rows / CHIP_ROWS)
    return col_tiles, row_tiles, col_tiles * row_tiles


def print_topology():
    """Print and save the multi-chip topology for this MNIST architecture."""
    l0_ct, l0_rt, l0_n = _chip_tiles(N_L0, N_IN)
    l1_ct, l1_rt, l1_n = _chip_tiles(N_L1, N_L0)
    total   = l0_n + l1_n
    cells   = total * CHIP_ROWS * CHIP_COLS
    cap_nF  = cells * 200e-15 * 1e9   # 200 fF per cell → nF

    lines = [
        "",
        "━" * 62,
        f"PCN {DATASET.upper()} Demo  —  analog multi-chip inference + Hebbian GHA",
        "━" * 62,
        "",
        f"Dataset        : {DATASET}  ({N_CLASSES} classes: "
            f"{CLASS_NAMES[0]}-{CLASS_NAMES[-1]})",
        f"Hardware topology  (Sky130A, {CHIP_ROWS}×{CHIP_COLS} MAC cells per chip):",
        "",
        f"  Layer 0 — pixel projection  ({N_IN} → {N_L0})",
        f"    Column tiles : ⌈{N_IN}/{CHIP_COLS}⌉ = {l0_ct:2d}  chips"
            f"  ({CHIP_COLS} pixel inputs each, partial KCL sums on row bus)",
        f"    Row tiles    : ⌈{N_L0}/{CHIP_ROWS}⌉ = {l0_rt:2d}  chips"
            f"  ({CHIP_ROWS} output feature rows each)",
        f"    Total (L0)   : {l0_ct} × {l0_rt} = {l0_n} chips",
        "",
        f"  Layer 1 — feature abstraction  ({N_L0} → {N_L1})",
        f"    Column tiles : ⌈{N_L0}/{CHIP_COLS}⌉ = {l1_ct}  chips",
        f"    Row tiles    : ⌈{N_L1}/{CHIP_ROWS}⌉ = {l1_rt}  chip",
        f"    Total (L1)   : {l1_ct} × {l1_rt} = {l1_n} chips",
        "",
        f"  Total chips    : {total}  physical MAC arrays  ({l0_n} L0 + {l1_n} L1)",
        f"  Total cells    : {cells:,d}  = {total} chips × {CHIP_ROWS * CHIP_COLS} cells",
        f"  Weight storage : {cap_nF:.2f} nF on-chip"
            f"  (Vw = 0.50–1.35 V, 200 fF × cell)",
        f"  Off-chip weight bandwidth : 0 bit/s",
        "",
        f"  Classifier     : {CLASSIFIER}  (supervised, host-side only)",
        f"  8-bit comparison : {'enabled' if QUANTISE_INF else 'disabled'}",
        "",
        "Tunable parameters:",
        "",
        f"  N_L0={N_L0}, N_L1={N_L1}"
            f"   — increase for more features (each +16 L0 adds 49 chips)",
        f"  N_EPOCHS_L0={N_EPOCHS_L0}, N_EPOCHS_L1={N_EPOCHS_L1}"
            f"   — more epochs → better GHA convergence, linear time cost",
        f"  LR_L0={LR_L0}→{LR_L0_MIN}, LR_L1={LR_L1}→{LR_L1_MIN}"
            f"   — maps to I_hebb×t_pulse/Cw (cosine decay)",
        "",
        "Fixed by Sky130A tape-out:",
        f"  {CHIP_ROWS}×{CHIP_COLS} MAC cells/chip   "
            f"8-bit weight DAC (codes 71–192)   ReLU clamp (V1 PMOS)",
        "  V2 upgrade path: Gilbert cell → signed outputs → ~3 pp accuracy gain",
        "",
    ]
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(RESULTS_DIR, f'{RESULTS_TAG}_topology.txt'), 'w') as f:
        f.write(text)
    return {'l0_chips': l0_n, 'l1_chips': l1_n, 'total_chips': total,
            'total_cells': cells}


# ── MNIST loader ───────────────────────────────────────────────────────────────

def load_mnist():
    """
    Load MNIST training and test sets.

    Tries sklearn.datasets.fetch_openml (downloads & caches ~55 MB on first run),
    then torchvision.  Raises RuntimeError with install instructions if both fail.

    Returns
    -------
    X_train : (60000, 784) float32 in [0, 1]
    y_train : (60000,) int
    X_test  : (10000, 784) float32 in [0, 1]
    y_test  : (10000,) int
    """
    print("Loading MNIST ...")

    # sklearn — most convenient; caches in ~/scikit_learn_data/
    try:
        from sklearn.datasets import fetch_openml
        try:
            mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        except TypeError:
            mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        X = mnist['data'].astype(np.float32) / 255.0
        y = mnist['target'].astype(int)
        X_train, y_train = X[:60000], y[:60000]
        X_test,  y_test  = X[60000:], y[60000:]
        print(f"  sklearn: {X_train.shape[0]:,} train / {X_test.shape[0]:,} test")
        return X_train, y_train, X_test, y_test
    except Exception as exc:
        print(f"  sklearn failed ({exc})")

    # torchvision — alternative
    try:
        import torchvision.datasets as tvd
        import torchvision.transforms as tvt
        cache = os.path.join(os.path.expanduser('~'), '.cache', 'mnist')
        tr = tvd.MNIST(cache, train=True,  download=True, transform=tvt.ToTensor())
        te = tvd.MNIST(cache, train=False, download=True, transform=tvt.ToTensor())
        X_train = np.array([s[0].numpy().flatten() for s in tr], dtype=np.float32)
        y_train = np.array([s[1] for s in tr])
        X_test  = np.array([s[0].numpy().flatten() for s in te], dtype=np.float32)
        y_test  = np.array([s[1] for s in te])
        print(f"  torchvision: {X_train.shape[0]:,} train / {X_test.shape[0]:,} test")
        return X_train, y_train, X_test, y_test
    except Exception as exc:
        print(f"  torchvision failed ({exc})")

    raise RuntimeError(
        "\nCannot load MNIST.  Install either:\n"
        "  pip install scikit-learn\n"
        "  pip install torchvision\n"
    )


def load_emnist_letters():
    """
    Load the EMNIST 'letters' split: 26 balanced classes (a-z, merged case),
    124,800 train / 20,800 test, 28×28 grayscale.

    Requires torchvision; downloads and caches the full EMNIST archive
    (~560 MB, one-time) under ~/.cache/emnist/.

    Reads the raw .data/.targets tensors directly (bypassing __getitem__/
    ToTensor) since no transform is needed beyond the orientation fix below —
    this also avoids a slow 145,600-iteration Python loop.

    EMNIST images ship transposed relative to the MNIST pixel convention;
    confirmed by inspection (sample 0, label 'w', renders correctly only
    after a row/col transpose) — corrected here so reshaping back to 28×28
    for filter plots gives upright letters.  Labels are 1-indexed in the
    official split (1=a … 26=z); remapped to 0-indexed to match CLASS_NAMES.

    Returns
    -------
    X_train : (124800, 784) float32 in [0, 1]
    y_train : (124800,) int, 0-25
    X_test  : (20800, 784) float32 in [0, 1]
    y_test  : (20800,) int, 0-25
    """
    print("Loading EMNIST letters ...")
    import torchvision.datasets as tvd

    cache = os.path.join(os.path.expanduser('~'), '.cache', 'emnist')
    tr = tvd.EMNIST(cache, split='letters', train=True,  download=True)
    te = tvd.EMNIST(cache, split='letters', train=False, download=True)

    def _to_arrays(ds):
        imgs = ds.data.numpy().transpose(0, 2, 1)        # fix row/col orientation
        X    = (imgs.reshape(len(ds), -1).astype(np.float32)) / 255.0
        y    = ds.targets.numpy().astype(int) - 1        # 1-indexed -> 0-indexed
        return X, y

    X_train, y_train = _to_arrays(tr)
    X_test,  y_test  = _to_arrays(te)
    print(f"  torchvision: {X_train.shape[0]:,} train / {X_test.shape[0]:,} test"
          f"  (26 classes, a-z)")
    return X_train, y_train, X_test, y_test


def preprocess(X_train, X_test):
    """
    Subtract training-set pixel mean (removes mean illumination) and
    L2-normalise each sample (gives each input unit norm, consistent scale
    for GHA regardless of digit stroke density).

    Returns (X_train_pp, X_test_pp, pixel_mean).
    Both test and train share the same pixel_mean (fitted on train only).
    """
    mean = X_train.mean(axis=0)
    X_tr = (X_train - mean).astype(np.float64)
    X_te = (X_test  - mean).astype(np.float64)
    tr_norms = np.linalg.norm(X_tr, axis=1, keepdims=True)
    te_norms = np.linalg.norm(X_te, axis=1, keepdims=True)
    X_tr /= np.maximum(tr_norms, 1e-8)
    X_te /= np.maximum(te_norms, 1e-8)
    return X_tr, X_te, mean


# ── GHA layer ─────────────────────────────────────────────────────────────────

class GHALayer:
    """
    Sanger's Generalised Hebbian Algorithm layer.

    Implements the learning rule native to the PCN chip: each physical MAC row
    fires a Hebbian pulse (HEBB_EN via Wishbone CTRL register), then firmware
    normalises the updated weight via a SPI write-back.  The GHA deflation step
    corresponds to firmware subtracting the current row's DAC output from the
    activation register before the next row's MAC computation — i.e., the
    predictive coding residual pathway.

    Parameters
    ----------
    n_rows : int   — number of output features (logical row count).
    n_cols : int   — number of inputs (logical column count).
    lr     : float — initial learning rate (overridden per-step by cosine schedule).
    seed   : int   — RNG seed for weight initialisation.
    """

    def __init__(self, n_rows, n_cols, lr=0.01, seed=0):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.lr     = lr
        rng = np.random.default_rng(seed)
        W = rng.standard_normal((n_rows, n_cols)) * W_INIT_STD
        norms = np.linalg.norm(W, axis=1, keepdims=True)
        self.W = W / np.maximum(norms, 1e-8)   # start with unit-norm rows

    @property
    def n_chips(self):
        return _chip_tiles(self.n_rows, self.n_cols)[2]

    def forward(self, x):
        """KCL summation: y = W @ x.  Models tiled chip MAC operation."""
        return self.W @ x

    def update_gha(self, x, lr):
        """
        One GHA step: sequential deflation over all n_rows.

        For row i (i = 0, 1, …, n_rows-1):
          y_i     = W_i · x_resid        (MAC on deflated residual)
          W_i    += lr × y_i × x_resid   (Hebbian outer product; HEBB_EN pulse)
          W_i    /= ||W_i||               (Oja normalisation; firmware SPI write)
          proj    = W_i · x              (full-input projection for deflation)
          x_resid -= proj × W_i          (subtract this row's contribution = PC residual)

        The deflation is equivalent to the predictive coding hierarchy: each row
        sees only the variance unexplained by all preceding rows.  After convergence,
        row i carries the i-th principal component of the training data distribution.

        Hardware note: deflation ≡ firmware subtracts W_i × (W_i · x) from the
        activation register (INP_DAC_DATA at Wishbone 0x2C) before the next row MAC.
        """
        x_resid = x.copy()
        for i in range(self.n_rows):
            y_i = float(np.dot(self.W[i], x_resid))
            self.W[i] += lr * y_i * x_resid
            nrm = np.linalg.norm(self.W[i])
            if nrm > 1e-8:
                self.W[i] /= nrm
            proj = float(np.dot(self.W[i], x))   # full x, not x_resid
            x_resid -= proj * self.W[i]

    def recon_mse(self, X_eval):
        """
        Mean normalised reconstruction MSE:  E[ ||x - W^T (W x)||² / ||x||² ].
        Measures how well the learned subspace captures the input variance.
        At convergence, trained samples → 0; out-of-distribution → ||x||²/n.
        """
        total = 0.0
        for x in X_eval:
            y     = self.W @ x
            x_hat = self.W.T @ y
            total += np.sum((x - x_hat) ** 2) / (np.sum(x ** 2) + 1e-8)
        return total / len(X_eval)


# ── Training ───────────────────────────────────────────────────────────────────

def _cosine_lr(step, total_steps, lr_max, lr_min):
    t = step / max(total_steps - 1, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * t))


def train_gha(layer, X_train, n_epochs, lr_max, lr_min,
              label, X_eval=None):
    """
    Train GHALayer for n_epochs passes over X_train with cosine LR decay.

    Parameters
    ----------
    layer    : GHALayer
    X_train  : (N, n_cols) training samples.
    n_epochs : int
    lr_max   : float — peak LR at step 0.
    lr_min   : float — floor LR at the final step.
    label    : str   — display name for progress lines.
    X_eval   : (M, n_cols) or None — subset used for mid-training recon_mse logging.

    Returns
    -------
    log : list of (step, recon_mse, lr) tuples.
    """
    N           = len(X_train)
    total_steps = n_epochs * N
    rng         = np.random.default_rng(0)
    log         = []
    step        = 0
    t0          = time.time()

    for epoch in range(n_epochs):
        idx = rng.permutation(N)
        for ii in range(N):
            lr = _cosine_lr(step, total_steps, lr_max, lr_min)
            layer.update_gha(X_train[idx[ii]], lr)

            if step % EVAL_EVERY == 0:
                mse     = layer.recon_mse(X_eval) if X_eval is not None else float('nan')
                elapsed = time.time() - t0
                eta     = elapsed / max(step, 1) * (total_steps - step) if step else 0.0
                log.append((step, mse, lr))
                print(f"  {label}  epoch {epoch + 1}/{n_epochs}"
                      f"  step {step:>7,}/{total_steps:,}"
                      f"  recon_mse={mse:.5f}"
                      f"  lr={lr:.5f}"
                      f"  {elapsed:5.0f}s  ETA {eta:5.0f}s")

            step += 1

    # Final eval
    mse_final = layer.recon_mse(X_eval) if X_eval is not None else float('nan')
    log.append((total_steps, mse_final, lr_min))
    print(f"  {label}  DONE  recon_mse={mse_final:.5f}  "
          f"total={time.time() - t0:.1f}s")
    return log


# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_features(layer, X, relu=True):
    """
    Batched forward pass: F = ReLU(W @ X^T)^T.

    ReLU models the V1 PMOS clamp (negative MAC outputs are blocked).
    Parameters
    ----------
    layer : GHALayer  (W may be float or 8-bit quantised).
    X     : (N, n_cols)
    relu  : bool — apply ReLU (True = hardware faithful).

    Returns
    -------
    F : (N, n_rows) float32
    """
    F = (layer.W @ X.T).T.astype(np.float32)
    if relu:
        np.maximum(F, 0.0, out=F)
    return F


# ── Classifiers ───────────────────────────────────────────────────────────────

def _onehot(y, n=N_CLASSES):
    Y = np.zeros((len(y), n), dtype=np.float64)
    Y[np.arange(len(y)), y] = 1.0
    return Y


def classify_lstsq(F_train, y_train, F_test, y_test):
    """
    Least-squares linear classifier (numpy only; no sklearn required).

    Solves:  min_W ||F_train @ W - Y_onehot||²_F  via the pseudoinverse.
    Bias column is prepended so the solution has an intercept.

    Hardware analogue: a small host-side processor (e.g. Raspberry Pi) with
    a precomputed weight table; inference is a single matrix-vector multiply.

    Returns (accuracy, predictions, W_cls).
    """
    # Prepend bias column
    ones_tr = np.ones((len(F_train), 1), dtype=np.float64)
    ones_te = np.ones((len(F_test),  1), dtype=np.float64)
    F_tr = np.hstack([ones_tr, F_train.astype(np.float64)])
    F_te = np.hstack([ones_te, F_test.astype(np.float64)])

    Y_onehot = _onehot(y_train)
    W_cls, _, _, _ = np.linalg.lstsq(F_tr, Y_onehot, rcond=None)
    pred = np.argmax(F_te @ W_cls, axis=1)
    acc  = float(np.mean(pred == y_test))
    return acc, pred, W_cls


def classify_logistic(F_train, y_train, F_test, y_test):
    """
    Multi-class logistic regression via sklearn (L-BFGS, max_iter=2000).

    Includes StandardScaler because L0 feature magnitudes are small (~0.036)
    and logistic regression converges faster with unit-variance inputs.

    Returns (accuracy, predictions, fitted_clf).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    F_tr_sc = scaler.fit_transform(F_train.astype(np.float64))
    F_te_sc = scaler.transform(F_test.astype(np.float64))

    clf = LogisticRegression(
        solver='lbfgs', C=10.0, max_iter=5000, random_state=0, n_jobs=-1,
    )
    clf.fit(F_tr_sc, y_train)
    pred = clf.predict(F_te_sc)
    acc  = float(np.mean(pred == y_test))
    return acc, pred, clf


# ── Visualisation ─────────────────────────────────────────────────────────────

def _get_plt():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def plot_filters(W, img_shape, n_grid_cols, title, fname):
    """
    Render each row of W as a filter image in a grid.
    Diverging colour map (RdBu_r) shows positive (blue) and negative (red) weights.
    """
    plt = _get_plt()
    if plt is None:
        print("  matplotlib not available — skipping filter plot")
        return

    n_filters   = len(W)
    n_grid_rows = math.ceil(n_filters / n_grid_cols)
    h, w        = img_shape

    fig, axes = plt.subplots(n_grid_rows, n_grid_cols,
                              figsize=(n_grid_cols * 1.4, n_grid_rows * 1.4))
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        if i < n_filters:
            filt = W[i].reshape(h, w)
            vmax = max(abs(filt.max()), abs(filt.min())) + 1e-8
            ax.imshow(filt, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                      interpolation='nearest')
        ax.axis('off')

    fig.suptitle(title, fontsize=10, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(RESULTS_DIR, fname)
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  -> {path}")


def plot_training(logs, labels, fname='mnist_training.png'):
    """Reconstruction MSE (log-scale) and cosine LR schedule over training steps."""
    plt = _get_plt()
    if plt is None:
        print("  matplotlib not available — skipping training plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    colours = ['steelblue', 'coral', 'seagreen']

    for log, label, col in zip(logs, labels, colours):
        if not log:
            continue
        steps = [r[0] for r in log]
        mses  = [r[1] for r in log]
        lrs   = [r[2] for r in log]
        ax1.semilogy(steps, mses, '-', color=col, lw=2, label=label)
        ax2.plot(steps, lrs, '-', color=col, lw=2, label=label)

    ax1.set_xlabel('Training step')
    ax1.set_ylabel('Reconstruction MSE  E[||x - WᵀWx||² / ||x||²]')
    ax1.set_title('GHA training convergence')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Training step')
    ax2.set_ylabel('Learning rate')
    ax2.set_title('Cosine LR schedule')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, fname)
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  -> {path}")


def plot_confusion(y_true, y_pred, title, fname='mnist_confusion.png'):
    """Normalised N_CLASSES × N_CLASSES confusion matrix.

    Per-cell text annotations are skipped above 12 classes (e.g. the 26-class
    EMNIST letters task) since they become unreadable clutter at that size.
    """
    plt = _get_plt()
    if plt is None:
        print("  matplotlib not available — skipping confusion matrix")
        return

    n  = N_CLASSES
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 6) if n <= 12 else (10, 9))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Fraction of true class')

    if n <= 12:
        for i in range(n):
            for j in range(n):
                v = cm_norm[i, j]
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=7, color='white' if v > 0.55 else 'black')

    ax.set_xticks(range(n))
    ax.set_xticklabels(CLASS_NAMES, fontsize=9 if n <= 12 else 7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(CLASS_NAMES, fontsize=9 if n <= 12 else 7)
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('True class')
    ax.set_title(title)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, fname)
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  -> {path}")


def plot_l1_pixel(W_l0, W_l1, img_shape, fname='mnist_filters_l1.png'):
    """
    Project L1 filters back to pixel space: W_l1 @ W_l0 → (N_L1, 784).
    Each row is the weighted combination of L0 edge detectors that this L1
    code cell responds to most strongly.
    """
    W_pixel = W_l1 @ W_l0   # (N_L1, 784)
    plot_filters(
        W_pixel, img_shape, n_grid_cols=4,
        title=f'L1 filters projected to pixel space  ({N_L1} codes, each = Σ L0 weights)',
        fname=fname,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    topo = print_topology()

    # ── Data ──────────────────────────────────────────────────────────────────
    if DATASET == 'mnist':
        X_train_raw, y_train, X_test_raw, y_test = load_mnist()
    else:
        X_train_raw, y_train, X_test_raw, y_test = load_emnist_letters()
    X_train, X_test, _ = preprocess(X_train_raw, X_test_raw)
    print(f"\n  Preprocessing: pixel mean subtracted, per-sample L2 normalised")
    print(f"  X_train {X_train.shape}  X_test {X_test.shape}  dtype={X_train.dtype}")

    rng       = np.random.default_rng(42)
    eval_idx  = rng.choice(len(X_train), size=EVAL_N, replace=False)
    X_eval    = X_train[eval_idx]

    # ── Layer 0 GHA ───────────────────────────────────────────────────────────
    print(f"\n── L0 GHA training  ({N_IN} → {N_L0})"
          f"  [{topo['l0_chips']} chips]  {N_EPOCHS_L0} epochs ──")
    L0 = GHALayer(N_L0, N_IN, lr=LR_L0, seed=0)
    log_l0 = train_gha(L0, X_train, N_EPOCHS_L0, LR_L0, LR_L0_MIN,
                        label='L0', X_eval=X_eval)

    # ── Layer 1 GHA ───────────────────────────────────────────────────────────
    # L1 is trained on the L0 feature space, centred and normalised
    print(f"\n── L0 feature extraction for L1 input ──")
    F_for_l1 = extract_features(L0, X_train, relu=True)   # (60K, 64)
    # Centre so GHA has zero-mean inputs (required for correct PC extraction)
    F_l1_mean  = F_for_l1.mean(axis=0)
    F_l1_ctrd  = F_for_l1 - F_l1_mean
    l1_norms   = np.linalg.norm(F_l1_ctrd, axis=1, keepdims=True)
    F_l1_ctrd /= np.maximum(l1_norms, 1e-8)
    eval_l1    = F_l1_ctrd[eval_idx]

    print(f"\n── L1 GHA training  ({N_L0} → {N_L1})"
          f"  [{topo['l1_chips']} chips]  {N_EPOCHS_L1} epochs ──")
    L1 = GHALayer(N_L1, N_L0, lr=LR_L1, seed=1)
    log_l1 = train_gha(L1, F_l1_ctrd, N_EPOCHS_L1, LR_L1, LR_L1_MIN,
                        label='L1', X_eval=eval_l1)

    # ── Feature extraction for classifier ─────────────────────────────────────
    print("\n── Feature extraction ──")
    F_tr_fl  = extract_features(L0, X_train, relu=True)   # (60K, 64) float weights
    F_te_fl  = extract_features(L0, X_test,  relu=True)

    if QUANTISE_INF:
        W_q_saved  = L0.W.copy()
        L0.W       = _quantise(W_q_saved)
        F_tr_q     = extract_features(L0, X_train, relu=True)
        F_te_q     = extract_features(L0, X_test,  relu=True)
        L0.W       = W_q_saved                          # restore float weights
        delta_W    = float(np.mean(np.abs(_quantise(W_q_saved) - W_q_saved)))
        print(f"  L0 weight quantisation:  mean |ΔW| = {delta_W:.5f}"
              f"  (step = {1/CODE_SCALE:.4f}, element scale ≈ {1/math.sqrt(N_IN):.4f})")

    # ── Classification ────────────────────────────────────────────────────────
    print(f"\n── Classification  ({CLASSIFIER.upper()}) ──")
    results = {}

    if CLASSIFIER == 'logistic':
        print("  Training LogisticRegression on float L0 features ...")
        acc, pred, _ = classify_logistic(F_tr_fl, y_train, F_te_fl, y_test)
        results['logistic  / float L0'] = (acc, pred)
        print(f"  logistic / float   {acc * 100:.2f}%")

        if QUANTISE_INF:
            print("  Training LogisticRegression on 8-bit L0 features ...")
            acc_q, pred_q, _ = classify_logistic(F_tr_q, y_train, F_te_q, y_test)
            results['logistic  / 8-bit L0'] = (acc_q, pred_q)
            print(f"  logistic / 8-bit   {acc_q * 100:.2f}%")
    else:
        print("  Least-squares classifier on float L0 features ...")
        acc, pred, _ = classify_lstsq(F_tr_fl, y_train, F_te_fl, y_test)
        results['lstsq     / float L0'] = (acc, pred)
        print(f"  lstsq / float      {acc * 100:.2f}%")

        if QUANTISE_INF:
            print("  Least-squares classifier on 8-bit L0 features ...")
            acc_q, pred_q, _ = classify_lstsq(F_tr_q, y_train, F_te_q, y_test)
            results['lstsq     / 8-bit L0'] = (acc_q, pred_q)
            print(f"  lstsq / 8-bit      {acc_q * 100:.2f}%")

    # ── Results summary ────────────────────────────────────────────────────────
    print()
    print("━" * 62)
    print("RESULTS SUMMARY")
    print("━" * 62)
    print(f"  Architecture    : PCN  {N_IN} → {N_L0} → {N_L1}")
    print(f"  Chips           : {topo['total_chips']}"
          f"  ({topo['l0_chips']} L0 + {topo['l1_chips']} L1,  Sky130A)")
    print(f"  Weight cells    : {topo['total_cells']:,}")
    print(f"  Learning        : GHA  {N_EPOCHS_L0} + {N_EPOCHS_L1} epochs"
          f"  (unsupervised, {len(X_train):,} samples)")
    print()
    best_acc, best_pred = 0.0, None
    best_label = ''
    for label, (acc, pred) in results.items():
        hw_note = '  ← hardware-faithful' if '8-bit' in label else ''
        print(f"  {label}  :  {acc * 100:6.2f}%{hw_note}")
        if acc > best_acc:
            best_acc, best_pred, best_label = acc, pred, label
    print()
    print(f"  Best result     : {best_acc * 100:.2f}%  ({best_label.strip()})")
    print()
    if QUANTISE_INF and len(results) >= 2:
        vals = list(results.values())
        drop = (vals[0][0] - vals[1][0]) * 100
        print(f"  8-bit overhead  : {drop:+.2f} pp  "
              f"({'better' if drop < 0 else 'worse'} than float)")
    print()

    # Per-class accuracy for the best result
    print("  Per-class accuracy (best result):")
    per_class = []
    for d in range(N_CLASSES):
        mask = y_test == d
        class_acc = float(np.mean(best_pred[mask] == d))
        per_class.append(class_acc)
        print(f"    class {CLASS_NAMES[d]:>2} : {class_acc * 100:5.1f}%  ({mask.sum()} samples)")
    worst  = int(np.argmin(per_class))
    best_c = int(np.argmax(per_class))
    print(f"  Hardest class : {CLASS_NAMES[worst]} ({per_class[worst]*100:.1f}%)   "
          f"Easiest : {CLASS_NAMES[best_c]} ({per_class[best_c]*100:.1f}%)")
    print()

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("── Saving plots ──")
    plot_filters(
        L0.W, (28, 28), n_grid_cols=8,
        title=(f'L0 learned filters  ({N_IN}→{N_L0}, GHA, {N_EPOCHS_L0} epochs)  '
               f'— {topo["l0_chips"]} Sky130A chips'),
        fname=f'{RESULTS_TAG}_filters_l0.png',
    )
    plot_l1_pixel(L0.W, L1.W, (28, 28), fname=f'{RESULTS_TAG}_filters_l1.png')
    plot_training(
        [log_l0, log_l1],
        [f'L0  ({N_L0} features, {topo["l0_chips"]} chips)',
         f'L1  ({N_L1} codes,    {topo["l1_chips"]} chips)'],
        fname=f'{RESULTS_TAG}_training.png',
    )
    plot_confusion(
        y_test, best_pred,
        title=f'PCN {DATASET.upper()}  {best_label.strip()}  —  {best_acc * 100:.2f}%',
        fname=f'{RESULTS_TAG}_confusion.png',
    )
    print("\n  All outputs written to sim/results/")


if __name__ == '__main__':
    main()
