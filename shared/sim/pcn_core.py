"""
pcn_core.py — Hardware-faithful software model of the PCN chip MAC layer.

Models:
  - Current-mode matrix-vector multiply (KCL summation)
  - Precision-gated Hebbian weight update
  - 8-bit weight quantisation (CMOS TG range, codes 71-192)
  - Three learning modes:
      'v1'  : LTP-only (current hardware, one-quadrant hebbian_mult)
               Only underprediction fires; overprediction clamped at zero.
      'bcm' : V1 hardware + BCM firmware threshold (practical V1 workaround)
               Disables updates for cells whose running activity exceeds a target.
      'v2'  : Four-quadrant (Gilbert cell upgrade, full LTP+LTD)
               Signed error outer-product; the correct PCN Hebbian rule.

Hardware constants (Sky130A, VDD=1.8V):
  CODE_MIN = 71   → Vw = 0.50 V  (CMOS TG lower limit)
  CODE_MID = 128  → Vw = 0.90 V  (zero-weight operating point = Vcm)
  CODE_MAX = 192  → Vw = 1.35 V  (CMOS TG upper limit)
  CODE_SCALE = 64 codes per unit weight (maps code→weight linearly about mid)
  Resulting weight range: (71-128)/64 = -0.891 to (192-128)/64 = +1.000

The asymmetry (-0.891 to +1.000) reflects the real hardware: the CMOS TG
allows more upward headroom from mid-scale than downward.

Reference: pred_code_networks.md §§25-29 (hebbian_mult), §5902 (R16, LTP risk).
"""

import numpy as np


# ── Hardware constants ────────────────────────────────────────────────────────

CODE_MIN   = 71    # lowest usable code (Vw = 0.50 V)
CODE_MID   = 128   # mid-scale / zero-weight (Vw = 0.90 V = Vcm)
CODE_MAX   = 192   # highest usable code (Vw = 1.35 V)
CODE_SCALE = 64.0  # codes per unit weight; weight = (code - CODE_MID) / CODE_SCALE

W_MIN = (CODE_MIN - CODE_MID) / CODE_SCALE   # = -0.891
W_MAX = (CODE_MAX - CODE_MID) / CODE_SCALE   # = +1.000


def _quantise(W):
    """Round weights to 8-bit hardware codes and clip to usable range."""
    codes = np.round(W * CODE_SCALE + CODE_MID).astype(int)
    codes = np.clip(codes, CODE_MIN, CODE_MAX)
    return (codes - CODE_MID) / CODE_SCALE


# ── PCN layer ─────────────────────────────────────────────────────────────────

class PCNLayer:
    """
    One physical 16×16 (or N_rows×N_cols) PCN MAC array.

    Parameters
    ----------
    n_rows : int
        Output dimension (number of prediction rows).
    n_cols : int
        Input dimension (number of MAC columns).
    lr : float
        Learning rate. Hardware equivalent: I_hebb × t_pulse / Cw per step.
    threshold : float
        Precision-gate threshold in normalised weight units.
        Hardware equivalent: V_pi / R_err mapped to the normalised domain.
    mode : str
        'v1'  — LTP-only (current silicon).
        'bcm' — V1 + BCM firmware threshold.
        'v2'  — Four-quadrant signed Hebbian (Gilbert cell upgrade).
    quantise : bool
        Apply 8-bit hardware quantisation after each weight update.
    bcm_target : float
        For 'bcm' mode: target mean squared activation; BCM threshold
        Θ_m = bcm_decay × Θ_m + (1-bcm_decay) × r_m²; update disabled when r_m² > Θ_m.
    bcm_decay : float
        Exponential decay rate for the per-row BCM threshold.
    seed : int or None
        RNG seed for reproducible weight initialisation.
    """

    def __init__(self, n_rows, n_cols, lr=0.01, threshold=0.05,
                 mode='v2', quantise=True,
                 bcm_target=0.25, bcm_decay=0.99,
                 w_init_scale=0.05, normalize_rows=False,
                 k_wta=None,
                 seed=None):
        if mode not in ('v1', 'bcm', 'v2'):
            raise ValueError("mode must be 'v1', 'bcm', or 'v2'")

        self.n_rows     = n_rows
        self.n_cols     = n_cols
        self.lr         = lr
        self.threshold  = threshold
        self.mode       = mode
        self.quantise   = quantise
        self.bcm_target     = bcm_target
        self.bcm_decay      = bcm_decay
        self.normalize_rows = normalize_rows
        self.k_wta          = k_wta

        rng = np.random.default_rng(seed)
        # Start near mid-scale; w_init_scale controls symmetry-breaking diversity
        self.W = rng.standard_normal((n_rows, n_cols)) * w_init_scale
        if quantise:
            self.W = _quantise(self.W)

        # BCM per-row sliding threshold (initialised to target)
        self._bcm_theta = np.full(n_rows, bcm_target)

        # Metrics accumulated during training
        self.history = {
            'step':        [],
            'pred_error':  [],   # mean |error| before gate
            'gate_frac':   [],   # fraction of rows where gate fired
            'w_std':       [],   # std of weight matrix (diversity measure)
        }

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, x):
        """
        Current-mode MAC: y = W @ x.
        Each row i accumulates Σ_j W_ij × x_j on the KCL bus.

        Parameters
        ----------
        x : (n_cols,) array — input activations.

        Returns
        -------
        y : (n_rows,) array — row output currents (normalised).
        """
        return self.W @ x

    def predict_down(self, y):
        """
        Top-down reconstruction: x̂ = W^T @ y.
        Uses the transpose weight matrix to generate a prediction of the input.
        Note: the current V1 chip approximates this with a resistor divider
        (layer_link); a full transpose is a V2 capability.

        Parameters
        ----------
        y : (n_rows,) array — hidden representation.

        Returns
        -------
        x_hat : (n_cols,) array — predicted input.
        """
        return self.W.T @ y

    def error(self, y, y_pred):
        """
        Row-level prediction error: ε_i = y_i - y_pred_i.

        In the hardware: current subtractor computes I_out - I_pred per row.
        Positive ε: actual output > prediction (underprediction → LTP candidate).
        Negative ε: actual output < prediction (overprediction → LTD candidate,
                    clamped to 0 in V1 hardware).

        Parameters
        ----------
        y      : (n_rows,) — actual MAC output.
        y_pred : (n_rows,) — top-down prediction current (may be zero).

        Returns
        -------
        eps : (n_rows,) — signed prediction error.
        """
        return y - y_pred

    # ── Weight update ─────────────────────────────────────────────────────────

    def update(self, x, y, y_pred, record=False):
        """
        Precision-gated Hebbian weight update.

        V1  mode: ε_i = max(0, y_i - y_pred_i)   [PMOS clamp; LTP only]
                  gate_i = ε_i > threshold
                  ΔW_ij = lr × gate_i × ε_i × x_j

        BCM mode: same as V1 but with BCM disable mask:
                  disable_i = (y_i² > Θ_i)
                  Θ_i ← decay × Θ_i + (1-decay) × y_i²

        V2  mode: ε_i = y_i - y_pred_i            [signed; full LTP+LTD]
                  gate_i = |ε_i| > threshold
                  ΔW_ij = lr × gate_i × ε_i × x_j

        Parameters
        ----------
        x      : (n_cols,) — input activations (pre-synaptic).
        y      : (n_rows,) — MAC output (post-synaptic activation).
        y_pred : (n_rows,) — top-down prediction; zeros if no prediction fed in.
        record : bool      — if True, append metrics to self.history.
        """
        eps_signed = y - y_pred

        if self.mode == 'v1':
            eps = np.maximum(0.0, eps_signed)     # PMOS clamp: no LTD
            gate = eps > self.threshold
            dW = self.lr * np.outer(gate * eps, x)

        elif self.mode == 'bcm':
            eps = np.maximum(0.0, eps_signed)
            gate = eps > self.threshold
            # BCM: disable rows whose output² exceeds running threshold
            bcm_mask = (y ** 2) <= self._bcm_theta
            self._bcm_theta = (self.bcm_decay * self._bcm_theta
                               + (1 - self.bcm_decay) * y ** 2)
            dW = self.lr * np.outer(gate * bcm_mask * eps, x)

        else:  # v2
            eps = eps_signed
            gate = np.abs(eps) > self.threshold
            dW = self.lr * np.outer(gate * eps, x)

        # Optional k-WTA override: keep only the k rows with highest |eps|.
        # Models a hardware row-arbiter that selects top-k error signals.
        if self.k_wta is not None and self.k_wta < self.n_rows:
            k = int(self.k_wta)
            abs_eps = np.abs(eps_signed)
            cutoff = np.sort(abs_eps)[-k]  # k-th largest |eps|
            wta_mask = abs_eps >= cutoff
            dW = dW * wta_mask[:, np.newaxis]

        self.W += dW
        if self.normalize_rows:
            # Oja weight normalization: normalise each row to unit L2 norm,
            # then re-quantise.  Models periodic firmware normalisation pass.
            norms = np.linalg.norm(self.W, axis=1, keepdims=True)
            self.W = self.W / np.maximum(norms, 1e-8)
        if self.quantise:
            self.W = _quantise(self.W)

        if record:
            self.history['pred_error'].append(float(np.mean(np.abs(eps_signed))))
            self.history['gate_frac'].append(float(np.mean(gate)))
            self.history['w_std'].append(float(np.std(self.W)))

    def reset_history(self):
        for k in self.history:
            self.history[k] = []


# ── Temporal reuse stack ──────────────────────────────────────────────────────

class PCNTemporalStack:
    """
    N_virt virtual PCN layers time-multiplexed on one physical MAC array.

    Each virtual layer has its own weight matrix stored in 'SRAM'
    (a Python list of PCNLayer objects sharing the same n_rows/n_cols).
    Inference proceeds sequentially: activations from VL_k are digitised
    (quantised to 8 bits, mimicking the SAR ADC) and presented as inputs
    to VL_{k+1}.

    Parameters
    ----------
    n_rows, n_cols : int   — physical array dimensions.
    n_virt         : int   — number of virtual layers (up to 8 in hardware).
    adc_bits       : int   — activation quantisation bits (8 in hardware).
    kwargs              — passed to each PCNLayer constructor.
    """

    def __init__(self, n_rows, n_cols, n_virt=4,
                 adc_bits=8, **kwargs):
        self.n_rows  = n_rows
        self.n_cols  = n_cols
        self.n_virt  = n_virt
        self.adc_bits = adc_bits
        self.layers  = [PCNLayer(n_rows, n_cols, **kwargs)
                        for _ in range(n_virt)]

    def _adc_quantise(self, act, v_min=0.0, v_max=1.8):
        """Simulate 8-bit SAR ADC: uniform quantisation over [v_min, v_max]."""
        levels = 2 ** self.adc_bits
        act_clipped = np.clip(act, v_min, v_max)
        codes = np.round((act_clipped - v_min) / (v_max - v_min) * (levels - 1))
        return codes / (levels - 1) * (v_max - v_min) + v_min

    def forward(self, x, quantise_activations=True):
        """
        Run one forward pass through all N_virt virtual layers.

        Returns
        -------
        activations : list of (n_rows,) arrays — one per virtual layer.
        """
        act = x
        activations = []
        for layer in self.layers:
            y = layer.forward(act)
            if quantise_activations:
                y = self._adc_quantise(y)
            activations.append(y)
            act = y   # output of VL_k becomes input to VL_{k+1}
        return activations

    def update_all(self, x, predictions=None, record=False):
        """
        Update all virtual layers given input x and optional predictions.

        predictions : list of (n_rows,) arrays or None.
            If None, all predictions are zero (no top-down signal — V1 mode).
        """
        if predictions is None:
            predictions = [np.zeros(self.n_rows)] * self.n_virt

        act = x
        for k, (layer, pred) in enumerate(zip(self.layers, predictions)):
            y = layer.forward(act)
            layer.update(act, y, pred, record=record)
            act = self._adc_quantise(y)
