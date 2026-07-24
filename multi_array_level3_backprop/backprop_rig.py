#!/usr/bin/env python3
"""
Backprop ceiling rig — the "what is possible" reference for the forwards-only PCN.

Fixes L0 exactly as every other test (frozen 576-dim ReLU'd PCA features, loaded
from the small sim's l0_cache_emnist), and trains L1/L2/L3 by BACKPROP on the SAME
chip-factored connectivity as the small model:

    L0(576) → L1(12 chips, 48→16) → L2(4 chips, 48→16) → L3(2 chips, 32→16) → clf(26)

Each chip reads a contiguous, non-overlapping window of its layer's input (block-
grouped), leaky-ReLU (α=0.1), no hidden bias (matches the chip MAC, which has no
bias term); the 26-way readout has a bias (matches fit_clf).  This isolates the ONE
variable we care about — the learning rule — against our forwards-only rule run on
the identical architecture (fixed L0, same sizes, same connectivity).

Reports the best test accuracy = the backprop ceiling for this architecture.

Caveat: FLOAT weights (no 8-bit quantisation).  This is the clean learning ceiling;
our forwards-only rig is quantised, so its gap to this number = (learning rule) +
(quantisation).  Run --quant to add straight-through 8-bit weight quantisation and
separate the two.  --dense gives the unconstrained (full-connectivity) ceiling.

Usage:
    python3 backprop_rig.py                 # chip-factored, float
    python3 backprop_rig.py --dense         # full-connectivity ceiling
    python3 backprop_rig.py --quant         # 8-bit STE weight quantisation
"""
import argparse, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DIR = os.path.dirname(os.path.abspath(__file__))
L0_CACHE = os.path.join(os.path.dirname(DIR), 'multi_array_level3_L0A', 'l0_cache_emnist')

N_ROWS = 16
LEAKY = 0.1
# chip-factored spec: (n_chips, chip_in, chip_out) per layer; each chip reads a
# contiguous window of size chip_in from the previous layer's output.
LAYERS = [(12, 48, 16),   # L1: 576 → 192
          (4,  48, 16),   # L2: 192 → 64
          (2,  32, 16)]   # L3: 64  → 32
N_CLASSES = 26


class GroupedLinear(nn.Module):
    """Block-grouped linear: input (N, n_chips*chip_in) reshaped to (N,n_chips,chip_in),
    each chip applies its own (chip_out,chip_in) weight → (N, n_chips*chip_out).
    No bias (matches the chip MAC)."""
    def __init__(self, n_chips, chip_in, chip_out, quant=False):
        super().__init__()
        self.n_chips, self.chip_in, self.chip_out = n_chips, chip_in, chip_out
        self.quant = quant
        self.w = nn.Parameter(torch.empty(n_chips, chip_out, chip_in))
        nn.init.kaiming_uniform_(self.w, a=5 ** 0.5)

    def _wq(self):
        if not self.quant:
            return self.w
        # straight-through 8-bit: signed weight = round(w*64) clipped to [-57,64], /64
        q = torch.clamp(torch.round(self.w * 64.0), -57, 64) / 64.0
        return self.w + (q - self.w).detach()

    def forward(self, x):
        x = x.view(x.shape[0], self.n_chips, self.chip_in)
        y = torch.einsum('coi,nci->nco', self._wq(), x)   # w:(chips,out,in) x:(N,chips,in)
        return y.reshape(x.shape[0], self.n_chips * self.chip_out)


class ChipNet(nn.Module):
    def __init__(self, in_dim, quant=False, linear=False):
        super().__init__()
        self.linear = linear
        self.l1 = GroupedLinear(*LAYERS[0], quant=quant)
        self.l2 = GroupedLinear(*LAYERS[1], quant=quant)
        self.l3 = GroupedLinear(*LAYERS[2], quant=quant)
        self.clf = nn.Linear(LAYERS[2][0] * LAYERS[2][2], N_CLASSES)   # 32 → 26, with bias

    def _act(self, x):
        return x if self.linear else F.leaky_relu(x, LEAKY)

    def forward(self, x):
        x = self._act(self.l1(x))
        x = self._act(self.l2(x))
        x = self._act(self.l3(x))
        return self.clf(x)


class DenseNet(nn.Module):
    """Unconstrained full-connectivity MLP with the SAME layer sizes (192/64/32)."""
    def __init__(self, in_dim, linear=False):
        super().__init__()
        act = (lambda: nn.Identity()) if linear else (lambda: nn.LeakyReLU(LEAKY))
        self.net = nn.Sequential(
            nn.Linear(in_dim, 192), act(),
            nn.Linear(192, 64),     act(),
            nn.Linear(64, 32),      act(),
            nn.Linear(32, N_CLASSES))

    def forward(self, x):
        return self.net(x)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dense', action='store_true', help='full-connectivity ceiling')
    p.add_argument('--linear', action='store_true', help='identity activations = the (restricted) linear model for this connectivity')
    p.add_argument('--quant', action='store_true', help='8-bit STE weight quantisation (chip-factored only)')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    Xtr = np.load(os.path.join(L0_CACHE, 'F_l0_train.npy')).astype(np.float32)
    Xte = np.load(os.path.join(L0_CACHE, 'F_l0_test.npy')).astype(np.float32)
    ytr = np.load(os.path.join(L0_CACHE, 'y_train.npy')).astype(np.int64)
    yte = np.load(os.path.join(L0_CACHE, 'y_test.npy')).astype(np.int64)
    in_dim = Xtr.shape[1]
    print(f"L0 features: train {Xtr.shape} / test {Xte.shape}  (fixed, frozen PCA)", flush=True)

    Xtr_t = torch.tensor(Xtr, device=dev); ytr_t = torch.tensor(ytr, device=dev)
    Xte_t = torch.tensor(Xte, device=dev); yte_t = torch.tensor(yte, device=dev)

    lin = ' LINEAR' if args.linear else ''
    if args.dense:
        model = DenseNet(in_dim, linear=args.linear).to(dev)
        tag = f'DENSE (full connectivity){lin}'
    else:
        model = ChipNet(in_dim, quant=args.quant, linear=args.linear).to(dev)
        tag = f"CHIP-FACTORED (matched){' +8bit-STE' if args.quant else ' float'}{lin}"
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {tag}  |  trainable params: {n_par:,}", flush=True)
    print(f"Architecture: 576(fixed L0)→192(L1)→64(L2)→32(L3)→clf(26), leaky α={LEAKY}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n = len(Xtr_t); best = 0.0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad()
            loss = F.cross_entropy(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            tr = (model(Xtr_t).argmax(1) == ytr_t).float().mean().item() * 100
            te = (model(Xte_t).argmax(1) == yte_t).float().mean().item() * 100
        best = max(best, te)
        if ep % 5 == 0 or ep == 1:
            print(f"  ep {ep:3d}  train {tr:5.2f}%  test {te:5.2f}%  (best {best:5.2f}%)  "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    print(f"\n{'='*60}")
    print(f"  BACKPROP CEILING [{tag}]: best test = {best:.2f}%")
    print(f"  (fixed L0; L1/L2/L3 trained by backprop; leaky α={LEAKY})")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
