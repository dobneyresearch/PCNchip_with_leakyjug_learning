#!/usr/bin/env python3
"""backprop_rig_big.py — the FULL-BACKPROP CEILING on the BIG rig.

WHY: on smallBIG we assumed we were near the ceiling, built an honest torch backprop reference
on the IDENTICAL topology, and found 4pp of headroom we could not see from inside our own rig
(and, chasing it, found the degenerate-softmax bug). Do the same on BIG before we believe our
79.19%.

Reproduces the BIG chip EXACTLY:
  L1 : 24 chips, 48 -> 16   (contiguous windows of the 1152 split-sign L0 feats)
  L2 :  8 chips, 48 -> 16   (contiguous: each L2 chip reads its 3 L1 chips)
  L3 : 16 chips, 32 -> 16   *** OVERLAPPING ROUTING ***
         L3_ROUTING = [[k, (k+1)%8] for k in 0..7] + [[k, (k+2)%8] for k in 0..7]
         every adjacent pair + every skip-2 pair of the 8 L2 chips; each L2 chip feeds
         exactly 4 L3 chips. This is a GATHER, not a contiguous reshape — smallBIG's rig
         could get away with a reshape, BIG cannot.
  readout: 256 -> 26 (with bias).  leaky-0.1, NO hidden bias (matches the chip MAC).

  --quant : straight-through 8-bit weight quantisation on the hardware grid
            clamp(round(w*64), -57, 64)/64  ==  our [-0.89, 1.0] CODE_MID=132 rail.
  --dense : full-connectivity control (what the chip-factoring costs)
  --linear: identity activations (what the nonlinearity buys)

Usage:
    python3 backprop_rig_big.py --quant                       # the honest ceiling
    python3 backprop_rig_big.py --quant --sgd --signsgd --momentum 0.9 --lr 1e-3
    python3 backprop_rig_big.py --dense                       # unconstrained ceiling
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

N_ROWS = 16
LEAKY = 0.1
N_CLASSES = 26
N_L1_CHIPS, N_L2_CHIPS, N_L3_CHIPS = 24, 8, 16
N_L1_PER_L2 = 3
L3_ROUTING = ([[k, (k + 1) % N_L2_CHIPS] for k in range(N_L2_CHIPS)] +
              [[k, (k + 2) % N_L2_CHIPS] for k in range(N_L2_CHIPS)])

DIR = os.path.dirname(os.path.abspath(__file__))
BIG_W = os.path.join(os.path.dirname(DIR), 'multi_array_level3', 'weights_big_emnist')


def q8(w, quant):
    """Straight-through 8-bit on the HARDWARE weight grid (CODE_MID=132 => [-0.89, 1.0])."""
    if not quant:
        return w
    q = torch.clamp(torch.round(w * 64.0), -57, 64) / 64.0
    return w + (q - w).detach()


class Grouped(nn.Module):
    """Block-grouped linear over CONTIGUOUS windows (L1, L2). No bias — matches the chip MAC."""
    def __init__(self, n_chips, chip_in, chip_out, quant):
        super().__init__()
        self.n, self.i, self.o, self.quant = n_chips, chip_in, chip_out, quant
        self.w = nn.Parameter(torch.empty(n_chips, chip_out, chip_in))
        nn.init.kaiming_uniform_(self.w, a=5 ** 0.5)

    def forward(self, x):
        x = x.view(x.shape[0], self.n, self.i)
        y = torch.einsum('coi,nci->nco', q8(self.w, self.quant), x)
        return y.reshape(x.shape[0], self.n * self.o)


class RoutedL3(nn.Module):
    """L3: 16 chips, each GATHERING 2 of the 8 L2 chips per L3_ROUTING (OVERLAPPING).

    Each L2 chip feeds 4 different L3 chips, so this cannot be a reshape. We build a gather
    index once and use it every forward — the same fan-out the chip's router has."""
    def __init__(self, quant):
        super().__init__()
        self.quant = quant
        self.w = nn.Parameter(torch.empty(N_L3_CHIPS, N_ROWS, 2 * N_ROWS))
        nn.init.kaiming_uniform_(self.w, a=5 ** 0.5)
        idx = []
        for g in range(N_L3_CHIPS):
            a, b = L3_ROUTING[g]
            idx += list(range(a * N_ROWS, (a + 1) * N_ROWS))
            idx += list(range(b * N_ROWS, (b + 1) * N_ROWS))
        self.register_buffer('idx', torch.tensor(idx, dtype=torch.long))   # (16*32,)

    def forward(self, x):                       # x: (N, 128)
        g = x[:, self.idx].view(x.shape[0], N_L3_CHIPS, 2 * N_ROWS)
        y = torch.einsum('coi,nci->nco', q8(self.w, self.quant), g)
        return y.reshape(x.shape[0], N_L3_CHIPS * N_ROWS)                  # (N, 256)


class BigChipNet(nn.Module):
    def __init__(self, in_dim, quant=False, linear=False):
        super().__init__()
        act = (lambda: nn.Identity()) if linear else (lambda: nn.LeakyReLU(LEAKY))
        self.l1, self.a1 = Grouped(N_L1_CHIPS, 48, N_ROWS, quant), act()
        self.l2, self.a2 = Grouped(N_L2_CHIPS, 48, N_ROWS, quant), act()
        self.l3, self.a3 = RoutedL3(quant), act()
        self.clf = nn.Linear(N_L3_CHIPS * N_ROWS, N_CLASSES)

    def forward(self, x):
        x = self.a1(self.l1(x))
        x = self.a2(self.l2(x))
        x = self.a3(self.l3(x))
        return self.clf(x)


class Dense(nn.Module):
    """Full-connectivity control — the same widths, no chip-factoring. Shows what the
    block-grouping costs."""
    def __init__(self, in_dim, linear=False):
        super().__init__()
        act = (lambda: nn.Identity()) if linear else (lambda: nn.LeakyReLU(LEAKY))
        self.net = nn.Sequential(nn.Linear(in_dim, 384), act(),
                                 nn.Linear(384, 128), act(),
                                 nn.Linear(128, 256), act(),
                                 nn.Linear(256, N_CLASSES))

    def forward(self, x):
        return self.net(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--quant', action='store_true', help='8-bit STE on the hardware weight grid')
    p.add_argument('--dense', action='store_true', help='full-connectivity ceiling')
    p.add_argument('--linear', action='store_true', help='identity activations')
    p.add_argument('--sgd', action='store_true')
    p.add_argument('--momentum', type=float, default=0.0)
    p.add_argument('--signsgd', action='store_true')
    a = p.parse_args()

    F_tr = np.load(os.path.join(BIG_W, 'F_l0_train.npy')).astype(np.float32)
    F_te = np.load(os.path.join(BIG_W, 'F_l0_test.npy')).astype(np.float32)
    y_tr = np.load(os.path.join(BIG_W, 'y_train.npy')).astype(np.int64)
    y_te = np.load(os.path.join(BIG_W, 'y_test.npy')).astype(np.int64)
    # the chip sees ACT CODES (0..255 over 0..1.8V), so match that, then scale to O(1)
    enc = lambda F: np.clip(np.round(np.clip(F, 0, 1.8) / 1.8 * 255), 0, 255) / 255.0
    Xtr = torch.tensor(enc(F_tr)); Xte = torch.tensor(enc(F_te))
    Ytr = torch.tensor(y_tr);      Yte = torch.tensor(y_te)
    in_dim = Xtr.shape[1]

    if a.dense:
        model = Dense(in_dim, a.linear); tag = 'DENSE (full connectivity)'
    else:
        model = BigChipNet(in_dim, a.quant, a.linear)
        tag = f"BIG CHIP-FACTORED (24/8/16, overlapping L3){' +8bit-STE' if a.quant else ' float'}"
    if a.linear:
        tag += ' [LINEAR]'

    if a.sgd:
        opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=a.momentum)
        oname = f'SGD lr={a.lr} mom={a.momentum}' + (' SIGN' if a.signsgd else '')
    else:
        opt = torch.optim.Adam(model.parameters(), lr=a.lr)
        oname = f'Adam lr={a.lr}'

    print(f"\n  {tag}\n  {oname}  batch={a.batch}  in={in_dim}  "
          f"params={sum(p.numel() for p in model.parameters()):,}", flush=True)

    lossf = nn.CrossEntropyLoss()
    best = 0.0
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), a.batch):
            b = perm[i:i + a.batch]
            opt.zero_grad()
            lossf(model(Xtr[b]), Ytr[b]).backward()
            if a.signsgd:
                for prm in model.parameters():
                    if prm.grad is not None:
                        prm.grad = prm.grad.sign()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = sum((model(Xte[i:i+2048]).argmax(1) == Yte[i:i+2048]).sum().item()
                      for i in range(0, len(Xte), 2048)) / len(Xte)
        best = max(best, acc)
        if ep % 5 == 0 or ep == 1:
            print(f"    ep{ep:3d}  test={acc*100:5.2f}%  (best {best*100:.2f}%)  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    print(f"\n  >>> {tag} | {oname} : BEST TEST = {best*100:.2f}%\n", flush=True)


if __name__ == '__main__':
    main()
