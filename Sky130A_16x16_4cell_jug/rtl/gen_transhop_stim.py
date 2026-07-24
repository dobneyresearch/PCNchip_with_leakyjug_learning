#!/usr/bin/env python3
"""gen_transhop_stim.py — golden for tb_transpose_hop (the transpose-at-source composition test).

Proves the NEW split composes: pcn_transpose (chip: Wᵀδ from its own W-SRAM) feeding router_gather
(router: accumulate-per-dest + avg_bp) must reproduce the OLD fused hop (twin_hop_generic), which
hop_engine used to compute in one module.

HOP1: 4 sources → 2 dests, fan-in 2 (sources 0,1→dest0 ; 2,3→dest1).

★ Tied to CODE_MID = PCN_WGT_ZERO = 117 (read from pcn_weight_params.vh), exactly like
gen_backproj_stim.py — the on-chip transpose uses 117, not the multi-array twin's default 132.

Writes: hop1_w.hex (4×256 codes), hop1_src.hex (4×16 int8), l2_expected.hex (2×16 int8).
"""
import os, sys, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'hw_multi_array_l3_fable', 'sim'))
import pcn_router_backproj as twin                                    # noqa: E402
from pcn_router_backproj import twin_hop_generic, WGT_MIN, WGT_MAX    # noqa: E402


def _read_wgt_zero():
    vh = os.path.join(HERE, 'pcn_weight_params.vh')
    m = re.search(r'`define\s+PCN_WGT_ZERO\s+(\d+)', open(vh).read())
    if not m:
        raise RuntimeError("PCN_WGT_ZERO not found in pcn_weight_params.vh")
    return int(m.group(1))


twin.CODE_MID = _read_wgt_zero()      # 117 — same zero the on-chip transpose uses
print(f"  CODE_MID tied to PCN_WGT_ZERO = {twin.CODE_MID}")

N = 16
rng = np.random.default_rng(3)

# HOP1: 4 sources → 2 dests, fan-in 2
W1  = [rng.integers(WGT_MIN, WGT_MAX + 1, (N, N)) for _ in range(4)]
S1  = [rng.integers(-20, 21, N) for _ in range(4)]
dst1_ids, fanin1 = [0, 0, 1, 1], [2, 2]
L2 = twin_hop_generic(W1, S1, dst1_ids, fanin1, 2)                    # (2,16) int8


def wblocks(path, blocks):
    with open(os.path.join(HERE, path), 'w') as f:
        for W in blocks:
            for i in range(N):
                for j in range(N):
                    f.write("%02x\n" % int(W[i][j]))

def vecs(path, rows):
    with open(os.path.join(HERE, path), 'w') as f:
        for r in rows:
            for v in r:
                f.write("%02x\n" % (int(v) & 0xff))


wblocks('hop1_w.hex', W1);  vecs('hop1_src.hex', S1);  vecs('l2_expected.hex', L2)
print("wrote transhop stimulus: hop1_w.hex(4×256), hop1_src.hex(4×16), l2_expected.hex(2×16)")
