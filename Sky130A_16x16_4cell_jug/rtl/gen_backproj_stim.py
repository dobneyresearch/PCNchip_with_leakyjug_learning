#!/usr/bin/env python3
"""Generate co-sim stimulus for tb_router_backproj from the validated B1 twin.

Reuses hw_multi_array_l3_fable/sim/pcn_router_backproj.py::twin_block so the RTL is
checked against the SAME golden the twin was validated to (±1 LSB vs the FABLE sim).
Writes bp_w.hex (256×NCASE codes), bp_delta.hex (16×NCASE int8), bp_expected.hex.
"""
import os, sys, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'hw_multi_array_l3_fable', 'sim'))
import pcn_router_backproj as twin                             # noqa: E402
from pcn_router_backproj import twin_block, WGT_MIN, WGT_MAX   # noqa: E402

# ★ TIE CODE_MID TO THE RTL's SINGLE SOURCE OF TRUTH (pcn_weight_params.vh).
# The shared multi-array twin defaults CODE_MID=132 (the OLD cell). The jug cell's zero is 117.
# Read PCN_WGT_ZERO from the .vh and override the twin's global so the golden this generator
# writes is referenced to the SAME zero the RTL uses. One number, one place — see the .vh.
def _read_wgt_zero():
    vh = os.path.join(HERE, 'pcn_weight_params.vh')
    m = re.search(r'`define\s+PCN_WGT_ZERO\s+(\d+)', open(vh).read())
    if not m:
        raise RuntimeError("PCN_WGT_ZERO not found in pcn_weight_params.vh")
    return int(m.group(1))

twin.CODE_MID = _read_wgt_zero()          # twin_block reads this global at call time
print(f"  CODE_MID tied to PCN_WGT_ZERO = {twin.CODE_MID} (from pcn_weight_params.vh)")

N, NCASE = 16, 64
rng = np.random.default_rng(2)

with open(os.path.join(HERE, 'bp_w.hex'), 'w') as fw, \
     open(os.path.join(HERE, 'bp_delta.hex'), 'w') as fd, \
     open(os.path.join(HERE, 'bp_expected.hex'), 'w') as fe:
    for _ in range(NCASE):
        W = rng.integers(WGT_MIN, WGT_MAX + 1, (N, N))
        d = rng.integers(-20, 21, N)               # rms ~12 — within int8 headroom (B1 finding)
        partial, _ = twin_block(W, d)
        for i in range(N):
            for j in range(N):
                fw.write("%02x\n" % int(W[i][j]))
        for i in range(N):
            fd.write("%02x\n" % (int(d[i]) & 0xff))
        for j in range(N):
            fe.write("%02x\n" % (int(partial[j]) & 0xff))

print(f"wrote bp_w.hex / bp_delta.hex / bp_expected.hex  ({NCASE} cases)")
