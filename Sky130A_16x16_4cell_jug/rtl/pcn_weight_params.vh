`ifndef PCN_WEIGHT_PARAMS_VH
`define PCN_WEIGHT_PARAMS_VH
// =============================================================================
// pcn_weight_params.vh — SINGLE SOURCE OF TRUTH for the analog cell's ZERO code.
// =============================================================================
//
// PCN_WGT_ZERO is the DAC code whose V_w gives ZERO differential MAC current. It is a
// PROPERTY OF THE ANALOG CELL, measured in SPICE — NOT a free digital constant. If the cell
// (MN3_w, the weight-pair sizing, the DAC span) changes, RE-MEASURE it and change it HERE only.
//
//   SPICE 2026-07-14, MN3_w=2 fix:  zero-current code = 117  (V_w = 0.823 V)
//   OLD broken cell (MN3_w=10):      132  — sat on a gm PEAK; the positive range was INVERTED.
//   See ../circuit/THE_WEIGHT_IS_NOT_LINEAR.md.
//
// ⚠ WHY THIS IS SHARED — the two users MUST agree:
//   * cap_array.v      — the W-cap reset / MAC zero (the FORWARD analog operator's zero).
//   * router_backproj.v — CODE_MID, the signed-weight reference for the TRANSPOSE backprojection
//                         (signed_W = shadow_code − CODE_MID).
//   The backprojection is only the true transpose of the forward operator if BOTH reference the
//   SAME zero. If they disagree by Δ codes, the router backprojects through a weight offset by Δ
//   — a spurious +Δ·Σδ bias on EVERY output. (This was live: router_backproj shipped CODE_MID=132
//   against the jug cell's 117 — a 15-code bias — until 2026-07-15.)
//
//   The Python co-sim generator (gen_backproj_stim.py) READS THIS FILE, so the golden the RTL is
//   checked against is tied here too. One number, one place.
// =============================================================================
`define PCN_WGT_ZERO 117
`endif
