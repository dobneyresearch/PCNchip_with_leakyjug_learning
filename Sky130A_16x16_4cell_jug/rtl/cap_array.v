`default_nettype none
`timescale 1ns/1ps
`include "pcn_weight_params.vh"
// Behavioral JUG cap model: W-cap (the weight) + Ce (the EVIDENCE ACCUMULATOR).
// Not synthesizable — models the analog storage for RTL simulation.
//
// ★★ TWO THINGS CHANGED FROM cap_array (emx). BOTH ARE STRUCTURAL. ★★
//
// 1. **THE MAC USES W ONLY.  Ce IS NOT IN THE SIGNAL PATH.**
//    The old cell was DUAL-TAIL (I_ntail = I_tail_W + I_tail_E) so the MAC computed (W + E).
//    MN3_e IS DELETED (../circuit/mac_cell_jug.spice).  Under the jug, E is an ACCUMULATOR,
//    not a residual, and its presence in the forward path is a POSITIVE-FEEDBACK RUNAWAY:
//        E in MAC:  0     0.2    0.4*   1.0    2.0        (weight codes at threshold)
//        accuracy: 81.96 81.32  80.46  76.83  COLLAPSE (47->12%)     * = the OLD design point
//    SPICE confirms: sweeping V_Ce over its FULL range moves the MAC output by 0.4 pA
//    (it moved TENS OF MICROAMPS in the dual-tail cell).
//
// 2. **ABSORB IS GONE.  THE JUG SUBTRACTS; IT DOES NOT DISCHARGE.**
//        OLD:  W_cap += E_cap ;  E_cap := 0            (DISCHARGE — lossy)
//        JUG:  if |E| >= theta:  W_code += ±1 ;  E -= sign*theta   (SUBTRACT — a sigma-delta)
//    ★ The residue-preserving subtraction is LOAD-BEARING.  Zeroing E throws away exactly the
//      sub-threshold charge the mechanism exists to accumulate.  It is ALSO what lets the
//      comparator be sloppy: a wrong-signed fire puts the charge BACK and the next correct
//      fire undoes it.  CHARGE IS CONSERVED. (20% wrong fires cost 0.34pp.)
//
// ⚠ THERE IS NO CHARGE PUMP INTO Cw.  A FIRE IS A ±1 INCREMENT ON THE W-SRAM CODE (jug_ctrl),
//   and refresh_ctrl repaints W_cap from W-SRAM through the PRE-DISTORTION LUT (wgt_lut).
//
// ⚠⚠ THE MAC MODEL BELOW IS LINEAR IN THE CODE — AND THAT IS ONLY TRUE BECAUSE OF THE LUT.
//   The raw cell's weight is SIGMOIDAL in V_w (28x sensitivity variation; and at the inherited
//   MN3_w=10 it was NON-MONOTONIC and the whole positive range was INVERTED).  wgt_lut
//   pre-distorts code -> V_w so the WEIGHT is uniform.  See ../circuit/THE_WEIGHT_IS_NOT_LINEAR.md.
//   ** DO NOT trust this linear model without the LUT. That assumption, shared between the
//      Python sim and this RTL, is exactly why the bug survived for months: TWO MODELS THAT
//      SHARE AN ASSUMPTION CANNOT TEST IT. **
//
// Load:   DAC code → W-cap voltage.  (weight_fsm initial load, and refresh_ctrl.)
// Inject: Ce += delta × x × (bh+1) / 2^LR_SHIFT.   Ce is NEVER discharged.
// Compare: |Ce| >= theta ?  -> fire_up / fire_dn   (the SHARED, SWEPT comparator)
// JugSub: Ce -= sign*theta  (the pulsed-current residue subtraction — Q = I*t)
// Save:   W-cap → W-code (1-cycle latency).  Used by save_ctrl (checkpoints only).
// MAC:    W_cap → eff_code (combinational).  ** E IS NOT INCLUDED. **

module cap_array #(
    parameter N_CELLS  = 4,
    parameter N_ELEMS  = 256,   // N_ROWS × N_COLS per cell
    parameter WGT_MIN  = 71,
    parameter WGT_MAX  = 192,
    parameter real E_MAX_V = 0.150, // ⚠ Ce's swing is now FREE (MN3_e deleted => Ce is OUT of
                                    // the signal path). The old ±55 mV was a MAC constraint.
                                    // The jug needs ~±150 mV (theta + overshoot).
                                    // ** The kT/C noise budget only closes because of this. **
    parameter real THETA     = 0.050, // the FIRE THRESHOLD, in Ce volts (fallback when jug_theta=0).
    parameter real THETA_LSB = 0.001  // jug_theta code (in mV) -> Ce volts. 50 -> 0.050.
                                    // Set by the analog: Q_theta = Ce*theta = 5 fC.
                                    // A per-layer CONFIG constant (like the ADC gain) — the
                                    // boss writes it at load time. Sim optimum: theta = 8x the
                                    // per-fold rms of |E| (a clean inverted-U:
                                    // 4->81.08, 8->81.96, 16->80.37, 32->73.95).
) (
    input  wire        clk,
    input  wire        rst_n,
    // runtime θ config (in mV, JUG_THETA reg); 0 => fall back to the THETA parameter.
    // ★ Models the analog comparator threshold the boss programs (an analog bias); in the pure-
    //   digital behavioural model it just scales the fire threshold. See ../INTERCONNECT_PROTOCOL.md.
    input  wire  [7:0] jug_theta,

    // ── Load path (W-code → W-cap) ────────────────────────────────────────
    input  wire        load_en,
    input  wire  [1:0] load_cell,
    input  wire  [7:0] load_addr,
    input  wire  [7:0] load_code,

    // ── Inject path (ΔE outer-product element) ────────────────────────────
    input  wire        inject_en,
    input  wire  [1:0] inject_cell,
    input  wire  [7:0] inject_addr,
    input  wire  [7:0] inject_delta, // signed 8-bit (twos-complement)
    input  wire  [7:0] inject_x,     // unsigned 8-bit activation
    input  wire  [2:0] inject_bh,    // boss_h [0..7]; scale = (bh+1)
    input  wire  [5:0] lr_shift,     // E learning rate: divide by 2^lr_shift.
                                     // 6-bit [0..63]: the deep layer (L2, ~90x
                                     // compressor) needs ~7 more shift-bits than
                                     // L1; 5-bit[0..31] overflowed. See CROSS_CHECK §C2.

    // ── COMPARE port: the SHARED, SWEPT comparator (jug_ctrl drives it) ───
    // One comparator, time-multiplexed across the array. Each cell is checked ONCE PER SWEEP
    // and can therefore fire AT MOST ONCE. ** THE SWEEP PERIOD IS A FREE REFRACTORY PERIOD. **
    // (A per-cell comparator would fire repeatedly on a crossing and RUN AWAY: 271%/fold.)
    input  wire        cmp_en,
    input  wire  [1:0] cmp_cell,
    input  wire  [7:0] cmp_addr,
    output wire        fire_up,      // Ce >= +theta
    output wire        fire_dn,      // Ce <= -theta

    // ── RESIDUE SUBTRACT port (the pulsed current source: Q = I*t) ────────
    // ⚠ jug_pulse is a FIXED-WIDTH one-shot from jug_ctrl. Its width is part of the ANALOG
    //   spec (500 nA x 10 ns = 5 fC = 50 mV on Ce = 100 fF). See ../circuit/jug_compare.spice.
    input  wire        jug_pulse,
    input  wire        jug_dn,       // 0 = subtract theta (fired +), 1 = add theta (fired -)
    input  wire  [1:0] jug_cell,
    input  wire  [7:0] jug_addr,

    // ── Save path (W-cap → W-code, 1-cycle latency) ───────────────────────
    input  wire        save_req,
    input  wire  [1:0] save_cell,
    input  wire  [7:0] save_addr,
    output reg   [7:0] save_code,
    output reg         save_valid,

    // ── MAC read (combinational) ──────────────────────────────────────────
    input  wire  [1:0] mac_cell,
    input  wire  [7:0] mac_addr,
    output reg   [7:0] mac_eff_code
);
    localparam TOTAL    = N_CELLS * N_ELEMS;
    localparam real SPAN = WGT_MAX - WGT_MIN;
    localparam real WGT_ZERO_R = `PCN_WGT_ZERO;   // analog zero-current code (SPICE), tied to
                                                  // router_backproj's CODE_MID via the shared .vh

    real W_cap [0:TOTAL-1];
    real E_cap [0:TOTAL-1];

    integer k;

    // effective fire threshold (Ce volts): runtime jug_theta if configured, else the parameter.
    real th_eff;
    always @(*) th_eff = (jug_theta != 8'd0) ? ($itor(jug_theta) * THETA_LSB) : THETA;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (k = 0; k < TOTAL; k = k + 1) begin
                // ⚠⚠ WGT_ZERO = 117, **NOT 132**. It moved because THE CELL CHANGED.
                // WGT_ZERO is the code whose V_w gives ZERO differential MAC current — it is a
                // PROPERTY OF THE CELL, not a free digital constant. MN3_w went 10 -> 2 um (it
                // was in TRIODE: ntail had collapsed to 15 mV, the weight PEAKED at V_w~0.90 V
                // and FELL above it, and the ENTIRE POSITIVE CODE RANGE was INVERTED).
                // Re-measured in SPICE for the new cell: zero-current code = 117 (V_w=0.823 V).
                // ★ TIED to router_backproj's CODE_MID via pcn_weight_params.vh (PCN_WGT_ZERO).
                W_cap[k] = (WGT_ZERO_R - WGT_MIN) / SPAN;   // analog zero-current code (SPICE 2026-07-14)
                E_cap[k] = 0.0;
            end
            save_valid <= 1'b0;
            save_code  <= 8'h00;
        end else begin
            save_valid <= 1'b0;

            // Load: W-code → W-cap voltage
            if (load_en)
                W_cap[load_cell * N_ELEMS + load_addr] =
                    ($itor(load_code) - WGT_MIN) / SPAN;

            // Inject: E_cap += delta × x × (bh+1) / 2^lr_shift
            if (inject_en) begin : blk_inj
                real dv, d_r, x_r, bh_r;
                integer idx;
                idx  = inject_cell * N_ELEMS + inject_addr;
                d_r  = $itor($signed(inject_delta));        // signed [-128..127]
                x_r  = $itor({24'b0, inject_x});            // unsigned [0..255]
                bh_r = $itor({29'b0, inject_bh}) + 1.0;    // [1..8], no 3-bit overflow
                dv   = d_r * x_r * bh_r / $itor(64'd1 << lr_shift);  // 64-bit shift: lr_shift up to 63
                E_cap[idx] = E_cap[idx] + dv;
                if (E_cap[idx] >  E_MAX_V) E_cap[idx] =  E_MAX_V;
                if (E_cap[idx] < -E_MAX_V) E_cap[idx] = -E_MAX_V;
            end

            // ★★ THE JUG: SUBTRACT theta. DO NOT DISCHARGE. ★★
            // The old absorb did `W_cap += E_cap; E_cap = 0` — it threw the residue away.
            // Here the WEIGHT is written by jug_ctrl as a ±1 CODE INCREMENT on W-SRAM (there
            // is NO charge pump into Cw), and all this block does is remove exactly theta of
            // charge from Ce, KEEPING the remainder.
            //
            // A cell that overshot to 3*theta fires once, KEEPS 2*theta, and fires again on the
            // next sweep. ** A BURST BECOMES A TRAIN. Delay the boost; never drop it. **
            // (SPICE: driven to ~4*theta it fires on FOUR successive sweeps and then stops of
            //  its own accord, with a constant step. tb_jug_fire.spice T3/T4.)
            if (jug_pulse) begin : blk_jug
                integer idx;
                idx = jug_cell * N_ELEMS + jug_addr;
                if (jug_dn) E_cap[idx] = E_cap[idx] + th_eff;  // fired NEGATIVE: add theta back
                else        E_cap[idx] = E_cap[idx] - th_eff;  // fired POSITIVE: subtract theta
                // the analog rail on Ce (inject clamp / comparator range) — NOT a MAC constraint
                if (E_cap[idx] >  E_MAX_V) E_cap[idx] =  E_MAX_V;
                if (E_cap[idx] < -E_MAX_V) E_cap[idx] = -E_MAX_V;
            end

            // Save: read W_cap, convert to W-code, assert valid next cycle
            if (save_req) begin : blk_save
                real v_w;
                integer code, idx;
                idx  = save_cell * N_ELEMS + save_addr;
                v_w  = W_cap[idx];
                code = $rtoi(v_w * SPAN + 0.5) + WGT_MIN;
                if (code > WGT_MAX) code = WGT_MAX;
                if (code < WGT_MIN) code = WGT_MIN;
                save_code  <= code[7:0];
                save_valid <= 1'b1;
            end
        end
    end

    // ── THE SHARED, SWEPT COMPARATOR ──────────────────────────────────────
    // Combinational on the address jug_ctrl is currently visiting.
    // ⚠ IT IS ALLOWED TO BE SLOPPY: offset may be a large fraction of theta (x2 threshold
    //   mismatch costs 0.76pp), and it may get the SIGN WRONG 20% of the time (0.34pp) —
    //   because the residue subtraction CONSERVES CHARGE, so a wrong fire is undone by the
    //   next right one.  ** The sigma-delta can forgive a bad DECISION.  It cannot forgive a
    //   LIE ABOUT HOW MUCH CHARGE IS THERE. ** (Hence: kT/C noise on Ce is the ONE tight spec,
    //   sigma <= 0.1*theta. See ../circuit/PHYSICAL_SPEC.md §3.)
    wire real e_now = E_cap[cmp_cell * N_ELEMS + cmp_addr];
    assign fire_up = cmp_en && (e_now >=  th_eff);
    assign fire_dn = cmp_en && (e_now <= -th_eff);

    // ── MAC eff_code: **W ONLY.  Ce IS NOT IN THE SIGNAL PATH.** ───────────
    // MN3_e is DELETED. Adding E here would reintroduce the positive-feedback runaway
    // (0.91%/fold -> 19.3% -> collapse). SPICE: V_Ce moves the MAC by 0.4 pA across its
    // whole range. See the header.
    always @(*) begin : blk_mac
        real v_eff;
        integer code;
        v_eff = W_cap[mac_cell * N_ELEMS + mac_addr];   // ** NO E TERM **
        if (v_eff > 1.0) v_eff = 1.0;
        if (v_eff < 0.0) v_eff = 0.0;
        code = $rtoi(v_eff * SPAN + 0.5) + WGT_MIN;
        if (code > WGT_MAX) code = WGT_MAX;
        if (code < WGT_MIN) code = WGT_MIN;
        mac_eff_code = code[7:0];
    end

endmodule
