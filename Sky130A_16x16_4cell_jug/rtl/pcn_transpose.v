`default_nettype none
`timescale 1ns/1ps
`include "pcn_weight_params.vh"
// =============================================================================
// pcn_transpose — TRANSPOSE-AT-SOURCE.  Computes  Wᵀ·δ  on the PCN chip, from the chip's OWN
// W-SRAM.  Replaces the router-held SHADOW_W: the weights never leave the die.
// =============================================================================
//
// This is the on-chip analogue of the router's SRC→TRIG→DST round-trip (router_node 0x04/08/0C).
// It reads one CELL's 256 weight codes (a 16×16 block) straight from the W-SRAM, then runs the
// VALIDATED router_backproj datapath (int transpose-MAC + per-block RMS renorm) on them.
//
//   δ in  ──► [read the cell_sel's 16×16 W block from W-SRAM] ──► router_backproj ──► partial out
//
// ★ WHY THIS IS NOT A SHADOW: the shadow was a COPY in the router, kept coherent over the
//   interconnect (the whole sync/drift problem). This reads the LIVE W-SRAM on the same die —
//   a plain memory read, always current, no coherence. See ../INTERCONNECT_PROTOCOL.md (C1/C2)
//   and memory project_pcnchip_transpose_at_source.
//
// ★ CODE_MID = PCN_WGT_ZERO (117), TIED to the forward MAC's zero via pcn_weight_params.vh, so
//   the transpose is the true transpose of the forward operator (no constant bias). The same
//   incoming δ ALSO drives E-inject in the top — δ is used twice (relay + absorb).
//
// W-SRAM read: 1 code/cycle, 1-cycle latency, pipelined → 256 reads to assemble the block, then
// the combinational router_backproj compute. The read port is arbitrated in the top (the jug RMW
// and refresh also use it); here it is a simple master port.
// =============================================================================

// ⚠ SIZE PARAMETERS: the cell is N×N (SQUARE — router_backproj transposes a square block).
// N_ELEMS is DERIVED, never passed: it used to be an independent parameter, which allowed a
// caller to set N_ELEMS ≠ N*N and get silently wrong silicon. Every width below is derived
// from N / N_CELLS, so this module re-parameterises cleanly.
module pcn_transpose #(
    parameter integer N        = 16,                // cell is N×N
    parameter integer N_CELLS  = 4,
    parameter integer CODE_MID = `PCN_WGT_ZERO      // 117 — TIED to cap_array's zero
) (
    input  wire              clk,
    input  wire              rst_n,

    // ── control: pulse start to transpose the chosen cell_sel using delta_flat ────────
    input  wire              start,
    input  wire  [$clog2(N_CELLS)-1:0] cell_sel,    // which cell (an N×N block)
    input  wire  [N*8-1:0]   delta_flat,            // N × int8 source δ (held stable is fine; latched)

    output reg               busy,
    output reg               done,                  // 1-cycle pulse
    output reg   [N*8-1:0]   partial_flat,          // N × int8 renormed partial

    // ── W-SRAM read port (arbitrated in the top; mock in the TB) ────────────
    output reg   [$clog2(N_CELLS)+$clog2(N*N)-1:0] sram_addr,   // {cell, elem}
    output reg               sram_re,
    input  wire  [7:0]       sram_rdata
);
    localparam integer N_ELEMS = N * N;             // DERIVED — cannot disagree with N
    localparam integer ELEM_AW = $clog2(N_ELEMS);
    localparam integer CELL_AW = $clog2(N_CELLS);

    // assembled N×N weight block, row-major [i*N+j] — exactly router_backproj's w_flat
    reg  [N*N*8-1:0] w_buf;
    reg  [N*8-1:0]   delta_r;
    reg  [CELL_AW-1:0] cell_r;
    reg  [ELEM_AW-1:0] rd;                          // 0..N_ELEMS-1 issue counter

    reg              bp_start;
    wire [N*8-1:0]   bp_partial;
    wire             bp_done;

    router_backproj #(.N(N), .CODE_MID(CODE_MID)) u_bp (
        .clk(clk), .rst_n(rst_n), .start(bp_start),
        .w_flat(w_buf), .delta_flat(delta_r),
        .partial_flat(bp_partial), .done(bp_done)
    );

    // Issue→WAIT→CAP per element (2 cycles/elem = obviously correct against the 1-cycle SRAM
    // latency). The transpose runs once per fold, so 512 cycles is negligible; a 1-cycle/elem
    // pipeline is a later optimisation, not worth the off-by-one risk now.
    localparam IDLE = 3'd0, WAIT = 3'd1, CAP = 3'd2, COMP = 3'd3;
    reg [2:0] state;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE; busy <= 1'b0; done <= 1'b0;
            sram_re <= 1'b0; bp_start <= 1'b0; rd <= {ELEM_AW{1'b0}};
        end else begin
            done     <= 1'b0;
            bp_start <= 1'b0;
            case (state)
                IDLE: if (start) begin
                    delta_r   <= delta_flat;
                    cell_r    <= cell_sel;
                    rd        <= {ELEM_AW{1'b0}};
                    busy      <= 1'b1;
                    sram_addr <= {cell_sel, {ELEM_AW{1'b0}}};   // issue read for element 0
                    sram_re   <= 1'b1;
                    state     <= WAIT;
                end

                // ── 1-cycle SRAM read latency: addr issued last cycle, data valid next ──
                WAIT: begin
                    sram_re <= 1'b0;
                    state   <= CAP;
                end

                // ── capture element rd; issue the next, or finish ─────────────
                CAP: begin
                    w_buf[rd*8 +: 8] <= sram_rdata;  // elem rd valid now
                    if (rd == N_ELEMS-1) begin
                        bp_start <= 1'b1;            // whole block loaded → transpose it
                        state    <= COMP;
                    end else begin
                        rd        <= rd + 1'b1;
                        sram_addr <= {cell_r, rd + 1'b1};   // rd is still the OLD value here
                        sram_re   <= 1'b1;
                        state     <= WAIT;
                    end
                end

                // ── router_backproj computes (combinational, done pulses) ─────
                COMP: if (bp_done) begin
                    partial_flat <= bp_partial;
                    done         <= 1'b1;
                    busy         <= 1'b0;
                    state        <= IDLE;
                end
            endcase
        end
    end
endmodule
`default_nettype wire
