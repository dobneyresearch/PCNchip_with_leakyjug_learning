`default_nettype none
`timescale 1ns/1ps
// =============================================================================
// router_gather — the GATHER-ONLY router for TRANSPOSE-AT-SOURCE.
// =============================================================================
//
// Under transpose-at-source the Wᵀδ is computed ON THE SOURCE CHIP (pcn_transpose), so the router
// holds NO weights and does NO transpose. It only GATHERS the pre-computed, already-renormed
// partials from the source chips and averages them per destination (avg_bp).
//
// This is `hop_engine` MINUS the `router_backproj` instance (and its w_flat / FIRE / CAP stages):
//   hop_engine:    for s: partial = Wᵀδ(w[s], src[s]); accum[dest[s]] += partial;  dst = accum/fanin
//   router_gather: for s:                               accum[dest[s]] += partial[s]; dst = accum/fanin
//
// ★ C1/C2/C3 of ../INTERCONNECT_PROTOCOL.md: no SHADOW_W, transpose is on-chip, and this is the
//   FIXED-DIVIDE gather (÷ nominal fanin — no wait-for-all barrier; a missing partial just
//   under-drives that dest, which the jug tolerates).
//
// Golden: pcn_router_backproj.py::twin_hop_generic (the avg_bp half; the transpose half is now
// on the chip and covered by tb_pcn_transpose).
// =============================================================================
module router_gather #(
    parameter integer S = 4,           // source blocks (partials arriving)
    parameter integer D = 2            // dest cells
)(
    input  wire              clk, rst_n, start,
    input  wire [S*128-1:0]  partial_flat,  // S pre-computed partials (16 int8 each) FROM THE CHIPS
    input  wire [S*8-1:0]    dest_id_flat,  // per source: dest cell 0..D-1
    input  wire [D*8-1:0]    fanin_flat,    // per dest: divisor (nominal fan-in — FIXED-DIVIDE)
    output reg  [D*128-1:0]  dst_flat,       // D dest δ (16 int8 each)
    output reg               done
);
    localparam [1:0] IDLE=0, ACC=1, AVG=2, FIN=3;
    reg [1:0]  state;
    reg [7:0]  s;
    reg signed [23:0] accum [0:D*16-1];
    reg signed [23:0] a_s;
    reg signed [31:0] q;
    reg signed [8:0]  fin_s, half;
    integer k, di;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE; done <= 0; s <= 0;
            for (k = 0; k < D*16; k = k + 1) accum[k] <= 0;
        end else begin
            case (state)
                IDLE: if (start) begin
                        for (k = 0; k < D*16; k = k + 1) accum[k] <= 0;
                        s <= 0; done <= 0; state <= ACC;
                    end
                // ── accumulate one source's partial per cycle into its dest ─────
                // (one-per-cycle keeps the NBA accumulate correct when two sources
                //  target the same dest — same reason hop_engine iterates.)
                ACC: begin
                        di = dest_id_flat[s*8 +: 8];
                        for (k = 0; k < 16; k = k + 1)   // $signed: preserve sign of partial + accum
                            accum[di*16+k] <=
                                $signed(accum[di*16+k]) + $signed(partial_flat[(s*16+k)*8 +: 8]);
                        if (s == S-1) state <= AVG;
                        else s <= s + 1;
                    end
                // ── avg_bp: dst[d] = sat_int8(round(accum[d] / fanin[d])) ───────
                AVG: begin
                        for (di = 0; di < D; di = di + 1) begin
                            fin_s = $signed({1'b0, fanin_flat[di*8 +: 8]});
                            half  = fin_s >>> 1;
                            for (k = 0; k < 16; k = k + 1) begin
                                a_s = $signed(accum[di*16+k]);
                                if (fin_s <= 9'sd1)   q = a_s;                  // fanin 0/1: passthrough
                                else if (a_s >= 0)    q = (a_s + half) / fin_s; // round half away from 0
                                else                  q = (a_s - half) / fin_s;
                                if      (q >  32'sd127) dst_flat[(di*16+k)*8 +: 8] <=  8'sd127;
                                else if (q < -32'sd128) dst_flat[(di*16+k)*8 +: 8] <= -8'sd128;
                                else                    dst_flat[(di*16+k)*8 +: 8] <=  q[7:0];
                            end
                        end
                        state <= FIN;
                    end
                FIN: begin done <= 1'b1; state <= IDLE; end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
`default_nettype wire
