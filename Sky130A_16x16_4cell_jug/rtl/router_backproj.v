`default_nettype none
`timescale 1ns/1ps
`include "pcn_weight_params.vh"
// router_backproj.v — S3 per-block backprojection datapath (Phase-2 B1 RTL).
//
// Reference twin (co-sim golden): ../../hw_multi_array_l3_fable/sim/pcn_router_backproj.py
//   signed_W[i][j] = w_code[i][j]  (from the chip's W-SRAM, via pcn_transpose) - CODE_MID   (CODE_MID = PCN_WGT_ZERO = 117 for the
//   jug cell — it MUST equal the FORWARD MAC's zero, or the transpose carries a constant bias.
//   See pcn_weight_params.vh.)
//   acc[j]         = sum_i signed_W[i][j] * delta[i]         (integer; ÷64 dropped into renorm)
//   sc             = sqrt( (sum_i delta[i]^2) / (sum_j acc[j]^2) )   (RMS-preserving renorm)
//   partial[j]     = sat_int8( round( acc[j] * sc ) )
//
// Fixed-point: sc held Q(F) via sc = isqrt( (ssd << 2F) / ssa ).  See router_backproj_spec.md.
// Combinational compute triggered on `start` (one-shot); `done` pulses when ready.
module router_backproj #(
    parameter integer N        = 16,
    parameter integer CODE_MID = `PCN_WGT_ZERO,  // = 117 (jug cell). TIED to cap_array's zero.
    parameter integer F        = 16       // sc fractional bits (Q16)
)(
    input  wire              clk,
    input  wire              rst_n,
    input  wire              start,
    input  wire [N*N*8-1:0]  w_flat,  // 16x16 uint8 weight codes, row-major [i*N+j]
    input  wire [N*8-1:0]    delta_flat,     // 16 int8 source δ
    output reg  [N*8-1:0]    partial_flat,   // 16 int8 renormed partial
    output reg               done
);
    integer i, j;
    reg signed [8:0]  dl  [0:N-1];
    reg signed [31:0] acc [0:N-1];
    reg signed [63:0] ssd, ssa;             // SIGNED: else the unsigned context reinterprets
    reg [63:0]        ratio;                //   negative δ/acc as large positive before squaring
    reg [31:0]        sc;                    // Q(F)
    reg signed [63:0] prod, rnd;

    // floor integer sqrt of a 64-bit value → 32-bit
    function [31:0] isqrt;
        input [63:0] num;
        reg [63:0] a, q, r;
        integer k;
        begin
            a = num; q = 0; r = 0;
            for (k = 0; k < 32; k = k + 1) begin
                r = (r << 2) | (a >> 62);
                a = a << 2;
                q = q << 1;
                if (r >= ((q << 1) | 64'd1)) begin
                    r = r - ((q << 1) | 64'd1);
                    q = q + 1;
                end
            end
            isqrt = q[31:0];
        end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done <= 1'b0; partial_flat <= {N*8{1'b0}};
        end else if (start) begin
            // δ (int8 → 9b signed) and Σδ²
            ssd = 0;
            for (i = 0; i < N; i = i + 1) begin
                dl[i] = $signed(delta_flat[i*8 +: 8]);
                ssd   = ssd + dl[i]*dl[i];
            end
            // transpose MAC: acc[j] = Σ_i (w[i][j]-CODE_MID)·δ[i]   (CODE_MID = jug zero = 117)
            for (j = 0; j < N; j = j + 1) begin
                acc[j] = 0;
                for (i = 0; i < N; i = i + 1)
                    acc[j] = acc[j] +
                        ($signed({1'b0, w_flat[(i*N+j)*8 +: 8]}) - CODE_MID) * dl[i];
            end
            // Σacc²
            ssa = 0;
            for (j = 0; j < N; j = j + 1)
                ssa = ssa + acc[j]*acc[j];
            // renorm + saturate
            if (ssa == 0) begin
                partial_flat <= {N*8{1'b0}};
            end else begin
                ratio = ($unsigned(ssd) << (2*F)) / $unsigned(ssa);   // Q(2F) of ssd/ssa (both ≥0)
                sc    = isqrt(ratio);                  // Q(F) of sqrt(ssd/ssa)
                for (j = 0; j < N; j = j + 1) begin
                    prod = $signed({{32{acc[j][31]}}, acc[j]}) * $signed({32'd0, sc});
                    rnd  = (prod + (64'sd1 <<< (F-1))) >>> F;   // round-nearest, arithmetic
                    if      (rnd >  127) partial_flat[j*8 +: 8] <=  8'sd127;
                    else if (rnd < -128) partial_flat[j*8 +: 8] <= -8'sd128;
                    else                 partial_flat[j*8 +: 8] <= rnd[7:0];
                end
            end
            done <= 1'b1;
        end else begin
            done <= 1'b0;
        end
    end
endmodule
`default_nettype wire
