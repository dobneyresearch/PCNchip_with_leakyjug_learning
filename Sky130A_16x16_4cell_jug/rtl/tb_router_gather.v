`timescale 1ns/1ps
`default_nettype none
// =============================================================================
// tb_router_gather — self-checking test of the gather-only router.
// Random partials + dest_ids + fanins; the expected avg_bp (accumulate per dest, round-nearest
// divide by fanin, sat_int8) is recomputed independently in the TB and compared to the DUT.
//   iverilog -g2012 -o /tmp/tbrg.vvp tb_router_gather.v router_gather.v && vvp /tmp/tbrg.vvp
// =============================================================================
module tb_router_gather;
    localparam integer S = 4, D = 2, NCASE = 200;

    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    reg  [S*128-1:0] partial_flat;
    reg  [S*8-1:0]   dest_id_flat;
    reg  [D*8-1:0]   fanin_flat;
    reg              start = 0;
    wire [D*128-1:0] dst_flat;
    wire             done;

    router_gather #(.S(S), .D(D)) dut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .partial_flat(partial_flat), .dest_id_flat(dest_id_flat),
        .fanin_flat(fanin_flat), .dst_flat(dst_flat), .done(done)
    );

    // reference model
    integer pbyte [0:S*16-1];   // signed int8 partials
    integer did   [0:S-1];
    integer fin   [0:D-1];
    integer accum [0:D*16-1];
    integer exp   [0:D*16-1];

    integer c, s, k, d, cnt, av, q, half, errors, got;

    task build_case;
        begin
            // random signed int8 partials
            for (s = 0; s < S; s = s + 1)
                for (k = 0; k < 16; k = k + 1) begin
                    pbyte[s*16+k] = $random % 128;            // -127..127
                    partial_flat[(s*16+k)*8 +: 8] = pbyte[s*16+k][7:0];
                end
            // random dest per source
            for (s = 0; s < S; s = s + 1) begin
                did[s] = (($random % D) + D) % D;             // 0..D-1
                dest_id_flat[s*8 +: 8] = did[s][7:0];
            end
            // fanin[d] = number of sources targeting d (>=1) — a true average
            for (d = 0; d < D; d = d + 1) begin
                cnt = 0;
                for (s = 0; s < S; s = s + 1) if (did[s] == d) cnt = cnt + 1;
                if (cnt == 0) cnt = 1;
                fin[d] = cnt;
                fanin_flat[d*8 +: 8] = cnt[7:0];
            end
            // reference: accumulate then avg_bp (round half away from zero), sat_int8
            for (k = 0; k < D*16; k = k + 1) accum[k] = 0;
            for (s = 0; s < S; s = s + 1)
                for (k = 0; k < 16; k = k + 1)
                    accum[did[s]*16+k] = accum[did[s]*16+k] + pbyte[s*16+k];
            for (d = 0; d < D; d = d + 1) begin
                half = fin[d] / 2;
                for (k = 0; k < 16; k = k + 1) begin
                    av = accum[d*16+k];
                    if (fin[d] <= 1)      q = av;
                    else if (av >= 0)     q = (av + half) / fin[d];
                    else                  q = (av - half) / fin[d];
                    if      (q >  127) q =  127;
                    else if (q < -128) q = -128;
                    exp[d*16+k] = q;
                end
            end
        end
    endtask

    initial begin
        $display("\n==============================================================");
        $display("  tb_router_gather — accumulate-per-dest + avg_bp (fixed-divide)");
        $display("==============================================================");
        errors = 0;
        #20 rst_n = 1; #20;

        for (c = 0; c < NCASE; c = c + 1) begin
            build_case;
            @(negedge clk); start = 1;
            @(negedge clk); start = 0;
            wait (done);
            @(posedge clk);
            for (d = 0; d < D; d = d + 1)
                for (k = 0; k < 16; k = k + 1) begin
                    got = $signed(dst_flat[(d*16+k)*8 +: 8]);
                    if (got !== exp[d*16+k]) begin
                        errors = errors + 1;
                        if (errors <= 8)
                            $display("  MISMATCH case %0d dest %0d elem %0d: got %0d exp %0d",
                                     c, d, k, got, exp[d*16+k]);
                    end
                end
        end

        $display("\n  %0d cases × %0d outputs checked; %0d mismatches", NCASE, D*16, errors);
        $display("==============================================================");
        if (errors == 0) $display("  ALL PASS — gather + avg_bp matches the reference exactly.");
        else             $display("  *** %0d FAILURE(S) ***", errors);
        $display("==============================================================\n");
        $finish;
    end
    initial begin #5_000_000; $display("*** TIMEOUT ***"); $finish; end
endmodule
`default_nettype wire
