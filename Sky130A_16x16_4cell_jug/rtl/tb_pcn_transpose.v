`timescale 1ns/1ps
`default_nettype none
// =============================================================================
// tb_pcn_transpose — proves TRANSPOSE-AT-SOURCE: pcn_transpose computes Wᵀ·δ by reading a
// mock W-SRAM (the chip's own weights), and matches the SAME golden tb_router_backproj uses.
//
// The golden (bp_w/bp_delta/bp_expected.hex, from gen_backproj_stim.py) is tied to
// CODE_MID = PCN_WGT_ZERO = 117 via pcn_weight_params.vh — so this also confirms the on-chip
// transpose uses the correct zero. The ONLY difference vs tb_router_backproj is that the weights
// arrive from a 1-cycle-latency W-SRAM read instead of a flat shadow bus — i.e. it proves the
// read path, not just the arithmetic.
//
//   (run gen_backproj_stim.py first)
//   iverilog -g2012 -o /tmp/tbpt.vvp tb_pcn_transpose.v pcn_transpose.v router_backproj.v \
//     && vvp /tmp/tbpt.vvp
// =============================================================================

module tb_pcn_transpose;
    localparam N = 16, NCASE = 64, N_CELLS = 4, N_ELEMS = N*N;
    localparam integer TOL = 2;                    // ±2 LSB vs the twin (as tb_router_backproj)

    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    // ── mock W-SRAM (1024×8, 1-cycle read) — stands in for the chip's own W-SRAM ──
    reg  [7:0] wsram [0:1023];
    wire [9:0] t_addr;  wire t_re;  reg [7:0] t_rdata;
    always @(posedge clk) if (t_re) t_rdata <= wsram[t_addr];

    // ── DUT ──────────────────────────────────────────────────────────────
    reg          start = 0;
    reg  [1:0]   cell_sel = 0;
    reg  [N*8-1:0] delta_flat;
    wire         busy, done;
    wire [N*8-1:0] partial_flat;

    pcn_transpose #(.N(N), .N_CELLS(N_CELLS)) dut (
        .clk(clk), .rst_n(rst_n),
        .start(start), .cell_sel(cell_sel), .delta_flat(delta_flat),
        .busy(busy), .done(done), .partial_flat(partial_flat),
        .sram_addr(t_addr), .sram_re(t_re), .sram_rdata(t_rdata)
    );

    // ── golden vectors (same files tb_router_backproj loads) ───────────────
    reg [7:0] bp_w        [0:N_ELEMS*NCASE-1];
    reg [7:0] bp_delta    [0:N*NCASE-1];
    reg [7:0] bp_expected [0:N*NCASE-1];

    integer c, k, errors, worst;
    integer av, ev, d;

    initial begin
        $display("\n==============================================================");
        $display("  tb_pcn_transpose — Wᵀ·δ from the chip's OWN W-SRAM vs golden");
        $display("==============================================================");
        $readmemh("bp_w.hex",        bp_w);
        $readmemh("bp_delta.hex",    bp_delta);
        $readmemh("bp_expected.hex", bp_expected);

        #20 rst_n = 1; #20;
        errors = 0; worst = 0;

        for (c = 0; c < NCASE; c = c + 1) begin
            // load this case's 16×16 block into the mock W-SRAM (cell_sel 0, elems 0..255)
            for (k = 0; k < N_ELEMS; k = k + 1) wsram[k] = bp_w[c*N_ELEMS + k];
            // pack δ
            for (k = 0; k < N; k = k + 1) delta_flat[k*8 +: 8] = bp_delta[c*N + k];

            // run one transpose
            @(negedge clk); start = 1; cell_sel = 2'd0;
            @(negedge clk); start = 0;
            wait (done);
            @(posedge clk);

            // compare (signed int8, ±TOL)
            for (k = 0; k < N; k = k + 1) begin
                av = $signed(partial_flat[k*8 +: 8]);
                ev = $signed(bp_expected[c*N + k]);
                d  = (av > ev) ? (av - ev) : (ev - av);
                if (d > worst) worst = d;
                if (d > TOL) begin
                    errors = errors + 1;
                    if (errors <= 8)
                        $display("  MISMATCH case %0d elem %0d: got %0d exp %0d (d=%0d)",
                                 c, k, av, ev, d);
                end
            end
        end

        $display("\n  %0d cases × %0d elems checked; worst |err| = %0d LSB; %0d over ±%0d",
                 NCASE, N, worst, errors, TOL);
        $display("==============================================================");
        if (errors == 0)
            $display("  ALL PASS — on-chip transpose (from W-SRAM) matches the twin within ±%0d LSB.", TOL);
        else
            $display("  *** %0d mismatches ***", errors);
        $display("==============================================================\n");
        $finish;
    end

    initial begin #20_000_000; $display("*** TIMEOUT ***"); $finish; end
endmodule
`default_nettype wire
