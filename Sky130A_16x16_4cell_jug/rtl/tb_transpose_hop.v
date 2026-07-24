`timescale 1ns/1ps
`default_nettype none
// =============================================================================
// tb_transpose_hop — THE COMPOSITION TEST for transpose-at-source.
//
// Proves that the NEW split reproduces the OLD fused hop (retired hop_engine / twin_hop_generic):
//   for each of 4 source blocks: pcn_transpose computes Wᵀδ from its OWN W-SRAM (the chip);
//   router_gather then accumulates the 4 partials per dest and avg_bp's them (the router).
//   Result must match l2_expected.hex (twin_hop_generic, tied to CODE_MID=117).
//
// This replaces tb_hop_chain's coverage under the new architecture: the transpose and the gather
// are now separate modules on separate dies, and this checks they COMPOSE correctly end to end.
//
//   (run gen_transhop_stim.py first)
//   iverilog -g2012 -o /tmp/tbth.vvp tb_transpose_hop.v pcn_transpose.v router_backproj.v \
//     router_gather.v && vvp /tmp/tbth.vvp
// =============================================================================
module tb_transpose_hop;
    localparam integer N = 16, S = 4, D = 2, TOL = 2, N_CELLS = 4;

    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    // ── mock W-SRAM holding the 4 source blocks (cells 0..3) ───────────────
    reg  [7:0] wsram [0:1023];
    wire [9:0] t_addr;  wire t_re;  reg [7:0] t_rdata;
    always @(posedge clk) if (t_re) t_rdata <= wsram[t_addr];

    // ── one pcn_transpose (time-shared over the 4 source blocks) ───────────
    reg          tr_start = 0;
    reg  [1:0]   tr_cell = 0;
    reg  [N*8-1:0] tr_delta;
    wire         tr_busy, tr_done;
    wire [N*8-1:0] tr_partial;

    pcn_transpose #(.N(N), .N_CELLS(N_CELLS)) u_tr (
        .clk(clk), .rst_n(rst_n),
        .start(tr_start), .cell_sel(tr_cell), .delta_flat(tr_delta),
        .busy(tr_busy), .done(tr_done), .partial_flat(tr_partial),
        .sram_addr(t_addr), .sram_re(t_re), .sram_rdata(t_rdata)
    );

    // ── one router_gather (S=4, D=2) ──────────────────────────────────────
    reg  [S*128-1:0] partials;
    reg  [S*8-1:0]   dest_ids;
    reg  [D*8-1:0]   fanins;
    reg              rg_start = 0;
    wire [D*128-1:0] dst;
    wire             rg_done;

    router_gather #(.S(S), .D(D)) u_rg (
        .clk(clk), .rst_n(rst_n), .start(rg_start),
        .partial_flat(partials), .dest_id_flat(dest_ids),
        .fanin_flat(fanins), .dst_flat(dst), .done(rg_done)
    );

    // golden
    reg [7:0] hop1_w   [0:S*256-1];
    reg [7:0] hop1_src [0:S*16-1];
    reg [7:0] l2_exp   [0:D*16-1];

    integer s, k, d, errors, worst, got, exp, diff;

    initial begin
        $display("\n==============================================================");
        $display("  tb_transpose_hop — pcn_transpose + router_gather == twin_hop");
        $display("==============================================================");
        $readmemh("hop1_w.hex",     hop1_w);
        $readmemh("hop1_src.hex",   hop1_src);
        $readmemh("l2_expected.hex",l2_exp);

        // load the 4 source blocks into the mock W-SRAM (cell s = block s)
        for (s = 0; s < S; s = s + 1)
            for (k = 0; k < 256; k = k + 1) wsram[s*256+k] = hop1_w[s*256+k];

        #20 rst_n = 1; #20;
        errors = 0; worst = 0;

        // ── run the transpose for each source block, collect the partials ──
        for (s = 0; s < S; s = s + 1) begin
            for (k = 0; k < N; k = k + 1) tr_delta[k*8 +: 8] = hop1_src[s*16+k];
            @(negedge clk); tr_start = 1; tr_cell = s[1:0];
            @(negedge clk); tr_start = 0;
            wait (tr_done);
            @(posedge clk);
            partials[s*128 +: 128] = tr_partial;    // store this source's partial
        end
        $display("  computed 4 transpose partials on the chip");

        // ── gather: sources 0,1→dest0 ; 2,3→dest1 ; fan-in 2 ───────────────
        dest_ids[0*8 +: 8] = 8'd0; dest_ids[1*8 +: 8] = 8'd0;
        dest_ids[2*8 +: 8] = 8'd1; dest_ids[3*8 +: 8] = 8'd1;
        fanins[0*8 +: 8] = 8'd2;   fanins[1*8 +: 8] = 8'd2;
        @(negedge clk); rg_start = 1;
        @(negedge clk); rg_start = 0;
        wait (rg_done);
        @(posedge clk);

        // ── compare vs twin_hop_generic golden ─────────────────────────────
        for (d = 0; d < D; d = d + 1)
            for (k = 0; k < N; k = k + 1) begin
                got = $signed(dst[(d*16+k)*8 +: 8]);
                exp = $signed(l2_exp[d*16+k]);
                diff = (got > exp) ? (got - exp) : (exp - got);
                if (diff > worst) worst = diff;
                if (diff > TOL) begin
                    errors = errors + 1;
                    if (errors <= 8)
                        $display("  MISMATCH dest %0d elem %0d: got %0d exp %0d (Δ=%0d)",
                                 d, k, got, exp, diff);
                end
            end

        $display("\n  %0d outputs checked; worst |Δ| = %0d LSB vs twin_hop; %0d over ±%0d",
                 D*16, worst, errors, TOL);
        $display("==============================================================");
        if (errors == 0)
            $display("  ALL PASS — chip transpose + router gather == the fused hop (within ±%0d LSB).", TOL);
        else
            $display("  *** %0d mismatches ***", errors);
        $display("==============================================================\n");
        $finish;
    end
    initial begin #10_000_000; $display("*** TIMEOUT ***"); $finish; end
endmodule
`default_nettype wire
