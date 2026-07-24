`timescale 1ns/1ps
`default_nettype none
// =============================================================================
// tb_absorb_match — RTL-T3: jug VALUE-CORRECTNESS on crafted elements.
//
// ★ REWRITTEN FOR THE JUG (2026-07-15).  The old one drove absorb_ctrl and asserted
//   `W = clip(W+E); E = 0`.  That rule is DEAD.  This isolates the one thing the epoch test
//   (random) and tb_jug_fire (mechanism, all mid-range) do NOT pin down: BEHAVIOUR AT THE RAIL.
//
// Under the jug the weight is an 8-bit CODE that moves ±1 per fire, CLAMPED to [WGT_MIN,WGT_MAX].
// The subtle, load-bearing case is a fire that lands ON a rail:
//
//   ★ THE CODE CLAMPS, BUT θ IS STILL SUBTRACTED FROM Ce.   (jug_pulse fires unconditionally.)
//     This is ANTI-WINDUP.  If Ce kept accumulating at a saturated rail it would wind up to
//     E_MAX (3·θ) and its SIGN would be destroyed; a later gradient reversal would then have to
//     discharge a full windup before the cell could move.  Draining θ each sweep keeps Ce near
//     ±θ, so the cell comes off the rail after ~1–2 sweeps.
//     ** This MATCHES the validated sim: pcn_jug.py clips Wm at the rail (line 804) but does
//        `E = e - s·th` UNCONDITIONALLY (line 807). **
//
// The five crafted elements (all cell 0):
//   addr 0: code=WGT_MAX, E=+1.6θ  → code STAYS 192, E→+0.6θ   (rail up, θ still drained)
//   addr 1: code=WGT_MIN, E=-1.6θ  → code STAYS 71,  E→-0.6θ   (rail dn, θ still drained)
//   addr 2: code=WGT_ZERO,E=+1.6θ  → code→118,       E→+0.6θ   (normal fire, control)
//   addr 3: code=WGT_ZERO,E=-0.8θ  → code STAYS 117, E=-0.8θ   (sub-threshold: charge HELD)
//   addr 4: code=WGT_MAX, E=+2.8θ  → code PINNED at 192, fires 2×, E 0.140→0.090→0.040
//                                     (BURST→TRAIN entirely AT the rail: θ drained each fire,
//                                      code never moves — anti-windup across a train)
//
//   iverilog -g2012 -Wall -o /tmp/tb_absorb_match.vvp tb_absorb_match.v jug_ctrl.v cap_array.v \
//     && vvp /tmp/tb_absorb_match.vvp
// =============================================================================

module tb_absorb_match;

    localparam N_CELLS = 4, N_ELEMS = 256;
    localparam WGT_MIN = 71, WGT_MAX = 192, WGT_ZERO = 117;
    localparam real THETA   = 0.050;
    localparam real E_TOL_V = 1.0e-6;

    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    // ── jug_ctrl <-> cap_array ────────────────────────────────────────────
    wire        cmp_en;  wire [1:0] cmp_cell;  wire [7:0] cmp_addr;
    wire        fire_up, fire_dn;
    wire        jug_pulse, jug_dn;
    wire [9:0]  sram_addr;  wire sram_re, sram_we;  wire [7:0] sram_wdata;
    reg  [7:0]  sram_rdata;
    wire        busy, done;
    reg         start_wb = 0;

    reg [7:0] wsram [0:N_CELLS*N_ELEMS-1];
    always @(posedge clk) begin
        if (sram_we) wsram[sram_addr] <= sram_wdata;
        sram_rdata <= wsram[sram_addr];
    end

    jug_ctrl #(.N_CELLS(N_CELLS), .N_ELEMS(N_ELEMS),
               .WGT_MIN(WGT_MIN), .WGT_MAX(WGT_MAX), .PULSE_CYC(1)) u_jug (
        .clk(clk), .rst_n(rst_n), .start_wb(start_wb), .sweep_n_in(1'b1),
        .busy(busy), .done(done),
        .cmp_en(cmp_en), .cmp_cell(cmp_cell), .cmp_addr(cmp_addr),
        .fire_up(fire_up), .fire_dn(fire_dn),
        .jug_pulse(jug_pulse), .jug_dn(jug_dn),
        .sram_addr(sram_addr), .sram_re(sram_re), .sram_rdata(sram_rdata),
        .sram_we(sram_we), .sram_wdata(sram_wdata)
    );

    reg [1:0] jug_cell_r;  reg [7:0] jug_addr_r;
    always @(posedge clk) if (cmp_en) begin jug_cell_r <= cmp_cell; jug_addr_r <= cmp_addr; end

    cap_array #(.N_CELLS(N_CELLS), .N_ELEMS(N_ELEMS), .WGT_MIN(WGT_MIN), .WGT_MAX(WGT_MAX),
                .THETA(THETA)) u_cap (
        .clk(clk), .rst_n(rst_n),
        .jug_theta(8'd0),
        .load_en(1'b0), .load_cell(2'd0), .load_addr(8'd0), .load_code(8'd0),
        .inject_en(1'b0), .inject_cell(2'd0), .inject_addr(8'd0),
        .inject_delta(8'd0), .inject_x(8'd0), .inject_bh(3'd0), .lr_shift(6'd0),
        .cmp_en(cmp_en), .cmp_cell(cmp_cell), .cmp_addr(cmp_addr),
        .fire_up(fire_up), .fire_dn(fire_dn),
        .jug_pulse(jug_pulse), .jug_dn(jug_dn),
        .jug_cell(jug_cell_r), .jug_addr(jug_addr_r),
        .save_req(1'b0), .save_cell(2'd0), .save_addr(8'd0),
        .save_code(), .save_valid(),
        .mac_cell(2'd0), .mac_addr(8'd0), .mac_eff_code()
    );

    integer errors = 0, i;

    task sweep;
        begin
            @(negedge clk); start_wb = 1;
            @(negedge clk); start_wb = 0;
            wait (done);
            @(posedge clk);
        end
    endtask

    task check(input [127:0] name, input [7:0] addr,
               input [7:0] exp_code, input real exp_e);
        real e_meas, e_diff;
        begin
            e_meas = u_cap.E_cap[addr];
            e_diff = (e_meas > exp_e) ? (e_meas - exp_e) : (exp_e - e_meas);
            if (wsram[addr] === exp_code && e_diff <= E_TOL_V)
                $display("  PASS %-22s code=%0d (exp %0d)  E=%+.4f (exp %+.4f)",
                         name, wsram[addr], exp_code, e_meas, exp_e);
            else begin
                $display("  FAIL %-22s code=%0d (exp %0d)  E=%+.6f (exp %+.6f)",
                         name, wsram[addr], exp_code, e_meas, exp_e);
                errors = errors + 1;
            end
        end
    endtask

    initial begin
        $display("\n==============================================================");
        $display("  tb_absorb_match (RTL) — jug value correctness AT THE RAIL");
        $display("==============================================================");

        for (i = 0; i < N_CELLS*N_ELEMS; i = i + 1) wsram[i] = WGT_ZERO;
        #20 rst_n = 1; #20;

        // craft the five elements (cell 0)
        wsram[0] = WGT_MAX;    u_cap.E_cap[0] = 0.080;   // +1.6θ, at the top rail
        wsram[1] = WGT_MIN;    u_cap.E_cap[1] = -0.080;  // -1.6θ, at the bottom rail
        wsram[2] = WGT_ZERO;   u_cap.E_cap[2] = 0.080;   // +1.6θ, mid-range (control)
        wsram[3] = WGT_ZERO;   u_cap.E_cap[3] = -0.040;  // -0.8θ, sub-threshold
        wsram[4] = WGT_MAX;    u_cap.E_cap[4] = 0.140;   // +2.8θ, already AT the top rail
        #1;

        // ── SWEEP 1 ─────────────────────────────────────────────────────────
        $display("\n-- after sweep 1 --");
        sweep;
        check("rail-up   addr0", 8'd0, WGT_MAX,      0.030);  // clamped, θ STILL drained
        check("rail-dn   addr1", 8'd1, WGT_MIN,     -0.030);  // clamped, θ STILL drained
        check("normal    addr2", 8'd2, WGT_ZERO+1,   0.030);  // moved 117→118
        check("subthresh addr3", 8'd3, WGT_ZERO,    -0.040);  // NO fire, charge HELD
        check("train@rail addr4",8'd4, WGT_MAX,      0.090);  // fire at rail: code pinned, E→0.090

        // ── SWEEP 2 ─────────────────────────────────────────────────────────
        $display("\n-- after sweep 2 --");
        sweep;
        check("rail-up   addr0", 8'd0, WGT_MAX,      0.030);  // sub-threshold now, stays
        check("normal    addr2", 8'd2, WGT_ZERO+1,   0.030);  // sub-threshold now, stays
        check("train@rail addr4",8'd4, WGT_MAX,      0.040);  // 2nd fire at rail: E→0.040

        // ── SWEEP 3 ─────────────────────────────────────────────────────────
        $display("\n-- after sweep 3 --");
        sweep;
        check("stop@rail addr4", 8'd4, WGT_MAX,      0.040);  // 0.040<θ: no fire, residue HELD

        $display("\n==============================================================");
        if (errors == 0)
            $display("  ALL PASS — code clamps at the rail; θ is drained anyway (anti-windup).");
        else
            $display("  *** %0d FAILURE(S) ***", errors);
        $display("==============================================================\n");
        $finish;
    end

    initial begin #10_000_000; $display("*** TIMEOUT ***"); $finish; end

endmodule
`default_nettype wire
