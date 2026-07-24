`timescale 1ns/1ps
`default_nettype none
// =============================================================================
// tb_epoch_match — RTL-T4: full multi-sweep EPOCH parity vs the Python jug model.
//
// ★ REWRITTEN FOR THE JUG (2026-07-15).  The old one drove the DEAD absorb_ctrl and demanded
//   `W += E; E = 0`.  This drives jug_ctrl + cap_array over a realistic epoch (injects
//   interleaved with array sweeps) and demands parity with the sigma-delta rule:
//       inject:  E += d*x*(bh+1)/2^shift        (clamped ±E_MAX_V)
//       sweep:   if |E| >= theta: W_code += ±1 ; E -= sign*theta      (RESIDUE SURVIVES)
//
// The stimulus (gen_epoch_stimulus.py) front-loads BURSTS (elements pushed to ~3*theta),
// then runs BARE sweeps so those bursts drain across successive sweeps — "a burst becomes a
// train" — then does several inject+sweep rounds.  So this exercises the train, the rails,
// both signs, and the sub-threshold hold, end to end.
//
// TWO checks, BOTH load-bearing:
//   1. W-SRAM codes must match w_expected.hex EXACTLY.  The fire is an integer ±1 increment;
//      given identical IEEE-double E arithmetic there is NO rounding slop — demand 100%.
//   2. ★ Ce residues must match e_expected.hex (signed micro-volts).  THIS is the jug: the
//      sub-threshold charge that the old absorb threw away must SURVIVE.  A model that
//      discharged Ce would pass check 1 on the fired elements and FAIL every residue.
//
//   iverilog -g2012 -Wall -o /tmp/tb_epoch_match.vvp tb_epoch_match.v jug_ctrl.v cap_array.v \
//     && vvp /tmp/tb_epoch_match.vvp
// (run gen_epoch_stimulus.py first to (re)generate the .hex files)
// =============================================================================

module tb_epoch_match;

    localparam N_CELLS  = 4;
    localparam N_ELEMS  = 256;
    localparam TOTAL    = N_CELLS * N_ELEMS;   // 1024
    localparam N_OPS    = 137;                  // ⚠ MUST equal gen_epoch_stimulus N_OPS
    localparam WGT_MIN  = 71;
    localparam WGT_MAX  = 192;
    localparam BH       = 3'd3;
    localparam LR_SHIFT = 6'd20;
    localparam real THETA   = 0.050;
    localparam real E_TOL_V = 2.0e-6;           // 2 uV — round(E*1e6) slop only

    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    // ── jug_ctrl <-> cap_array wiring ─────────────────────────────────────
    wire        cmp_en;  wire [1:0] cmp_cell;  wire [7:0] cmp_addr;
    wire        fire_up, fire_dn;
    wire        jug_pulse, jug_dn;
    wire [9:0]  sram_addr;  wire sram_re, sram_we;  wire [7:0] sram_wdata;
    reg  [7:0]  sram_rdata;
    wire        busy, done;
    reg         start_wb = 0;

    // inject port (driven directly by the schedule, NOT through jug_ctrl)
    reg        inject_en = 0;
    reg [1:0]  inject_cell = 0;
    reg [7:0]  inject_addr = 0, inject_delta = 0, inject_x = 0;

    // ── the mock W-SRAM: THIS holds the weight the jug modifies (±1 code) ──
    reg [7:0] wsram [0:TOTAL-1];
    always @(posedge clk) begin
        if (sram_we) wsram[sram_addr] <= sram_wdata;
        sram_rdata <= wsram[sram_addr];       // 1-cycle read latency (jug_ctrl RDW covers it)
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

    // the residue subtractor addresses the element the comparator just visited
    reg [1:0] jug_cell_r;  reg [7:0] jug_addr_r;
    always @(posedge clk) if (cmp_en) begin jug_cell_r <= cmp_cell; jug_addr_r <= cmp_addr; end

    cap_array #(.N_CELLS(N_CELLS), .N_ELEMS(N_ELEMS), .WGT_MIN(WGT_MIN), .WGT_MAX(WGT_MAX),
                .THETA(THETA)) u_cap (
        .clk(clk), .rst_n(rst_n),
        .jug_theta(8'd0),
        .load_en(1'b0), .load_cell(2'd0), .load_addr(8'd0), .load_code(8'd0),
        .inject_en(inject_en), .inject_cell(inject_cell), .inject_addr(inject_addr),
        .inject_delta(inject_delta), .inject_x(inject_x), .inject_bh(BH), .lr_shift(LR_SHIFT),
        .cmp_en(cmp_en), .cmp_cell(cmp_cell), .cmp_addr(cmp_addr),
        .fire_up(fire_up), .fire_dn(fire_dn),
        .jug_pulse(jug_pulse), .jug_dn(jug_dn),
        .jug_cell(jug_cell_r), .jug_addr(jug_addr_r),
        .save_req(1'b0), .save_cell(2'd0), .save_addr(8'd0),
        .save_code(), .save_valid(),
        .mac_cell(2'd0), .mac_addr(8'd0), .mac_eff_code()
    );

    // ── stimulus memories ─────────────────────────────────────────────────
    reg [7:0]  w_init     [0:TOTAL-1];
    reg [7:0]  w_expected [0:TOTAL-1];
    reg [31:0] schedule   [0:N_OPS-1];
    reg signed [31:0] e_expected [0:TOTAL-1];   // signed micro-volts

    integer i, op;
    integer w_exact, w_bad;
    integer e_ok, e_bad;
    real    e_meas, e_want, e_diff;

    task inject(input [1:0] c, input [7:0] a, input [7:0] d, input [7:0] x);
        begin
            @(negedge clk);
            inject_en = 1; inject_cell = c; inject_addr = a; inject_delta = d; inject_x = x;
            @(negedge clk);
            inject_en = 0;
        end
    endtask

    task sweep;
        begin
            @(negedge clk); start_wb = 1;
            @(negedge clk); start_wb = 0;
            wait (done);
            @(posedge clk);
        end
    endtask

    initial begin
        $display("\n==============================================================");
        $display("  tb_epoch_match (RTL) — full EPOCH parity vs the Python jug");
        $display("==============================================================");

        $readmemh("w_init.hex",     w_init);
        $readmemh("w_expected.hex", w_expected);
        $readmemh("stim.hex",       schedule);
        $readmemh("e_expected.hex", e_expected);

        // load the initial weight codes into the mock W-SRAM
        for (i = 0; i < TOTAL; i = i + 1) wsram[i] = w_init[i];

        #20 rst_n = 1; #20;

        // ── replay the schedule ────────────────────────────────────────────
        for (op = 0; op < N_OPS; op = op + 1) begin
            if (schedule[op][31]) begin
                sweep;                                   // OP = sweep the whole array once
            end else begin
                inject(schedule[op][25:24], schedule[op][23:16],
                       schedule[op][15:8],  schedule[op][7:0]);
            end
        end
        $display("  replayed %0d ops (injects + sweeps)", N_OPS);

        // ── CHECK 1: W-SRAM codes must match EXACTLY ───────────────────────
        w_exact = 0; w_bad = 0;
        for (i = 0; i < TOTAL; i = i + 1) begin
            if (wsram[i] === w_expected[i]) w_exact = w_exact + 1;
            else begin
                w_bad = w_bad + 1;
                if (w_bad <= 10)
                    $display("  W MISMATCH idx=%0d actual=%0d expected=%0d",
                             i, wsram[i], w_expected[i]);
            end
        end
        $display("\n  CHECK 1 (W codes): %0d/%0d exact, %0d mismatched", w_exact, TOTAL, w_bad);

        // ── CHECK 2: ★ Ce residues must SURVIVE (match to the micro-volt) ──
        e_ok = 0; e_bad = 0;
        for (i = 0; i < TOTAL; i = i + 1) begin
            e_meas = u_cap.E_cap[i];
            e_want = $itor(e_expected[i]) / 1.0e6;
            e_diff = (e_meas > e_want) ? (e_meas - e_want) : (e_want - e_meas);
            if (e_diff <= E_TOL_V) e_ok = e_ok + 1;
            else begin
                e_bad = e_bad + 1;
                if (e_bad <= 10)
                    $display("  E MISMATCH idx=%0d actual=%.6f expected=%.6f",
                             i, e_meas, e_want);
            end
        end
        $display("  CHECK 2 (Ce residue): %0d/%0d within %.0f uV, %0d off",
                 e_ok, TOTAL, E_TOL_V*1e6, e_bad);

        $display("\n==============================================================");
        if (w_bad == 0 && e_bad == 0)
            $display("  ALL PASS — codes AND residues match the jug model exactly.");
        else
            $display("  *** FAIL: %0d code / %0d residue mismatches ***", w_bad, e_bad);
        $display("==============================================================\n");
        $finish;
    end

    initial begin #50_000_000; $display("*** TIMEOUT ***"); $finish; end

endmodule
`default_nettype wire
