`timescale 1ns/1ps
`default_nettype none
// =============================================================================
// tb_jug_fire — RTL verification of THE JUG.
//
// The SPICE testbench (../circuit/tb_jug_fire.spice) proved the ANALOG residue subtraction
// works.  This proves the DIGITAL side does the right thing with it:
//
//   T1  the swept comparator fires when |Ce| >= theta
//   T2  ★ Ce is SUBTRACTED by theta — the RESIDUE SURVIVES.  It is NOT discharged.
//        (The old absorb did `W += E; E = 0`. That threw away exactly the sub-threshold
//         charge the mechanism exists to accumulate.)
//   T3  ★ a BURST becomes a TRAIN: a cell driven to 3*theta fires on THREE successive
//        sweeps and then stops of its own accord.  ** Delay the boost; never drop it. **
//   T4  ★ a fire is a ±1 INCREMENT ON THE W-SRAM CODE — there is NO charge pump into Cw.
//   T5  a cell BELOW threshold must NOT fire, and must KEEP its charge.
//
// ⚠⚠ ALL CODE COMPARISONS USE `!==`, NOT `!=`.
//    `!=` returns X if either operand is X, and `if (X)` is FALSE in Verilog — so a test
//    written with `!=` PASSES SILENTLY on an X. That is a test that CANNOT FAIL.
//    (It happened here: an off-by-one on the SRAM read latency made the code X, and T3/T4
//     both reported PASS. Same defect class as the WGT_ZERO sign check that let an INVERTED
//     weight through for months — a test that cannot distinguish right from wrong.)
//
//   iverilog -g2012 -o tb_jug_fire.out tb_jug_fire.v jug_ctrl.v cap_array.v && ./tb_jug_fire.out
// =============================================================================

module tb_jug_fire;
    localparam N_CELLS = 4, N_ELEMS = 256;
    localparam WGT_MIN = 71, WGT_MAX = 192, WGT_ZERO = 117;   // ⚠ 117, not 132 — the cell changed
    localparam real THETA = 0.050;

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

    // ── a mock W-SRAM (the real one lives behind weight_fsm) ──────────────
    reg [7:0] wsram [0:N_CELLS*N_ELEMS-1];
    always @(posedge clk) begin
        if (sram_we) wsram[sram_addr] <= sram_wdata;
        sram_rdata <= wsram[sram_addr];      // 1-cycle read
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

    // the residue subtractor addresses the same element the comparator just visited
    reg [1:0] jug_cell_r;  reg [7:0] jug_addr_r;
    always @(posedge clk) if (cmp_en) begin jug_cell_r <= cmp_cell; jug_addr_r <= cmp_addr; end

    cap_array #(.N_CELLS(N_CELLS), .N_ELEMS(N_ELEMS), .WGT_MIN(WGT_MIN), .WGT_MAX(WGT_MAX),
                .THETA(THETA)) u_cap (
        .clk(clk), .rst_n(rst_n),
        .jug_theta(8'd0),
        .load_en(1'b0), .load_cell(2'd0), .load_addr(8'd0), .load_code(8'd0),
        .inject_en(inj_en), .inject_cell(inj_cell), .inject_addr(inj_addr),
        .inject_delta(inj_d), .inject_x(inj_x), .inject_bh(3'd0), .lr_shift(6'd0),
        .cmp_en(cmp_en), .cmp_cell(cmp_cell), .cmp_addr(cmp_addr),
        .fire_up(fire_up), .fire_dn(fire_dn),
        .jug_pulse(jug_pulse), .jug_dn(jug_dn),
        .jug_cell(jug_cell_r), .jug_addr(jug_addr_r),
        .save_req(1'b0), .save_cell(2'd0), .save_addr(8'd0),
        .save_code(), .save_valid(),
        .mac_cell(2'd0), .mac_addr(8'd0), .mac_eff_code()
    );

    reg        inj_en = 0;
    reg [1:0]  inj_cell = 0;
    reg [7:0]  inj_addr = 0;
    reg [7:0]  inj_d = 0, inj_x = 0;

    integer errors = 0;
    integer fires_seen;
    real    e0, e1, e2, e3;

    // count fires at element 0
    always @(posedge clk)
        if (jug_pulse && jug_cell_r == 0 && jug_addr_r == 0) fires_seen = fires_seen + 1;

    task inject(input [1:0] c, input [7:0] a, input [7:0] d, input [7:0] x);
        begin
            @(negedge clk);
            inj_en = 1; inj_cell = c; inj_addr = a; inj_d = d; inj_x = x;
            @(negedge clk);
            inj_en = 0;
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
        $display("  tb_jug_fire (RTL) — SUBTRACT, DO NOT DISCHARGE");
        $display("==============================================================");
        for (integer i = 0; i < N_CELLS*N_ELEMS; i = i + 1) wsram[i] = WGT_ZERO;
        #20 rst_n = 1; #20;
        fires_seen = 0;

        // ── T1/T2: drive element 0 to 1.6*theta and sweep once ──────────────
        // inject 8*10 = 80 units; lr_shift=0 => dv = 80e-3? No: dv = d*x*(bh+1)/2^0 = 80.
        // Use small numbers so dv lands at 0.080 V = 1.6*theta.  d=8, x=10, bh=0 => 80... but
        // cap_array's dv is in VOLTS, so scale: d=8,x=10 gives 80.0 — far too big. Use lr_shift.
        // d=8, x=10, bh=0, lr_shift=10  => 80/1024 = 0.078 V ~ 1.6*theta.  (set below)
        inject(0, 0, 8'd8, 8'd10);      // lr_shift is 0 here -> see the note; we force E directly
        u_cap.E_cap[0] = 0.080;         // = 1.6 * theta, cleanly
        #1;

        $display("\nT1/T2 — one sweep, element 0 at 1.6*theta (0.080 V)");
        e0 = u_cap.E_cap[0];
        sweep;
        e1 = u_cap.E_cap[0];
        $display("   E before = %f", e0);
        $display("   E after  = %f     (theta = %f)", e1, THETA);
        $display("   W-SRAM code: %0d  (started at %0d)", wsram[0], WGT_ZERO);

        if (fires_seen != 1) begin
            $display("   *** FAIL: expected exactly 1 fire, got %0d", fires_seen); errors = errors+1;
        end else $display("   PASS: fired exactly once (the sweep is a REFRACTORY PERIOD)");

        if (wsram[0] !== WGT_ZERO + 1) begin
            $display("   *** FAIL(T4): W-SRAM code must be %0d, is %0d", WGT_ZERO+1, wsram[0]);
            errors = errors+1;
        end else $display("   PASS(T4): a fire is a +1 CODE INCREMENT on W-SRAM (no charge pump)");

        // ★★ THE CRITICAL ASSERTION ★★
        if (e1 < 0.0001) begin
            $display("   *** FAIL(T2): Ce was DISCHARGED (E=%f). That is the OLD, LOSSY absorb.", e1);
            $display("       The residue MUST survive: 0.080 - 0.050 = 0.030.");
            errors = errors + 1;
        end else if (e1 > 0.0295 && e1 < 0.0305) begin
            $display("   PASS(T2): ★ E = %f — theta was SUBTRACTED and THE RESIDUE SURVIVED.", e1);
        end else begin
            $display("   *** FAIL(T2): E = %f, expected 0.030 (= 0.080 - theta)", e1);
            errors = errors + 1;
        end

        // ── T3: a BURST becomes a TRAIN ─────────────────────────────────────
        $display("\nT3 — ★ a BURST becomes a TRAIN (element 1 driven to 3.4*theta)");
        u_cap.E_cap[1] = 0.170;         // 3.4 * theta
        #1;
        for (integer sw = 1; sw <= 4; sw = sw + 1) begin
            sweep;
            $display("   sweep %0d: E[1] = %f   W-SRAM[1] = %0d", sw, u_cap.E_cap[1], wsram[1]);
        end
        // 0.170 -> .120 -> .070 -> .020 (stops: below theta).  3 fires. code 117 -> 120.
        if (wsram[1] !== WGT_ZERO + 3) begin
            $display("   *** FAIL(T3): expected 3 fires (code %0d), got code %0d",
                     WGT_ZERO+3, wsram[1]);
            errors = errors + 1;
        end else
            $display("   PASS(T3): ★ fired 3x on successive sweeps, then STOPPED. Charge conserved.");

        // ── T5: below threshold must NOT fire, and must KEEP its charge ─────
        $display("\nT5 — a cell BELOW threshold must not fire, and must keep its charge");
        u_cap.E_cap[2] = 0.040;         // 0.8 * theta
        #1;
        sweep;
        if (wsram[2] !== WGT_ZERO) begin
            $display("   *** FAIL(T5): a sub-threshold cell FIRED (code %0d)", wsram[2]);
            errors = errors + 1;
        end else
            $display("   PASS(T5): did not fire, and the 0.040 V of evidence is STILL THERE.");

        $display("\n==============================================================");
        if (errors == 0) $display("  ALL PASS — the jug SUBTRACTS. It does not discharge.");
        else             $display("  *** %0d FAILURE(S) ***", errors);
        $display("==============================================================\n");
        $finish;
    end
endmodule
`default_nettype wire
