`timescale 1ns/1ps
`default_nettype none
// tb_e_inject_match.v  —  RTL-T2: e_inject_ctrl outer-product vs the Python formula.
//
// ★ UPDATED FOR THE JUG (2026-07-15). The old version verified injects by reading mac_eff_code —
//   but the JUG MAC is W ONLY (E is out of the signal path, MN3_e deleted), so injects do NOT
//   change mac_eff_code. Verification now reads E_cap directly (as tb_epoch_match/absorb_match do).
//   Also dropped the retired `.absorb_*` cap_array ports; tied the jug/compare ports inactive.
//
// Wires e_inject_ctrl → cap_array. delta_flat row0 = 0x10 (+16); act col0 = 0xFF, col1 = 0x80.
// bh=3 (bh_r=4), lr_shift=20.  dv = delta * x * 4 / 2^20:
//   (row0,col0): 16*255*4/2^20 = 0.0155640   → E_cap[0]
//   (row0,col1): 16*128*4/2^20 = 0.0078125   → E_cap[1]
//   (row0,col≥2): 0 (act=0);  (row≥1,·): 0 (delta=0)  → E_cap = 0
//
//   iverilog -g2012 -o /tmp/tbei.vvp tb_e_inject_match.v e_inject_ctrl.v cap_array.v && vvp /tmp/tbei.vvp

module tb_e_inject_match;
    localparam N_ROWS = 16, N_COLS = 16;
    localparam real TOL = 1.0e-6;

    reg  clk = 0, rst_n;
    always #5 clk = ~clk;

    // e_inject_ctrl ports
    reg                    start;
    reg  [1:0]             inject_cell;
    wire                   busy, done;
    reg  [N_ROWS*8-1:0]    delta_flat;
    reg  [N_COLS*8-1:0]    act_flat;
    reg  [2:0]             bh;
    reg  [5:0]             lr_shift;

    // e_inject_ctrl → cap_array wires
    wire        inj_en;
    wire [1:0]  inj_cell_o;
    wire [7:0]  inj_addr, inj_delta, inj_x;

    // cap_array load/save/MAC ports
    reg        load_en;
    reg  [1:0] load_cell;
    reg  [7:0] load_addr, load_code;

    integer pass_cnt = 0, fail_cnt = 0, i;
    integer bad;
    real    e_meas, e_want, e_diff;

    e_inject_ctrl #(.N_ROWS(N_ROWS), .N_COLS(N_COLS)) u_einj (
        .clk(clk), .rst_n(rst_n),
        .start(start), .inject_cell(inject_cell), .busy(busy), .done(done),
        .delta_flat(delta_flat), .act_flat(act_flat),
        .bh(bh), .lr_shift(lr_shift),
        .inject_en(inj_en), .inject_cell_o(inj_cell_o),
        .inject_addr(inj_addr), .inject_delta(inj_delta), .inject_x(inj_x)
    );

    cap_array #(.N_CELLS(4), .N_ELEMS(256), .WGT_MIN(71), .WGT_MAX(192)) u_cap (
        .clk(clk), .rst_n(rst_n), .jug_theta(8'd0),
        .load_en(load_en), .load_cell(load_cell),
        .load_addr(load_addr), .load_code(load_code),
        .inject_en(inj_en), .inject_cell(inj_cell_o),
        .inject_addr(inj_addr), .inject_delta(inj_delta),
        .inject_x(inj_x), .inject_bh(bh), .lr_shift(lr_shift),
        // jug/compare ports inactive — this test only exercises the INJECT path
        .cmp_en(1'b0), .cmp_cell(2'd0), .cmp_addr(8'd0), .fire_up(), .fire_dn(),
        .jug_pulse(1'b0), .jug_dn(1'b0), .jug_cell(2'd0), .jug_addr(8'd0),
        .save_req(1'b0), .save_cell(2'd0), .save_addr(8'd0), .save_code(), .save_valid(),
        .mac_cell(2'd0), .mac_addr(8'd0), .mac_eff_code()
    );

    task check_e(input [127:0] name, input [7:0] addr, input real exp);
        begin
            e_meas = u_cap.E_cap[addr];
            e_diff = (e_meas > exp) ? (e_meas - exp) : (exp - e_meas);
            if (e_diff <= TOL) begin
                $display("  PASS %-14s E_cap[%0d]=%.7f (exp %.7f)", name, addr, e_meas, exp);
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("  FAIL %-14s E_cap[%0d]=%.7f (exp %.7f)", name, addr, e_meas, exp);
                fail_cnt = fail_cnt + 1;
            end
        end
    endtask

    initial begin : tb
        start = 0; inject_cell = 2'd0; bh = 3'd3; lr_shift = 6'd20; load_en = 0;
        delta_flat = 128'h0000_0000_0000_0000_0000_0000_0000_0010;  // row0 = 0x10
        act_flat   = 128'h0000_0000_0000_0000_0000_0000_0000_80FF;  // col0=0xFF col1=0x80

        rst_n = 0; repeat(2) @(posedge clk); #1;
        rst_n = 1; @(posedge clk); #1;
        $display("\n=== RTL-T2: e_inject_ctrl outer-product parity (jug: verify via E_cap) ===");

        // load cell 0 with W=128 (a neutral baseline; E is independent, starts at 0)
        for (i = 0; i < 256; i = i + 1) begin
            @(negedge clk); load_en=1; load_cell=2'd0; load_addr=i[7:0]; load_code=8'd128;
            @(posedge clk); #1; load_en=0;
        end

        // run the inject FSM (256 outer-product elements)
        @(negedge clk); start=1; inject_cell=2'd0;
        @(posedge clk); #1; start=0;
        @(posedge done); @(posedge clk); #1;
        $display("  inject FSM completed (256 cycles)");

        check_e("T2-A r0c0", 8'd0,  0.0155640);   // 16*255*4/2^20
        check_e("T2-B r0c1", 8'd1,  0.0078125);   // 16*128*4/2^20
        check_e("T2-C r0c2", 8'd2,  0.0);         // act[2]=0
        check_e("T2-D r1c0", 8'd16, 0.0);         // delta[1]=0

        // every other element must still be exactly 0
        bad = 0;
        for (i = 0; i < 256; i = i + 1)
            if (i != 0 && i != 1) begin
                e_meas = u_cap.E_cap[i];
                if (e_meas > TOL || e_meas < -TOL) bad = bad + 1;
            end
        if (bad == 0) begin
            $display("  PASS T2-E: all other 254 elements E_cap = 0"); pass_cnt = pass_cnt + 1;
        end else begin
            $display("  FAIL T2-E: %0d elements have non-zero E_cap", bad); fail_cnt = fail_cnt + 1;
        end

        $display("\n=== Results: %0d passed, %0d failed ===", pass_cnt, fail_cnt);
        if (fail_cnt == 0) $display("ALL PASS"); else $display("FAILURES DETECTED");
        $finish;
    end
    initial begin #1000000; $display("TIMEOUT"); $finish; end
endmodule
`default_nettype wire
