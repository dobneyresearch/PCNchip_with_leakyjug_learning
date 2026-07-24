`timescale 1ns/1ps
// tb_cap_array_mac.v  —  RTL-T1: MAC weight round-trip parity
//
// Part A: load all 122 valid codes (71–192) into addr 0–121 of cell 0 and check
//         mac_eff_code == the loaded code for every entry (zero tolerance).
//         Verifies the round-trip  load_code → W_cap → mac_eff_code.
//
// Part B (W+E combined MAC) was DELETED 2026-07-17 along with MN3_e.
//   It asserted mac_eff_code == f(W, E) — e.g. W=128, E=+E_MAX_V → 135 — i.e. the
//   DUAL-TAIL cell in which Ce injected current into the MAC and the effective weight
//   was (W+E). Under the jug the MAC is **W ONLY**: Ce is a storage-and-compare node,
//   out of the signal path entirely. E in the forward path is not merely unmodelled,
//   it is HARMFUL (measured: 81.96% at e_fwd=0 vs 80.46% at the old design point, and
//   a positive-feedback collapse to 12% at 2 codes). Part B therefore asserted
//   behaviour that is now wrong BY DESIGN, and could only ever fail.
//   See DESIGN.md §2 and cap_array.v's header.
//
// The jug behaviour of cap_array is covered by tb_jug_fire, tb_cap_theta,
// tb_epoch_match, tb_absorb_match and tb_e_inject_match.

module tb_cap_array_mac;

    reg        clk, rst_n;
    reg        load_en;
    reg  [1:0] load_cell;
    reg  [7:0] load_addr, load_code;
    reg        inject_en;
    reg  [1:0] inject_cell;
    reg  [7:0] inject_addr, inject_delta, inject_x;
    reg  [2:0] inject_bh;
    reg  [5:0] lr_shift;          // 6 bits — must match cap_array (a [4:0] here silently truncates)
    reg  [7:0] jug_theta;
    reg        jug_pulse, jug_dn;
    reg  [1:0] jug_cell;
    reg  [7:0] jug_addr;
    reg        save_req;
    reg  [1:0] save_cell;
    reg  [7:0] save_addr;
    wire [7:0] save_code;
    wire       save_valid;
    wire       fire_up, fire_dn;
    reg  [1:0] mac_cell;
    reg  [7:0] mac_addr;
    wire [7:0] mac_eff_code;

    integer pass_cnt, fail_cnt, i, mismatch_count;

    cap_array #(.N_CELLS(4), .N_ELEMS(256), .WGT_MIN(71), .WGT_MAX(192)) dut (
        .clk(clk), .rst_n(rst_n),
        .load_en(load_en), .load_cell(load_cell),
        .load_addr(load_addr), .load_code(load_code),
        .inject_en(inject_en), .inject_cell(inject_cell),
        .inject_addr(inject_addr), .inject_delta(inject_delta),
        .inject_x(inject_x), .inject_bh(inject_bh), .lr_shift(lr_shift),
        .jug_theta(jug_theta),
        .fire_up(fire_up), .fire_dn(fire_dn),
        .jug_pulse(jug_pulse), .jug_dn(jug_dn),
        .jug_cell(jug_cell), .jug_addr(jug_addr),
        .save_req(save_req), .save_cell(save_cell), .save_addr(save_addr),
        .save_code(save_code), .save_valid(save_valid),
        .mac_cell(mac_cell), .mac_addr(mac_addr), .mac_eff_code(mac_eff_code)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    initial begin : tb
        pass_cnt = 0; fail_cnt = 0; mismatch_count = 0;
        load_en=0; inject_en=0; save_req=0;
        jug_pulse=0; jug_dn=0; jug_cell=0; jug_addr=0; jug_theta=8'd0;
        load_cell=0; load_addr=0; load_code=8'd128;
        inject_cell=0; inject_addr=0; inject_delta=0; inject_x=0;
        inject_bh=3'd3; lr_shift=6'd20;
        mac_cell=2'd0; mac_addr=8'd0;

        rst_n=0; repeat(2) @(posedge clk); #1;
        rst_n=1; @(posedge clk); #1;
        $display("\n=== RTL-T1: MAC weight round-trip parity ===");

        // ── Part A: all 122 valid codes, zero tolerance ────────────────────
        $display("\n-- Part A: round-trip all codes 71..192 (zero tolerance) --");
        mismatch_count = 0;

        for (i = 71; i <= 192; i = i + 1) begin
            @(negedge clk);
            load_en=1; load_cell=2'd0; load_addr=i[7:0]-8'd71; load_code=i[7:0];
            @(posedge clk); #1; load_en=0;
        end

        @(posedge clk); #1;
        for (i = 71; i <= 192; i = i + 1) begin
            mac_cell = 2'd0;
            mac_addr = i[7:0] - 8'd71;
            #1;
            if (mac_eff_code !== i[7:0]) begin
                $display("  MISMATCH code=%0d: mac_eff_code=%0d", i, mac_eff_code);
                mismatch_count = mismatch_count + 1;
            end
        end

        if (mismatch_count == 0) begin
            $display("  PASS Part-A: 122 codes, 0 mismatches"); pass_cnt=pass_cnt+1;
        end else begin
            $display("  FAIL Part-A: %0d mismatches", mismatch_count); fail_cnt=fail_cnt+1;
        end

        // ── Part A2: the MAC is W-ONLY — an injected E must NOT move it ────
        // The positive control for Part B's deletion: charge Ce hard and prove
        // mac_eff_code does not budge. If a future change puts E back in the
        // signal path, THIS is what catches it.
        $display("\n-- Part A2: E is out of the signal path (MAC is W only) --");
        `define LOADC(c, a, cd) \
            @(negedge clk); load_en=1; load_cell=c; load_addr=a; load_code=cd; \
            @(posedge clk); #1; load_en=0;
        `LOADC(2'd1, 8'd0, 8'd128)
        mac_cell=2'd1; mac_addr=8'd0; #1;
        if (mac_eff_code !== 8'd128) begin
            $display("  FAIL A2-pre: mac_eff_code=%0d (exp 128)", mac_eff_code);
            fail_cnt=fail_cnt+1;
        end
        // Slam Ce to its positive rail: delta=+127, x=255, bh=7, no shift.
        @(negedge clk);
        inject_en=1; inject_cell=2'd1; inject_addr=8'd0;
        inject_delta=8'h7F; inject_x=8'hFF; inject_bh=3'd7; lr_shift=6'd0;
        @(posedge clk); #1; inject_en=0;
        @(posedge clk); #1;
        mac_cell=2'd1; mac_addr=8'd0; #1;
        if (mac_eff_code === 8'd128) begin
            $display("  PASS A2: E at rail, mac_eff_code still 128 (W only)");
            pass_cnt=pass_cnt+1;
        end else begin
            $display("  FAIL A2: mac_eff_code=%0d — E LEAKED INTO THE MAC (dual-tail is back?)",
                     mac_eff_code);
            fail_cnt=fail_cnt+1;
        end
        inject_delta=0; inject_x=0; inject_bh=3'd3; lr_shift=6'd20;

        $display("\n=== Results: %0d passed, %0d failed ===", pass_cnt, fail_cnt);
        if (fail_cnt == 0) $display("ALL PASS"); else $display("FAILURES DETECTED");
        $finish;
    end

endmodule
