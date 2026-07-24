`timescale 1ns/1ps
// tb_save_ctrl.v — save_ctrl FSM + cap_array save-port integration
//
// T1: Pre-load W-caps with known codes, run save_ctrl, verify SRAM writes
//     match expected codes (±1 tolerance for float rounding)
// T2: SRAM write count = 1024 (N_CELLS × N_ELEMS)
// T3: SRAM addresses sequential 0..1023
// T4: busy deasserts and done pulses correctly

module tb_save_ctrl;
    localparam N_CELLS = 4;
    localparam N_ELEMS = 256;
    localparam TOTAL   = N_CELLS * N_ELEMS;

    reg  clk, rst_n;

    // cap_array ports
    reg        load_en;
    reg  [1:0] load_cell;
    reg  [7:0] load_addr, load_code;
    wire [7:0] save_code_cap;
    wire       save_valid_cap;

    // save_ctrl → cap_array
    wire        save_req_sc;
    wire [1:0]  save_cell_sc;
    wire [7:0]  save_addr_sc;

    // save_ctrl ports
    reg         start;
    wire        busy, done;
    wire [9:0]  sram_addr_sc;
    wire [7:0]  sram_wdata_sc;
    wire        sram_we_sc;

    // Tracking
    integer pass_cnt, fail_cnt;
    integer write_count, addr_err, data_err;
    reg [7:0] sram_mem [0:TOTAL-1];
    integer i;

    cap_array #(.N_CELLS(N_CELLS), .N_ELEMS(N_ELEMS)) u_caps (
        .clk(clk), .rst_n(rst_n),
        .load_en(load_en), .load_cell(load_cell),
        .load_addr(load_addr), .load_code(load_code),
        .inject_en(1'b0), .inject_cell(2'b0), .inject_addr(8'b0),
        .inject_delta(8'b0), .inject_x(8'b0), .inject_bh(3'b0), .lr_shift(6'b0),
        // absorb_* retired with absorb_ctrl; the jug port set replaces it. Held inert:
        // this TB exercises save_ctrl, so no fire must occur (jug_theta=0 ⇒ cap_array
        // falls back to its compile-time THETA, and jug_pulse is never asserted).
        .jug_theta(8'd0), .fire_up(), .fire_dn(),
        .jug_pulse(1'b0), .jug_dn(1'b0), .jug_cell(2'b0), .jug_addr(8'b0),
        .save_req(save_req_sc), .save_cell(save_cell_sc), .save_addr(save_addr_sc),
        .save_code(save_code_cap), .save_valid(save_valid_cap),
        .mac_cell(2'b0), .mac_addr(8'b0), .mac_eff_code()
    );

    save_ctrl #(.N_CELLS(N_CELLS), .N_ELEMS(N_ELEMS)) u_save (
        .clk(clk), .rst_n(rst_n),
        .start(start), .busy(busy), .done(done),
        .save_req(save_req_sc), .save_cell_o(save_cell_sc), .save_addr_o(save_addr_sc),
        .save_code(save_code_cap), .save_valid(save_valid_cap),
        .sram_addr(sram_addr_sc), .sram_wdata(sram_wdata_sc), .sram_we(sram_we_sc)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    `define PASS(msg) begin $display("  PASS %s", msg); pass_cnt=pass_cnt+1; end
    `define FAIL(msg) begin $display("  FAIL %s", msg); fail_cnt=fail_cnt+1; end

    // Capture SRAM writes into sram_mem and count them
    always @(posedge clk) begin
        if (sram_we_sc) begin
            write_count = write_count + 1;
            // Check address is sequential
            if (sram_addr_sc != write_count - 1)
                addr_err = addr_err + 1;
            sram_mem[sram_addr_sc] = sram_wdata_sc;
        end
    end

    initial begin : tb
        pass_cnt = 0; fail_cnt = 0;
        load_en = 0; start = 0;
        write_count = 0; addr_err = 0; data_err = 0;
        for (i = 0; i < TOTAL; i = i + 1) sram_mem[i] = 0;

        rst_n = 0; repeat(2) @(posedge clk); #1;
        rst_n = 1; @(posedge clk); #1;
        $display("\n=== save_ctrl tests ===");

        // ── Pre-load W-caps with code = 100 + (flat_index mod 50) ─────────
        // Simple pattern: cell 0 → 100..149 (first 50), then wraps; etc.
        // For verification, we just check a few known entries.
        // Load cell 0 addr 0 → code 100; cell 1 addr 0 → 200 (clamped to 192); etc.
        for (i = 0; i < N_ELEMS; i = i + 1) begin
            @(negedge clk);
            load_en=1; load_cell=2'd0; load_addr=i[7:0];
            load_code = 8'd100 + i[6:0];  // 100..227 (clamped at 192 for i>=92)
            @(posedge clk); #1;
        end
        @(negedge clk);
        load_en=1; load_cell=2'd1; load_addr=8'd0; load_code=8'd150;
        @(posedge clk); #1;
        @(negedge clk);
        load_en=1; load_cell=2'd2; load_addr=8'd0; load_code=8'd71;  // WGT_MIN
        @(posedge clk); #1;
        @(negedge clk);
        load_en=1; load_cell=2'd3; load_addr=8'd255; load_code=8'd192; // WGT_MAX
        @(posedge clk); #1;
        load_en = 0;
        repeat(3) @(posedge clk); #1;

        // ── T1-T4: Run save_ctrl and capture results ───────────────────────
        write_count = 0; addr_err = 0;
        @(negedge clk); start = 1;
        @(posedge clk); #1; start = 0;

        @(posedge done); @(posedge clk); #1;

        // T2: write count
        if (write_count == TOTAL) begin
            $display("  PASS T2-count: %0d SRAM writes", write_count);
            pass_cnt = pass_cnt + 1;
        end else begin
            $display("  FAIL T2-count: %0d SRAM writes (expected %0d)",
                     write_count, TOTAL);
            fail_cnt = fail_cnt + 1;
        end

        // T3: sequential addresses
        if (addr_err == 0) `PASS("T3-addrs: sequential 0..1023")
        else begin
            $display("  FAIL T3-addrs: %0d address errors", addr_err);
            fail_cnt = fail_cnt + 1;
        end

        // T4: busy deasserted after done
        if (!busy) `PASS("T4-idle: busy=0 after done")
        else `FAIL("T4-idle: still busy after done")

        // T1: verify spot-check data values (±1 tolerance)
        // cell 0 addr 0: loaded with code 100 → save_code should be 100
        begin : blk_check
            integer ok;
            ok = 1;
            if (sram_mem[0] < 8'd99 || sram_mem[0] > 8'd101) begin
                $display("  FAIL T1-data[0]: sram[0]=%0d expected ~100", sram_mem[0]);
                ok = 0; fail_cnt = fail_cnt + 1;
            end
            // cell 0 addr 1: code 101
            if (sram_mem[1] < 8'd100 || sram_mem[1] > 8'd102) begin
                $display("  FAIL T1-data[1]: sram[1]=%0d expected ~101", sram_mem[1]);
                ok = 0; fail_cnt = fail_cnt + 1;
            end
            // cell 1 addr 0: code 150 → sram_mem[256]
            if (sram_mem[256] < 8'd149 || sram_mem[256] > 8'd151) begin
                $display("  FAIL T1-data[256]: sram[256]=%0d expected ~150", sram_mem[256]);
                ok = 0; fail_cnt = fail_cnt + 1;
            end
            // cell 2 addr 0 (WGT_MIN=71) → sram_mem[512]
            if (sram_mem[512] < 8'd70 || sram_mem[512] > 8'd72) begin
                $display("  FAIL T1-data[512]: sram[512]=%0d expected ~71", sram_mem[512]);
                ok = 0; fail_cnt = fail_cnt + 1;
            end
            // cell 3 addr 255 (WGT_MAX=192) → sram_mem[1023]
            if (sram_mem[1023] < 8'd191 || sram_mem[1023] > 8'd192) begin
                $display("  FAIL T1-data[1023]: sram[1023]=%0d expected ~192", sram_mem[1023]);
                ok = 0; fail_cnt = fail_cnt + 1;
            end
            if (ok) `PASS("T1-data: spot-check codes correct (±1)")
        end

        $display("\n=== Results: %0d passed, %0d failed ===",
                 pass_cnt, fail_cnt);
        if (fail_cnt == 0) $display("ALL PASS");
        else               $display("FAILURES DETECTED");
        $finish;
    end

    initial begin
        #2000000; $display("TIMEOUT"); $finish;
    end

endmodule
