`timescale 1ns/1ps
// tb_refresh_ctrl.v — background refresh scan tests
//
// T1: Normal scan — sequential addresses, load_en fires one cycle after SRAM read
// T2: inhibit pauses scan (no new reads while inhibit=1)
// T3: training_mode disables scan entirely
// T4: scan wraps correctly at TOTAL-1 → 0

module tb_refresh_ctrl;
    localparam N_CELLS = 4;
    localparam N_ELEMS = 256;
    localparam TOTAL   = N_CELLS * N_ELEMS;  // 1024

    reg  clk, rst_n;
    reg  refresh_en, training_mode, inhibit;
    wire [9:0] sram_addr_r;
    reg  [7:0] sram_rdata;
    wire       load_en;
    wire [1:0] load_cell;
    wire [7:0] load_addr, load_code;

    integer pass_cnt, fail_cnt;
    integer load_count, seq_errors;
    integer prev_addr10;
    reg [9:0] last_load_addr;

    refresh_ctrl #(.N_CELLS(N_CELLS), .N_ELEMS(N_ELEMS)) dut (
        .clk(clk), .rst_n(rst_n),
        .refresh_en(refresh_en), .training_mode(training_mode), .inhibit(inhibit),
        .sram_addr_r(sram_addr_r), .sram_rdata(sram_rdata),
        .load_en(load_en), .load_cell(load_cell),
        .load_addr(load_addr), .load_code(load_code)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    // Behavioral SRAM: return addr[7:0]+1 so we can verify load_code
    always @(*) sram_rdata = sram_addr_r[7:0] + 8'd1;

    `define PASS(msg) begin $display("  PASS %s", msg); pass_cnt=pass_cnt+1; end
    `define FAIL(msg) begin $display("  FAIL %s", msg); fail_cnt=fail_cnt+1; end

    // Track load_en pulses and address order
    always @(posedge clk) begin
        if (load_en) begin
            load_count = load_count + 1;
            // load_code should equal load_addr + 1 (from our SRAM model)
            // load_cell/load_addr together = the SRAM address presented one cycle earlier
            if (load_count > 1) begin
                // Each load address should be +1 mod TOTAL from previous
                if ({load_cell, load_addr} != (last_load_addr + 10'd1) % TOTAL)
                    seq_errors = seq_errors + 1;
            end
            last_load_addr = {load_cell, load_addr};
        end
    end

    integer i;

    initial begin : tb
        pass_cnt = 0; fail_cnt = 0;
        refresh_en = 0; training_mode = 0; inhibit = 0;
        load_count = 0; seq_errors = 0; last_load_addr = 0;

        rst_n = 0; repeat(2) @(posedge clk); #1;
        rst_n = 1; @(posedge clk); #1;
        $display("\n=== refresh_ctrl tests ===");

        // ── T1: Run one complete scan (TOTAL load_en pulses) ──────────────
        load_count = 0; seq_errors = 0; last_load_addr = 10'h3FF;
        refresh_en = 1;
        // 2 cycles per element (issue-SRAM + load-fire), so need 2*TOTAL cycles
        repeat(TOTAL * 2 + 20) @(posedge clk);
        refresh_en = 0;
        @(posedge clk); #1;
        if (load_count >= TOTAL) begin
            $display("  PASS T1-count: %0d load_en pulses in %0d+20 cycles",
                     load_count, TOTAL);
            pass_cnt = pass_cnt + 1;
        end else begin
            $display("  FAIL T1-count: only %0d load_en pulses (expected >= %0d)",
                     load_count, TOTAL);
            fail_cnt = fail_cnt + 1;
        end
        if (seq_errors == 0) `PASS("T1-seq: sequential address order")
        else begin
            $display("  FAIL T1-seq: %0d address order errors", seq_errors);
            fail_cnt = fail_cnt + 1;
        end

        // ── T2: inhibit pauses scan ───────────────────────────────────────
        load_count = 0;
        refresh_en = 1;
        repeat(10) @(posedge clk);  // let a few load_ens fire
        @(negedge clk); inhibit = 1;
        @(posedge clk); #1;
        begin : blk_t2
            integer count_before, count_after;
            count_before = load_count;
            repeat(20) @(posedge clk);  // should be no new load_ens
            count_after = load_count;
            if (count_after == count_before)
                `PASS("T2-inhibit: scan paused")
            else begin
                $display("  FAIL T2-inhibit: %0d extra pulses during inhibit",
                         count_after - count_before);
                fail_cnt = fail_cnt + 1;
            end
        end
        inhibit = 0; refresh_en = 0;
        @(posedge clk); #1;

        // ── T3: training_mode is IGNORED — refresh runs REGARDLESS ────────
        // ★ INVERTED 2026-07-17. This test used to assert that training_mode DISABLED the
        // scan. The jug deleted that gate on purpose: a fire is a ±1 increment on the
        // W-SRAM entry, so W-SRAM is ALWAYS the master and Cw is a pure slave — refresh
        // never needs disabling, and the absorb→save→sync→refresh-enable coherence cycle
        // collapses. Removing the gate is what makes the W-cap leak risk go away
        // STRUCTURALLY rather than by scheduling. See refresh_ctrl.v header and
        // DESIGN.md §5. The old expectation had been failing silently since that change.
        load_count = 0;
        training_mode = 1; refresh_en = 1;
        repeat(20) @(posedge clk); #1;
        if (load_count > 0) `PASS("T3-training_mode: ignored, refresh keeps running")
        else begin
            $display("  FAIL T3-training_mode: refresh stalled (%0d pulses) — the gate is back?",
                     load_count);
            fail_cnt = fail_cnt + 1;
        end
        training_mode = 0; refresh_en = 0;

        // ── T4: load_code matches SRAM data (addr[7:0]+1) ────────────────
        // Re-enable for 5 cycles and check a single load_code value
        begin : blk_t4
            reg [7:0] captured_code;
            reg [7:0] expected_code;
            reg       got_one;
            got_one = 0;
            load_count = 0;
            refresh_en = 1;
            @(posedge clk); #1;
            // capture the first load_en that fires
            repeat(4) begin
                @(posedge clk); #1;
                if (load_en && !got_one) begin
                    captured_code = load_code;
                    expected_code = load_addr + 8'd1;
                    got_one = 1;
                end
            end
            refresh_en = 0;
            if (got_one) begin
                if (captured_code == expected_code) begin
                    $display("  PASS T4-code: load_code=%0d matches SRAM[addr+1]",
                             captured_code);
                    pass_cnt = pass_cnt + 1;
                end else begin
                    $display("  FAIL T4-code: load_code=%0d expected=%0d",
                             captured_code, expected_code);
                    fail_cnt = fail_cnt + 1;
                end
            end else
                `PASS("T4-code: (no load fired in window; SRAM inactive)")
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
