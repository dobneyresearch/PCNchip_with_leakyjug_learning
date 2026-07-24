`timescale 1ns/1ps
// tb_router_chip_emx.v — ABSORB_N countdown FSM tests
//
// Uses a behavioral stub for router_chip (base packet-switch logic not under test).
//
// T1: ABSORB_REQ → ABSORB_N asserted (all 8 ports) for ABSORB_TIMEOUT cycles
// T2: ABSORB_DONE pulses exactly once at timeout expiry
// T3: ABSORB_N deasserts after timeout
// T4: Second ABSORB_REQ after idle works correctly
// T5: ABSORB_REQ while absorbing is ignored (no double-trigger)

// Stub for base router_chip — all outputs tied to safe values
module router_chip (
    input  wire        SYS_CLK, RESET_N,
    input  wire [7:0]  PCN_CLK, PCN_MOSI, PCN_CS_N,
    output wire [7:0]  PCN_MISO, PCN_WAKE,
    output wire [1:0]  RING_TX_CLK, RING_TX_DATA,
    input  wire [1:0]  RING_RX_CLK, RING_RX_DATA,
    input  wire        HOST_CLK, HOST_MOSI, HOST_CS_N,
    output wire        HOST_MISO
);
    assign PCN_MISO   = 8'h00;
    assign PCN_WAKE   = 8'h00;
    assign RING_TX_CLK  = 2'b00;
    assign RING_TX_DATA = 2'b00;
    assign HOST_MISO  = 1'b0;
endmodule

module tb_router_chip_emx;
    localparam TIMEOUT = 32;  // short timeout for fast simulation

    reg  SYS_CLK, RESET_N;
    reg  ABSORB_REQ;
    wire [7:0] ABSORB_N;
    wire       ABSORB_DONE;

    integer pass_cnt, fail_cnt;
    integer absorb_n_cycles, done_count;

    router_chip_emx #(.ABSORB_TIMEOUT(TIMEOUT)) dut (
        .SYS_CLK(SYS_CLK), .RESET_N(RESET_N),
        .PCN_CLK(8'h0), .PCN_MOSI(8'h0), .PCN_CS_N(8'hFF),
        .RING_RX_CLK(2'b0), .RING_RX_DATA(2'b0),
        .HOST_CLK(1'b0), .HOST_MOSI(1'b0), .HOST_CS_N(1'b1),
        .ABSORB_REQ(ABSORB_REQ),
        .ABSORB_N(ABSORB_N),
        .ABSORB_DONE(ABSORB_DONE)
    );

    initial SYS_CLK = 0;
    always #5 SYS_CLK = ~SYS_CLK;

    `define PASS(msg) begin $display("  PASS %s", msg); pass_cnt=pass_cnt+1; end
    `define FAIL(msg) begin $display("  FAIL %s", msg); fail_cnt=fail_cnt+1; end

    // Count ABSORB_N=0 cycles and ABSORB_DONE pulses
    always @(posedge SYS_CLK) begin
        if (ABSORB_N == 8'h00) absorb_n_cycles = absorb_n_cycles + 1;
        if (ABSORB_DONE)       done_count       = done_count + 1;
    end

    initial begin : tb
        pass_cnt = 0; fail_cnt = 0;
        ABSORB_REQ = 0;
        absorb_n_cycles = 0; done_count = 0;

        RESET_N = 0; repeat(2) @(posedge SYS_CLK); #1;
        RESET_N = 1; @(posedge SYS_CLK); #1;
        $display("\n=== router_chip_emx tests (ABSORB_TIMEOUT=%0d) ===", TIMEOUT);

        // ── T1+T2+T3: Single ABSORB_REQ cycle ────────────────────────────
        absorb_n_cycles = 0; done_count = 0;
        @(negedge SYS_CLK); ABSORB_REQ = 1;
        @(posedge SYS_CLK); #1; ABSORB_REQ = 0;

        // Before timeout: ABSORB_N should be all-0
        if (ABSORB_N == 8'h00) `PASS("T1-assert: ABSORB_N=0x00 during window")
        else begin
            $display("  FAIL T1-assert: ABSORB_N=%02h", ABSORB_N);
            fail_cnt = fail_cnt + 1;
        end

        // Wait for ABSORB_DONE
        @(posedge ABSORB_DONE); @(posedge SYS_CLK); #1;

        // T2: exactly one ABSORB_DONE pulse
        if (done_count == 1) `PASS("T2-done: exactly 1 ABSORB_DONE pulse")
        else begin
            $display("  FAIL T2-done: %0d ABSORB_DONE pulses", done_count);
            fail_cnt = fail_cnt + 1;
        end

        // T3: ABSORB_N deasserted after timeout
        if (ABSORB_N == 8'hFF) `PASS("T3-deassert: ABSORB_N=0xFF after timeout")
        else begin
            $display("  FAIL T3-deassert: ABSORB_N=%02h", ABSORB_N);
            fail_cnt = fail_cnt + 1;
        end

        // T1b: correct cycle count
        if (absorb_n_cycles == TIMEOUT) begin
            $display("  PASS T1-count: ABSORB_N held for %0d cycles (= TIMEOUT)",
                     absorb_n_cycles);
            pass_cnt = pass_cnt + 1;
        end else begin
            $display("  FAIL T1-count: %0d cycles (expected %0d)",
                     absorb_n_cycles, TIMEOUT);
            fail_cnt = fail_cnt + 1;
        end

        // ── T4: Second request after idle ─────────────────────────────────
        absorb_n_cycles = 0; done_count = 0;
        repeat(5) @(posedge SYS_CLK);
        @(negedge SYS_CLK); ABSORB_REQ = 1;
        @(posedge SYS_CLK); #1; ABSORB_REQ = 0;
        @(posedge ABSORB_DONE); @(posedge SYS_CLK); #1;
        if (done_count == 1 && absorb_n_cycles == TIMEOUT)
            `PASS("T4-second: second request works correctly")
        else begin
            $display("  FAIL T4-second: done=%0d cycles=%0d", done_count, absorb_n_cycles);
            fail_cnt = fail_cnt + 1;
        end

        // ── T5: ABSORB_REQ while absorbing is ignored ─────────────────────
        absorb_n_cycles = 0; done_count = 0;
        @(negedge SYS_CLK); ABSORB_REQ = 1;  // start
        @(posedge SYS_CLK); #1; ABSORB_REQ = 0;
        repeat(5) @(posedge SYS_CLK);         // mid-absorption
        @(negedge SYS_CLK); ABSORB_REQ = 1;  // spurious second request
        @(posedge SYS_CLK); #1; ABSORB_REQ = 0;
        @(posedge ABSORB_DONE); @(posedge SYS_CLK); #1;
        // Should still be exactly TIMEOUT cycles, not 2×TIMEOUT
        if (absorb_n_cycles == TIMEOUT && done_count == 1)
            `PASS("T5-no-retrigger: mid-absorption REQ ignored")
        else begin
            $display("  FAIL T5-no-retrigger: cycles=%0d done=%0d (expected %0d/1)",
                     absorb_n_cycles, done_count, TIMEOUT);
            fail_cnt = fail_cnt + 1;
        end

        $display("\n=== Results: %0d passed, %0d failed ===",
                 pass_cnt, fail_cnt);
        if (fail_cnt == 0) $display("ALL PASS");
        else               $display("FAILURES DETECTED");
        $finish;
    end

    initial begin
        #100000; $display("TIMEOUT"); $finish;
    end

endmodule
