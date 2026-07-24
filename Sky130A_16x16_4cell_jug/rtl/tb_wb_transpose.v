`timescale 1ns/1ps
`default_nettype none
// =============================================================================
// tb_wb_transpose — proves the new WB register path for transpose-at-source:
//   BP_TRIG (0xA0) write: [0]=start_transpose pulse, [5:4]=transpose_cell, resets DST ptr
//   BP_TRIG (0xA0) read : [0]=transpose_busy
//   BP_DST  (0xA4) read : partial byte at an auto-incrementing pointer (16 int8)
//
//   iverilog -g2012 -o /tmp/tbwt.vvp tb_wb_transpose.v pcn_wb_regs_4cell_emx.v && vvp /tmp/tbwt.vvp
// =============================================================================

module tb_wb_transpose;
    localparam N_ROWS = 16, N_CELLS = 4;

    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    reg  [31:0] wb_addr_i, wb_dat_i;
    reg  [3:0]  wb_sel_i;
    reg         wb_we_i, wb_cyc_i, wb_stb_i;
    wire [31:0] wb_dat_o;
    wire        wb_ack_o;

    reg  [N_ROWS*8-1:0] transpose_partial;
    reg                 transpose_busy;
    wire                start_transpose;
    wire [1:0]          transpose_cell;
    // LUT streaming outputs
    wire [7:0]          wgt_lut_addr;
    wire [9:0]          wgt_lut_wdata;
    wire                wgt_lut_we;

    // only the ports this test needs are connected; the rest float (harmless here).
    pcn_wb_regs_4cell_emx #(.N_ROWS(N_ROWS), .N_CELLS(N_CELLS)) u_regs (
        .clk(clk), .rst_n(rst_n),
        .wb_addr_i(wb_addr_i), .wb_dat_i(wb_dat_i), .wb_sel_i(wb_sel_i),
        .wb_we_i(wb_we_i), .wb_cyc_i(wb_cyc_i), .wb_stb_i(wb_stb_i),
        .wb_dat_o(wb_dat_o), .wb_ack_o(wb_ack_o),
        .start_transpose(start_transpose), .transpose_cell(transpose_cell),
        .transpose_partial(transpose_partial), .transpose_busy(transpose_busy),
        .wgt_lut_addr(wgt_lut_addr), .wgt_lut_wdata(wgt_lut_wdata), .wgt_lut_we(wgt_lut_we),
        // inputs that must not float into the logic we exercise:
        .irq_frame_done(1'b0), .router_busy(1'b0),
        .e_update_busy(1'b0), .absorb_busy(1'b0), .save_busy(1'b0),
        .sram_rdata(8'h0), .act_rdata_wb_i(8'h0), .ierr_dig_i(16'h0),
        .routing_matrix_hebb_i({N_CELLS*N_CELLS{1'b0}}), .routing_matrix_hebb_we(1'b0)
    );

    integer errors = 0, k;
    integer trig_pulses;
    reg [7:0] got;

    // WGT_LUT commit monitor: record (addr, data) on each wgt_lut_we pulse
    integer   lut_n = 0;
    reg [7:0] lut_a [0:7];
    reg [9:0] lut_d [0:7];
    always @(posedge clk) if (wgt_lut_we && lut_n < 8) begin
        lut_a[lut_n] = wgt_lut_addr;  lut_d[lut_n] = wgt_lut_wdata;  lut_n = lut_n + 1;
    end

    // single-beat WB write: request for one cycle, wait ack
    task wb_write(input [31:0] a, input [31:0] d);
        begin
            @(negedge clk); wb_addr_i=a; wb_dat_i=d; wb_we_i=1; wb_sel_i=4'hF; wb_cyc_i=1; wb_stb_i=1;
            @(negedge clk); wb_cyc_i=0; wb_stb_i=0; wb_we_i=0;
        end
    endtask

    // single-beat WB read: request one cycle, capture dat_o when ack
    task wb_read(input [31:0] a, output [31:0] d);
        begin
            @(negedge clk); wb_addr_i=a; wb_we_i=0; wb_sel_i=4'hF; wb_cyc_i=1; wb_stb_i=1;
            @(posedge clk);              // module registers dat_o + ack here
            @(negedge clk); wb_cyc_i=0; wb_stb_i=0;
            d = wb_dat_o;
        end
    endtask

    // count start_transpose pulses
    always @(posedge clk) if (start_transpose) trig_pulses = trig_pulses + 1;

    reg [31:0] rd;
    initial begin
        $display("\n==============================================================");
        $display("  tb_wb_transpose — the BP_TRIG / BP_DST register path");
        $display("==============================================================");
        wb_addr_i=0; wb_dat_i=0; wb_sel_i=0; wb_we_i=0; wb_cyc_i=0; wb_stb_i=0;
        transpose_busy=0; trig_pulses=0;
        // partial pattern: byte k = k+1  (0x01..0x10)
        for (k = 0; k < N_ROWS; k = k + 1) transpose_partial[k*8 +: 8] = k + 1;

        #20 rst_n = 1; #20;

        // ── BP_TRIG: start=1, cell=2 → data = (2<<4)|1 = 0x21 ──────────────
        wb_write(32'hA0, 32'h21);
        @(posedge clk); @(posedge clk); #1;   // let the 1-cycle start_transpose pulse be counted
        if (trig_pulses != 1) begin
            $display("  *** FAIL: start_transpose pulsed %0d times (expected 1)", trig_pulses);
            errors = errors + 1;
        end else $display("  PASS: BP_TRIG pulsed start_transpose exactly once");
        if (transpose_cell !== 2'd2) begin
            $display("  *** FAIL: transpose_cell = %0d (expected 2)", transpose_cell);
            errors = errors + 1;
        end else $display("  PASS: transpose_cell latched = 2");

        // ── BP_DST: read 16 bytes, expect 0x01..0x10 in order (ptr auto-inc) ──
        $write("  BP_DST bytes:");
        for (k = 0; k < N_ROWS; k = k + 1) begin
            wb_read(32'hA4, rd);
            got = rd[7:0];
            $write(" %0d", got);
            if (got !== (k+1)) begin
                $display("\n  *** FAIL: BP_DST byte %0d = %0d (expected %0d)", k, got, k+1);
                errors = errors + 1;
            end
        end
        $display("");
        if (errors == 0) $display("  PASS: BP_DST returned all 16 partial bytes in order");

        // ── BP_TRIG read = transpose_busy ────────────────────────────────
        transpose_busy = 1; #1;
        wb_read(32'hA0, rd);
        if (rd[0] !== 1'b1) begin
            $display("  *** FAIL: BP_TRIG read busy bit = %b (expected 1)", rd[0]);
            errors = errors + 1;
        end else $display("  PASS: BP_TRIG read reports transpose_busy");

        // ── WGT_LUT streaming: set addr=5, then stream 3 entries ─────────────
        // Proves the fix: wgt_lut_we pulses ONCE per write and the addr auto-increments (5,6,7).
        wb_write(32'h94, 32'd5);                       // WGT_LUT_ADDR = 5
        lut_n = 0;
        wb_write(32'h98, 32'h111);                     // WGT_LUT_DATA
        wb_write(32'h98, 32'h222);
        wb_write(32'h98, 32'h333);
        @(posedge clk); @(posedge clk); #1;
        // expect 3 commits at addr 5,6,7 with data 0x111,0x222,0x333
        if (lut_n != 3) begin
            $display("  *** FAIL: WGT_LUT committed %0d times (expected 3 — we stuck?)", lut_n);
            errors = errors + 1;
        end else if (lut_a[0]!==8'd5 || lut_a[1]!==8'd6 || lut_a[2]!==8'd7) begin
            $display("  *** FAIL: WGT_LUT addr seq = %0d,%0d,%0d (expected 5,6,7)",
                     lut_a[0], lut_a[1], lut_a[2]);
            errors = errors + 1;
        end else if (lut_d[0]!==10'h111 || lut_d[1]!==10'h222 || lut_d[2]!==10'h333) begin
            $display("  *** FAIL: WGT_LUT data seq wrong");
            errors = errors + 1;
        end else
            $display("  PASS: WGT_LUT streamed 3 entries — we pulses, addr 5→6→7 (bug fixed)");

        $display("\n==============================================================");
        if (errors == 0) $display("  ALL PASS — the transpose WB register path works.");
        else             $display("  *** %0d FAILURE(S) ***", errors);
        $display("==============================================================\n");
        $finish;
    end
    initial begin #100000; $display("*** TIMEOUT ***"); $finish; end
endmodule
`default_nettype wire
