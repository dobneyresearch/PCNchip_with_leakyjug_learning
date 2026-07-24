`timescale 1ns/1ps
// =============================================================================
// tb_param_smoke.v — does the design actually RE-PARAMETERISE?
//
// Every other TB runs at the 16×16×4 default, so they prove only that N=16 still
// works — they cannot catch a hardcoded width. This one instantiates the geometry
// -dependent modules at a DELIBERATELY NON-DEFAULT size (32×32, 8 cells) and checks
// the structural properties that the derived widths control:
//
//   S1  pcn_transpose reads EVERY element of the selected cell EXACTLY ONCE, in
//       order — addr must sweep {cell, 0}..{cell, N²-1}. This is the direct test of
//       the derived ELEM_AW / CELL_AW / sram_addr width.
//   S2  a NON-ZERO cell_sel addresses the right SRAM region (tests CELL_AW: with the
//       old [1:0] cell_sel, cell 5 would alias to cell 1).
//   S3  done fires after exactly N² reads, and busy behaves.
//   S4  router_chip_emx derives its ABSORB_TIMEOUT from the geometry (the window must
//       span a full comparator sweep) rather than the old literal 1280.
//
// It deliberately does NOT re-derive router_backproj's arithmetic — a test that
// mirrors the implementation proves nothing. Correctness of the maths at N=16 is
// tb_pcn_transpose's job (golden vectors); this is about geometry.
// =============================================================================

// Behavioural stub for router_chip (the base packet switch lives outside this folder and
// is not under test here) — same approach as tb_router_chip_emx.
module router_chip (
    input  wire        SYS_CLK, RESET_N,
    input  wire [7:0]  PCN_CLK, PCN_MOSI, PCN_CS_N,
    output wire [7:0]  PCN_MISO, PCN_WAKE,
    output wire [1:0]  RING_TX_CLK, RING_TX_DATA,
    input  wire [1:0]  RING_RX_CLK, RING_RX_DATA,
    input  wire        HOST_CLK, HOST_MOSI, HOST_CS_N,
    output wire        HOST_MISO
);
    assign PCN_MISO     = 8'h00;
    assign PCN_WAKE     = 8'h00;
    assign RING_TX_CLK  = 2'b00;
    assign RING_TX_DATA = 2'b00;
    assign HOST_MISO    = 1'b0;
endmodule

module tb_param_smoke;

    // ── a geometry that is NOT the default, in BOTH dimensions ────────────────
    localparam integer N       = 32;                  // vs default 16
    localparam integer N_CELLS = 8;                   // vs default 4
    localparam integer N_ELEMS = N * N;               // 1024
    localparam integer CELL_AW = $clog2(N_CELLS);     // 3
    localparam integer ELEM_AW = $clog2(N_ELEMS);     // 10
    localparam integer AW      = CELL_AW + ELEM_AW;   // 13
    localparam integer PICK    = 5;                   // a cell that needs >2 bits to address

    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    // mock W-SRAM spanning ALL cells
    reg  [7:0] wsram [0:N_CELLS*N_ELEMS-1];
    wire [AW-1:0] t_addr;
    wire          t_re;
    reg  [7:0]    t_rdata;
    always @(posedge clk) if (t_re) t_rdata <= wsram[t_addr];

    reg                    start = 0;
    reg  [CELL_AW-1:0]     cell_sel = 0;
    reg  [N*8-1:0]         delta_flat;
    wire                   busy, done;
    wire [N*8-1:0]         partial_flat;

    pcn_transpose #(.N(N), .N_CELLS(N_CELLS)) dut (
        .clk(clk), .rst_n(rst_n),
        .start(start), .cell_sel(cell_sel), .delta_flat(delta_flat),
        .busy(busy), .done(done), .partial_flat(partial_flat),
        .sram_addr(t_addr), .sram_re(t_re), .sram_rdata(t_rdata)
    );

    // ── observe the read stream ───────────────────────────────────────────────
    integer nread, out_of_order, wrong_cell, k;
    integer fails;
    reg [AW-1:0] expect_addr;

    always @(posedge clk) begin
        if (t_re && busy) begin
            if (t_addr !== expect_addr) begin
                if (out_of_order < 5)
                    $display("  MISMATCH read %0d: addr=%0h expected %0h", nread, t_addr, expect_addr);
                out_of_order = out_of_order + 1;
            end
            if (t_addr[AW-1 -: CELL_AW] !== PICK[CELL_AW-1:0]) wrong_cell = wrong_cell + 1;
            nread       = nread + 1;
            expect_addr = expect_addr + 1'b1;
        end
    end

    initial begin
        $display("\n==============================================================");
        $display("  tb_param_smoke — re-parameterisation at %0dx%0d, %0d cells", N, N, N_CELLS);
        $display("  (defaults are 16x16, 4 cells — a hardcoded width cannot survive this)");
        $display("==============================================================");
        fails = 0; nread = 0; out_of_order = 0; wrong_cell = 0;
        for (k = 0; k < N_CELLS*N_ELEMS; k = k + 1) wsram[k] = k[7:0];
        for (k = 0; k < N; k = k + 1) delta_flat[k*8 +: 8] = 8'd0;

        #20 rst_n = 1; #20;

        // ── S1/S2/S3: sweep cell PICK ─────────────────────────────────────────
        expect_addr = {PICK[CELL_AW-1:0], {ELEM_AW{1'b0}}};
        @(negedge clk); start = 1; cell_sel = PICK[CELL_AW-1:0];
        @(negedge clk); start = 0;
        wait (done);
        @(posedge clk);

        if (nread == N_ELEMS)
            $display("  PASS S3: done after exactly %0d reads (= N*N)", nread);
        else begin
            $display("  FAIL S3: %0d reads, expected %0d", nread, N_ELEMS);
            fails = fails + 1;
        end

        if (out_of_order == 0)
            $display("  PASS S1: address swept {cell,0}..{cell,N*N-1} in order");
        else begin
            $display("  FAIL S1: %0d out-of-order/incorrect addresses", out_of_order);
            fails = fails + 1;
        end

        if (wrong_cell == 0)
            $display("  PASS S2: all reads targeted cell %0d (CELL_AW=%0d derived)", PICK, CELL_AW);
        else begin
            $display("  FAIL S2: %0d reads hit the wrong cell — cell_sel too narrow?", wrong_cell);
            fails = fails + 1;
        end

        // ── S4: ABSORB_TIMEOUT must span a full sweep at THIS geometry ────────
        if (u_rtr.ABSORB_TIMEOUT >= N_CELLS*N_ELEMS)
            $display("  PASS S4: ABSORB_TIMEOUT=%0d covers the %0d-element sweep",
                     u_rtr.ABSORB_TIMEOUT, N_CELLS*N_ELEMS);
        else begin
            $display("  FAIL S4: ABSORB_TIMEOUT=%0d < sweep %0d — window truncates the sweep",
                     u_rtr.ABSORB_TIMEOUT, N_CELLS*N_ELEMS);
            fails = fails + 1;
        end

        $display("==============================================================");
        if (fails == 0) $display("  ALL PASS — geometry re-parameterises cleanly.");
        else            $display("  *** %0d failures ***", fails);
        $display("==============================================================\n");
        $finish;
    end

    // ── S4 DUT: the router's absorb window, at the same non-default geometry ──
    wire [7:0] absorb_n_w;
    router_chip_emx #(.N_ROWS(N), .N_COLS(N), .N_CELLS(N_CELLS)) u_rtr (
        .SYS_CLK(clk), .RESET_N(rst_n),
        .PCN_CLK(8'd0), .PCN_MOSI(8'd0), .PCN_MISO(), .PCN_CS_N(8'hFF), .PCN_WAKE(),
        .RING_TX_CLK(), .RING_TX_DATA(), .RING_RX_CLK(2'b00), .RING_RX_DATA(2'b00),
        .HOST_CLK(1'b0), .HOST_MOSI(1'b0), .HOST_MISO(), .HOST_CS_N(1'b1),
        .ABSORB_REQ(1'b0), .ABSORB_N(absorb_n_w), .ABSORB_DONE()
    );

endmodule
