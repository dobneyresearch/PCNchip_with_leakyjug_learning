`timescale 1ns/1ps
// tb_router_backproj — co-sim router_backproj.v vs the B1 twin (gen_backproj_stim.py).
// PASS = every partial within TOL LSB of the twin (TOL=2 covers isqrt-floor + round-mode).
module tb_router_backproj;
    localparam integer N = 16, NCASE = 64, TOL = 2;

    reg clk = 0, rst_n = 0, start = 0;
    reg  [N*N*8-1:0] w_flat;
    reg  [N*8-1:0]   delta_flat;
    wire [N*8-1:0]   partial_flat;
    wire done;

    reg [7:0] wmem [0:N*N*NCASE-1];
    reg [7:0] dmem [0:N*NCASE-1];
    reg [7:0] emem [0:N*NCASE-1];
    integer c, i, j, d, errs, maxd, got, exp;

    router_backproj dut (.clk(clk), .rst_n(rst_n), .start(start),
                         .w_flat(w_flat), .delta_flat(delta_flat),
                         .partial_flat(partial_flat), .done(done));
    always #5 clk = ~clk;

    initial begin
        $readmemh("bp_w.hex",        wmem);
        $readmemh("bp_delta.hex",    dmem);
        $readmemh("bp_expected.hex", emem);
        errs = 0; maxd = 0;
        @(negedge clk); rst_n = 1; @(negedge clk);

        for (c = 0; c < NCASE; c = c + 1) begin
            for (i = 0; i < N; i = i + 1)
                for (j = 0; j < N; j = j + 1)
                    w_flat[(i*N+j)*8 +: 8] = wmem[c*N*N + i*N + j];
            for (i = 0; i < N; i = i + 1)
                delta_flat[i*8 +: 8] = dmem[c*N + i];

            @(negedge clk); start = 1; @(negedge clk); start = 0;
            while (!done) @(negedge clk);

            for (j = 0; j < N; j = j + 1) begin
                got = $signed(partial_flat[j*8 +: 8]);
                exp = $signed(emem[c*N + j]);
                d   = (got > exp) ? (got - exp) : (exp - got);
                if (d > maxd) maxd = d;
                if (d > TOL) begin
                    errs = errs + 1;
                    if (errs <= 8)
                        $display("  case %0d row %0d: rtl=%0d twin=%0d (Δ=%0d)", c, j, got, exp, d);
                end
            end
        end

        $display("RTL-B1 router_backproj co-sim: %0d cases x %0d rows, max|Δ|=%0d LSB, %0d over TOL(%0d)",
                 NCASE, N, maxd, errs, TOL);
        if (errs == 0) $display("RTL-B1: PASS (all within %0d LSB of the twin)", TOL);
        else           $display("RTL-B1: FAIL");
        $finish;
    end
endmodule
