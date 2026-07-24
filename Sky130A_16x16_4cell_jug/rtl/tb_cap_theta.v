`timescale 1ns/1ps
`default_nettype none
// =============================================================================
// tb_cap_theta — proves the RUNTIME θ path (JUG_THETA) actually moves the fire threshold.
// cap_array now takes jug_theta (mV; 0 => THETA param). th_eff = jug_theta*THETA_LSB (0.001).
// With Ce = 0.075 V held on element 0:
//   jug_theta =  0  → th_eff = THETA(0.050) → 0.075 >= 0.050 → FIRE
//   jug_theta = 50  → th_eff = 0.050        → 0.075 >= 0.050 → FIRE
//   jug_theta = 100 → th_eff = 0.100        → 0.075 <  0.100 → NO FIRE
// If θ were still a fixed parameter, the 100 case would (wrongly) still fire.
//
//   iverilog -g2012 -o /tmp/tbct.vvp tb_cap_theta.v cap_array.v && vvp /tmp/tbct.vvp
// =============================================================================
module tb_cap_theta;
    localparam N_CELLS = 4, N_ELEMS = 256;
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    reg  [7:0] jug_theta;
    reg        cmp_en;
    wire       fire_up, fire_dn;

    cap_array #(.N_CELLS(N_CELLS), .N_ELEMS(N_ELEMS), .WGT_MIN(71), .WGT_MAX(192),
                .THETA(0.050)) u_cap (
        .clk(clk), .rst_n(rst_n), .jug_theta(jug_theta),
        .load_en(1'b0), .load_cell(2'd0), .load_addr(8'd0), .load_code(8'd0),
        .inject_en(1'b0), .inject_cell(2'd0), .inject_addr(8'd0),
        .inject_delta(8'd0), .inject_x(8'd0), .inject_bh(3'd0), .lr_shift(6'd0),
        .cmp_en(cmp_en), .cmp_cell(2'd0), .cmp_addr(8'd0),
        .fire_up(fire_up), .fire_dn(fire_dn),
        .jug_pulse(1'b0), .jug_dn(1'b0), .jug_cell(2'd0), .jug_addr(8'd0),
        .save_req(1'b0), .save_cell(2'd0), .save_addr(8'd0),
        .save_code(), .save_valid(),
        .mac_cell(2'd0), .mac_addr(8'd0), .mac_eff_code()
    );

    integer errors = 0;
    task chk(input [7:0] th, input exp_fire);
        begin
            jug_theta = th; #1;
            if (fire_up !== exp_fire) begin
                $display("  FAIL jug_theta=%0d: fire_up=%b th_eff=%.4f (expected fire=%b)",
                         th, fire_up, u_cap.th_eff, exp_fire);
                errors = errors + 1;
            end else
                $display("  PASS jug_theta=%0d: th_eff=%.4f  fire_up=%b", th, u_cap.th_eff, fire_up);
        end
    endtask

    initial begin
        $display("\n==============================================================");
        $display("  tb_cap_theta — runtime JUG_THETA moves the fire threshold");
        $display("==============================================================");
        jug_theta = 0; cmp_en = 0;
        #20 rst_n = 1; #20;
        u_cap.E_cap[0] = 0.075;      // hold Ce at 0.075 V on element 0
        cmp_en = 1; #1;

        chk(8'd0,   1'b1);           // fallback to THETA=0.050 → fire
        chk(8'd50,  1'b1);           // 0.050 → fire
        chk(8'd100, 1'b0);           // 0.100 → NO fire  ← proves θ is runtime, not fixed
        chk(8'd70,  1'b1);           // 0.070 → fire (0.075>=0.070)
        chk(8'd80,  1'b0);           // 0.080 → no fire

        $display("\n==============================================================");
        if (errors == 0) $display("  ALL PASS — JUG_THETA is wired to the comparator at runtime.");
        else             $display("  *** %0d FAILURE(S) ***", errors);
        $display("==============================================================\n");
        $finish;
    end
endmodule
`default_nettype wire
