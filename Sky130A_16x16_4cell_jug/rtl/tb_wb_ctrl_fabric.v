`timescale 1ns/1ps
// tb_wb_ctrl_fabric — self-checking B3 test of the S2 control fabric.
// Config (4 nodes): 0,1 = chips group1(L1); 2 = chip group2(L2); 3 = router group4.
// Checks: unicast targeting, group broadcast, ALL broadcast, router broadcast,
//         barrier wired-AND (status read at NODE_ID=0xFF).
module tb_wb_ctrl_fabric;
    localparam integer N = 4;
    reg clk = 0, rst_n = 0;
    reg  [31:0] m_addr = 0, m_wdata = 0;
    reg         m_we = 0, m_stb = 0, m_cyc = 0;
    wire [31:0] m_rdata; wire m_ack;

    // node identity: {kind,id,group}  n0:chip,0,1  n1:chip,1,1  n2:chip,2,2  n3:router,0,4
    wire [N-1:0]   node_kind  = 4'b1000;                       // n3 = router
    wire [8*N-1:0] node_id    = {8'd0, 8'd2, 8'd1, 8'd0};      // n3,n2,n1,n0
    wire [3*N-1:0] node_group = {3'd4, 3'd2, 3'd1, 3'd1};      // n3,n2,n1,n0

    wire [7:0]  n_reg; wire [31:0] n_wdata; wire n_we;
    wire [N-1:0] n_stb;
    wire [32*N-1:0] n_rdata;
    reg  [N-1:0] n_done = 0, part_mask = 0;
    wire barrier_done;
    wire [31:0] obs [0:N-1];

    wb_ctrl_fabric #(.N(N)) fab (
        .clk(clk), .rst_n(rst_n), .m_addr(m_addr), .m_wdata(m_wdata),
        .m_we(m_we), .m_stb(m_stb), .m_cyc(m_cyc), .m_rdata(m_rdata), .m_ack(m_ack),
        .node_kind(node_kind), .node_id(node_id), .node_group(node_group),
        .n_reg(n_reg), .n_wdata(n_wdata), .n_we(n_we), .n_stb(n_stb), .n_rdata(n_rdata),
        .n_done(n_done), .part_mask(part_mask), .barrier_done(barrier_done));

    genvar g;
    generate for (g = 0; g < N; g = g + 1) begin : nodes
        wb_mock_node nd (.clk(clk), .rst_n(rst_n), .stb(n_stb[g]), .we(n_we),
                         .reg_addr(n_reg), .wdata(n_wdata), .rdata(n_rdata[g*32 +: 32]),
                         .observe0(obs[g]));
    end endgenerate

    always #5 clk = ~clk;
    integer errs = 0;

    task wb_write(input [31:0] a, input [31:0] d);
        begin @(negedge clk); m_addr=a; m_wdata=d; m_we=1; m_stb=1; m_cyc=1;
              @(posedge clk); @(negedge clk); m_stb=0; m_we=0; m_cyc=0; @(negedge clk); end
    endtask
    task wb_read(input [31:0] a, output [31:0] d);
        begin @(negedge clk); m_addr=a; m_we=0; m_stb=1; m_cyc=1;
              @(posedge clk); @(negedge clk); d=m_rdata; m_stb=0; m_cyc=0; end
    endtask
    task chk(input [255:0] name, input cond);
        begin if (cond) $display("  PASS  %0s", name);
              else begin $display("  FAIL  %0s", name); errs=errs+1; end end
    endtask

    reg [31:0] rd;
    initial begin
        @(negedge clk); rst_n = 1; @(negedge clk);

        // T1 unicast → (chip,id=1) reg0 = 0xAA : only node1
        wb_write(32'h0000_0100, 32'hAA);
        chk("unicast hits node1",        obs[1]==32'hAA);
        chk("unicast misses node0",      obs[0]!=32'hAA);
        chk("unicast misses node2/3",    obs[2]!=32'hAA && obs[3]!=32'hAA);

        // T2 broadcast group1 (L1) reg0 = 0xBB : node0 & node1, not node2/3
        wb_write(32'h8000_0100, 32'hBB);
        chk("bcast L1 hits node0&1",     obs[0]==32'hBB && obs[1]==32'hBB);
        chk("bcast L1 skips node2 (L2)", obs[2]!=32'hBB);
        chk("bcast L1 skips node3 (rtr)",obs[3]!=32'hBB);

        // T3 broadcast ALL (group0) reg0 = 0xCC : every node
        wb_write(32'h8000_0000, 32'hCC);
        chk("bcast ALL hits all 4",      obs[0]==32'hCC && obs[1]==32'hCC &&
                                          obs[2]==32'hCC && obs[3]==32'hCC);

        // T4 broadcast group4 (ROUTERS) reg0 = 0xDD : only node3; chips stay 0xCC
        wb_write(32'h8000_0400, 32'hDD);
        chk("bcast ROUTERS hits node3",  obs[3]==32'hDD);
        chk("bcast ROUTERS skips chips", obs[0]==32'hCC && obs[1]==32'hCC && obs[2]==32'hCC);

        // T5 unicast read back node2 reg0 (should be 0xCC)
        wb_read(32'h0000_0200, rd);
        chk("unicast read node2 = 0xCC", rd==32'hCC);

        // T6 barrier wired-AND: participants = all; status read at NODE_ID=0xFF
        part_mask = 4'b1111; n_done = 4'b1011; @(negedge clk);
        wb_read(32'h0000_FF00, rd);
        chk("barrier NOT done (n2 busy)", rd[0]==1'b0);
        n_done = 4'b1111; @(negedge clk);
        wb_read(32'h0000_FF00, rd);
        chk("barrier done (all ready)",   rd[0]==1'b1);

        // T7 barrier ignores non-participants: only router participates (done); chips busy
        part_mask = 4'b1000; n_done = 4'b1000; @(negedge clk);
        wb_read(32'h0000_FF00, rd);
        chk("barrier done: only router participates & is done (chips busy, ignored)", rd[0]==1'b1);

        $display("RTL-B3 wb_ctrl_fabric: %0d checks failed", errs);
        if (errs==0) $display("RTL-B3: PASS");
        else         $display("RTL-B3: FAIL");
        $finish;
    end
endmodule
