`default_nettype none
`timescale 1ns/1ps
// wb_mock_node.v — minimal WB slave for the B3 fabric test: a 4-word register file.
// Latches on (stb & we); exposes reg[0] as `observe0` so the TB can verify which
// nodes a unicast/broadcast actually reached. Stands in for a real PCN-chip / router
// WB slave (whose per-chip addr[7:2] decode is unchanged, per S2).
module wb_mock_node (
    input  wire        clk, rst_n,
    input  wire        stb, we,
    input  wire [7:0]  reg_addr,
    input  wire [31:0] wdata,
    output wire [31:0] rdata,
    output wire [31:0] observe0
);
    reg [31:0] mem [0:3];
    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) for (i = 0; i < 4; i = i + 1) mem[i] <= 32'h0;
        else if (stb & we) mem[reg_addr[3:2]] <= wdata;   // word-addressed
    end
    assign rdata    = mem[reg_addr[3:2]];
    assign observe0 = mem[0];
endmodule
`default_nettype wire
