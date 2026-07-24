`default_nettype none
// E-inject FSM: runs one N_ROWS×N_COLS outer-product injection for a single cell.
//
// Host loads delta_flat[N_ROWS*8-1:0] (16 signed bytes, row-major) into WB regs,
// sets act_flat[N_COLS*8-1:0] (captured from inp_dac writes by inp_act_latch),
// specifies inject_cell, then pulses start.  FSM runs N_ROWS×N_COLS = 256 cycles.
//
// Each cycle fires cap_array inject_en for one (row, col) element.
// Shared bh and lr_shift come from WB regs.

module e_inject_ctrl #(
    parameter N_ROWS = 16,
    parameter N_COLS = 16
) (
    input  wire        clk,
    input  wire        rst_n,

    // Control
    input  wire        start,
    input  wire  [1:0] inject_cell,
    output reg         busy,
    output reg         done,          // one-cycle pulse

    // Delta buffer (N_ROWS × 8-bit signed, packed LSB-first: row0 in [7:0])
    input  wire [N_ROWS*8-1:0] delta_flat,

    // Input activation latch (N_COLS × 8-bit unsigned, packed LSB-first)
    input  wire [N_COLS*8-1:0] act_flat,

    // Boss-h and LR shift from WB regs
    input  wire  [2:0] bh,
    input  wire  [5:0] lr_shift,   // 6-bit (see cap_array.v)

    // cap_array inject port
    output reg         inject_en,
    output reg   [1:0] inject_cell_o,
    output reg   [7:0] inject_addr,
    output reg   [7:0] inject_delta,
    output reg   [7:0] inject_x
);
    localparam N_ELEMS = N_ROWS * N_COLS;  // 256

    reg [7:0] cnt;   // element index [0..255]
    reg [3:0] row;   // [0..N_ROWS-1]
    reg [3:0] col;   // [0..N_COLS-1]

    localparam IDLE = 1'b0, RUN = 1'b1;
    reg state;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;  busy <= 1'b0;  done <= 1'b0;
            inject_en <= 1'b0;
            cnt <= 8'd0;  row <= 4'd0;  col <= 4'd0;
        end else begin
            done      <= 1'b0;
            inject_en <= 1'b0;

            case (state)
                IDLE: if (start) begin
                    state <= RUN;  busy <= 1'b1;
                    cnt <= 8'd0;  row <= 4'd0;  col <= 4'd0;
                end

                RUN: begin
                    inject_en     <= 1'b1;
                    inject_cell_o <= inject_cell;
                    inject_addr   <= {row, col};  // row*N_COLS + col (N_COLS=16=2^4)
                    inject_delta  <= delta_flat[row*8 +: 8];
                    inject_x      <= act_flat[col*8 +: 8];

                    if (cnt == N_ELEMS - 1) begin
                        state <= IDLE;  busy <= 1'b0;  done <= 1'b1;
                        cnt <= 8'd0;  row <= 4'd0;  col <= 4'd0;
                    end else begin
                        cnt <= cnt + 8'd1;
                        if (col == N_COLS - 1) begin
                            col <= 4'd0;
                            row <= row + 4'd1;
                        end else begin
                            col <= col + 4'd1;
                        end
                    end
                end
            endcase
        end
    end

endmodule
