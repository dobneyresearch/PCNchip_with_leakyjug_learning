`default_nettype none
// Save FSM: reads all W-caps via cap_array save interface, writes codes to W-SRAM.
//
// Triggered by EMX_CTRL[2] (host pulse).  Runs N_CELLS × N_ELEMS = 1024 cycles
// (one element per REQ→WAIT→WRITE mini-sequence = 3 cycles per element).
// Should not run concurrently with absorb_ctrl or refresh_ctrl (both driven by
// the top-level inhibit signal).

module save_ctrl #(
    parameter N_CELLS = 4,
    parameter N_ELEMS = 256
) (
    input  wire        clk,
    input  wire        rst_n,

    input  wire        start,
    output reg         busy,
    output reg         done,      // one-cycle pulse

    // cap_array save port
    output reg         save_req,
    output reg   [1:0] save_cell_o,
    output reg   [7:0] save_addr_o,
    input  wire  [7:0] save_code,
    input  wire        save_valid,

    // W-SRAM write port (top-level must MUX; save_ctrl has highest write priority)
    output reg   [9:0] sram_addr,
    output reg   [7:0] sram_wdata,
    output reg         sram_we
);
    localparam TOTAL = N_CELLS * N_ELEMS;  // 1024

    reg [9:0] cnt;

    localparam IDLE  = 2'd0,
               REQ   = 2'd1,
               WAIT  = 2'd2,
               WRITE = 2'd3;
    reg [1:0] state;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state   <= IDLE;  busy <= 1'b0;  done <= 1'b0;
            save_req <= 1'b0; sram_we <= 1'b0;
            cnt <= 10'd0;
        end else begin
            done     <= 1'b0;
            save_req <= 1'b0;
            sram_we  <= 1'b0;

            case (state)
                IDLE: if (start) begin
                    state <= REQ;  busy <= 1'b1;  cnt <= 10'd0;
                end

                REQ: begin
                    save_req    <= 1'b1;
                    save_cell_o <= cnt[9:8];
                    save_addr_o <= cnt[7:0];
                    state       <= WAIT;
                end

                WAIT: if (save_valid) begin
                    sram_addr  <= {save_cell_o, save_addr_o};
                    sram_wdata <= save_code;
                    sram_we    <= 1'b1;
                    state      <= WRITE;
                end

                WRITE: begin
                    if (cnt == TOTAL - 1) begin
                        state <= IDLE;  busy <= 1'b0;  done <= 1'b1;
                        cnt   <= 10'd0;
                    end else begin
                        cnt   <= cnt + 10'd1;
                        state <= REQ;
                    end
                end
            endcase
        end
    end

endmodule
