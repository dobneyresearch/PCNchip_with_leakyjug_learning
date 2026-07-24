`default_nettype none
`timescale 1ns/1ps
// rtr_adc_bridge.v — per-cell ADC sweep sequencer for router_ctrl.
//
// router_ctrl pulses adc_sweep_req[src] for one cycle per TDM cell slot.
// This module captures the request, sequences N_ROWS SAR ADC conversions
// for that cell, writes the N_ROWS results to the caller's rtr_act_sram
// write port, then asserts adc_sweep_done[src] for one cycle.
//
// Only one cell is swept at a time; router_ctrl issues slots sequentially.
// Concurrent requests are resolved by priority-encoding (lowest index wins);
// in normal TDM operation this case never occurs.

module rtr_adc_bridge #(
    parameter N_CELLS = 4,
    parameter N_ROWS  = 16,
    parameter ACT_W   = 8
) (
    input  wire clk,
    input  wire rst_n,

    // ── Router request/done handshake ─────────────────────────────────────
    input  wire [N_CELLS-1:0]             adc_sweep_req,
    output reg  [N_CELLS-1:0]             adc_sweep_done,

    // ── SAR ADC interface ─────────────────────────────────────────────────
    output reg                             adc_sample,   // level: asserted until adc_done
    input  wire                            adc_done,     // 1-cycle pulse from sar_adc
    input  wire [ACT_W-1:0]               adc_data,

    // ── rtr_act_sram write port ───────────────────────────────────────────
    // waddr = {cell_sel[$clog2(N_CELLS)-1:0], row_idx[$clog2(N_ROWS)-1:0]}
    output reg  [$clog2(N_CELLS)+$clog2(N_ROWS)-1:0] rtr_act_waddr,
    output reg  [ACT_W-1:0]               rtr_act_wdata,
    output reg                             rtr_act_we,

    // ── dac_addr override hints (Gap 1+2) ────────────────────────────────
    // rtr_sweep_active: asserted during SWEEP and WRMEM states so pcn_digital_top
    //   can substitute rtr_row_idx_o into dac_addr[7:4], ensuring cap_eff_code
    //   reads the correct row of the SenderID-selected weight page.
    output wire                            rtr_sweep_active,
    output wire [$clog2(N_ROWS)-1:0]      rtr_row_idx_o
);
    localparam SEL_W  = $clog2(N_CELLS);   // 2
    localparam ROW_W  = $clog2(N_ROWS);    // 4
    localparam ADDR_W = SEL_W + ROW_W;     // 6

    localparam IDLE  = 2'd0;
    localparam SWEEP = 2'd1;
    localparam WRMEM = 2'd2;
    localparam DONE  = 2'd3;

    reg [1:0]       state;
    reg [SEL_W-1:0] cell_sel;
    reg [ROW_W-1:0] row_idx;

    integer c;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= IDLE;
            cell_sel       <= {SEL_W{1'b0}};
            row_idx        <= {ROW_W{1'b0}};
            adc_sample     <= 1'b0;
            adc_sweep_done <= {N_CELLS{1'b0}};
            rtr_act_we     <= 1'b0;
            rtr_act_waddr  <= {ADDR_W{1'b0}};
            rtr_act_wdata  <= {ACT_W{1'b0}};
        end else begin
            adc_sweep_done <= {N_CELLS{1'b0}};
            rtr_act_we     <= 1'b0;
            adc_sample     <= 1'b0;

            case (state)
                IDLE: begin
                    if (|adc_sweep_req) begin
                        // Priority-encode: highest-index requester wins on tie
                        cell_sel <= {SEL_W{1'b0}};
                        for (c = N_CELLS-1; c >= 0; c = c-1)
                            if (adc_sweep_req[c])
                                cell_sel <= c[SEL_W-1:0];
                        row_idx <= {ROW_W{1'b0}};
                        state   <= SWEEP;
                    end
                end

                SWEEP: begin
                    adc_sample <= 1'b1;
                    if (adc_done) begin
                        adc_sample <= 1'b0;
                        state      <= WRMEM;
                    end
                end

                WRMEM: begin
                    rtr_act_we    <= 1'b1;
                    rtr_act_waddr <= {cell_sel, row_idx};
                    rtr_act_wdata <= adc_data;
                    if (row_idx == N_ROWS - 1) begin
                        state <= DONE;
                    end else begin
                        row_idx <= row_idx + 1'b1;
                        state   <= SWEEP;
                    end
                end

                DONE: begin
                    adc_sweep_done[cell_sel] <= 1'b1;
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

    assign rtr_sweep_active = (state == SWEEP) | (state == WRMEM);
    assign rtr_row_idx_o    = row_idx;
endmodule
