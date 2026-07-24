`default_nettype none
`timescale 1ns/1ps
// routing_hebb.v — Hebbian routing-weight update for dynamic topology mode.
//
// When hebb_en_route=1 (topo_mode=1 in router_ctrl), this module tracks which
// source cells had ADC sweeps and which destination cells received dispatched
// activations within each TDM frame.  On irq_frame_done it rewrites the bits
// of routing_matrix that are permitted by loop_mask: a bit is SET when both
// the source and destination cells were co-active in the same frame, CLEARED
// otherwise.  Bits not covered by loop_mask are preserved from routing_matrix_cur.
//
// The update is written back to the WB routing_matrix register via the
// routing_matrix_hebb_i / routing_matrix_hebb_we ports of pcn_wb_regs_4cell_emx,
// making it visible to the host for monitoring and diagnostics.
//
// N_SLOTS = N_CELLS + 1 (includes the SPI slot); loop_mask is N_SLOTS × N_SLOTS.
// Only the lower N_CELLS × N_CELLS bits of loop_mask are used here (cell-to-cell
// Hebbian edges; SPI-slot edges are handled externally).

module routing_hebb #(
    parameter N_CELLS = 4,
    parameter N_SLOTS = N_CELLS + 1   // 5 — matches router_ctrl localparam
) (
    input  wire clk,
    input  wire rst_n,

    // Enable: mirrors hebb_en_route from router_ctrl (= topo_mode)
    input  wire hebb_en_route,
    // Edge learning mask: loop_mask[src*N_SLOTS+dst] permits Hebbian updates
    input  wire [N_SLOTS*N_SLOTS-1:0] loop_mask,

    // Per-slot activity indicators (1-cycle pulses from router domain)
    input  wire [N_CELLS-1:0] adc_sweep_done,   // src cell measured this slot
    input  wire [N_CELLS-1:0] rtr_inp_dac_we,   // dst cell received dispatch

    // End-of-frame pulse from router_ctrl
    input  wire irq_frame_done,

    // Current routing matrix (for preserving non-learnable bits)
    input  wire [N_CELLS*N_CELLS-1:0] routing_matrix_cur,

    // Hebbian update output → pcn_wb_regs_4cell_emx write-back ports
    output reg  [N_CELLS*N_CELLS-1:0] routing_matrix_hebb,
    output reg                         routing_matrix_we
);
    // Per-frame activity accumulators
    reg [N_CELLS-1:0] frame_src;   // OR of adc_sweep_done across this frame
    reg [N_CELLS-1:0] frame_dst;   // OR of rtr_inp_dac_we across this frame

    integer h, k;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            frame_src            <= {N_CELLS{1'b0}};
            frame_dst            <= {N_CELLS{1'b0}};
            routing_matrix_hebb  <= {(N_CELLS*N_CELLS){1'b0}};
            routing_matrix_we    <= 1'b0;
        end else begin
            routing_matrix_we <= 1'b0;

            if (!irq_frame_done) begin
                // Accumulate activity within the current TDM frame
                frame_src <= frame_src | adc_sweep_done;
                frame_dst <= frame_dst | rtr_inp_dac_we;
            end else begin
                // End of frame: optionally update routing_matrix
                if (hebb_en_route) begin
                    for (h = 0; h < N_CELLS; h = h+1) begin
                        for (k = 0; k < N_CELLS; k = k+1) begin
                            if (loop_mask[h*N_SLOTS+k])
                                // Learnable edge: co-activity Hebbian rule
                                // Include current-cycle activity via the |-with-done
                                routing_matrix_hebb[h*N_CELLS+k] <=
                                    (frame_src[h] | adc_sweep_done[h]) &
                                    (frame_dst[k] | rtr_inp_dac_we[k]);
                            else
                                // Non-learnable edge: hold current WB value
                                routing_matrix_hebb[h*N_CELLS+k] <=
                                    routing_matrix_cur[h*N_CELLS+k];
                        end
                    end
                    routing_matrix_we <= 1'b1;
                end
                // Reset accumulators for next frame
                frame_src <= {N_CELLS{1'b0}};
                frame_dst <= {N_CELLS{1'b0}};
            end
        end
    end
endmodule
