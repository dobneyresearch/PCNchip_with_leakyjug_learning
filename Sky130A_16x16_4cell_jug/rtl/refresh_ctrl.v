`default_nettype none
// Background DRAM-style W-cap refresh from W-SRAM.
//
// Scans all N_CELLS × N_ELEMS addresses sequentially, issuing one SRAM read
// per cycle (when not inhibited).  The SRAM data is registered for one cycle
// then written to cap_array via the load port.
//
// ★★ THE `training_mode` GATE IS GONE. ★★
//
// It existed because the OLD absorb wrote the learned value into the ANALOG W-cap and left
// W-SRAM stale until an explicit `save` — so refreshing from SRAM would have overwritten the
// learning with old codes.
//
// UNDER THE JUG, A FIRE IS A ±1 INCREMENT ON THE W-SRAM CODE ITSELF (jug_ctrl). There is no
// charge pump into Cw. Therefore:
//   * ** W-SRAM IS ALWAYS THE CURRENT MASTER ** — it is never stale.
//   * refresh can run CONTINUOUSLY, during training, with no coherence hazard.
//   * `save` is not needed DURING training (only for checkpoints to NV).
//   * the absorb -> save -> SHADOW_W_SYNC -> refresh-enable coherence cycle COLLAPSES.
//   * ** the W-cap leak risk goes away STRUCTURALLY, not by scheduling. **
//
// `training_mode` is retained as a port for compatibility but is IGNORED here. (Kept rather
// than removed so the top-level wiring does not silently change meaning.)
//
// With 1 pA leakage, 100 fF cap drifts 1 LSB in ~400 ms.
// At 50 MHz, a full 1024-entry scan takes 2048 cycles = 41 µs — more than
// adequate margin.

module refresh_ctrl #(
    parameter N_CELLS = 4,
    parameter N_ELEMS = 256
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        refresh_en,    // WB reg REFRESH_EN[0]
    input  wire        training_mode, // ⚠ IGNORED under the jug (see header). Retained for
                                      //   port compatibility with the emx top-level.
    input  wire        inhibit,       // pause when absorb/inject/save/wfsm busy

    // W-SRAM read port (shared with weight_fsm via top-level mux; lowest priority)
    output reg   [9:0] sram_addr_r,   // registered address for SRAM
    input  wire  [7:0] sram_rdata,    // one-cycle-delayed SRAM output

    // cap_array load port
    output reg         load_en,
    output reg   [1:0] load_cell,
    output reg   [7:0] load_addr,
    output reg   [7:0] load_code
);
    localparam TOTAL = N_CELLS * N_ELEMS;  // 1024

    reg [9:0] scan_ptr;
    reg       read_pending;   // SRAM read issued; rdata arrives next cycle

    // Capture address presented to SRAM so we can tag the load correctly
    reg [9:0] read_addr_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            scan_ptr     <= 10'd0;
            read_pending <= 1'b0;
            load_en      <= 1'b0;
            sram_addr_r  <= 10'd0;
            read_addr_r  <= 10'd0;
        end else begin
            load_en <= 1'b0;

            if (read_pending) begin
                // SRAM data ready — write to cap
                load_en   <= 1'b1;
                load_cell <= read_addr_r[9:8];
                load_addr <= read_addr_r[7:0];
                load_code <= sram_rdata;
                read_pending <= 1'b0;
            // ⚠ NOTE: `training_mode` is deliberately NOT in this condition — see the header.
            end else if (refresh_en && !inhibit) begin
                // Issue next SRAM read
                sram_addr_r  <= scan_ptr;
                read_addr_r  <= scan_ptr;
                read_pending <= 1'b1;
                scan_ptr     <= (scan_ptr == TOTAL-1) ? 10'd0
                                                      : scan_ptr + 10'd1;
            end
        end
    end

endmodule
