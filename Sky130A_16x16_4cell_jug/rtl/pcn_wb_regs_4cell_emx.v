`default_nettype none
// Wishbone register file for the EMX 4-cell chip.
// Extends pcn_wb_regs_4cell.v with EMX-specific registers at 0x74+.
// All parent registers (0x00–0x70) are reproduced verbatim.
//
// New registers (word addresses 6'h1D–6'h24, byte offsets 0x74–0x90):
//   0x74  TRAINING_MODE [0]       — ⚠ IGNORED by refresh_ctrl under the jug (W-SRAM is always
//                                   the master, so refresh never needs disabling). Port retained.
//   0x78  LR_CFG        [8:6]=BH[2:0], [5:0]=LR_SHIFT[5:0]   (9 bits — NOT byte-packed)
//   0x7C  DELTA_IDX     [5:4]=cell[1:0], [3:0]=row[3:0]
//   0x80  DELTA_DATA    [7:0]     — writes delta_buf[cell][row]; auto-pointer
//   0x84  EMX_CTRL      [2]=start_save [1]=start_absorb [0]=start_e_update (pulses)
//   0x88  EMX_STATUS    [2]=save_busy [1]=absorb_busy [0]=e_update_busy (RO)
//   0x8C  REFRESH_EN    [0]       — enable background W-cap refresh
//   0x90  ROUTING_MX    [N_CELLS*N_CELLS-1:0] — per-src destination bitmask (default: funnel)

module pcn_wb_regs_4cell_emx #(
    parameter N_ROWS  = 16,
    parameter N_CELLS = 4,
    parameter VIRT_AW = 3
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [31:0] wb_addr_i,
    input  wire [31:0] wb_dat_i,
    input  wire  [3:0] wb_sel_i,
    input  wire        wb_we_i,
    input  wire        wb_cyc_i,
    input  wire        wb_stb_i,
    output reg  [31:0] wb_dat_o,
    output reg         wb_ack_o,

    // ── Parent: weight-cell interface ────────────────────────────────────
    output reg   [7:0] weight_data,
    output reg  [15:0] cell_addr,
    output reg   [5:0] ctrl,
    output reg  [N_ROWS-1:0] hebb_mask,
    output reg  [15:0] hebb_pw,
    input  wire  [3:0] status,
    output reg   [7:0] sram_wdata,
    output reg         sram_we,
    input  wire  [7:0] sram_rdata,
    output reg         start_load,
    output reg         load_all,
    output reg         rst_weights,
    output reg         start_temporal,
    output reg  [VIRT_AW:0] n_virt_layers,
    output reg  [N_ROWS-1:0] hebb_row_mask,
    input  wire [N_ROWS-1:0] ierr_dig_i,
    output reg         start_adc_sweep,
    output reg   [7:0] inp_dac_wb_data,
    output reg  [15:0] inp_dac_wb_addr,
    output reg         inp_dac_wb_we,
    input  wire  [7:0] act_rdata_wb_i,
    output reg   [7:0] act_wb_wdata,
    output reg         act_wb_we,

    // ── Parent: router registers ──────────────────────────────────────────
    output reg         topo_mode,
    output reg  [24:0] loop_mask,
    output reg  [4:0]  input_cell_mask,
    output reg  [4:0]  output_cell_mask,
    output reg  [15:0] dispatch_pw,
    output reg   [7:0] my_chip_id,
    output reg   [7:0] spi_clk_div,
    output reg         start_frame,
    input  wire        router_busy,
    input  wire        irq_frame_done,
    output reg  [31:0] next_hop_chip_id,
    output reg  [15:0] dest_cell_id_reg,
    output reg  [N_CELLS-1:0] peer_output_mask,
    output reg  [31:0] peer_next_hop_chip_id,
    output reg  [15:0] peer_dest_cell_id_reg,

    // ── EMX: new outputs ─────────────────────────────────────────────────
    output reg         training_mode,    // TRAINING_MODE[0]
    output reg   [2:0] bh,               // LR_CFG[7:5]
    output reg   [5:0] lr_shift,         // LR_CFG[5:0] (6-bit; deep-layer range)
    output reg   [1:0] inject_cell,      // DELTA_IDX[5:4]
    // delta_flat: 4 cells × N_ROWS rows × 8 bits, packed [cell*N_ROWS*8 + row*8 +: 8]
    output reg  [N_CELLS*N_ROWS*8-1:0] delta_flat,
    output reg         start_e_update,   // EMX_CTRL[0] pulse
    output reg         start_absorb,     // EMX_CTRL[1] pulse
    output reg         start_save,       // EMX_CTRL[2] pulse

    // ── ★ THE PRE-DISTORTION LUT (wgt_lut): code -> DAC drive ────────────────
    // weight_dac maps code -> V_w LINEARLY; the CELL maps V_w -> weight SIGMOIDALLY (a 28x
    // sensitivity variation). The LUT pre-distorts so the WEIGHT is uniform — which is what
    // every simulation has always assumed. ⚠ IT IS ONLY TRUE ONCE THE LUT IS LOADED.
    // ★ LOADABLE => this is the PER-DIE CALIBRATION HOOK for the whole weight path: measure
    //   the cell's sigmoid on each die, invert it, write the table.
    // 0x94 WGT_LUT_ADDR [7:0]
    // 0x98 WGT_LUT_DATA [9:0]  -- a WRITE here commits tbl[WGT_LUT_ADDR] and auto-increments
    output reg  [7:0]  wgt_lut_addr,
    output reg  [9:0]  wgt_lut_wdata,
    output reg         wgt_lut_we,

    // ── ★ THETA — the JUG FIRE THRESHOLD (a per-layer CONFIG constant) ───────
    // 0x9C JUG_THETA [7:0], in Ce-LSBs. The boss writes it at load time, exactly like the ADC
    // gain and the routing. A chip never discovers it at runtime.
    // Sim optimum: theta = 8x the per-fold rms of |E| (a clean inverted-U:
    //   4 -> 81.08%, 8 -> 81.96%, 16 -> 80.37%, 32 -> 73.95%).
    // ⚠⚠ DO NOT SERVO IT. The firing rate FALLS on its own as the network converges
    //   (2.69% -> 1.85%/fold) — an lr schedule emerging from the physics. A loop holding the
    //   rate constant by adjusting theta would CANCEL that annealing.
    output reg  [7:0]  jug_theta,
    output reg         refresh_en,       // REFRESH_EN[0]
    output reg  [N_CELLS*N_CELLS-1:0] routing_matrix, // ROUTING_MX — dest bitmask per src cell

    // ── ★ TRANSPOSE-AT-SOURCE (the backprojection δ round-trip) ──────────────
    // The router drives this like router_node's SRC→TRIG→DST, but the transpose runs ON THIS CHIP
    // from its OWN W-SRAM (pcn_transpose) — no shadow. δ-in reuses DELTA_IDX/DATA (it also drives
    // E-inject). See ../INTERCONNECT_PROTOCOL.md.
    //   BP_TRIG (0xA0) write: [0]=start_transpose (pulse), [5:4]=transpose_cell; resets DST ptr.
    //                   read: [0]=transpose_busy.
    //   BP_DST  (0xA4) read : partial byte at an auto-incrementing pointer (16 int8 = one δ block).
    output reg         start_transpose,  // BP_TRIG[0] pulse
    output reg   [1:0] transpose_cell,   // BP_TRIG[5:4]

    // ── EMX: status inputs ────────────────────────────────────────────────
    input  wire        e_update_busy,
    input  wire        absorb_busy,
    input  wire        save_busy,
    input  wire  [N_ROWS*8-1:0] transpose_partial,  // renormed Wᵀδ from pcn_transpose (16 int8)
    input  wire        transpose_busy,
    // ── Hebbian routing write-back (from routing_hebb) ────────────────────
    // When routing_matrix_hebb_we=1, routing_matrix is updated with the
    // Hebbian co-activity result.  A simultaneous WB host write takes priority.
    input  wire [N_CELLS*N_CELLS-1:0] routing_matrix_hebb_i,
    input  wire                        routing_matrix_hebb_we
);
    wire sel  = wb_cyc_i & wb_stb_i;
    wire [5:0] addr = wb_addr_i[7:2];

    // Internal DELTA_IDX pointer (cell + row within cell)
    reg [1:0] di_cell;
    reg [3:0] di_row;

    reg frame_done_latch;
    reg [3:0] bp_dptr;                    // BP_DST read pointer (0..15), resets on BP_TRIG

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // ── Parent defaults ───────────────────────────────────────────
            weight_data    <= 8'h80;  cell_addr <= 16'h0;  ctrl <= 6'h0;
            hebb_mask      <= {N_ROWS{1'b1}};  hebb_pw <= 16'd2500;
            wb_ack_o       <= 1'b0;   wb_dat_o <= 32'h0;
            start_load     <= 1'b0;   load_all  <= 1'b0;  rst_weights <= 1'b0;
            start_temporal <= 1'b0;   n_virt_layers <= {(VIRT_AW+1){1'b0}};
            sram_we        <= 1'b0;   sram_wdata <= 8'h0;
            hebb_row_mask  <= {N_ROWS{1'b1}};
            start_adc_sweep   <= 1'b0;
            inp_dac_wb_we     <= 1'b0;  inp_dac_wb_data <= 8'h80;
            inp_dac_wb_addr   <= 16'h0;
            act_wb_we         <= 1'b0;  act_wb_wdata <= 8'h0;
            next_hop_chip_id      <= 32'hFF_FF_FF_FF;
            dest_cell_id_reg      <= 16'h0000;
            peer_output_mask      <= {N_CELLS{1'b0}};
            peer_next_hop_chip_id <= 32'hFF_FF_FF_FF;
            peer_dest_cell_id_reg <= 16'h0000;
            topo_mode        <= 1'b0;
            loop_mask        <= 25'h1E79E;
            input_cell_mask  <= 5'b0_0001;
            output_cell_mask <= 5'b0_1000;
            dispatch_pw      <= 16'd2500;
            my_chip_id       <= 8'h00;
            spi_clk_div      <= 8'd3;
            start_frame      <= 1'b0;
            frame_done_latch <= 1'b0;
            // ── EMX defaults ─────────────────────────────────────────────
            training_mode  <= 1'b0;
            bh             <= 3'd3;     // neutral boss_h=3
            lr_shift       <= 6'd22;    // ΔE = product / 2^22
            inject_cell    <= 2'd0;
            di_cell        <= 2'd0;
            di_row         <= 4'd0;
            delta_flat     <= {(N_CELLS*N_ROWS*8){1'b0}};
            start_e_update <= 1'b0;
            start_absorb   <= 1'b0;
            start_save     <= 1'b0;
            wgt_lut_we     <= 1'b0;   // ★ was: an auto-increment block MISPLACED in reset (bug)
            wgt_lut_addr   <= 8'd0;   //   → wgt_lut_we stuck high & addr never advanced in normal op
            refresh_en     <= 1'b0;
            // Funnel topology default: cells 0,1,2 each send to cell 3; cell 3 silent
            routing_matrix <= 16'b0000_1000_1000_1000;
            jug_theta       <= 8'd50;   // default θ = 50 (mV) = the cap_array THETA=0.050 default
            start_transpose <= 1'b0;  transpose_cell <= 2'd0;  bp_dptr <= 4'd0;
        end else begin
            // Default pulse deasserts
            start_load      <= 1'b0;  load_all       <= 1'b0;  rst_weights <= 1'b0;
            start_temporal  <= 1'b0;  start_adc_sweep <= 1'b0;
            sram_we         <= 1'b0;  wb_ack_o       <= 1'b0;
            inp_dac_wb_we   <= 1'b0;  act_wb_we      <= 1'b0;
            start_frame     <= 1'b0;
            start_e_update  <= 1'b0;
            start_absorb    <= 1'b0;
            start_save      <= 1'b0;
            start_transpose <= 1'b0;
            // ★ THE LUT WRITE IS A ONE-CYCLE PULSE; the address auto-increments so the boss can
            //   stream all 256 entries without rewriting WGT_LUT_ADDR. (Moved here from the reset
            //   branch, where it never ran during operation — wgt_lut_we stuck high, addr frozen.)
            if (wgt_lut_we) begin
                wgt_lut_we   <= 1'b0;
                wgt_lut_addr <= wgt_lut_addr + 8'd1;
            end

            if (irq_frame_done) frame_done_latch <= 1'b1;

            // Hebbian update (host WB write wins if both arrive same cycle)
            if (routing_matrix_hebb_we) routing_matrix <= routing_matrix_hebb_i;

            if (sel) begin
                wb_ack_o <= 1'b1;
                if (wb_we_i) begin
                    case (addr)
                        // ── Parent write (0x00–0x70) ──────────────────────
                        6'h00: weight_data  <= wb_dat_i[7:0];
                        6'h01: cell_addr    <= wb_dat_i[15:0];
                        6'h02: begin
                            ctrl <= wb_dat_i[5:0];
                            if (wb_dat_i[0]) start_load      <= 1'b1;
                            if (wb_dat_i[1]) load_all        <= 1'b1;
                            if (wb_dat_i[4]) rst_weights     <= 1'b1;
                            if (wb_dat_i[5]) start_temporal  <= 1'b1;
                            if (wb_dat_i[6]) start_adc_sweep <= 1'b1;
                        end
                        6'h04: hebb_mask     <= wb_dat_i[N_ROWS-1:0];
                        6'h05: hebb_pw       <= wb_dat_i[15:0];
                        6'h06: begin sram_wdata <= wb_dat_i[7:0]; sram_we <= 1'b1; end
                        6'h08: n_virt_layers <= wb_dat_i[VIRT_AW:0];
                        6'h09: hebb_row_mask <= wb_dat_i[N_ROWS-1:0];
                        6'h0B: begin
                            inp_dac_wb_data <= wb_dat_i[7:0];
                            inp_dac_wb_addr <= cell_addr;
                            inp_dac_wb_we   <= 1'b1;
                        end
                        6'h0C: begin act_wb_wdata <= wb_dat_i[7:0]; act_wb_we <= 1'b1; end
                        6'h0D: topo_mode        <= wb_dat_i[0];
                        6'h0E: loop_mask        <= wb_dat_i[24:0];
                        6'h0F: input_cell_mask  <= wb_dat_i[4:0];
                        6'h10: output_cell_mask <= wb_dat_i[4:0];
                        6'h11: dispatch_pw      <= wb_dat_i[15:0];
                        6'h12: my_chip_id       <= wb_dat_i[7:0];
                        6'h13: spi_clk_div      <= wb_dat_i[7:0];
                        6'h15: if (wb_dat_i[0]) start_frame <= 1'b1;
                        6'h16: begin
                            next_hop_chip_id[7:0]  <= wb_dat_i[7:0];
                            next_hop_chip_id[15:8] <= wb_dat_i[15:8];
                        end
                        6'h17: begin
                            next_hop_chip_id[23:16] <= wb_dat_i[7:0];
                            next_hop_chip_id[31:24] <= wb_dat_i[15:8];
                        end
                        6'h18: dest_cell_id_reg <= wb_dat_i[15:0];
                        6'h19: peer_output_mask <= wb_dat_i[N_CELLS-1:0];
                        6'h1A: begin
                            peer_next_hop_chip_id[7:0]  <= wb_dat_i[7:0];
                            peer_next_hop_chip_id[15:8] <= wb_dat_i[15:8];
                        end
                        6'h1B: begin
                            peer_next_hop_chip_id[23:16] <= wb_dat_i[7:0];
                            peer_next_hop_chip_id[31:24] <= wb_dat_i[15:8];
                        end
                        6'h1C: peer_dest_cell_id_reg <= wb_dat_i[15:0];

                        // ── EMX write (0x74–0x8C) ─────────────────────────
                        6'h1D: training_mode <= wb_dat_i[0];
                        6'h1E: begin
                            // ⚠ BH sits ABOVE the 6-bit LR_SHIFT, not at [7:5]: lr_shift was
                            // widened 5→6 bits and bit 5 would otherwise alias into both fields
                            // (you could not set LR_SHIFT[5] without corrupting BH[0]). This
                            // packing matches the readback {23'h0, bh, lr_shift} exactly.
                            bh       <= wb_dat_i[8:6];
                            lr_shift <= wb_dat_i[5:0];
                        end
                        6'h1F: begin
                            di_cell     <= wb_dat_i[5:4];
                            di_row      <= wb_dat_i[3:0];
                            inject_cell <= wb_dat_i[5:4];
                        end
                        6'h20: begin
                            // Write signed delta into staging buffer at [di_cell][di_row]
                            delta_flat[(di_cell * N_ROWS + di_row) * 8 +: 8] <= wb_dat_i[7:0];
                        end
                        6'h21: begin
                            if (wb_dat_i[0]) start_e_update <= 1'b1;
                            if (wb_dat_i[1]) start_absorb   <= 1'b1;   // = start a JUG SWEEP
                            if (wb_dat_i[2]) start_save      <= 1'b1;
                        end
                        // ── the pre-distortion LUT ──────────────────────────
                        6'h25: wgt_lut_addr <= wb_dat_i[7:0];          // WGT_LUT_ADDR
                        6'h26: begin                                    // WGT_LUT_DATA (commits)
                            wgt_lut_wdata <= wb_dat_i[9:0];
                            wgt_lut_we    <= 1'b1;
                        end
                        6'h27: jug_theta   <= wb_dat_i[7:0];           // JUG_THETA
                        // 6'h22: EMX_STATUS read-only
                        6'h23: refresh_en     <= wb_dat_i[0];
                        6'h24: routing_matrix <= wb_dat_i[N_CELLS*N_CELLS-1:0];
                        // ── ★ transpose-at-source: BP_TRIG (0xA0) ────────────
                        6'h28: begin
                            if (wb_dat_i[0]) start_transpose <= 1'b1;   // TRIG a Wᵀδ on this cell
                            transpose_cell <= wb_dat_i[5:4];
                            bp_dptr        <= 4'd0;                      // reset the DST read ptr
                        end
                        default: ;
                    endcase
                end else begin
                    case (addr)
                        // ── Parent read ──────────────────────────────────
                        6'h00: wb_dat_o <= {24'h0, weight_data};
                        6'h01: wb_dat_o <= {16'h0, cell_addr};
                        6'h02: wb_dat_o <= {26'h0, ctrl};
                        6'h03: wb_dat_o <= {28'h0, status};
                        6'h04: wb_dat_o <= {{(32-N_ROWS){1'b0}}, hebb_mask};
                        6'h05: wb_dat_o <= {16'h0, hebb_pw};
                        6'h06: wb_dat_o <= {24'h0, sram_rdata};
                        6'h08: wb_dat_o <= {{(31-VIRT_AW){1'b0}}, n_virt_layers};
                        6'h09: wb_dat_o <= {{(32-N_ROWS){1'b0}}, hebb_row_mask};
                        6'h0A: wb_dat_o <= {{(32-N_ROWS){1'b0}}, ierr_dig_i};
                        6'h0C: wb_dat_o <= {24'h0, act_rdata_wb_i};
                        6'h0D: wb_dat_o <= {31'h0, topo_mode};
                        6'h0E: wb_dat_o <= {7'h0, loop_mask};
                        6'h0F: wb_dat_o <= {27'h0, input_cell_mask};
                        6'h10: wb_dat_o <= {27'h0, output_cell_mask};
                        6'h11: wb_dat_o <= {16'h0, dispatch_pw};
                        6'h12: wb_dat_o <= {24'h0, my_chip_id};
                        6'h13: wb_dat_o <= {24'h0, spi_clk_div};
                        6'h14: begin
                            wb_dat_o <= {30'h0, frame_done_latch, router_busy};
                            frame_done_latch <= 1'b0;
                        end
                        6'h16: wb_dat_o <= {16'h0, next_hop_chip_id[15:8], next_hop_chip_id[7:0]};
                        6'h17: wb_dat_o <= {16'h0, next_hop_chip_id[31:24], next_hop_chip_id[23:16]};
                        6'h18: wb_dat_o <= {16'h0, dest_cell_id_reg};
                        6'h19: wb_dat_o <= {28'h0, peer_output_mask};
                        6'h1A: wb_dat_o <= {16'h0, peer_next_hop_chip_id[15:8],
                                                     peer_next_hop_chip_id[7:0]};
                        6'h1B: wb_dat_o <= {16'h0, peer_next_hop_chip_id[31:24],
                                                     peer_next_hop_chip_id[23:16]};
                        6'h1C: wb_dat_o <= {16'h0, peer_dest_cell_id_reg};
                        // ── EMX read ──────────────────────────────────────
                        6'h1D: wb_dat_o <= {31'h0, training_mode};
                        6'h1E: wb_dat_o <= {23'h0, bh, lr_shift};
                        6'h1F: wb_dat_o <= {26'h0, di_cell, di_row};
                        6'h20: wb_dat_o <= {24'h0,
                                            delta_flat[(di_cell*N_ROWS+di_row)*8 +: 8]};
                        6'h21: wb_dat_o <= 32'h0;    // EMX_CTRL write-only
                        6'h22: wb_dat_o <= {29'h0, save_busy, absorb_busy, e_update_busy};
                        6'h23: wb_dat_o <= {31'h0, refresh_en};
                        6'h24: wb_dat_o <= {{(32-N_CELLS*N_CELLS){1'b0}}, routing_matrix};
                        // ── ★ transpose-at-source read-back ──────────────────
                        6'h28: wb_dat_o <= {31'h0, transpose_busy};    // BP_TRIG read = busy
                        6'h29: begin                                    // BP_DST: partial byte, auto-inc
                            wb_dat_o <= {24'h0, transpose_partial[bp_dptr*8 +: 8]};
                            bp_dptr  <= bp_dptr + 4'd1;
                        end
                        default: wb_dat_o <= 32'h0;
                    endcase
                end
            end
        end
    end
endmodule
