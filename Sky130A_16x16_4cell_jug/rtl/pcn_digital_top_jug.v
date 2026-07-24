`default_nettype none
// EMX 4-cell PCN chip top-level.
//
// Architecture: dual-cap per weight (W-cap + E-cap in analog domain).
// Digital logic controls cap access; analog MAC happens off-chip boundary.
//
// W-SRAM (1024 × 8-bit) is the persistence layer.  W-caps are the live weights.
// E-caps accumulate per-sample corrections; absorption transfers E → W, zeros E.
//
// DAC modes (EMX_MODE register):
//   0 = PARALLEL: dac_data comes from weight_fsm/SRAM (host test/load path)
//   1 = ANALOG:   dac_data comes from cap_array eff_code (inference path)
//
// inp_act_latch captures inp_dac writes for e_inject_ctrl's x_u8 values.

module pcn_digital_top_jug #(
    parameter N_ROWS   = 16,
    parameter N_COLS   = 16,
    parameter N_CELLS  = 4,
    parameter WGT_MIN  = 71,
    parameter WGT_MAX  = 192
) (
    input  wire        clk,
    input  wire        rst_n,
    // Wishbone
    input  wire [31:0] wb_addr_i,
    input  wire [31:0] wb_dat_i,
    input  wire  [3:0] wb_sel_i,
    input  wire        wb_we_i,
    input  wire        wb_cyc_i,
    input  wire        wb_stb_i,
    output wire [31:0] wb_dat_o,
    output wire        wb_ack_o,
    output wire  [2:0] user_irq,
    output wire [31:0] la_data_out,
    // Analog boundary: weight DAC
    output wire [15:0] dac_addr,
    output wire  [7:0] dac_data,  // muxed: cap eff_code or SRAM code
    output wire        dac_we,
    input  wire [N_ROWS-1:0] ierr,
    output wire [N_ROWS-1:0] we_out,
    output wire        full_power,
    output wire        keep_alive,
    // Analog boundary: SAR ADC
    output wire  [7:0] adc_dac_out,
    input  wire        adc_cmp,
    // Analog boundary: activation input DAC
    output wire [15:0] inp_dac_addr,
    output wire  [7:0] inp_dac_data,
    output wire        inp_dac_we,
    // Router absorb signal (from router_chip_emx)
    input  wire        absorb_n,   // active-low: assert to trigger absorption
    // ── Router SPI (this chip = SPI master to router) ────────────────────────
    output wire        spi_clk_o,
    output wire        spi_mosi,
    input  wire        spi_miso,
    output wire        spi_cs_n,
    // ── Peer SPI master (this chip → peer slave) ─────────────────────────────
    output wire        peer_cs_n_o,
    output wire        peer_clk_o,
    output wire        peer_mosi_o,
    input  wire        peer_miso_i,
    // ── Peer SPI slave (peer master → this chip) ─────────────────────────────
    input  wire        peer_cs_n_i,
    input  wire        peer_clk_i,
    input  wire        peer_mosi_i,
    output wire        peer_miso_o
);
    // ── Local parameters ─────────────────────────────────────────────────────
    localparam N_ELEMS  = N_ROWS * N_COLS;   // 256 per cell
    localparam SRAM_D   = N_CELLS * N_ELEMS; // 1024 total SRAM entries
    localparam VIRT_AW  = 3;
    localparam CELL_AW  = $clog2(N_CELLS);   // 2

    // ⚠ pcn_transpose / router_backproj transpose a SQUARE N×N block, so the cell must be
    // square. A rectangular cell would need separate row/col dims through the whole δ path.
    // Fail loudly at elaboration rather than produce wrong silicon.
    initial begin
        if (N_ROWS != N_COLS) begin
            $display("FATAL pcn_digital_top_jug: transpose requires a SQUARE cell, got %0dx%0d",
                     N_ROWS, N_COLS);
            $fatal;
        end
    end

    // ── WB register wires ────────────────────────────────────────────────────
    wire  [7:0] weight_data;
    wire [15:0] cell_addr;
    wire  [5:0] ctrl;
    wire [N_ROWS-1:0] hebb_mask, hebb_row_mask;
    wire [15:0] hebb_pw;
    wire  [3:0] status_in;
    wire  [7:0] sram_wdata_wb;
    wire        sram_we_wb;
    wire  [7:0] sram_rdata;
    wire        start_load, load_all, rst_weights, start_temporal, start_adc_sweep;
    wire [VIRT_AW:0] n_virt_layers;
    wire [N_ROWS-1:0] ierr_dig;
    wire  [7:0] inp_dac_wb_data;
    wire [15:0] inp_dac_wb_addr;
    wire        inp_dac_wb_we;
    wire  [7:0] act_rdata_wb;
    wire  [7:0] act_wb_wdata;
    wire        act_wb_we;
    // router wires (to/from WB regs → router_ctrl)
    wire        topo_mode, start_frame;
    wire [24:0] loop_mask;
    wire  [4:0] input_cell_mask, output_cell_mask;
    wire [15:0] dispatch_pw;
    wire  [7:0] my_chip_id, spi_clk_div;
    wire [31:0] next_hop_chip_id, peer_next_hop_chip_id;
    wire [15:0] dest_cell_id_reg, peer_dest_cell_id_reg;
    wire [N_CELLS-1:0] peer_output_mask;
    wire [N_CELLS*N_CELLS-1:0] routing_matrix;
    // router_ctrl status (replaces tied-off stubs)
    wire        router_ctrl_busy, router_ctrl_irq;
    // router_ctrl ADC sweep handshake
    wire [N_CELLS-1:0] adc_sweep_req_w, adc_sweep_done_w;
    // router_ctrl act_sram interface (router reads N_ROWS acts per cell)
    localparam RTR_CELL_AW = $clog2(N_ROWS);  // was hardcoded 4 — wrong bus width at N_ROWS≠16
    wire [N_CELLS*RTR_CELL_AW-1:0] rtr_act_raddr_w;
    reg  [7:0] rtr_act_rdat [0:N_CELLS-1];  // per-cell registered read data
    // router_ctrl inp_dac dispatch
    wire [N_CELLS*RTR_CELL_AW-1:0] rtr_inp_dac_addr_w;
    wire [N_CELLS*8-1:0]            rtr_inp_dac_data_w;
    wire [N_CELLS-1:0]              rtr_inp_dac_we_w;
    // router_ctrl sender_id (page selection hint for analog domain)
    wire [N_CELLS*2-1:0]  sender_id_w;
    wire [N_CELLS-1:0]    sender_id_valid_w;
    wire                   hebb_en_route_w;
    // router_ctrl HELLO result and received-packet signals (unused beyond routing)
    wire [7:0]  hello_chip_id_w;
    wire        hello_done_w;
    wire        spi_pkt_valid_w;
    wire [3:0]  spi_dest_cell_w, spi_src_cell_w;
    wire [N_ROWS*8-1:0] spi_act_data_w;
    wire        peer_pkt_valid_w;
    wire [3:0]  peer_dest_cell_w, peer_src_cell_w;
    wire [N_ROWS*8-1:0] peer_act_data_w;
    wire [2:0]  router_current_slot_w;
    // rtr_adc_bridge → rtr_act_sram
    wire [$clog2(N_CELLS)+RTR_CELL_AW-1:0] rtr_act_waddr_w;
    wire [7:0]  rtr_act_wdata_w;
    wire        rtr_act_we_w;
    wire        adc_sample_rtr;
    // Gap 1+2: rtr_adc_bridge sweep hints for dac_addr override
    wire        rtr_sweep_active_w;
    wire [RTR_CELL_AW-1:0] rtr_row_addr_w;
    // Gap 1: SenderID — use router's current slot as weight-page select
    wire        rtr_slot_active  = router_ctrl_busy && !router_current_slot_w[2];
    wire [1:0]  rtr_page_sel     = router_current_slot_w[1:0];
    // Unified dac_addr override: page bits from SenderID, row bits from bridge sweep
    wire [15:0] dac_addr_mux;
    assign dac_addr_mux = rtr_slot_active ?
        {dac_addr_fsm[15:10], rtr_page_sel,
         rtr_sweep_active_w ? {rtr_row_addr_w, 4'b0} : dac_addr_fsm[7:0]} :
        dac_addr_fsm;
    // Gap 3: routing_hebb Hebbian update output
    wire [N_CELLS*N_CELLS-1:0] routing_matrix_hebb_w;
    wire                        routing_matrix_hebb_we_w;
    // EMX control wires
    wire        training_mode;
    wire  [2:0] bh;
    wire  [5:0] lr_shift;   // 6-bit (deep-layer range)
    wire  [1:0] inject_cell_wb;
    wire [N_CELLS*N_ROWS*8-1:0] delta_flat;
    wire        start_e_update_wb, start_absorb_wb, start_save_wb;
    wire        refresh_en;

    assign ierr_dig = ierr;

    // ── Busy and done signals ─────────────────────────────────────────────────
    wire wfsm_busy, wfsm_ready, irq_load_done, irq_temporal_done;
    wire e_update_busy, e_update_done;
    wire absorb_busy,   absorb_done;
    wire save_busy,     save_done;
    wire hebb_actv, irq_hebb_ovf;
    wire sleep_ack, irq_sleep_ack;

    assign status_in = {sleep_ack, hebb_actv, wfsm_busy, wfsm_ready};

    // ── Wishbone register file ────────────────────────────────────────────────
    pcn_wb_regs_4cell_emx #(
        .N_ROWS(N_ROWS), .N_CELLS(N_CELLS), .VIRT_AW(VIRT_AW)
    ) u_regs (
        .clk              (clk),            .rst_n          (rst_n),
        .wb_addr_i        (wb_addr_i),      .wb_dat_i       (wb_dat_i),
        .wb_sel_i         (wb_sel_i),       .wb_we_i        (wb_we_i),
        .wb_cyc_i         (wb_cyc_i),       .wb_stb_i       (wb_stb_i),
        .wb_dat_o         (wb_dat_o),       .wb_ack_o       (wb_ack_o),
        .weight_data      (weight_data),    .cell_addr      (cell_addr),
        .ctrl             (ctrl),           .hebb_mask      (hebb_mask),
        .hebb_pw          (hebb_pw),        .status         (status_in),
        .sram_wdata       (sram_wdata_wb),  .sram_we        (sram_we_wb),
        .sram_rdata       (sram_rdata),     .start_load     (start_load),
        .load_all         (load_all),       .rst_weights    (rst_weights),
        .start_temporal   (start_temporal), .n_virt_layers  (n_virt_layers),
        .hebb_row_mask    (hebb_row_mask),  .ierr_dig_i     (ierr_dig),
        .start_adc_sweep  (start_adc_sweep),
        .inp_dac_wb_data  (inp_dac_wb_data),.inp_dac_wb_addr(inp_dac_wb_addr),
        .inp_dac_wb_we    (inp_dac_wb_we),  .act_rdata_wb_i (act_rdata_wb),
        .act_wb_wdata     (act_wb_wdata),   .act_wb_we      (act_wb_we),
        .topo_mode        (topo_mode),      .loop_mask      (loop_mask),
        .input_cell_mask  (input_cell_mask),.output_cell_mask(output_cell_mask),
        .dispatch_pw      (dispatch_pw),    .my_chip_id     (my_chip_id),
        .spi_clk_div      (spi_clk_div),    .start_frame    (start_frame),
        .router_busy      (router_ctrl_busy),.irq_frame_done (router_ctrl_irq),
        .next_hop_chip_id (next_hop_chip_id),.dest_cell_id_reg(dest_cell_id_reg),
        .peer_output_mask (peer_output_mask),
        .peer_next_hop_chip_id(peer_next_hop_chip_id),
        .peer_dest_cell_id_reg(peer_dest_cell_id_reg),
        .routing_matrix   (routing_matrix),
        .routing_matrix_hebb_i (routing_matrix_hebb_w),
        .routing_matrix_hebb_we(routing_matrix_hebb_we_w),
        .training_mode    (training_mode),  .bh             (bh),
        .lr_shift         (lr_shift),       .inject_cell    (inject_cell_wb),
        .delta_flat       (delta_flat),
        .start_e_update   (start_e_update_wb),
        .start_absorb     (start_absorb_wb),.start_save     (start_save_wb),
        .refresh_en       (refresh_en),
        // ★ FINISH-THE-TOP 2026-07-15: these EMX config outputs were declared-but-undriven
        //   (floating), so the LUT couldn't be WB-loaded and θ wasn't runtime-wired.
        .wgt_lut_addr     (wgt_lut_addr),   .wgt_lut_wdata  (wgt_lut_wdata),
        .wgt_lut_we       (wgt_lut_we),     .jug_theta      (jug_theta),
        .start_transpose  (start_transpose_wb), .transpose_cell(transpose_cell_wb),
        .transpose_partial(transpose_partial),  .transpose_busy(transpose_busy),
        .e_update_busy    (e_update_busy),  .absorb_busy    (absorb_busy),
        .save_busy        (save_busy)
    );

    // ── W-SRAM (1024 × 8-bit; sky130_sram_1kbyte for synthesis) ─────────────
    // SRAM address mux priority: JUG(RMW) > save_ctrl > weight_fsm > refresh_ctrl > WB
    // ⚠ THE JUG HAS A WRITE PORT NOW. A fire is a ±1 code increment on W-SRAM, so the
    //   jug must win the address mux while it is doing its read-modify-write.
    wire [9:0] sram_addr_fsm;
    wire [9:0] sram_addr_ref;   // from refresh_ctrl
    wire [9:0] sram_addr_save;
    wire [7:0] sram_wdata_save;
    wire       sram_we_save;

    // ── transpose-at-source read port (driven by u_transpose, below) ────────
    wire [9:0]  t_sram_addr;
    wire        t_sram_re;
    wire        transpose_busy, transpose_done;
    wire [N_ROWS*8-1:0] transpose_partial;
    wire        start_transpose_wb;
    wire  [1:0] transpose_cell_wb;

    wire inhibit_refresh = wfsm_busy | absorb_busy | save_busy | e_update_busy | transpose_busy;

    // ── SINGLE-PORT W-SRAM ARBITER ──────────────────────────────────────────
    // Priority: JUG(RMW) > save > weight_fsm > transpose(read) > refresh; WB writes when idle.
    // ⚠ FIXED 2026-07-15: the JUG's RMW port (jug_sram_addr/we/wdata) was declared and connected
    //   to jug_ctrl but NEVER routed into this mux — so in the assembled top, fires never reached
    //   the W-SRAM. Added here alongside the new transpose read port.
    wire [9:0] sram_addr_mux =
        absorb_busy    ? jug_sram_addr  :   // ★ jug read-modify-write (was missing)
        save_busy      ? sram_addr_save :
        wfsm_busy      ? sram_addr_fsm  :
        transpose_busy ? t_sram_addr    :   // ★ transpose read burst (256 codes/cell)
                         sram_addr_ref;     // refresh when idle

    wire       sram_we_mux =
        absorb_busy ? jug_sram_we   :       // ★ jug write-back of the ±1 code
        save_busy   ? sram_we_save  :
                      sram_we_wb;           // host WB write when neither jug nor save is active
    wire [7:0] sram_wdata_mux =
        absorb_busy ? jug_sram_wdata :
        save_busy   ? sram_wdata_save :
                      sram_wdata_wb;

    sram_if #(.N_CELLS(SRAM_D)) u_sram (
        .clk   (clk),            .rst_n (rst_n),
        .addr  (sram_addr_mux),  .wdata (sram_wdata_mux),
        .we    (sram_we_mux),    .rdata (sram_rdata)
    );

    // ── Dual-cap array (W-cap + E-cap per weight) ────────────────────────────
    // Load port: weight_fsm dac_we during load → also charges W-cap
    wire [15:0] dac_addr_fsm;
    wire  [7:0] dac_data_fsm;
    wire        dac_we_fsm;
    wire  [7:0] cap_eff_code;  // combinational eff_code at current dac_addr

    // absorb_ctrl port
    wire        absorb_en_cap;
    wire  [1:0] absorb_cell_cap;
    wire  [7:0] absorb_addr_cap;

    // inject port
    wire        inject_en_cap;
    wire  [1:0] inject_cell_cap;
    wire  [7:0] inject_addr_cap;
    wire  [7:0] inject_delta_cap;
    wire  [7:0] inject_x_cap;

    // save port
    wire        save_req_cap;
    wire  [1:0] save_cell_cap;
    wire  [7:0] save_addr_cap;
    wire  [7:0] save_code_cap;
    wire        save_valid_cap;

    // refresh / WB load port (shared: refresh_ctrl has priority when running)
    wire        load_en_ref;
    wire  [1:0] load_cell_ref;
    wire  [7:0] load_addr_ref;
    wire  [7:0] load_code_ref;

    // load_en during weight_fsm dac_we: charges W-cap from dac_data code
    wire load_en_fsm  = dac_we_fsm & ~training_mode;  // only in load (non-training) mode
    wire [1:0] load_cell_fsm = dac_addr_fsm[9:8];
    wire [7:0] load_addr_fsm_8 = dac_addr_fsm[7:0];

    // Mux: refresh_ctrl or weight_fsm drives cap_array load port
    wire load_en_mux    = load_en_ref  | load_en_fsm;
    wire [1:0] lc_mux   = load_en_ref ? load_cell_ref : load_cell_fsm;
    wire [7:0] la_mux   = load_en_ref ? load_addr_ref : load_addr_fsm_8;
    wire [7:0] lcode_mux= load_en_ref ? load_code_ref : dac_data_fsm;

    cap_array #(
        .N_CELLS(N_CELLS), .N_ELEMS(N_ELEMS),
        .WGT_MIN(WGT_MIN), .WGT_MAX(WGT_MAX)
    ) u_caps (
        .clk          (clk),            .rst_n        (rst_n),
        .jug_theta    (jug_theta),      // ★ runtime θ (JUG_THETA reg) — was unwired
        .load_en      (load_en_mux),    .load_cell    (lc_mux),
        .load_addr    (la_mux),         .load_code    (lcode_mux),
        .inject_en    (inject_en_cap),  .inject_cell  (inject_cell_cap),
        .inject_addr  (inject_addr_cap),.inject_delta (inject_delta_cap),
        .inject_x     (inject_x_cap),   .inject_bh    (bh),
        .lr_shift     (lr_shift),
        .cmp_en       (cmp_en_cap),     .cmp_cell     (cmp_cell_cap),
        .cmp_addr     (cmp_addr_cap),
        .fire_up      (fire_up),        .fire_dn      (fire_dn),
        .jug_pulse    (jug_pulse),      .jug_dn       (jug_dn),
        .jug_cell     (jug_cell_r),     .jug_addr     (jug_addr_r),
        .save_req     (save_req_cap),   .save_cell    (save_cell_cap),
        .save_addr    (save_addr_cap),  .save_code    (save_code_cap),
        .save_valid   (save_valid_cap),
        .mac_cell     (dac_addr_mux[9:8]), .mac_addr  (dac_addr_mux[7:0]),
        .mac_eff_code (cap_eff_code)
    );

    // ── inp_act_latch: captures x_u8 values from inp_dac writes ─────────────
    reg [N_COLS*8-1:0] inp_act_latch;
    wire [15:0] inp_dac_addr_fsm;
    wire  [7:0] inp_dac_data_fsm;
    wire        inp_dac_we_fsm;

    // Router dispatch mux: pick active destination cell's inp_dac
    reg [15:0] rtr_dac_addr_r;
    reg  [7:0] rtr_dac_data_r;
    reg        rtr_dac_we_r;
    always @(*) begin : rtr_dac_sel
        integer rd;
        rtr_dac_addr_r = 16'h0;
        rtr_dac_data_r = 8'h0;
        rtr_dac_we_r   = 1'b0;
        for (rd = 0; rd < N_CELLS; rd = rd+1) begin
            if (rtr_inp_dac_we_w[rd]) begin
                rtr_dac_addr_r = {{(16-RTR_CELL_AW){1'b0}},
                                   rtr_inp_dac_addr_w[rd*RTR_CELL_AW+:RTR_CELL_AW]};
                rtr_dac_data_r = rtr_inp_dac_data_w[rd*8+:8];
                rtr_dac_we_r   = 1'b1;
            end
        end
    end

    // inp_dac priority: weight_fsm > router dispatch > host WB
    assign inp_dac_addr = wfsm_busy    ? inp_dac_addr_fsm :
                          rtr_dac_we_r ? rtr_dac_addr_r   : inp_dac_wb_addr;
    assign inp_dac_data = wfsm_busy    ? inp_dac_data_fsm :
                          rtr_dac_we_r ? rtr_dac_data_r   : inp_dac_wb_data;
    assign inp_dac_we   = wfsm_busy    ? inp_dac_we_fsm   :
                          rtr_dac_we_r ? 1'b1             : inp_dac_wb_we;

    // Latch x_u8 on any inp_dac write (captures the input activation for this column)
    always @(posedge clk) begin
        if (inp_dac_we && inp_dac_addr[3:0] < N_COLS)
            inp_act_latch[inp_dac_addr[3:0]*8 +: 8] <= inp_dac_data;
    end

    // ── E-inject FSM ─────────────────────────────────────────────────────────
    // delta_flat slice for inject_cell: [cell*N_ROWS*8 +: N_ROWS*8]
    wire [N_ROWS*8-1:0] delta_slice =
        delta_flat[inject_cell_wb * N_ROWS * 8 +: N_ROWS*8];

    e_inject_ctrl #(.N_ROWS(N_ROWS), .N_COLS(N_COLS)) u_einj (
        .clk            (clk),             .rst_n        (rst_n),
        .start          (start_e_update_wb),.inject_cell (inject_cell_wb),
        .busy           (e_update_busy),   .done         (e_update_done),
        .delta_flat     (delta_slice),
        .act_flat       (inp_act_latch),
        .bh             (bh),              .lr_shift     (lr_shift),
        .inject_en      (inject_en_cap),   .inject_cell_o(inject_cell_cap),
        .inject_addr    (inject_addr_cap), .inject_delta  (inject_delta_cap),
        .inject_x       (inject_x_cap)
    );

    // ── ★ THE JUG FIRE SWEEP (replaces absorb_ctrl) ──────────────────────────
    // OLD absorb: W_cap += E_cap; E_cap := 0        (DISCHARGE — lossy)
    // JUG:        if |E| >= theta: W_code += ±1; E -= sign*theta   (SUBTRACT — a sigma-delta)
    // ** A FIRE IS A ±1 INCREMENT ON THE W-SRAM CODE. There is NO charge pump into Cw. **
    wire        jug_pulse, jug_dn;
    wire [9:0]  jug_sram_addr;
    wire        jug_sram_re, jug_sram_we;
    wire [7:0]  jug_sram_wdata;
    wire        fire_up, fire_dn;
    wire        cmp_en_cap;
    wire [1:0]  cmp_cell_cap;
    wire [7:0]  cmp_addr_cap;

    jug_ctrl #(.N_CELLS(N_CELLS), .N_ELEMS(N_ELEMS),
               .WGT_MIN(WGT_MIN), .WGT_MAX(WGT_MAX), .PULSE_CYC(1)) u_jug (
        .clk        (clk),             .rst_n      (rst_n),
        .start_wb   (start_absorb_wb), .sweep_n_in (absorb_n),
        .busy       (absorb_busy),     .done       (absorb_done),
        .cmp_en     (cmp_en_cap),      .cmp_cell   (cmp_cell_cap),
        .cmp_addr   (cmp_addr_cap),
        .fire_up    (fire_up),         .fire_dn    (fire_dn),
        .jug_pulse  (jug_pulse),       .jug_dn     (jug_dn),
        .sram_addr  (jug_sram_addr),   .sram_re    (jug_sram_re),
        .sram_rdata (sram_rdata),
        .sram_we    (jug_sram_we),     .sram_wdata (jug_sram_wdata)
    );

    // the residue subtractor addresses the element the comparator just visited
    reg [1:0] jug_cell_r;  reg [7:0] jug_addr_r;
    always @(posedge clk) if (cmp_en_cap) begin
        jug_cell_r <= cmp_cell_cap;  jug_addr_r <= cmp_addr_cap;
    end

    // ── ★ TRANSPOSE-AT-SOURCE: Wᵀδ computed ON THIS CHIP, from its OWN W-SRAM ──
    // Replaces the router's SHADOW_W (weights never leave the die). The incoming δ (the SAME
    // delta_flat that drives E-inject) is transposed here; the router reads back the partial and
    // does only the gather (fixed-divide). See ../INTERCONNECT_PROTOCOL.md.
    // δ slice for the selected cell: [cell*N_ROWS*8 +: N_ROWS*8].
    wire [N_ROWS*8-1:0] transpose_delta =
        delta_flat[transpose_cell_wb * N_ROWS * 8 +: N_ROWS*8];

    pcn_transpose #(.N(N_ROWS), .N_CELLS(N_CELLS)) u_transpose (
        .clk         (clk),               .rst_n       (rst_n),
        .start       (start_transpose_wb),.cell_sel    (transpose_cell_wb),
        .delta_flat  (transpose_delta),
        .busy        (transpose_busy),    .done        (transpose_done),
        .partial_flat(transpose_partial),
        .sram_addr   (t_sram_addr),       .sram_re     (t_sram_re),
        .sram_rdata  (sram_rdata)
    );

    // ── ★ THE PRE-DISTORTION LUT: code -> DAC drive, so the WEIGHT is linear ──
    // weight_dac maps code -> V_w LINEARLY; the CELL maps V_w -> weight SIGMOIDALLY (28x
    // sensitivity variation).  The LUT pre-distorts so the composition is LINEAR — which is
    // what every simulation has always assumed. ⚠ IT IS ONLY TRUE WITH THE LUT LOADED.
    // ★ And it is LOADABLE => it is the PER-DIE CALIBRATION HOOK for the whole weight path.
    // See ../circuit/THE_WEIGHT_IS_NOT_LINEAR.md.
    wire [9:0] wgt_dac_drive;
    wire       wgt_lut_loaded;
    wire [7:0] wgt_lut_addr;
    wire [9:0] wgt_lut_wdata;
    wire wgt_lut_we;
    wire [7:0] jug_theta;
    wgt_lut #(.DAC_W(10)) u_wlut (
        .clk(clk), .rst_n(rst_n),
        .code(lcode_mux), .dac_drive(wgt_dac_drive),
        .lut_we(wgt_lut_we), .lut_addr(wgt_lut_addr), .lut_wdata(wgt_lut_wdata),
        .lut_loaded(wgt_lut_loaded)
    );

    // ── Refresh FSM ─────────────────────────────────────────────────────────
    wire [9:0] refresh_sram_addr;
    refresh_ctrl #(.N_CELLS(N_CELLS), .N_ELEMS(N_ELEMS)) u_ref (
        .clk          (clk),             .rst_n        (rst_n),
        .refresh_en   (refresh_en),      .training_mode(training_mode),
        .inhibit      (inhibit_refresh),
        .sram_addr_r  (refresh_sram_addr),.sram_rdata  (sram_rdata),
        .load_en      (load_en_ref),     .load_cell    (load_cell_ref),
        .load_addr    (load_addr_ref),   .load_code    (load_code_ref)
    );
    assign sram_addr_ref = refresh_sram_addr;

    // ── Save FSM ─────────────────────────────────────────────────────────────
    save_ctrl #(.N_CELLS(N_CELLS), .N_ELEMS(N_ELEMS)) u_save (
        .clk        (clk),          .rst_n      (rst_n),
        .start      (start_save_wb),.busy       (save_busy),
        .done       (save_done),
        .save_req   (save_req_cap), .save_cell_o(save_cell_cap),
        .save_addr_o(save_addr_cap),.save_code  (save_code_cap),
        .save_valid (save_valid_cap),
        .sram_addr  (sram_addr_save),.sram_wdata(sram_wdata_save),
        .sram_we    (sram_we_save)
    );

    // ── Weight FSM ──────────────────────────────────────────────────────────
    // Handles initial W-SRAM load and host-override (parallel DAC path).
    // In training_mode: load_en_fsm suppressed → W-caps not touched.
    wire [VIRT_AW-1:0] virt_layer_idx;
    wire        adc_sample_fsm, adc_done;
    wire  [7:0] adc_data_w;
    wire        adc_sample_mux = adc_sample_fsm | adc_sample_rtr;
    wire [CELL_AW-1:0] act_addr_w;
    wire  [7:0] act_wdata_w, act_rdata_w;
    wire        act_we_w;
    wire [9:0]  sram_addr_fsm_w;
    assign sram_addr_fsm = sram_addr_fsm_w;

    weight_fsm #(.N_CELLS(N_ELEMS)) u_wfsm (  // N_CELLS param = SRAM depth per VL
        .clk              (clk),
        .rst_n            (rst_n),
        .start_load       (start_load),
        .load_all         (load_all),
        .rst_weights      (rst_weights),
        .cell_addr        (cell_addr),
        .weight_data      (weight_data),
        .hebb_pw          (hebb_pw),
        .sram_addr        (sram_addr_fsm_w),
        .sram_rdata       (sram_rdata),
        .dac_addr         (dac_addr_fsm),
        .dac_data         (dac_data_fsm),
        .dac_we           (dac_we_fsm),
        .busy             (wfsm_busy),
        .ready            (wfsm_ready),
        .irq_load_done    (irq_load_done),
        .start_temporal   (start_temporal),
        .start_adc_sweep  (start_adc_sweep),
        .n_virt_layers    (n_virt_layers),
        .adc_sample       (adc_sample_fsm),
        .adc_done         (adc_done),
        .adc_data         (adc_data_w),
        .act_addr         (act_addr_w),
        .act_wdata        (act_wdata_w),
        .act_we           (act_we_w),
        .act_rdata        (act_rdata_w),
        .inp_dac_addr     (inp_dac_addr_fsm),
        .inp_dac_data     (inp_dac_data_fsm),
        .inp_dac_we       (inp_dac_we_fsm),
        .virt_layer_idx   (virt_layer_idx),
        .irq_temporal_done(irq_temporal_done)
    );

    // ── DAC output mux ───────────────────────────────────────────────────────
    // ANALOG mode: cap_array eff_code drives the column current source.
    // PARALLEL mode: weight_fsm dac_data (from SRAM) drives the DAC (test/load).
    assign dac_addr = dac_addr_mux;
    assign dac_data = training_mode ? cap_eff_code : dac_data_fsm;
    assign dac_we   = dac_we_fsm;

    // ── SAR ADC ─────────────────────────────────────────────────────────────
    // adc_sample_mux: OR of weight_fsm and rtr_adc_bridge requests
    sar_adc #(.BITS(8)) u_adc (
        .clk     (clk),      .rst_n  (rst_n),
        .sample  (adc_sample_mux),.done (adc_done),
        .data    (adc_data_w),.dac_out(adc_dac_out),
        .cmp     (adc_cmp)
    );

    // ── Activation SRAM ──────────────────────────────────────────────────────
    wire [CELL_AW-1:0] act_addr_mux = wfsm_busy ? act_addr_w : cell_addr[CELL_AW-1:0];
    wire  [7:0] act_wdata_mux = wfsm_busy ? act_wdata_w : act_wb_wdata;
    wire        act_we_mux    = wfsm_busy ? act_we_w    : act_wb_we;
    assign act_rdata_wb = act_rdata_w;

    act_sram #(.DEPTH(N_CELLS)) u_act (
        .clk   (clk),          .addr  (act_addr_mux),
        .wdata (act_wdata_mux),.we    (act_we_mux),
        .rdata (act_rdata_w)
    );

    // ── Hebbian controller ───────────────────────────────────────────────────
    wire [N_ROWS-1:0] eff_hebb_mask = hebb_mask & hebb_row_mask;
    hebb_ctrl #(.N_ROWS(N_ROWS)) u_hebb (
        .clk         (clk),       .rst_n       (rst_n),
        .hebb_en     (ctrl[2]),   .hebb_mask   (eff_hebb_mask),
        .hebb_pw     (hebb_pw),   .ierr        (ierr),
        .we_out      (we_out),    .hebb_actv   (hebb_actv),
        .irq_hebb_ovf(irq_hebb_ovf)
    );

    // ── Power FSM ────────────────────────────────────────────────────────────
    power_fsm u_pwr (
        .clk          (clk),         .rst_n      (rst_n),
        .sleep_req    (ctrl[3]),     .busy       (wfsm_busy),
        .full_power   (full_power),  .keep_alive (keep_alive),
        .sleep_ack    (sleep_ack),   .irq_sleep_ack(irq_sleep_ack)
    );

    // ── Router act-SRAM: N_CELLS × N_ROWS × 8-bit ──────────────────────────
    // Written by rtr_adc_bridge during per-cell ADC sweeps.
    // Read by router_ctrl to build DATA packets for SPI dispatch.
    localparam RTR_ACT_DEPTH = N_CELLS * N_ROWS;  // 64
    reg [7:0] rtr_act_mem [0:RTR_ACT_DEPTH-1];

    always @(posedge clk) begin
        if (rtr_act_we_w)
            rtr_act_mem[rtr_act_waddr_w] <= rtr_act_wdata_w;
    end

    // Per-cell synchronous read (1-cycle latency; router_ctrl pre-addresses by 1)
    genvar rta;
    generate for (rta = 0; rta < N_CELLS; rta = rta+1) begin : rtr_rd
        always @(posedge clk)
            rtr_act_rdat[rta] <=
                rtr_act_mem[rta*N_ROWS + rtr_act_raddr_w[rta*RTR_CELL_AW+:RTR_CELL_AW]];
    end
    endgenerate

    // Pack per-cell read data into bus for router_ctrl
    wire [N_CELLS*8-1:0] rtr_act_rdata_bus;
    genvar rtb;
    generate for (rtb = 0; rtb < N_CELLS; rtb = rtb+1) begin : rtr_rdbus
        assign rtr_act_rdata_bus[rtb*8+:8] = rtr_act_rdat[rtb];
    end
    endgenerate

    // ── ADC bridge: sequences N_ROWS SAR conversions per cell for router ────
    rtr_adc_bridge #(.N_CELLS(N_CELLS), .N_ROWS(N_ROWS)) u_rtr_adc (
        .clk            (clk),              .rst_n          (rst_n),
        .adc_sweep_req  (adc_sweep_req_w),  .adc_sweep_done (adc_sweep_done_w),
        .adc_sample     (adc_sample_rtr),
        .adc_done       (adc_done),         .adc_data       (adc_data_w),
        .rtr_act_waddr  (rtr_act_waddr_w),  .rtr_act_wdata  (rtr_act_wdata_w),
        .rtr_act_we     (rtr_act_we_w),
        .rtr_sweep_active(rtr_sweep_active_w),
        .rtr_row_idx_o  (rtr_row_addr_w)
    );

    // ── Hebbian routing update (Gap 3: hebb_en_route connected) ─────────────
    // Tracks per-frame ADC sweep (src) and dispatch (dst) activity.
    // On irq_frame_done, rewrites loop_mask-permitted bits of routing_matrix
    // via the write-back ports on pcn_wb_regs_4cell_emx.
    routing_hebb #(.N_CELLS(N_CELLS)) u_rhebb (
        .clk                (clk),
        .rst_n              (rst_n),
        .hebb_en_route      (hebb_en_route_w),
        .loop_mask          (loop_mask),
        .adc_sweep_done     (adc_sweep_done_w),
        .rtr_inp_dac_we     (rtr_inp_dac_we_w),
        .irq_frame_done     (router_ctrl_irq),
        .routing_matrix_cur (routing_matrix),
        .routing_matrix_hebb(routing_matrix_hebb_w),
        .routing_matrix_we  (routing_matrix_hebb_we_w)
    );

    // ── Router controller: TDM polling FSM + SPI master ─────────────────────
    router_ctrl #(
        .N_CELLS        (N_CELLS),
        .N_ROWS         (N_ROWS),
        .ACT_W          (8),
        .CELL_AW        (RTR_CELL_AW),
        .HELLO_EN       (1)
    ) u_rtr (
        .clk                (clk),              .rst_n              (rst_n),
        // Frame control
        .start_frame        (start_frame),      .busy               (router_ctrl_busy),
        .irq_frame_done     (router_ctrl_irq),
        // ADC sweep handshake
        .adc_sweep_req      (adc_sweep_req_w),  .adc_sweep_done     (adc_sweep_done_w),
        // Act-SRAM read (per-cell)
        .act_raddr          (rtr_act_raddr_w),  .act_rdata          (rtr_act_rdata_bus),
        // inp_dac dispatch (per-dest-cell)
        .rtr_inp_dac_addr   (rtr_inp_dac_addr_w),
        .rtr_inp_dac_data   (rtr_inp_dac_data_w),
        .rtr_inp_dac_we     (rtr_inp_dac_we_w),
        // SenderID conditioning (page select hint for analog domain)
        .sender_id          (sender_id_w),      .sender_id_valid    (sender_id_valid_w),
        // Configuration
        .routing_matrix     (routing_matrix),
        .input_cell_mask    (input_cell_mask[N_CELLS-1:0]),
        .output_cell_mask   (output_cell_mask[N_CELLS-1:0]),
        .topo_mode          (topo_mode),        .dispatch_pw        (dispatch_pw),
        .hebb_en_route      (hebb_en_route_w),
        // SPI routing config
        .my_chip_id         (my_chip_id),       .spi_clk_div        (spi_clk_div),
        .next_hop_chip_id   (next_hop_chip_id), .dest_cell_id_reg   (dest_cell_id_reg),
        // Router SPI
        .spi_clk_o          (spi_clk_o),        .spi_mosi           (spi_mosi),
        .spi_miso           (spi_miso),         .spi_cs_n           (spi_cs_n),
        // HELLO result
        .hello_chip_id      (hello_chip_id_w),  .hello_done         (hello_done_w),
        // Received DATA from router
        .spi_pkt_valid      (spi_pkt_valid_w),  .spi_dest_cell      (spi_dest_cell_w),
        .spi_src_cell       (spi_src_cell_w),   .spi_act_data       (spi_act_data_w),
        // Peer SPI master
        .peer_cs_n_o        (peer_cs_n_o),      .peer_clk_o         (peer_clk_o),
        .peer_mosi_o        (peer_mosi_o),      .peer_miso_i        (peer_miso_i),
        // Peer SPI slave
        .peer_cs_n_i        (peer_cs_n_i),      .peer_clk_i         (peer_clk_i),
        .peer_mosi_i        (peer_mosi_i),      .peer_miso_o        (peer_miso_o),
        // Peer received packet
        .peer_pkt_valid     (peer_pkt_valid_w), .peer_dest_cell     (peer_dest_cell_w),
        .peer_src_cell      (peer_src_cell_w),  .peer_act_data      (peer_act_data_w),
        // Peer output config
        .peer_next_hop_chip_id(peer_next_hop_chip_id),
        .peer_dest_cell_id_reg(peer_dest_cell_id_reg),
        .peer_output_mask   (peer_output_mask),
        // Status
        .current_slot       (router_current_slot_w)
    );

    // ── IRQ and LA outputs ───────────────────────────────────────────────────
    assign user_irq  = {irq_sleep_ack, irq_hebb_ovf, irq_load_done | irq_temporal_done};
    assign la_data_out = {
        absorb_done, save_done, absorb_busy, e_update_busy, // [31:28]
        ierr[3:0],                                          // [27:24]
        dac_addr_fsm[7:0],                                  // [23:16]
        cap_eff_code,                                       // [15:8]
        sleep_ack, hebb_actv, wfsm_busy, wfsm_ready,       // [7:4]
        training_mode, 3'b000                               // [3:0]
    };

endmodule
