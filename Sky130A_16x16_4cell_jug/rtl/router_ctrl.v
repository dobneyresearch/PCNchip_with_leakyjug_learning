`default_nettype none
// router_ctrl.v — TDM polling router for EMX 4-cell PCN chip.
// Ported from Sky130A_16x16_4cell/temporal/rtl/router_ctrl.v — logic unchanged.
// EMX extension: bh[2:0] FEEDBACK packet field handled at WB-register level (EMX_DESIGN.md §3.3).
//
// Packet format matches ROUTER_PROTOCOL.md (DATA, Sky130A, N_ROWS=16):
//   22 bytes MSB-first over SPI:
//     B0 {pkt_type[7:6]=00, rsvd, frame_seq[3:0]}
//     B1  dest_chip_id
//     B2  src_chip_id
//     B3  checksum (XOR of all bytes except B3)
//     B4  {src_cell_id[7:4], dest_cell_id[3:0]}
//     B5  payload_length = N_ROWS
//     B6..B21  activations[0..N_ROWS-1]
//   PKT_BYTES = N_ROWS + 6 = 22.  PKT_BITS_SPI = 176.
//
// SPI transaction (CPOL=0, CPHA=0, this chip = SPI master):
//   CS_N falls → 1 dummy clock (pcn_spi.v initialises rx_active, pre-loads MISO)
//   → PKT_BITS_SPI full-duplex clocks (TX on MOSI falling-edge, RX on MISO rising-edge)
//   → CS_N rises (pcn_spi.v captures packet, fires dbuf_consumed)
//
// Startup HELLO exchange (once after reset, before any TDM frame):
//   1. TX: 9-byte HELLO padded to PKT_BYTES clocks (padding zeros — router discards them).
//   2. Wait HELLO_ACK_WAIT sys_clk cycles for router to write HELLO_ACK to delivery_buf.
//   3. RX: dummy MOSI (byte3=0xFF bad-checksum so router discards) while reading HELLO_ACK.
//   4. Parse: extract assigned_chip_id → hello_chip_id, assert hello_done.
//
// Peer port — two independent paths:
//   Peer MASTER TX: additional SPI transaction (peer_cs_n_o/clk_o/mosi_o) in SPI slot
//                   when peer_output_mask != 0, after router transaction.
//   Peer SLAVE RX:  separate always block on peer_clk_i; toggle-handshake CDC to sys_clk.
//
`timescale 1ns/1ps

module router_ctrl #(
    parameter N_CELLS        = 4,
    parameter N_ROWS         = 16,
    parameter ACT_W          = 8,
    parameter CELL_AW        = 4,         // $clog2(N_ROWS)
    parameter [15:0] TILE_LOAD_WAIT = 16'd256,
    parameter [15:0] HELLO_ACK_WAIT = 16'd600, // sys_clk cycles between HELLO TX and ACK read
    parameter        HELLO_EN        = 1        // 0 = skip HELLO exchange (testbench use)
) (
    input  wire clk,
    input  wire rst_n,

    // ── Frame control ──────────────────────────────────────────────────────────
    input  wire        start_frame,
    output reg         busy,
    output reg         irq_frame_done,

    // ── Per-cell ADC sweep ─────────────────────────────────────────────────────
    output reg  [N_CELLS-1:0]          adc_sweep_req,
    input  wire [N_CELLS-1:0]          adc_sweep_done,

    // ── Per-cell act_sram read ─────────────────────────────────────────────────
    output reg  [N_CELLS*CELL_AW-1:0]  act_raddr,
    input  wire [N_CELLS*ACT_W-1:0]    act_rdata,

    // ── Per-cell inp_dac write ─────────────────────────────────────────────────
    output reg  [N_CELLS*CELL_AW-1:0]  rtr_inp_dac_addr,
    output reg  [N_CELLS*ACT_W-1:0]    rtr_inp_dac_data,
    output reg  [N_CELLS-1:0]           rtr_inp_dac_we,

    // ── SenderID conditioning ──────────────────────────────────────────────────
    output reg  [N_CELLS*2-1:0]         sender_id,
    output reg  [N_CELLS-1:0]           sender_id_valid,

    // ── Configuration registers ────────────────────────────────────────────────
    input  wire [N_CELLS*N_CELLS-1:0]   routing_matrix,
    input  wire [N_CELLS-1:0]            input_cell_mask,
    input  wire [N_CELLS-1:0]            output_cell_mask,
    input  wire                           topo_mode,
    input  wire [15:0]                    dispatch_pw,
    output wire                           hebb_en_route,

    // ── SPI routing registers ──────────────────────────────────────────────────
    input  wire [7:0]               my_chip_id,
    input  wire [7:0]               spi_clk_div,
    input  wire [N_CELLS*8-1:0]     next_hop_chip_id,   // dest chip per output cell
    input  wire [N_CELLS*4-1:0]     dest_cell_id_reg,   // dest cell per output cell

    // ── Router SPI (this chip = SPI master) ───────────────────────────────────
    output reg  spi_clk_o,
    output reg  spi_mosi,
    input  wire spi_miso,
    output reg  spi_cs_n,

    // ── HELLO result (valid after hello_done=1) ────────────────────────────────
    output reg [7:0]  hello_chip_id,
    output reg        hello_done,

    // ── Received DATA from router (destined for this chip) ────────────────────
    output reg        spi_pkt_valid,
    output reg [3:0]  spi_dest_cell,
    output reg [3:0]  spi_src_cell,
    output reg [N_ROWS*ACT_W-1:0] spi_act_data,

    // ── Peer SPI master (this chip → peer slave) ──────────────────────────────
    output reg  peer_cs_n_o,
    output reg  peer_clk_o,
    output reg  peer_mosi_o,
    input  wire peer_miso_i,

    // ── Peer SPI slave (peer master → this chip) ──────────────────────────────
    input  wire peer_cs_n_i,
    input  wire peer_clk_i,
    input  wire peer_mosi_i,
    output reg  peer_miso_o,

    // ── Peer received packet ───────────────────────────────────────────────────
    output reg        peer_pkt_valid,
    output reg [3:0]  peer_dest_cell,
    output reg [3:0]  peer_src_cell,
    output reg [N_ROWS*ACT_W-1:0] peer_act_data,

    // ── Peer output configuration ──────────────────────────────────────────────
    input  wire [N_CELLS*8-1:0]  peer_next_hop_chip_id,
    input  wire [N_CELLS*4-1:0]  peer_dest_cell_id_reg,
    input  wire [N_CELLS-1:0]    peer_output_mask,

    // ── Status ─────────────────────────────────────────────────────────────────
    output reg [2:0]  current_slot
);

    // ── Packet constants ───────────────────────────────────────────────────────
    localparam PKT_BYTES    = N_ROWS + 6;
    localparam PKT_BITS_SPI = PKT_BYTES * 8;

    localparam PKT_DATA  = 2'b00;
    localparam PKT_HELLO = 2'b10;

    // ── State encoding ─────────────────────────────────────────────────────────
    localparam ST_IDLE           = 5'd0;
    localparam ST_SLOT_START     = 5'd1;
    localparam ST_ADC_REQ        = 5'd2;
    localparam ST_ADC_WAIT       = 5'd3;
    localparam ST_WEIGHT_LOAD    = 5'd4;
    localparam ST_DISP_INIT      = 5'd5;
    localparam ST_DISP_ACT       = 5'd6;
    localparam ST_DISP_WR        = 5'd7;
    localparam ST_DISP_NX        = 5'd8;
    // SPI transaction machine (shared by router TX, peer TX, HELLO TX, HELLO RX)
    localparam ST_SPI_TXBLD_INIT = 5'd9;
    localparam ST_SPI_TXBLD_ACT  = 5'd10;
    localparam ST_SPI_TXBLD_RD   = 5'd11;
    localparam ST_SPI_START      = 5'd12;  // pull CS_N low, arm tx_shift
    localparam ST_SPI_DUMMY      = 5'd13;  // 1 full SPI clock period
    localparam ST_SPI_BITS       = 5'd14;  // PKT_BITS_SPI full-duplex clocks
    localparam ST_SPI_PARSE      = 5'd15;  // parse rx_shift
    localparam ST_SPI_NEXT_CELL  = 5'd16;  // advance tx_cell_scan; loop or exit
    localparam ST_SLOT_NEXT      = 5'd17;
    localparam ST_DONE           = 5'd18;
    // HELLO exchange
    localparam ST_HELLO_INIT     = 5'd19;  // build HELLO tx_pkt_buf; enter SPI machine
    localparam ST_HELLO_WAIT     = 5'd20;  // wait HELLO_ACK_WAIT cycles
    localparam ST_HELLO_RX       = 5'd21;  // build dummy TX; enter SPI machine for HELLO_ACK
    localparam ST_HELLO_PARSE    = 5'd22;  // extract assigned_chip_id from rx_shift

    localparam N_SLOTS = N_CELLS + 1;
    localparam SPI_SLOT = N_CELLS;

    // ── Registers ──────────────────────────────────────────────────────────────
    reg [4:0]  state;
    reg [4:0]  spi_return_state; // where ST_SPI_BITS jumps after completing
    reg        is_peer_tx;       // 0=router SPI, 1=peer SPI for current transaction

    reg [2:0]  slot_ctr;
    reg [1:0]  src;
    reg [N_CELLS-1:0] dest_mask;

    reg [CELL_AW-1:0] row_idx;
    reg [15:0]         settle_cnt;
    reg [15:0]         tile_cnt;
    reg [15:0]         hello_wait_cnt;
    reg [3:0]          frame_seq;

    // SPI transaction registers
    reg [PKT_BITS_SPI-1:0] tx_pkt_buf;  // assembled outgoing packet
    reg [PKT_BITS_SPI-1:0] tx_shift;    // shift register for TX
    reg [PKT_BITS_SPI-1:0] rx_shift;    // shift register for RX (MISO)
    reg [7:0]  tx_csum;                 // checksum accumulator during build
    reg [CELL_AW-1:0] tx_row_idx;
    reg [N_CELLS-1:0] tx_cell_scan;     // one-hot scan through output cells
    reg [1:0]  tx_src_cell;             // source cell index for current TX
    reg [7:0]  spi_clk_cnt;
    reg [8:0]  spi_bit_cnt;             // up to PKT_BITS_SPI = 176

    // ── Combinational helpers ──────────────────────────────────────────────────
    wire [N_CELLS-1:0] row_dest = routing_matrix[src*N_CELLS +: N_CELLS];

    wire        tx_any = |tx_cell_scan;
    wire [1:0]  tx_sel = tx_cell_scan[0] ? 2'd0 :
                         tx_cell_scan[1] ? 2'd1 :
                         tx_cell_scan[2] ? 2'd2 : 2'd3;

    assign hebb_en_route = topo_mode;

    // ── RX parse helpers (combinational on rx_shift) ───────────────────────────
    wire [1:0] rx_pkt_type   = rx_shift[PKT_BITS_SPI-1 -: 2];
    wire [7:0] rx_dest_chip  = rx_shift[PKT_BITS_SPI-9  -: 8];
    wire [3:0] rx_src_cell_f = rx_shift[PKT_BITS_SPI-33 -: 4]; // byte4[7:4]
    wire [3:0] rx_dest_cell_f= rx_shift[PKT_BITS_SPI-37 -: 4]; // byte4[3:0]
    wire [N_ROWS*ACT_W-1:0] rx_act = rx_shift[N_ROWS*ACT_W-1 : 0]; // last N_ROWS bytes

    // HELLO_ACK fields (same rx_shift after HELLO_RX transaction)
    wire [7:0] rx_hello_type    = rx_shift[PKT_BITS_SPI-33 -: 8]; // byte4
    wire [7:0] rx_assigned_id   = rx_shift[PKT_BITS_SPI-41 -: 8]; // byte5

    // ── Main FSM ───────────────────────────────────────────────────────────────
    integer d;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= HELLO_EN ? ST_HELLO_INIT : ST_IDLE;
            spi_return_state<= ST_IDLE;
            is_peer_tx      <= 1'b0;
            slot_ctr        <= 3'h0;
            src             <= 2'h0;
            dest_mask       <= {N_CELLS{1'b0}};
            row_idx         <= {CELL_AW{1'b0}};
            settle_cnt      <= 16'h0;
            tile_cnt        <= 16'h0;
            hello_wait_cnt  <= 16'h0;
            frame_seq       <= 4'h0;
            busy            <= 1'b0;
            irq_frame_done  <= 1'b0;
            current_slot    <= 3'h0;
            adc_sweep_req   <= {N_CELLS{1'b0}};
            rtr_inp_dac_we  <= {N_CELLS{1'b0}};
            rtr_inp_dac_addr<= {(N_CELLS*CELL_AW){1'b0}};
            rtr_inp_dac_data<= {(N_CELLS*ACT_W){1'b0}};
            act_raddr       <= {(N_CELLS*CELL_AW){1'b0}};
            sender_id       <= {(N_CELLS*2){1'b0}};
            sender_id_valid <= {N_CELLS{1'b0}};
            spi_clk_o       <= 1'b0;
            spi_mosi        <= 1'b0;
            spi_cs_n        <= 1'b1;
            peer_cs_n_o     <= 1'b1;
            peer_clk_o      <= 1'b0;
            peer_mosi_o     <= 1'b0;
            spi_pkt_valid   <= 1'b0;
            hello_chip_id   <= 8'hFF;
            hello_done      <= 1'b0;
            tx_pkt_buf      <= {PKT_BITS_SPI{1'b0}};
            tx_shift        <= {PKT_BITS_SPI{1'b0}};
            rx_shift        <= {PKT_BITS_SPI{1'b0}};
            tx_csum         <= 8'h00;
            tx_row_idx      <= {CELL_AW{1'b0}};
            tx_cell_scan    <= {N_CELLS{1'b0}};
            tx_src_cell     <= 2'h0;
            spi_clk_cnt     <= 8'h0;
            spi_bit_cnt     <= 9'h0;
        end else begin
            // Default deasserts
            irq_frame_done  <= 1'b0;
            adc_sweep_req   <= {N_CELLS{1'b0}};
            rtr_inp_dac_we  <= {N_CELLS{1'b0}};
            spi_pkt_valid   <= 1'b0;

            case (state)

                // ── Idle ───────────────────────────────────────────────────────
                ST_IDLE: begin
                    busy     <= 1'b0;
                    spi_cs_n <= 1'b1;
                    if (start_frame) begin
                        busy         <= 1'b1;
                        slot_ctr     <= 3'h0;
                        current_slot <= 3'h0;
                        state        <= ST_SLOT_START;
                    end
                end

                // ── Slot dispatch ──────────────────────────────────────────────
                ST_SLOT_START: begin
                    current_slot <= slot_ctr;
                    if (slot_ctr == SPI_SLOT[2:0]) begin
                        // SPI slot: router TX/RX then optional peer TX
                        tx_cell_scan <= output_cell_mask;
                        is_peer_tx   <= 1'b0;
                        if (|output_cell_mask) begin
                            state <= ST_SPI_TXBLD_INIT;
                        end else if (|peer_output_mask) begin
                            // No router output cells; go straight to peer TX
                            tx_cell_scan <= peer_output_mask;
                            is_peer_tx   <= 1'b1;
                            state        <= ST_SPI_TXBLD_INIT;
                        end else begin
                            state <= ST_SLOT_NEXT;
                        end
                    end else begin
                        src   <= slot_ctr[1:0];
                        state <= ST_ADC_REQ;
                    end
                end

                // ── ADC sweep ──────────────────────────────────────────────────
                ST_ADC_REQ: begin
                    adc_sweep_req[src] <= 1'b1;
                    state <= ST_ADC_WAIT;
                end

                ST_ADC_WAIT: begin
                    if (adc_sweep_done[src]) begin
                        dest_mask <= row_dest & ~input_cell_mask;
                        tile_cnt  <= TILE_LOAD_WAIT;
                        state     <= ST_WEIGHT_LOAD;
                    end
                end

                // ── Weight tile load ───────────────────────────────────────────
                ST_WEIGHT_LOAD: begin
                    for (d = 0; d < N_CELLS; d = d + 1) begin
                        if (dest_mask[d]) begin
                            sender_id_valid[d]  <= 1'b1;
                            sender_id[d*2 +: 2] <= src;
                        end
                    end
                    if (tile_cnt == 16'h0) begin
                        sender_id_valid <= {N_CELLS{1'b0}};
                        row_idx         <= {CELL_AW{1'b0}};
                        act_raddr[src*CELL_AW +: CELL_AW] <= {CELL_AW{1'b0}};
                        state <= ST_DISP_INIT;
                    end else begin
                        tile_cnt <= tile_cnt - 1'b1;
                    end
                end

                ST_DISP_INIT: begin
                    state <= ST_DISP_ACT;
                end

                // ── Dispatch loop ──────────────────────────────────────────────
                ST_DISP_ACT: begin
                    for (d = 0; d < N_CELLS; d = d + 1) begin
                        rtr_inp_dac_addr[d*CELL_AW +: CELL_AW] <= row_idx;
                        rtr_inp_dac_data[d*ACT_W   +: ACT_W]   <=
                            act_rdata[src*ACT_W +: ACT_W];
                    end
                    settle_cnt <= dispatch_pw;
                    state      <= ST_DISP_WR;
                end

                ST_DISP_WR: begin
                    rtr_inp_dac_we <= dest_mask;
                    if (settle_cnt == 16'h0) begin
                        rtr_inp_dac_we <= {N_CELLS{1'b0}};
                        state <= ST_DISP_NX;
                    end else begin
                        settle_cnt <= settle_cnt - 1'b1;
                    end
                end

                ST_DISP_NX: begin
                    if (row_idx == N_ROWS - 1) begin
                        state <= ST_SLOT_NEXT;
                    end else begin
                        row_idx <= row_idx + 1'b1;
                        act_raddr[src*CELL_AW +: CELL_AW] <= row_idx + 1'b1;
                        state <= ST_DISP_ACT;
                    end
                end

                // ── Slot counter ───────────────────────────────────────────────
                ST_SLOT_NEXT: begin
                    if (slot_ctr == N_SLOTS[2:0] - 1) begin
                        state <= ST_DONE;
                    end else begin
                        slot_ctr <= slot_ctr + 1'b1;
                        state    <= ST_SLOT_START;
                    end
                end

                ST_DONE: begin
                    irq_frame_done <= 1'b1;
                    busy           <= 1'b0;
                    state          <= ST_IDLE;
                end

                // ================================================================
                // SPI TRANSACTION MACHINE
                // Shared by: router DATA TX, peer DATA TX, HELLO TX, HELLO RX.
                //
                // Before entry, caller must set:
                //   tx_pkt_buf    — packet bytes to transmit
                //   spi_return_state — state after ST_SPI_BITS completes
                //   is_peer_tx    — 0=router ports, 1=peer ports
                //
                // After ST_SPI_BITS, rx_shift holds received bits (MSB-first).
                // ================================================================

                // ── Build TX packet from act_sram ──────────────────────────────
                // tx_sel: current output cell; tx_cell_scan: remaining cells
                ST_SPI_TXBLD_INIT: begin
                    // Header (byte3=0, filled with checksum at end of TXBLD_RD)
                    begin : txbld_hdr
                        reg [7:0] b0, b1, b2, b4, b5;
                        reg [7:0] nhop;
                        reg [3:0] dcell;
                        nhop  = is_peer_tx ? peer_next_hop_chip_id[tx_sel*8 +: 8]
                                           : next_hop_chip_id[tx_sel*8 +: 8];
                        dcell = is_peer_tx ? peer_dest_cell_id_reg[tx_sel*4 +: 4]
                                           : dest_cell_id_reg[tx_sel*4 +: 4];
                        b0 = {PKT_DATA, 2'b00, frame_seq};
                        b1 = nhop;
                        b2 = my_chip_id;
                        b4 = {{(4-2){1'b0}}, tx_sel, dcell};
                        b5 = N_ROWS[7:0];
                        tx_pkt_buf[PKT_BITS_SPI-1  -: 8] <= b0;
                        tx_pkt_buf[PKT_BITS_SPI-9  -: 8] <= b1;
                        tx_pkt_buf[PKT_BITS_SPI-17 -: 8] <= b2;
                        tx_pkt_buf[PKT_BITS_SPI-25 -: 8] <= 8'h00; // byte3 placeholder
                        tx_pkt_buf[PKT_BITS_SPI-33 -: 8] <= b4;
                        tx_pkt_buf[PKT_BITS_SPI-41 -: 8] <= b5;
                        tx_csum <= b0 ^ b1 ^ b2 ^ b4 ^ b5;
                    end
                    tx_src_cell <= tx_sel;
                    tx_row_idx  <= {CELL_AW{1'b0}};
                    act_raddr[tx_sel*CELL_AW +: CELL_AW] <= {CELL_AW{1'b0}};
                    spi_return_state <= ST_SPI_PARSE;
                    state <= ST_SPI_TXBLD_ACT;
                end

                ST_SPI_TXBLD_ACT: begin
                    // 1-cycle act_sram read latency
                    state <= ST_SPI_TXBLD_RD;
                end

                ST_SPI_TXBLD_RD: begin
                    begin : txbld_rd
                        reg [7:0] act_byte;
                        act_byte = act_rdata[tx_src_cell*ACT_W +: ACT_W];
                        // Byte 6+tx_row_idx in the packet
                        tx_pkt_buf[PKT_BITS_SPI - 49 - tx_row_idx*8 -: 8] <= act_byte;
                        if (tx_row_idx == N_ROWS - 1) begin
                            // Last byte: close checksum and write byte3
                            tx_pkt_buf[PKT_BITS_SPI-25 -: 8] <= tx_csum ^ act_byte;
                            state <= ST_SPI_START;
                        end else begin
                            tx_csum    <= tx_csum ^ act_byte;
                            tx_row_idx <= tx_row_idx + 1'b1;
                            act_raddr[tx_src_cell*CELL_AW +: CELL_AW] <= tx_row_idx + 1'b1;
                            state <= ST_SPI_TXBLD_ACT;
                        end
                    end
                end

                // ── SPI transaction: CS_N low, arm first MOSI bit ──────────────
                ST_SPI_START: begin
                    rx_shift    <= {PKT_BITS_SPI{1'b0}};
                    // Pre-load tx_shift: first bit goes to MOSI directly;
                    // tx_shift is pre-shifted by 1 so falling-edge logic uses [MSB].
                    if (is_peer_tx) begin
                        peer_cs_n_o  <= 1'b0;
                        peer_mosi_o  <= tx_pkt_buf[PKT_BITS_SPI-1]; // bit 7 of byte0
                        peer_clk_o   <= 1'b0;
                    end else begin
                        spi_cs_n  <= 1'b0;
                        spi_mosi  <= tx_pkt_buf[PKT_BITS_SPI-1];
                        spi_clk_o <= 1'b0;
                    end
                    tx_shift    <= {tx_pkt_buf[PKT_BITS_SPI-2:0], 1'b0}; // pre-shift
                    spi_clk_cnt <= 8'h0;
                    state       <= ST_SPI_DUMMY;
                end

                // ── Dummy clock: 1 full SPI period (pcn_spi init + MISO pre-load) ──
                ST_SPI_DUMMY: begin
                    if (spi_clk_cnt == spi_clk_div) begin
                        spi_clk_cnt <= 8'h0;
                        if (is_peer_tx)
                            peer_clk_o <= ~peer_clk_o;
                        else
                            spi_clk_o  <= ~spi_clk_o;
                        // After 2 half-periods (one full dummy clock), move to data phase
                        if (is_peer_tx ? peer_clk_o : spi_clk_o) begin
                            // Was high → going low = dummy negedge completed
                            spi_bit_cnt <= PKT_BITS_SPI[8:0];
                            state       <= ST_SPI_BITS;
                        end
                    end else begin
                        spi_clk_cnt <= spi_clk_cnt + 1'b1;
                    end
                end

                // ── Full-duplex data clocks ────────────────────────────────────
                // Rising edge (clk_o was 0): sample MISO → rx_shift; decrement bit_cnt.
                // Falling edge (clk_o was 1): shift out next MOSI bit.
                // Terminates after PKT_BITS_SPI rising edges; CS_N rises; → spi_return_state.
                ST_SPI_BITS: begin
                    if (spi_clk_cnt == spi_clk_div) begin
                        spi_clk_cnt <= 8'h0;
                        if (is_peer_tx) begin
                            peer_clk_o <= ~peer_clk_o;
                            if (!peer_clk_o) begin  // rising edge
                                rx_shift <= {rx_shift[PKT_BITS_SPI-2:0], peer_miso_i};
                                if (spi_bit_cnt == 9'd1) begin
                                    spi_bit_cnt <= 9'h0;
                                    peer_clk_o  <= 1'b0;
                                    peer_cs_n_o <= 1'b1;
                                    frame_seq   <= frame_seq + 1'b1;
                                    state       <= spi_return_state;
                                end else begin
                                    spi_bit_cnt <= spi_bit_cnt - 1'b1;
                                end
                            end else begin  // falling edge
                                peer_mosi_o <= tx_shift[PKT_BITS_SPI-1];
                                tx_shift    <= {tx_shift[PKT_BITS_SPI-2:0], 1'b0};
                            end
                        end else begin
                            spi_clk_o <= ~spi_clk_o;
                            if (!spi_clk_o) begin  // rising edge
                                rx_shift <= {rx_shift[PKT_BITS_SPI-2:0], spi_miso};
                                if (spi_bit_cnt == 9'd1) begin
                                    spi_bit_cnt <= 9'h0;
                                    spi_clk_o   <= 1'b0;
                                    spi_cs_n    <= 1'b1;
                                    frame_seq   <= frame_seq + 1'b1;
                                    state       <= spi_return_state;
                                end else begin
                                    spi_bit_cnt <= spi_bit_cnt - 1'b1;
                                end
                            end else begin  // falling edge
                                spi_mosi <= tx_shift[PKT_BITS_SPI-1];
                                tx_shift <= {tx_shift[PKT_BITS_SPI-2:0], 1'b0};
                            end
                        end
                    end else begin
                        spi_clk_cnt <= spi_clk_cnt + 1'b1;
                    end
                end

                // ── Parse received packet ──────────────────────────────────────
                ST_SPI_PARSE: begin
                    // Fire spi_pkt_valid / peer_pkt_valid if the received packet is
                    // a DATA packet addressed to this chip.
                    if (rx_pkt_type == PKT_DATA && rx_dest_chip == my_chip_id) begin
                        if (is_peer_tx) begin
                            peer_pkt_valid <= 1'b1;
                            peer_dest_cell <= rx_dest_cell_f;
                            peer_src_cell  <= rx_src_cell_f;
                            peer_act_data  <= rx_act;
                        end else begin
                            spi_pkt_valid <= 1'b1;
                            spi_dest_cell <= rx_dest_cell_f;
                            spi_src_cell  <= rx_src_cell_f;
                            spi_act_data  <= rx_act;
                        end
                    end
                    state <= ST_SPI_NEXT_CELL;
                end

                // ── Advance to next output cell ────────────────────────────────
                ST_SPI_NEXT_CELL: begin
                    // Clear the just-transmitted cell from scan mask
                    tx_cell_scan <= tx_cell_scan &
                                    ~({{(N_CELLS-1){1'b0}}, 1'b1} << tx_sel);
                    begin : next_cell_chk
                        reg [N_CELLS-1:0] remaining;
                        remaining = tx_cell_scan & ~({{(N_CELLS-1){1'b0}}, 1'b1} << tx_sel);
                        if (|remaining) begin
                            // More cells in this pass (router or peer)
                            state <= ST_SPI_TXBLD_INIT;
                        end else if (!is_peer_tx && |peer_output_mask) begin
                            // Router pass done; now do peer TX pass
                            tx_cell_scan <= peer_output_mask;
                            is_peer_tx   <= 1'b1;
                            state        <= ST_SPI_TXBLD_INIT;
                        end else begin
                            // Both passes done
                            state <= ST_SLOT_NEXT;
                        end
                    end
                end

                // ================================================================
                // HELLO EXCHANGE (startup, before first TDM frame)
                // ================================================================

                // ── Build HELLO packet ─────────────────────────────────────────
                ST_HELLO_INIT: begin
                    begin : hello_bld
                        reg [7:0] b0, b3;
                        b0 = {PKT_HELLO, 2'b00, frame_seq};
                        // XOR of all bytes except byte3:
                        // B0 ^ B1(0xFF) ^ B2(my_chip_id) ^ B4(0x00) ^
                        // B5(N_CELLS) ^ B6(N_ROWS) ^ B7(0x00) ^ B8(my_chip_id)
                        // my_chip_id appears twice → cancels
                        b3 = b0 ^ 8'hFF ^ 8'h00 ^ N_CELLS[7:0] ^ N_ROWS[7:0] ^ 8'h00 ^ 8'h00;
                        tx_pkt_buf                         <= {PKT_BITS_SPI{1'b0}};
                        tx_pkt_buf[PKT_BITS_SPI-1  -: 8]  <= b0;           // B0
                        tx_pkt_buf[PKT_BITS_SPI-9  -: 8]  <= 8'hFF;        // B1 dest=bcast
                        tx_pkt_buf[PKT_BITS_SPI-17 -: 8]  <= my_chip_id;   // B2 src
                        tx_pkt_buf[PKT_BITS_SPI-25 -: 8]  <= b3;           // B3 checksum
                        tx_pkt_buf[PKT_BITS_SPI-33 -: 8]  <= 8'h00;        // B4 hello_type=HELLO
                        tx_pkt_buf[PKT_BITS_SPI-41 -: 8]  <= N_CELLS[7:0]; // B5 cell_count
                        tx_pkt_buf[PKT_BITS_SPI-49 -: 8]  <= N_ROWS[7:0];  // B6 cell_rows
                        tx_pkt_buf[PKT_BITS_SPI-57 -: 8]  <= 8'h00;        // B7 wake_mode=push
                        tx_pkt_buf[PKT_BITS_SPI-65 -: 8]  <= my_chip_id;   // B8 remembered_id
                        // Bytes 9..PKT_BYTES-1 remain 0 (padding)
                    end
                    is_peer_tx       <= 1'b0;
                    spi_return_state <= ST_HELLO_WAIT;
                    state            <= ST_SPI_START;
                end

                // ── Wait for router to prepare HELLO_ACK ───────────────────────
                ST_HELLO_WAIT: begin
                    if (hello_wait_cnt == HELLO_ACK_WAIT) begin
                        hello_wait_cnt <= 16'h0;
                        state <= ST_HELLO_RX;
                    end else begin
                        hello_wait_cnt <= hello_wait_cnt + 1'b1;
                    end
                end

                // ── Read HELLO_ACK from router delivery_buf ────────────────────
                // TX: poison dummy (byte3=0xFF) so router discards the inbound frame.
                ST_HELLO_RX: begin
                    tx_pkt_buf <= {PKT_BITS_SPI{1'b0}};
                    tx_pkt_buf[PKT_BITS_SPI-25 -: 8] <= 8'hFF; // B3=0xFF → bad checksum
                    is_peer_tx       <= 1'b0;
                    spi_return_state <= ST_HELLO_PARSE;
                    state            <= ST_SPI_START;
                end

                // ── Extract assigned_chip_id from HELLO_ACK ────────────────────
                ST_HELLO_PARSE: begin
                    // rx_shift holds up to PKT_BYTES bytes from MISO.
                    // HELLO_ACK is 8 bytes starting at rx_shift[PKT_BITS_SPI-1].
                    if (rx_pkt_type == PKT_HELLO && rx_hello_type == 8'h01) begin
                        hello_chip_id <= rx_assigned_id;
                    end
                    hello_done <= 1'b1;
                    state      <= ST_IDLE;
                end

                default: state <= ST_IDLE;

            endcase
        end
    end

    // =========================================================================
    // PEER SPI SLAVE — asynchronous receive on peer_clk_i domain
    //
    // Mirrors pcn_spi.v RX path.  Uses toggle-handshake CDC to sys_clk.
    // MISO is pre-loaded from peer_miso_pkt when peer_cs_n_i falls.
    // =========================================================================

    // Peer clock domain registers
    reg [PKT_BITS_SPI-1:0] peer_rx_shreg;
    reg [7:0]  peer_rx_bit_cnt;
    reg [5:0]  peer_rx_byte_cnt;
    reg        peer_rx_active;
    reg        peer_cs_n_d1;
    wire peer_cs_n_fall_w = peer_cs_n_d1  && !peer_cs_n_i;
    wire peer_cs_n_rise_w = !peer_cs_n_d1 && peer_cs_n_i;
    reg peer_cs_n_fall_pend;

    reg [PKT_BITS_SPI-1:0] peer_cap_data;
    reg        peer_done_pcn;  // toggle: set on CS_N rise after full packet

    // 2-flop synchroniser: peer_done_pcn → sys_clk domain
    reg peer_done_s1, peer_done_s, peer_done_prev;
    wire peer_done_edge = peer_done_s ^ peer_done_prev;

    // MISO packet register (sys_clk domain, updated after each TDM frame)
    // The TDM FSM writes peer_miso_pkt whenever it builds a peer TX packet.
    // On peer CS_N fall, the slave pre-loads from this register.
    reg [PKT_BITS_SPI-1:0] peer_miso_pkt;

    // MISO shift register (peer_clk negedge domain)
    reg [PKT_BITS_SPI-1:0] peer_miso_shreg;

    initial begin
        peer_rx_shreg    = {PKT_BITS_SPI{1'b0}};
        peer_rx_bit_cnt  = 8'h0;
        peer_rx_byte_cnt = 6'h0;
        peer_rx_active   = 1'b0;
        peer_cs_n_d1     = 1'b1;
        peer_cs_n_fall_pend = 1'b0;
        peer_cap_data    = {PKT_BITS_SPI{1'b0}};
        peer_done_pcn    = 1'b0;
        peer_miso_shreg  = {PKT_BITS_SPI{1'b0}};
        peer_miso_o      = 1'b0;
    end

    // RX shift (posedge peer_clk_i)
    always @(posedge peer_clk_i) begin
        peer_cs_n_d1        <= peer_cs_n_i;
        peer_cs_n_fall_pend <= peer_cs_n_fall_w;

        if (peer_cs_n_i) begin
            peer_rx_active   <= 1'b0;
            peer_rx_bit_cnt  <= 8'h0;
            peer_rx_byte_cnt <= 6'h0;
        end else begin
            if (!peer_rx_active) begin
                peer_rx_active   <= 1'b1;
                peer_rx_bit_cnt  <= 8'h0;
                peer_rx_byte_cnt <= 6'h0;
            end else begin
                peer_rx_shreg   <= {peer_rx_shreg[PKT_BITS_SPI-2:0], peer_mosi_i};
                peer_rx_bit_cnt <= peer_rx_bit_cnt + 1'b1;
                if (peer_rx_bit_cnt == 8'd7)
                    peer_rx_byte_cnt <= peer_rx_byte_cnt + 1'b1;
            end
        end

        // Capture packet on CS_N rise
        if (peer_cs_n_rise_w) begin
            if (peer_rx_byte_cnt >= PKT_BYTES[5:0]) begin
                peer_cap_data <= peer_rx_shreg;
                peer_done_pcn <= ~peer_done_pcn;
            end
        end
    end

    // MISO shift (negedge peer_clk_i)
    always @(negedge peer_clk_i) begin
        if (peer_cs_n_fall_pend) begin
            // Pre-load MISO from peer_miso_pkt; present bit 7 of byte0 immediately.
            peer_miso_shreg <= {peer_miso_pkt[PKT_BITS_SPI-2:0], 1'b0};
            peer_miso_o     <= peer_miso_pkt[PKT_BITS_SPI-1];
        end else if (!peer_cs_n_i) begin
            peer_miso_o     <= peer_miso_shreg[PKT_BITS_SPI-1];
            peer_miso_shreg <= {peer_miso_shreg[PKT_BITS_SPI-2:0], 1'b0};
        end else begin
            peer_miso_o <= 1'b0;
        end
    end

    // CDC: peer_done_pcn → sys_clk
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            peer_done_s1   <= 1'b0;
            peer_done_s    <= 1'b0;
            peer_done_prev <= 1'b0;
            peer_pkt_valid <= 1'b0;
            peer_dest_cell <= 4'h0;
            peer_src_cell  <= 4'h0;
            peer_act_data  <= {(N_ROWS*ACT_W){1'b0}};
        end else begin
            peer_done_s1   <= peer_done_pcn;
            peer_done_s    <= peer_done_s1;
            peer_done_prev <= peer_done_s;

            peer_pkt_valid <= 1'b0;
            if (peer_done_edge) begin
                // peer_cap_data: byte0=[PKT_BITS-1:PKT_BITS-8], etc.
                if (peer_cap_data[PKT_BITS_SPI-1 -: 2] == PKT_DATA &&
                    peer_cap_data[PKT_BITS_SPI-9  -: 8] == my_chip_id) begin
                    peer_pkt_valid <= 1'b1;
                    peer_dest_cell <= peer_cap_data[PKT_BITS_SPI-37 -: 4];
                    peer_src_cell  <= peer_cap_data[PKT_BITS_SPI-33 -: 4];
                    peer_act_data  <= peer_cap_data[N_ROWS*ACT_W-1:0];
                end
            end
        end
    end

    // Update peer_miso_pkt from the last tx_pkt_buf assembled for peer TX.
    // (peer_miso_pkt is held stable throughout the peer_clk transaction and
    //  only changes between TDM frames, so no CDC hazard.)
    always @(posedge clk) begin
        if (state == ST_SPI_START && is_peer_tx)
            peer_miso_pkt <= tx_pkt_buf;
    end

endmodule
