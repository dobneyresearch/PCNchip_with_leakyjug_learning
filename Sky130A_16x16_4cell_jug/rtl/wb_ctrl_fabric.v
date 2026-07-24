`default_nettype none
`timescale 1ns/1ps
// wb_ctrl_fabric.v — S2 control-plane interconnect (Phase-2 B3).
//
// Implements the control_address_map_spec.md address map + broadcast + barrier:
//   addr[31]    BROADCAST   0=unicast(acked)  1=broadcast(posted, write-only)
//   addr[30]    NODE_KIND   0=chip 1=router                 (unicast)
//   addr[15:8]  NODE_ID     0..255, 0xFF = fabric status port(unicast)
//   addr[10:8]  GROUP       0=ALL, else CTRL_GROUP          (broadcast)
//   addr[7:0]   REG         per-node register offset (unchanged single-chip decode)
//
// Broadcast fans stb to every node whose CTRL_GROUP matches GROUP (0=ALL), posted
// (immediate ack, no per-node wait). Barrier aggregate = wired-AND of participating
// nodes' done (BP_BARRIER_DONE / ABSORB_DONE_ALL / EPOCH_BARRIER) read at NODE_ID=0xFF.
module wb_ctrl_fabric #(
    parameter integer N = 4
)(
    input  wire        clk, rst_n,
    // WB master (boss)
    input  wire [31:0] m_addr, m_wdata,
    input  wire        m_we, m_stb, m_cyc,
    output reg  [31:0] m_rdata,
    output reg         m_ack,
    // per-node identity (config): kind(1b), id(8b), CTRL_GROUP(3b)
    input  wire [N-1:0]    node_kind,
    input  wire [8*N-1:0]  node_id,
    input  wire [3*N-1:0]  node_group,
    // to node slaves (shared reg/wdata/we, per-node stb)
    output wire [7:0]      n_reg,
    output wire [31:0]     n_wdata,
    output wire           n_we,
    output reg  [N-1:0]    n_stb,
    input  wire [32*N-1:0] n_rdata,
    // barrier
    input  wire [N-1:0]    n_done,
    input  wire [N-1:0]    part_mask,        // 1 = node participates in the barrier
    output wire           barrier_done
);
    localparam [7:0] STATUS_ID = 8'hFF;
    wire       bcast = m_addr[31];
    wire       kind  = m_addr[30];
    wire [7:0] id    = m_addr[15:8];
    wire [2:0] grp   = m_addr[10:8];
    assign n_reg   = m_addr[7:0];
    assign n_wdata = m_wdata;
    assign n_we    = m_we;

    // wired-AND over participating nodes (non-participants forced to 1)
    assign barrier_done = &( n_done | ~part_mask );

    integer k;
    reg [31:0] sel_rdata;
    always @(*) begin
        n_stb     = {N{1'b0}};
        sel_rdata = 32'h0;
        if (m_stb && m_cyc) begin
            if (bcast) begin                              // posted broadcast
                for (k = 0; k < N; k = k + 1)
                    n_stb[k] = (grp == 3'd0) || (grp == node_group[k*3 +: 3]);
            end else if (id == STATUS_ID) begin           // fabric status read
                sel_rdata = {31'h0, barrier_done};
            end else begin                                // unicast
                for (k = 0; k < N; k = k + 1)
                    if (kind == node_kind[k] && id == node_id[k*8 +: 8]) begin
                        n_stb[k]  = 1'b1;
                        sel_rdata = n_rdata[k*32 +: 32];
                    end
            end
        end
    end

    // single-cycle ack (posted broadcast acks immediately too), registered rdata
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin m_ack <= 1'b0; m_rdata <= 32'h0; end
        else begin
            m_ack   <= m_stb & m_cyc & ~m_ack;
            m_rdata <= sel_rdata;
        end
    end
endmodule
`default_nettype wire
