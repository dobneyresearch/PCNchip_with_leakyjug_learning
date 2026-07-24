`default_nettype none
// router_chip_emx: extends router_chip with an ABSORB broadcast mechanism.
//
// New interface vs router_chip:
//   ABSORB_REQ  (input)  — host pulses high for one SYS_CLK cycle to start absorption
//   ABSORB_N[7:0] (output) — driven low to all 8 PCN ports during absorption window
//   ABSORB_DONE (output) — pulses high for one cycle when timeout expires
//
// Timing: ABSORB_N is asserted immediately on ABSORB_REQ, held for ABSORB_TIMEOUT
// cycles, then deasserted.  The PCN chip's jug sweep uses its own internal counter;
// the router just provides the window.
//
// ⚠ ABSORB_TIMEOUT must COVER THE WHOLE SWEEP: the shared comparator visits every one of
// the chip's N_CELLS×N_ELEMS elements once, and that single-visit-per-sweep property IS
// the refractory period that keeps the jug stable. A window shorter than the sweep would
// truncate it and starve the last elements of the array. It is therefore DERIVED from the
// cell geometry (+ MARGIN), not a literal — it was 1280, silently hardcoding 4×256+256.
//
// No modification to router_chip's packet-switching logic is needed.

module router_chip_emx #(
    parameter integer N_ROWS  = 16,
    parameter integer N_COLS  = 16,
    parameter integer N_CELLS = 4,
    parameter integer MARGIN  = 256,
    // DERIVED (see header): the window must span a full N_CELLS×N_ELEMS comparator sweep.
    parameter integer ABSORB_TIMEOUT = N_CELLS * N_ROWS * N_COLS + MARGIN   // 1280 @16×16×4
) (
    input  wire        SYS_CLK,
    input  wire        RESET_N,

    // ── router_chip pass-through ports ───────────────────────────────────────
    input  wire [7:0]  PCN_CLK,
    input  wire [7:0]  PCN_MOSI,
    output wire [7:0]  PCN_MISO,
    input  wire [7:0]  PCN_CS_N,
    output wire [7:0]  PCN_WAKE,

    output wire [1:0]  RING_TX_CLK,
    output wire [1:0]  RING_TX_DATA,
    input  wire [1:0]  RING_RX_CLK,
    input  wire [1:0]  RING_RX_DATA,

    input  wire        HOST_CLK,
    input  wire        HOST_MOSI,
    output wire        HOST_MISO,
    input  wire        HOST_CS_N,

    // ── New EMX ports ─────────────────────────────────────────────────────────
    input  wire        ABSORB_REQ,   // one-cycle pulse from host GPIO
    output wire [7:0]  ABSORB_N,     // active-low absorption window to all PCN chips
    output wire        ABSORB_DONE   // one-cycle pulse on window expiry
);

    // ── Base router ───────────────────────────────────────────────────────────
    router_chip u_base (
        .SYS_CLK      (SYS_CLK),
        .RESET_N      (RESET_N),
        .PCN_CLK      (PCN_CLK),
        .PCN_MOSI     (PCN_MOSI),
        .PCN_MISO     (PCN_MISO),
        .PCN_CS_N     (PCN_CS_N),
        .PCN_WAKE     (PCN_WAKE),
        .RING_TX_CLK  (RING_TX_CLK),
        .RING_TX_DATA (RING_TX_DATA),
        .RING_RX_CLK  (RING_RX_CLK),
        .RING_RX_DATA (RING_RX_DATA),
        .HOST_CLK     (HOST_CLK),
        .HOST_MOSI    (HOST_MOSI),
        .HOST_MISO    (HOST_MISO),
        .HOST_CS_N    (HOST_CS_N)
    );

    // ── Absorb FSM ────────────────────────────────────────────────────────────
    // IDLE: ABSORB_N = 1 (inactive)
    // ABSORBING: ABSORB_N = 0, countdown timer
    // Done: pulse ABSORB_DONE, return to IDLE

    localparam CTR_W = $clog2(ABSORB_TIMEOUT + 1);

    reg [CTR_W-1:0] ctr;
    reg             absorbing;
    reg             done_r;

    always @(posedge SYS_CLK or negedge RESET_N) begin
        if (!RESET_N) begin
            absorbing <= 1'b0;
            ctr       <= {CTR_W{1'b0}};
            done_r    <= 1'b0;
        end else begin
            done_r <= 1'b0;

            if (!absorbing) begin
                if (ABSORB_REQ) begin
                    absorbing <= 1'b1;
                    ctr       <= ABSORB_TIMEOUT - 1;
                end
            end else begin
                if (ctr == {CTR_W{1'b0}}) begin
                    absorbing <= 1'b0;
                    done_r    <= 1'b1;
                end else begin
                    ctr <= ctr - {{(CTR_W-1){1'b0}}, 1'b1};
                end
            end
        end
    end

    assign ABSORB_N    = {8{~absorbing}};  // all 8 PCN ports see same signal
    assign ABSORB_DONE = done_r;

endmodule
