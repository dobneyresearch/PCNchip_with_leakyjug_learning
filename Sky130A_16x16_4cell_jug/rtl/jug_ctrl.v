`default_nettype none
// =============================================================================
// jug_ctrl — THE JUG FIRE SWEEP.  Replaces absorb_ctrl.
// =============================================================================
//
// THE OLD absorb_ctrl DID:   W_cap += E_cap ;  E_cap := 0        (DISCHARGE — lossy)
// THE JUG DOES:              if |E| >= theta:  W_code += ±1 ;  E -= sign*theta
//                                                                (SUBTRACT — a sigma-delta)
//
// ── WHY ──────────────────────────────────────────────────────────────────────
// The current learning rule writes lr*sign(E) with lr = 3e-4, which is 0.019 of ONE weight
// LSB.  Written naively to an 8-bit analog cell that is a NO-OP, ~51 times in a row.  The old
// absorb never hit this because it wrote the MAGNITUDE of E (clamped to ±6.7 codes — ~350x
// larger) — but that rule plateaued at 64.09%.  The jug is how the BETTER rule (82.50% float)
// runs on the cell:  81.96% bit-accurate, vs 75.25% for the new rule written naively.
//
// ★★ THE RESIDUE-PRESERVING SUBTRACTION IS LOAD-BEARING.  Discharging E throws away exactly
//    the sub-threshold charge the mechanism exists to accumulate.  Subtracting theta makes it
//    a true SIGMA-DELTA: no charge, and therefore no learning signal, is ever lost.
//    It is ALSO what makes the comparator allowed to be sloppy: a WRONG-SIGNED fire moves W
//    the wrong way AND puts the charge back, and the next correct fire undoes both.
//    CHARGE IS CONSERVED.  (Measured: 20% wrong fires cost 0.34pp.)
//
// ── THE SWEEP IS A FREE REFRACTORY PERIOD ────────────────────────────────────
// One SHARED comparator, time-multiplexed across the array (1024x cheaper than one per cell),
// so each cell is checked ONCE PER SWEEP and can fire AT MOST ONCE.  That single-fire limit is
// THE DESIGN, not a simulation shortcut.
// ⚠ DO NOT put a comparator in every cell: it fires repeatedly on a crossing and RUNS AWAY
//   (measured 271%/fold, network collapses to 18%).
// A cell that overshot to 3*theta fires once, KEEPS 2*theta, and fires again next sweep.
// ** A BURST BECOMES A TRAIN.  Delay the boost; never drop it. **
//
// ── THERE IS NO CHARGE PUMP INTO Cw ──────────────────────────────────────────
// A FIRE IS A ±1 INCREMENT ON THE 8-BIT W-SRAM CODE.  The existing weight_dac + refresh path
// repaints W_cap through the pre-distortion LUT (wgt_lut).  Consequences:
//   * W-SRAM is ALWAYS the current master  => refresh NEVER has to be disabled
//   * `save` is not needed DURING training => the absorb->save->sync->refresh cycle collapses
//   * the W-cap leak risk goes away STRUCTURALLY, not by scheduling
//
// ⚠ THE FIRE PULSE WIDTH IS PART OF THE ANALOG SPEC.  The residue subtraction is a PULSED
//   CURRENT SOURCE: Q = I * t (500 nA x 10 ns = 5 fC = 50 mV on Ce = 100 fF).  So `jug_pulse`
//   must be a FIXED-WIDTH one-shot, NOT the whole sweep strobe.  Gating the current source with
//   a 100 ns strobe instead of a 10 ns one-shot would remove 10x the intended charge.
//   PULSE_CYC sets it.  See ../circuit/jug_compare.spice.
//
// Per element the FSM runs:  CMP -> (RMW the W-SRAM code) -> FIRE(pulse) -> next
// =============================================================================

module jug_ctrl #(
    parameter N_CELLS   = 4,
    parameter N_ELEMS   = 256,
    parameter WGT_MIN   = 71,        // code rail (the analog cell's usable range)
    parameter WGT_MAX   = 192,
    parameter PULSE_CYC = 1          // fire one-shot width, in clocks. Q = I*t — SEE ABOVE.
) (
    input  wire        clk,
    input  wire        rst_n,

    input  wire        start_wb,     // WB reg: start a sweep
    input  wire        sweep_n_in,   // from router (active-low), edge-detected

    output reg         busy,
    output reg         done,         // one-cycle pulse

    // ── cap_array COMPARE port (the shared, swept comparator) ────────────────
    output reg         cmp_en,
    output reg   [1:0] cmp_cell,
    output reg   [7:0] cmp_addr,
    input  wire        fire_up,      // |E| >= theta  and  E > 0
    input  wire        fire_dn,      // |E| >= theta  and  E < 0

    // ── cap_array RESIDUE-SUBTRACT port (the pulsed current source) ──────────
    output reg         jug_pulse,    // FIXED-WIDTH one-shot. Q = I*t.
    output reg         jug_dn,       // 0 = subtract theta (fired +), 1 = add theta (fired -)

    // ── W-SRAM read-modify-write: a fire is a ±1 CODE INCREMENT ──────────────
    output reg   [9:0] sram_addr,
    output reg         sram_re,
    input  wire  [7:0] sram_rdata,
    output reg         sram_we,
    output reg   [7:0] sram_wdata
);
    localparam TOTAL = N_CELLS * N_ELEMS;   // 1024

    reg [9:0] cnt;
    reg [2:0] pulse_cnt;
    reg       fire_up_r, fire_dn_r;

    reg sweep_n_r;
    wire sweep_fall = sweep_n_r & ~sweep_n_in;

    // ⚠ RDW is a WAIT STATE for the W-SRAM read latency. Without it, WR uses sram_rdata
    //   one cycle early and increments GARBAGE. (Caught by tb_jug_fire — but only after
    //   the assertion was made x-safe: `if (x != 118)` is FALSE in Verilog, so the test
    //   PASSED on an `x`. A test that cannot fail is not a test. Use `!==`.)
    localparam IDLE = 3'd0, CMP = 3'd1, RD = 3'd2, RDW = 3'd5, WR = 3'd3, FIRE = 3'd4;
    reg [2:0] state;

    // ★ the ±1 code increment, CLAMPED to the analog cell's rails.
    // The cell physically cannot go beyond [WGT_MIN, WGT_MAX], so a fire AT a rail leaves the
    // code where it is. But θ IS STILL SUBTRACTED from Ce (jug_pulse fires unconditionally in
    // WR) — this is ANTI-WINDUP: if the charge kept accumulating at a saturated rail, E would
    // wind up to E_MAX (3·θ) and its SIGN would be destroyed. Draining θ each sweep keeps E
    // hovering near ±θ, so when the gradient reverses the cell moves off the rail after only
    // ~1–2 sweeps instead of first discharging a full windup.
    // ★ THIS MATCHES THE VALIDATED SIM: pcn_jug.py clips Wm at [WMIN,WMAX] (line 804) but does
    //   `E = e - s·th` UNCONDITIONALLY (line 807). tb_absorb_match asserts it explicitly.
    wire [7:0] code_up = (sram_rdata >= WGT_MAX) ? WGT_MAX[7:0] : (sram_rdata + 8'd1);
    wire [7:0] code_dn = (sram_rdata <= WGT_MIN) ? WGT_MIN[7:0] : (sram_rdata - 8'd1);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= IDLE;  busy   <= 1'b0;  done   <= 1'b0;
            cmp_en    <= 1'b0;  cnt    <= 10'd0; sweep_n_r <= 1'b1;
            jug_pulse <= 1'b0;  jug_dn <= 1'b0;  pulse_cnt <= 3'd0;
            sram_re   <= 1'b0;  sram_we <= 1'b0;
        end else begin
            sweep_n_r <= sweep_n_in;
            done      <= 1'b0;
            cmp_en    <= 1'b0;
            sram_re   <= 1'b0;
            sram_we   <= 1'b0;

            case (state)
                IDLE: if (start_wb | sweep_fall) begin
                    cnt   <= 10'd0;
                    busy  <= 1'b1;
                    state <= CMP;
                end

                // ── COMPARE: the shared comparator visits this element ───────
                CMP: begin
                    cmp_en   <= 1'b1;
                    cmp_cell <= cnt[9:8];
                    cmp_addr <= cnt[7:0];
                    state    <= RD;
                end

                // ── latch the comparator, and read the W-SRAM code ───────────
                RD: begin
                    fire_up_r <= fire_up;
                    fire_dn_r <= fire_dn;
                    if (fire_up | fire_dn) begin
                        sram_addr <= cnt;
                        sram_re   <= 1'b1;
                        state     <= RDW;      // ⚠ wait for the SRAM read (see RDW)
                    end else begin
                        // NO FIRE: nothing happens. The charge STAYS on Ce.
                        if (cnt == TOTAL - 1) begin
                            busy <= 1'b0;  done <= 1'b1;  state <= IDLE;
                        end else begin
                            cnt <= cnt + 10'd1;  state <= CMP;
                        end
                    end
                end

                // ── WAIT: the W-SRAM read has 1-cycle latency ───────────────
                RDW: begin
                    sram_addr <= cnt;          // hold the address
                    state     <= WR;
                end

                // ── ±1 CODE INCREMENT, written back to W-SRAM ───────────────
                // ** THIS IS THE WEIGHT WRITE. There is NO charge pump into Cw. **
                // refresh_ctrl repaints W_cap from W-SRAM through wgt_lut, continuously.
                WR: begin
                    sram_addr  <= cnt;
                    sram_we    <= 1'b1;
                    sram_wdata <= fire_up_r ? code_up : code_dn;
                    // and fire the one-shot at the residue subtractor
                    jug_pulse  <= 1'b1;
                    jug_dn     <= fire_dn_r;
                    pulse_cnt  <= PULSE_CYC[2:0];
                    state      <= FIRE;
                end

                // ── the FIXED-WIDTH fire pulse (Q = I*t — the analog spec) ───
                FIRE: begin
                    if (pulse_cnt > 3'd1) begin
                        pulse_cnt <= pulse_cnt - 3'd1;
                        jug_pulse <= 1'b1;
                    end else begin
                        jug_pulse <= 1'b0;
                        if (cnt == TOTAL - 1) begin
                            busy <= 1'b0;  done <= 1'b1;  state <= IDLE;
                        end else begin
                            cnt <= cnt + 10'd1;  state <= CMP;
                        end
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end
endmodule
`default_nettype wire
