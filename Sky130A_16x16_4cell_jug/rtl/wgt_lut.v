`default_nettype none
// =============================================================================
// wgt_lut — THE PRE-DISTORTION LUT.  code -> DAC drive, so the WEIGHT is linear.
// =============================================================================
//
// ── THE PROBLEM ──────────────────────────────────────────────────────────────
// `weight_dac` maps code -> V_w **LINEARLY** (that is the hardware).
// The CELL maps V_w -> weight **SIGMOIDALLY** (that is the physics).
// Compose them and the WEIGHT IS NOT LINEAR IN THE CODE — which every simulation in this
// project has assumed since the beginning.
//
// SPICE-measured (mac_cell_jug, MN3_w=2):  dW/dcode is ~0.05 uA/V at the rails and ~1.42 in
// the middle — a **28x SENSITIVITY VARIATION**.  The jug's whole mechanism is
// "ONE FIRE = ONE CODE", so a fire would be worth 28x more weight in the middle of the range
// than at the rails.  Measured cost of the raw sigmoid: **-1.45pp**.
//
// (At the INHERITED MN3_w=10 it was far worse than non-linear: gm PEAKED at V_w ~ 0.90 V and
//  FELL above it, WGT_ZERO sat ON the peak, and codes 119..192 — THE ENTIRE POSITIVE WEIGHT
//  RANGE — were dead or INVERTED.  See ../circuit/THE_WEIGHT_IS_NOT_LINEAR.md.)
//
// ── THE FIX ──────────────────────────────────────────────────────────────────
// PRE-DISTORT THE DAC.  Choose V_w per code so the WEIGHT steps are uniform.  The composition
// (LUT -> linear DAC -> sigmoidal cell) is then LINEAR, and the sim's assumption becomes TRUE
// OF THE SILICON rather than a convenient fiction.
//
// Linearising the WHOLE sigmoid is impossible (the rails would need ~74 mV of V_w per code,
// far beyond the 850 mV budget).  But it does not have to be:
//
//     uniform +-28 units (88% of the sigmoid's span)  needs only 427 mV of the 850 mV budget,
//     with FULL 121-code resolution.
//
// MEASURED (pcn_jug.py):
//     linear DAC  => sigmoid weight, full range (theta=16, its best)   80.51%
//     ** PRE-DISTORTED DAC => LINEAR weight, +-0.83 **                 81.46%
//     (the old, UNACHIEVABLE linear assumption, full +-0.95 rail)      81.96%
// ** 81.46 vs 81.96 is INSIDE the 0.7pp noise floor. The LUT recovers essentially everything. **
//
// ── ★ AND IT IS LOADABLE, WHICH MAKES IT A PER-DIE CALIBRATION ───────────────
// The sigmoid is a property of the CELL AND THE PROCESS.  A hard-wired non-linear resistor
// ladder could not track process variation.  A LOADABLE LUT can: measure the cell's curve on
// each die, invert it, write the table.  ** The LUT is not just a fix — it is the calibration
// hook for the whole weight path. **
//
// ── WIDTH ────────────────────────────────────────────────────────────────────
// 8-bit stored code -> DAC_W-bit DAC drive.  The DAC must be WIDER than the code so the LUT has
// somewhere to put the pre-distortion: uniform WEIGHT steps need NON-uniform V_w steps (small in
// the steep region, large at the rails).  10 bits gives 4x the V_w resolution — ample.
// ⚠ THE ANALOG weight_dac MUST THEREFORE BE 10-BIT, not 8. That is a SPICE change.
//
// Reset default = IDENTITY (code << (DAC_W-8)), i.e. the old LINEAR behaviour, so an
// uncalibrated part still runs (badly). The boss MUST load the real table.
// =============================================================================

module wgt_lut #(
    parameter DAC_W = 10        // ⚠ the analog weight_dac must be DAC_W bits (was 8)
) (
    input  wire              clk,
    input  wire              rst_n,

    // ── lookup (combinational) ────────────────────────────────────────────
    input  wire [7:0]        code,        // the 8-bit weight code from W-SRAM
    output wire [DAC_W-1:0]  dac_drive,   // the pre-distorted DAC input

    // ── load port (WB): the boss writes the calibrated table ──────────────
    input  wire              lut_we,
    input  wire [7:0]        lut_addr,
    input  wire [DAC_W-1:0]  lut_wdata,

    output wire              lut_loaded   // 0 = still the identity default (UNCALIBRATED)
);
    reg [DAC_W-1:0] tbl [0:255];
    reg             loaded;
    integer i;

    assign dac_drive  = tbl[code];
    assign lut_loaded = loaded;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // IDENTITY default = the OLD linear behaviour. An uncalibrated part still runs,
            // but with the sigmoidal weight (-1.45pp) — and at MN3_w=10 it would be far worse.
            // ⚠ `lut_loaded` stays LOW until the boss writes the table. CHECK IT.
            for (i = 0; i < 256; i = i + 1)
                tbl[i] <= i[7:0] << (DAC_W - 8);
            loaded <= 1'b0;
        end else if (lut_we) begin
            tbl[lut_addr] <= lut_wdata;
            loaded        <= 1'b1;
        end
    end
endmodule
`default_nettype wire
