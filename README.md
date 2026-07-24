# A Forwards-Only, Asynchronous Analog Architecture for Predictive-Coding Networks

**Local transpose and sigma–delta weight update for on-die supervised learning.**

Design files, RTL, SPICE netlists and simulation scripts for the **W&E** paper (Part II of this
study). The paper itself is in [`paper/`](paper/).

> **All results are pre-silicon.** The evidence here is Python behavioural simulation, Sky130A SPICE,
> and a bit-faithful RTL implementation. Nothing has been fabricated.

---

## What this is

An analog predictive-coding substrate that trains **in place** under three constraints usually
treated as obstacles: no per-device calibration, no reverse analog signal path, and no global clock.
The organisation the paper calls **W&E** keeps the weights `W` and the error store `E` in separate
populations linked along a one-way path:

- **Transpose-at-source** — `Wᵀδ` is computed on the chip that owns `W`, so no weight ever leaves the
  die and the router holds no weights.
- **The leaky jug** — the error store is a leaking capacitor plus a shared threshold comparator,
  forming a first-order sigma–delta modulator whose cumulative quantisation error is bounded
  *independently of the number of updates*. This is what lets a per-update step of 0.019 weight LSB
  drive a coarse 8-bit analog cell with **no per-synapse digital accumulator**.

On EMNIST letters with a 48-chip topology the forwards-only rule reaches **82.50%** against a
full-backpropagation ceiling of **82.85%** on identical topology.

---

## Repository structure

```
.
├── paper/                          The paper, its sources, and its build script
├── multi_array_level3_BIGspec/     Float rig — the learning rule (Table 1)
├── multi_array_level3_backprop/    Backprop decomposition on the small topology
├── hw_buildcheck/                  Ablation ladder — how the gap was closed (Sec. 7)
├── shared/sim/                     EMNIST loader and quantiser shared by the rigs
└── Sky130A_16x16_4cell_jug/        THE DESIGN — sim, RTL and SPICE for the jug chip
    ├── sim/                        Bit-faithful model (Tables 2 and 3)
    ├── rtl/                        Synthesisable Verilog + 17 test benches
    └── circuit/                    Sky130A SPICE netlists and measured output
```

The top-level directory names are **load-bearing**. The simulation scripts resolve each other by
relative path (`pcn_jug.py` reaches `../../multi_array_level3_BIGspec` for the float reference, which
in turn reaches `../shared/sim` for the data loader). Renaming or flattening these directories will
break the import chain.

### `paper/` — the paper

| file | role |
|---|---|
| `main_stage2_v4.pdf` | **the paper** |
| `main_stage2_v4.tex` | LaTeX source of the PDF |
| `main_stage2_v4.md` | editable prose twin in markdown |
| `main_stage2_v3.tex` | figure/equation source; `build_v4.py` splices the TikZ figures and numbered equations from here so numbering stays correct |
| `refs.bib` | bibliography |
| `build_v4.py` | assembles `v4.tex` from `v4.md` (prose) + `v3.tex` (figures, equations, preamble) |
| `mkmd.py` | regenerates the markdown twin from the `.tex` |

Rebuild:

```bash
cd paper
python3 build_v4.py          # writes main_stage2_v4.tex
pdflatex main_stage2_v4 && bibtex main_stage2_v4
pdflatex main_stage2_v4 && pdflatex main_stage2_v4
```

Requires `pandoc` (tested on 3.1.3) and a TeX distribution with `IEEEtran`.

### `multi_array_level3_BIGspec/` — the float rig (Table 1)

The learning-rule evidence, in floating point, on the BIG topology: 1152 split-sign input features →
384 → 128 → 256 across 48 chips, EMNIST letters.

| file | produces |
|---|---|
| `pcn_bigspec.py` | the forwards-only rule — **82.50%** with all hardware constraints, 82.89% with unconstrained error broadcast |
| `backprop_rig_big.py` | the backpropagation ceilings — **82.85%** chip-factored, **89.48%** dense |
| `big_spec.log`, `big_unconstrained.log`, `linbase_big.log` | the recorded runs, including the 77.14% linear baseline |

`pcn_bigspec.py` is also imported as a module by the bit-faithful rigs, which take their topology,
weight-code constants and EMNIST features from it. It is the single definition of the network.

### `Sky130A_16x16_4cell_jug/sim/` — the bit-faithful rig (Tables 2 and 3)

Quantises the weight cell, the activations, the error path and the update, and carries the jug.

| file | produces |
|---|---|
| `pcn_jug.py` | **81.96%** clean, against an 82.13% float ceiling for the same model; all of Table 2's robustness arms |
| `pcn_jug_skew.py` | Table 3 — forward/transpose weight skew |
| `jug_physical_spec.py` | translates the model's jug parameters into circuit quantities |
| `RESULTS.md`, `SKEW_RESULTS.md` | the recorded evidence, with the noise-floor caveat stated |
| `run_skew_focused.sh`, `run_skew_escalate.sh` | the skew sweeps as run |
| `*.log` | raw run logs |

**Read `RESULTS.md` before quoting any number from this rig.** The run-to-run noise band is ≈0.7 pp
and is *not* only seed variation — the float matmuls and the classifier fit are BLAS-dependent, so
the same configuration at a different thread count sums in a different order. Differences below
about 1 pp are not results.

Reproduction (defaults are the working design point: `--jug_theta 8`, `--jug_leak 0`, `--agc rms`):

```bash
cd Sky130A_16x16_4cell_jug/sim
python3 pcn_jug.py                              # clean devices        -> ~81.96%
python3 pcn_jug.py --float_ref                  # float ceiling        -> ~82.13%
python3 pcn_jug.py --jug_sign_err 0.20          # comparator wrong 20% -> ~81.62%
python3 pcn_jug.py --jug_theta_spread 2.0       # theta mismatch x2    -> ~81.20%
python3 pcn_jug.py --jug_tau 100 --jug_tau_spread 3.0    # leakage     -> ~81.53%
python3 pcn_jug.py --jug_e_fwd 1.0              # counter-test: E into the MAC (degrades)
python3 pcn_jug.py --jug_multifire              # counter-test: per-synapse comparator (collapses)
./run_skew_focused.sh                           # Table 3

# with the SPICE-measured, non-linear code->weight map instead of the linear
# assumption (Sec. 3.2)                          -> ~81.46%
python3 pcn_jug.py --w_map ../circuit/output/w_map.csv
```

### `hw_buildcheck/` — the ablation ladder

The paper reports (Sec. 7) that the hardware gap was closed **by elimination rather than
hypothesis**: every quantiser was ablated individually and each was innocent, which is what
identified the update path as the one component never isolated. This directory is that record.

| file | role |
|---|---|
| `pcn_hw.py` | the ablation rig; `--agc rms` reproduces the **75.25%** naive-write baseline that the jug improves on |
| `THE_JUG.md` | how the jug was arrived at |
| `HW_BUILD_CHECK.md` | the buildability audit |
| `RTL_RECONCILIATION.md` | model-versus-RTL reconciliation |
| `abl_*.log`, `fb_*.log`, `r2_*.log`, `hw_*.log` | the individual ablations |

### `Sky130A_16x16_4cell_jug/rtl/` — bit-faithful RTL

Synthesisable Verilog with a full regression. **All 17 test benches elaborate and pass:**

```bash
cd Sky130A_16x16_4cell_jug/rtl
./run_all_tb.sh              # 17 pass, 0 fail
./run_all_tb.sh tb_jug       # substring filter
```

Requires `iverilog` (`-g2012`). The directory is self-contained — the runner uses `-y .`, so no
external library path is needed.

New or changed for the jug design:

| module | role |
|---|---|
| `jug_ctrl.v` | the swept comparator, the ±1 weight-code increment, the fire one-shot |
| `wgt_lut.v` | the pre-distortion LUT (code → 10-bit DAC drive); loadable, so it is also the per-die calibration hook |
| `cap_array.v` | MAC is **W only** — `Ce` is out of the signal path; absorb subtracts θ rather than discharging |
| `pcn_transpose.v` | transpose-at-source; reproduces the model within ±1 LSB |
| `refresh_ctrl.v` | continuous refresh — the absorb→save→sync coherence cycle collapses |

`RTL_STATUS.md` records what changed against the earlier design and what was ported unchanged.

A note on the regression's design, since it is deliberate: a test bench counts as **failed** if it
does not elaborate, if its output contains a failure marker, *or if it never prints a pass marker*.
Silence is a failure. This runner was written after an earlier claim that "every block test bench
passes" turned out to rest on test benches that could not fail.

### `Sky130A_16x16_4cell_jug/circuit/` — Sky130A SPICE

The two transistor-level results the paper's central claims rest on, plus the weight-cell
characterisation.

| file | claim it supports |
|---|---|
| `tb_jug_fire.spice`, `output/jug_fire.csv` | the residue subtraction removes a **fixed charge** `Q = It` independent of capacitor voltage — the design's single tight analog specification |
| `tb_jug_comparator.spice`, `jug_compare.spice`, `output/jug_cmp_dc.csv` | the window comparator settles within 0.75 mV, dead zone ≈0.2 µV, in 2.5 ns — the component the theory *allows* to be loose is in fact precise |
| `sweep_mn3w.spice`, `analyse_weight.py`, `output/w_transfer.csv`, `output/w_map.csv` | the weight cell is **non-linear and non-monotonic** above mid-range; sizing plus a pre-distortion table restores monotonicity |
| `weight_dac_10b.spice`, `tb_weight_dac_10b.spice`, `output/dac10b.csv` | the pre-distorted weight DAC |
| `tb_wgt_zero_jug.spice`, `analyse_wgt_zero.py`, `output/wgt_zero_jug.csv` | the zero-weight operating point (`WGT_ZERO` 132 → 117) |
| `THE_WEIGHT_IS_NOT_LINEAR.md` | the finding, and why a single-operating-point check could not see it |
| `SPICE_RESULTS.md`, `PHYSICAL_SPEC.md` | measured results and the circuit specification |

`THE_WEIGHT_IS_NOT_LINEAR.md` is worth reading even if you skip the rest: the effective weight peaks
near mid-range and *falls* above it, so the entire positive weight range was compressed and, at the
extreme, inverted. Every test bench that pinned the weight voltage to a single healthy value passed
regardless. **Characterise the range, not the operating point.**

`output/jug_fire.csv` is the largest file in the repository (≈9.7 MB); it is the raw transient trace
behind the fixed-charge claim.

### Protocol documents

`DESIGN.md` is the design record for the jug chip. `INTERCONNECT_PROTOCOL.md` and
`ROUTER_PROTOCOL.md` specify the inter-chip link, including packets, the state machine, dormancy,
discovery, registers, electrical and timing. As the paper's limitations state, this protocol is
**specified but not yet realised** across the heterogeneous process nodes the architecture
anticipates.

---

## Where each headline number comes from

| paper | claim | produced by |
|---|---|---|
| Table 1 | 82.50% forwards-only, all hardware constraints | `multi_array_level3_BIGspec/pcn_bigspec.py` |
| Table 1 | 82.85% backprop, chip-factored / 89.48% dense | `multi_array_level3_BIGspec/backprop_rig_big.py` |
| Table 1 | 77.14% linear baseline on 1152 features | `multi_array_level3_BIGspec/linbase_big.log` |
| Table 1 | 64.09% prior fold/absorb rule | recorded in `Sky130A_16x16_4cell_jug/sim/RESULTS.md` |
| Sec. 6.3 | 81.96% bit-faithful, 82.13% float ceiling | `Sky130A_16x16_4cell_jug/sim/pcn_jug.py` |
| Sec. 6.3 | 75.25% — the same rule written naively to the 8-bit cell | `hw_buildcheck/pcn_hw.py` (`--agc rms`); see `HW_BUILD_CHECK.md` |
| Sec. 3.2 | 81.46% with the measured weight curve and pre-distortion | `pcn_jug.py --w_map ../circuit/output/w_map.csv`; see `circuit/THE_WEIGHT_IS_NOT_LINEAR.md` |
| Table 2 | leakage, θ mismatch, comparator sign errors | `Sky130A_16x16_4cell_jug/sim/pcn_jug.py` (flags above) |
| Table 3 | forward/transpose weight skew | `Sky130A_16x16_4cell_jug/sim/pcn_jug_skew.py` |
| Sec. 6.5 | fixed-charge pulse; comparator margin | `Sky130A_16x16_4cell_jug/circuit/` |
| Sec. 6.5 | transpose within ±1 LSB, residue within 2 µV | `Sky130A_16x16_4cell_jug/rtl/` |
| Sec. 7 | every quantiser innocent under ablation | `hw_buildcheck/` |

---

## Data and caches

The EMNIST feature caches and the trained BIG weights are **not in this repository** — together they
are roughly 645 MB. They are regenerated, not downloaded:

- `multi_array_level3_BIGspec/l0_cache_emnist/` — the L0 (uncentered-PCA) features, built on first
  run from the EMNIST letters CSV.
- `multi_array_level3/weights_big_emnist/` — the trained BIG weights; the directory is created
  automatically.

You will need the **EMNIST letters** dataset (Cohen et al., 2017) in CSV form. `shared/sim/pcn_mnist.py`
loads it and honours the `DATASET` environment variable (`DATASET=emnist_letters`).

Expect the first run to spend significant time building the L0 cache before training begins.

---

## Requirements

| for | needs |
|---|---|
| Python rigs | Python 3, NumPy; `backprop_rig*.py` additionally needs PyTorch |
| RTL regression | `iverilog` with `-g2012` |
| SPICE | `ngspice` and the **Sky130A** PDK |
| Paper rebuild | `pandoc` (3.1.3 tested), a TeX distribution with `IEEEtran` |

---

## Provenance and honesty notes

Files are copied **verbatim** from the working tree. Some of the copied `.md` records therefore
contain relative cross-references written against the original layout; the directory names preserved
here resolve most of them, and the table above is authoritative for which file produces which number.

The design records in this repository are working documents, not marketing. They contain retractions,
failed predictions and superseded conclusions, kept deliberately — `THE_WEIGHT_IS_NOT_LINEAR.md` and
`RESULTS.md` in particular record results that overturned earlier claims of ours. Where a document
disagrees with the paper, the paper is the considered version and the document is the trail.

The largest known accuracy deficit is **not** the learning rule but the chip factoring, which costs
6.6 pp against a dense network of the same widths. That is a question of width and connectivity, and
it is open.

---

## Licence

**Source-available, not open source.** Full detail in [`NOTICE.md`](NOTICE.md).

| you are | licence | what you may do |
|---|---|---|
| academic, public research, charity, government, or an individual on a noncommercial project | [PolyForm Noncommercial 1.0.0](LICENSE.md) | use, modify, redistribute — for noncommercial purposes |
| a commercial company | [PolyForm Free Trial 1.0.0](LICENSE-COMMERCIAL-EVALUATION.md) | **evaluate only**, under 32 consecutive days, no redistribution |
| a commercial company wanting more | contact the licensor | negotiated terms |

Both licence texts are unmodified canonical PolyForm texts. No downstream commercial use is possible
without permission: the Noncommercial licence permits redistribution only for noncommercial
purposes, so commercial rights cannot be acquired from an intermediary.

**The paper itself is not under these licences.** `paper/main_stage2_v4.pdf` and its sources are ©
Saul Dobney, all rights reserved, subject to normal academic citation and fair use. The Sky130A PDK,
EMNIST, and all third-party tooling are separately licensed by their own authors and are not
included here.

## Citation

Part I of this study https://github.com/dobneyresearch/PredictiveCodingNetworks_AnalogVLSIdesign presents the analog cell and its unsupervised learning; this repository
accompanies Part II. See `paper/refs.bib`.

Correspondence: `saul.dobney@dobney.com`

## Acknowledgements

Architectural direction and design decisions are the author's. The simulators, SPICE netlists and RTL
were implemented with [Claude Code](https://claude.com/claude-code) (Opus and Fable models), whose
contribution was the rate at which code and tests could be produced to explore the topology,
algorithm and parameter space and to characterise the physics.
