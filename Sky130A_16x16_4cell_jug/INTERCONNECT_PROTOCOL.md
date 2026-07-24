# PCN Interconnect — transpose-at-source revision of the δ plane

**Status:** design spec (2026-07-15). Feeds item 3 (relocate `router_backproj` into the PCN top).

**This is NOT a new protocol.** It is a **targeted revision of `../ROUTER_PROTOCOL.md`
§"Backprojection (δ) Plane Protocol"** for the transpose-at-source architecture. Everything else in
`ROUTER_PROTOCOL.md` stands unchanged: the physical layer (SPI+WB per chip, ring between routers,
`ABSORB_N` broadcast), the packet fabric, `frame_seq`, the forward plane, discovery, dormancy. Read
that doc first; this changes only the four things below.

See also: `sim/SKEW_RESULTS.md` (the experiment that justifies the freshness window), and
`../../memory` → `project_pcnchip_transpose_at_source.md`.

---

## What transpose-at-source changes (vs `ROUTER_PROTOCOL.md` §Backprojection)

The old δ plane was **compute-and-gather in the router**: the router held a `SHADOW_W` copy of every
local chip's forward weights, ran `Wᵀ·δ` on that copy, and used a **per-hop wait-for-all barrier**
before `avg_bp`. Transpose-at-source changes exactly four things:

| # | old (`ROUTER_PROTOCOL.md` §Backprojection) | transpose-at-source |
|---|---|---|
| **C1** | router holds `SHADOW_W[chip][1024]` + `SHADOW_W_SYNC` (re-sync after every absorb) | **DELETED.** Weights never leave the chip. No weight message class exists. |
| **C2** | `Wᵀ·δ` (+ per-block RMS renorm) runs **in the router** on the shadow | runs **on the PCN chip**, reading its **own live W-SRAM** (`router_backproj` relocated into `pcn_digital_top_jug`) |
| **C3** | **per-hop barrier**: destination router waits for *all* partial sums, then `avg_bp` | **fixed-divide on fold-close**: divide by nominal `BP_FANIN` when the fold's `frame_seq` closes; ship what arrived. No wait-for-all. |
| **C4** | "PCN chip does not parse δ packets"; δ arrives only as `delta_flat` for E-inject | the chip **also** consumes δ to drive its transpose and **emits a δ_partial** back to the router |

Unchanged and reused as-is: `PLANE=1` δ packet format, `frame_seq`, `BP_DEST_CHIP_ID/CELL_ID`,
`BP_FANIN`, `avg_bp` (integer ÷ fan-in), per-source-block RMS renorm, fire-and-forget δ (a dropped
partial contributes 0), and the two-hop schedule (L3→L2, L2→L1; L1 fan-in = 1, no averaging).

---

## The revised per-hop sequence

One hop = one backprojection layer. **The transpose moves from the router to the source chip; the
router keeps only the gather.**

```
1. SEED    Router WRITES the hop's source δ into each source CHIP's δ register  (MOSI)
           — reuses the existing DELTA_IDX/DELTA_DATA path (0x7C/0x80).
2. TRIG    Router pulses the chip's transpose-start  (new chip-side register).
3. COMPUTE Chip computes  δ_partial = RMS_renorm_per_block( Wᵀ · δ )  from its OWN W-SRAM.
           ** Same incoming δ ALSO drives E-inject (start_e_update, 0x84) — δ is used twice:
              relay (transpose) AND absorb (into E). **
4. READ    Router READS δ_partial back from the chip  (MISO)  — new chip-side partial-out register.
5. GATHER  Router ADDS δ_partial (local + ring PLANE=1 packets) into gather[dest]. Order-irrelevant.
6. DIVIDE  On fold-close (frame_seq advances): δ_dest = gather[dest] / BP_FANIN[dest].  ** FIXED-DIVIDE:
           divide by NOMINAL fan-in and ship — NO wait-for-all barrier. A missing partial just
           under-drives δ_dest (jug-tolerant). **
7. DELIVER δ_dest becomes the next hop's source δ (→ step 1 for the layer below).
```

Router state is now just `gather[dest]` accumulators + `BP_FANIN` — **no weights**. Its RAM shrinks by
the whole `SHADOW_W` (8b×1024×ports).

---

## Fold tag = `frame_seq`; the freshness window

The fold tag is the **existing `frame_seq`** (`ROUTER_PROTOCOL.md` byte 0 [3:0], wraps at 16) — no new
field. It does two jobs:
1. **Grouping (correctness):** `gather[dest]` accumulates only same-`frame_seq` partials, so fold N and
   N+1 never blend.
2. **Skew bound (robustness):** it caps how many folds of backward state are in flight, which caps the
   forward↔transpose weight-skew.

**Freshness window = configurable, DEFAULT 4 folds** (a new `BP_FRESH_WINDOW` register, θ-like; the
boss sets it once). Policy:
- **Apply-late, do NOT drop.** A dropped δ loses a gradient; a late δ costs only the (tiny, bounded)
  skew. So the window bounds **buffer depth + grouping**, it is not a hard freshness gate. Drop only on
  genuine accumulator overflow (real backpressure).
- Default loose because the skew headroom is large (below) and the fabric is **heterogeneous**
  (Sky130A ↔ 28nm nodes at different rates; a tight window penalises the slow node every fold).

`frame_seq` wraps at 16, so any window ≤ 15 fits the existing field.

### Why loose is safe — the skew experiment (`sim/SKEW_RESULTS.md`)

| skew (folds) | 0 | 1 | 2 | 4 | 8 | 8 (rand) |
|---|---|---|---|---|---|---|
| accuracy (ep2/ch4000) | 67.47 | **67.47** | 66.33 | 65.75 | 66.34 | 67.13 |
| accuracy (ep3/ch8000) | 72.43 | **72.43** | 71.00 | — | — | — |

**k=1 is free** (identical, at two convergence levels), and the penalty is **bounded and does not grow**
(k=2/4/8 all within the ~0.7pp noise floor of each other; k=8 uncorrelated = −0.34pp). So even a window
of 8 costs ~1–2pp worst case; the default 4 is comfortably inside the free-to-cheap regime while giving
generous timing slack.

---

## The reliability split (unchanged in spirit, stated explicitly)

- **Forward activations: RELIABLE.** WB `ack` on the local link; ring framing/retransmit. Never dropped
  (the forward needs the complete input vector).
- **Backward δ / partials: BEST-EFFORT.** Fire-and-forget into accumulators (exactly as the old spec
  already had it — "a dropped partial contributes 0"). Justified by the jug: a lost/stale δ under-charges
  E slightly and self-corrects over folds.
- **boss_h: last-one-only** register (newest wins). **Config** (weights init, `JUG_THETA` 0x9C, ADC gain,
  `WGT_LUT` 0x94/0x98, `ROUTING_MX`/`BP_*` tables, `my_chip_id`): reliable, one-time boss phase.

The change from the old spec is **C3 only**: fixed-divide removes the *wait* (the barrier), not the
loss-tolerance. Dropping was already accepted; we now also stop synchronising on completion.

---

## Register delta (→ item 3)

**Router chip — REMOVE:** `SHADOW_W[chip][1024]`, `SHADOW_W_SYNC`. **KEEP:** `BP_DEST_CHIP_ID/CELL_ID`,
`BP_FANIN`, `gather` accumulators, `BP_ACTIVE_MASK`. **ADD:** `BP_FRESH_WINDOW` (fold window, default 4).

**PCN chip — ADD:** a transpose-start trigger and a δ_partial read-back register (the on-chip analogue of
`router_node`'s `TRIG`/`DST`), plus the relocated transpose datapath (`router_backproj` in the top,
reading W-SRAM through `pcn_weight_params.vh`'s `PCN_WGT_ZERO`). **REUSE:** `DELTA_IDX`/`DELTA_DATA`
(0x7C/0x80) for δ-in; `start_e_update` (0x84) for the E-inject use of the same δ.

---

## Invariants (for the implementer)

1. **Weights never leave the chip.** No weight message class; `SHADOW_W` is deleted.
2. **Forward reliable; backward best-effort.** Never drop FWD; drop BWD only on overflow.
3. **Fixed-divide on fold-close** — divide by nominal `BP_FANIN`, never wait-for-all.
4. **`frame_seq` groups; `BP_FRESH_WINDOW` bounds buffering, not correctness** — apply-late over drop.
5. **The `ABSORB_N` sweep barrier holds a chip's forward and its own transpose in one weight-epoch** —
   do not run a chip's next-fold sweep ahead of its current-fold backward. Cross-chip skew is then ≤
   window folds, which `SKEW_RESULTS.md` shows is safe.
