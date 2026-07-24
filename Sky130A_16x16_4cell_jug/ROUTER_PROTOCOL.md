# Router Inter-Chip SPI Protocol

## Core Principles

- Nothing finer than cell granularity crosses a chip boundary.
- PCN chips connect to a dedicated router chip via a single SPI link (SPI_ROUTER port).
- An optional second SPI port (SPI_PEER) supports direct chip-to-chip connection bypassing the router.
- Both ports use the same packet format; `dest_chip_id` determines delivery path.
- Topology is discovered automatically via HELLO. Logical connectivity (which chips send activations to which) is written by a controlling host via SET_REGISTER packets.
- Each router chip has exactly **8 PCN ports** and exactly **2 inter-router ports**.
- Routers connect in a **bidirectional ring**: each router connects to its left and right neighbour; the final router connects back to the first. Packets travel the shorter direction around the ring.
- Two inter-router ports provide ring redundancy: a single router or link failure causes traffic to reroute around the surviving arc without protocol changes.
- **chip_id is assigned by the router** at connection time. A PCN chip may present a remembered ID from a prior connection; the router confirms or reassigns. chip_id is never burned-in or derived from physical port position.
- **NOTIFY events are scoped to the local router.** They do not cross inter-router links unless the NOTIFY packet explicitly addresses a chip_id on another router (requiring the sender to know that chip's network ID).
- **DATA packets are fire-and-forget.** No acknowledgement, no retry. A dropped packet is superseded by the next TDM cycle's output. The network is a streaming compute fabric, not a lossless channel.
- **SPI_PEER links are invisible to the router.** Peer-connected chips are routed to via their normal PCN port; the router has no knowledge of the peer link.

---

## Packet Format

All packets share a 4-byte header. `packet_type` in header byte 0 determines the class.

### Common Header

```
Byte  Bits   Field           Description
0     [7:6]  packet_type     00=DATA  01=CTRL  10=HELLO  11=NOTIFY
      [5:4]  reserved        set to 00
      [3:0]  frame_seq       rolling counter, wraps at 16
1            dest_chip_id    target chip (0–255); 0xFF = broadcast
2            src_chip_id     source chip (0–255)
3            checksum        XOR of bytes 0–2 and all payload bytes
```

### DATA (packet_type = 00)

```
Header [0–3]
Byte 4:    [7:4] src_cell_id    source cell on src chip (0–15)
           [3:0] dest_cell_id   destination cell on dest chip (0–15)
Byte 5:    payload_length       number of activation bytes following (16 for Sky130A, 32 for 28nm)
Bytes 6…:  activations          payload_length × 8-bit activation values
```

payload_length allows the router to delimit packets without knowing the source
chip's cell geometry. A router receiving a DATA packet reads exactly
payload_length bytes after byte 5 and then expects the next packet header.
This is the sole mechanism for mixed-node (Sky130A + 28nm) networks to coexist
on the same ring without the router needing per-chip type tables.

Sky130A: 22 bytes total. 28nm: 38 bytes total.

**Inter-chip SenderID:** `src_chip_id` (byte 2) is the sender identity used by the
receiving chip's weight_fsm to select the correct weight page for this activation.
Each chip is the atomic unit of computation: it runs its full internal temporal
processing (all TDM cycles and in-chip virtual layer passes) and sends a single
consolidated output packet per cell. The receiving chip loads one weight page per
source chip, identified by `src_chip_id`. No additional sender-identity field is
needed. This design — chip as autonomous group — keeps the inter-chip protocol
simple and makes the network graph debuggable at chip granularity.

### CTRL (packet_type = 01)

```
Header [0–3]
Byte 4:    ctrl_type    0x01=ACK  0x02=NAK  0x10=SET_REGISTER  0x11=WAKE_REQUEST
Bytes 5…:  ctrl_payload (depends on ctrl_type)
```

SET_REGISTER payload: `[reg_id:8 | value:32]`

WAKE_REQUEST payload: `[requested_cell_mask:16]` — asks dormant chip to wake;
chip responds with CTRL/ACK when bias generators have settled.

ASSIGN_CHIP_ID payload: `[new_chip_id:8]` — host or router explicitly assigns
a chip_id outside the normal HELLO flow (used during fixed-topology startup).
Chip updates MY_CHIP_ID register and stores new_chip_id as remembered_id.

### HELLO (packet_type = 10)

HELLO is not forwarded beyond direct neighbours (dest_chip_id = 0xFF, one hop only).

**HELLO (hello_type = 0x00) — sent by PCN chip on power-up and every HELLO_INTERVAL:**
```
Header [0–3]  (src_chip_id = MY_CHIP_ID if already assigned, else 0xFF)
Byte 4:    hello_type = 0x00
Byte 5:    cell_count        cells on this chip
Byte 6:    cell_rows         rows per cell (16 or 32)
Byte 7:    wake_mode         0x00=push  0x01=pull  0x02=threshold
Byte 8:    remembered_id     previously assigned chip_id, or 0xFF if none
```

remembered_id = 0xFF: chip has no prior ID; router assigns from pool.
remembered_id = 0x00–0xFE: chip presents a remembered ID from a prior connection.
Router confirms the remembered_id if available, or assigns a new one if taken.

**HELLO_ACK (hello_type = 0x01) — sent by router in response:**
```
Header [0–3]  (dest_chip_id = newly assigned ID; src_chip_id = router's chip_id)
Byte 4:    hello_type = 0x01
Byte 5:    assigned_chip_id  the ID the chip should use (confirmed or newly assigned)
Byte 6:    router_id         which router is responding
Byte 7:    reserved
```

**REACHABILITY_AD (hello_type = 0x02) — sent by routers only, propagates around ring:**
```
Header [0–3]  (dest_chip_id = 0xFF; src_chip_id = advertising router's chip_id)
Byte 4:    hello_type = 0x02
Byte 5:    hop_count         hops this AD has already travelled
Byte 6:    chip_id_count     number of chip_ids in the list
Bytes 7…:  chip_id_list      chip_ids reachable via the advertising router
```

### NOTIFY (packet_type = 11)

```
Header [0–3]  (dest_chip_id = subscriber or 0xFF)
Byte 4:    notify_type    0x01=NEW_DEVICE  0x02=DEVICE_LOST  0x03=ROUTE_CHANGE
Byte 5:    event_chip_id  chip that triggered the event
Bytes 6–7: event_detail   (depends on notify_type)
```

Not forwarded beyond one router hop unless explicitly addressed.

---

## Protocol State Machine (PCN Chip)

### Transmit (TDM SPI slot)

```
For each set bit c in OUTPUT_CELL_MASK:
  1. Trigger ADC capture of cell c output
  2. Build DATA packet:
       dest_chip_id   = NEXT_HOP_CHIP_ID[c]
       src_chip_id    = MY_CHIP_ID
       src_cell_id    = c
       dest_cell_id   = DEST_CELL_ID[c]
       payload_length = cell_rows (16 or 32)
       activations    = ADC result
  3. Transmit on SPI_ROUTER (or SPI_PEER if dest is direct peer)
  4. Continue immediately — no ACK wait, no retry
```

### Receive (SPI_ROUTER or SPI_PEER)

```
1. Wait for packet header
2. Check packet_type:
   DATA:  if dest_chip_id = MY_CHIP_ID → deliver to dest_cell_id
          else → forward to router (store-and-forward)
   CTRL:  if WAKE_REQUEST → wake sequence; respond ACK when ready
          if SET_REGISTER → write register and ACK
   HELLO: respond HELLO_ACK; update topology table
   NOTIFY: deliver to local handler
3. For DATA delivery:
   a. Load activations into inp_dac for dest_cell_id
   b. Signal weight_fsm: senderID = src_chip_id
      (For in-chip senders the TDM slot index is already the senderID by timing;
      src_chip_id handles all inter-chip cases without a separate field.)
   c. weight_fsm loads weight page W^(senderID) and triggers MAC
   d. TDM FSM devotes one slot to this inter-chip activation, allowing analog settle
      before the next in-chip or inter-chip slot begins
```

---

## Chip-Level Dormancy

### Power Domains

**Always-on** (all states):
- SPI receiver front-end (listens on SPI_ROUTER and SPI_PEER)
- HELLO timer
- SRAM (static retention)
- Wake comparator

**Gated** (active only):
- TDM FSM and system clock
- Analog bias generators (Vbias_n, Vbias_p, VCM)
- SAR ADC, inp_dac
- MAC cell analog compute path

### Sleep Sequence

```
1. TDM FSM completes current cycle
2. ADC pass: all cell activations captured to SRAM
3. System clock gated; bias generators off
4. Always-on domain continues
```

### Wake Sequence

```
1. SPI receiver or HELLO timer asserts wake
2. Bias generators on; settle ~1–10 µs
3. Clock ungated; TDM FSM resumes at slot 0
4. If woken by DATA: load activations into dest cell(s)
5. If woken by WAKE_REQUEST: send CTRL/ACK; await DATA
```

### Wake Modes (WAKE_MODE register)

| Mode | Value | Wake condition |
|------|-------|----------------|
| push | 0x00 | Any DATA or WAKE_REQUEST packet addressed to this chip |
| pull | 0x01 | WAKE_REQUEST only; DATA packets buffered or discarded until requested |
| threshold | 0x02 | DATA packet where any activation > WAKE_THRESHOLD; else discard |

### Dormancy Trigger

LAST_TX_TIMER counts time since last packet transmitted. When LAST_TX_TIMER >
DORMANCY_TIMEOUT, sleep sequence initiates. In pull mode, LAST_RX_TIMER > 
DORMANCY_TIMEOUT also triggers sleep.

---

## Network Discovery (Topology Layer)

Topology is auto-populated by HELLO. No manual configuration of physical
connectivity is required.

### HELLO Exchange (on power-up and every HELLO_INTERVAL)

```
1. Broadcast HELLO on all physical ports (dest_chip_id = 0xFF)
2. Each direct neighbour responds with HELLO_ACK carrying its chip_id and capabilities
3. Add to topology table: port_n → chip_id_y
```

### REACHABILITY Advertisement (ring propagation)

```
1. Router R1 completes HELLO with its two ring neighbours
2. R1 sends REACHABILITY_AD on both ring ports:
      "I can reach chip_ids [list] via me, hop_count=1"
3. Neighbour R2 receives AD on port P:
      For each chip_id in list: add entry (next_hop=P, hop_count=1) if better than existing
      Re-advertise on the OTHER ring port only (split-horizon: never re-advertise
      back to the port the AD was received from)
4. Propagation continues around the ring; stops when hop_count > MAX_HOP_COUNT
```

When two paths exist to the same dest_chip_id (both directions around the ring),
the entry with the lower hop_count is used. On equal hop_count, left port wins
(tie-breaking rule; direction is consistent per ROUTER_ID assignment).

Split-horizon prevents routing loops: a router never advertises a route back
through the neighbour it learned that route from. Combined with MAX_HOP_COUNT,
this guarantees convergence.

**Fault recovery:** when a router or link fails, TTL expiry removes affected
entries. The surviving ring arc re-advertises reachability through the intact
direction. No protocol change is needed; the existing TTL mechanism handles it.

### Routing Table Entry

| Field | Width | Description |
|-------|-------|-------------|
| dest_chip_id | 8 | Destination chip |
| next_hop_port | 4 | Physical SPI port toward destination (left or right ring port, or direct PCN port) |
| hop_count | 4 | Distance in hops |
| ttl | 8 | Seconds until entry expires (reset on HELLO refresh) |

Entry expiry triggers NOTIFY/DEVICE_LOST to subscribed chips.

---

## Network Connectivity (Logical Layer)

Topology (§Network Discovery) and connectivity are separate concerns:

- **Topology**: can a packet reach chip 47? (auto-discovered via HELLO)
- **Connectivity**: should chip 12 cell 2 send activations to chip 47 cell 0? (host-configured)

Logical connections are written by a controlling host or router via SET_REGISTER
CTRL packets: NEXT_HOP_CHIP_ID[n] and DEST_CELL_ID[n] on each source chip.

### New Device Notification

```
1. New chip powers up; broadcasts HELLO on its port
2. Local router receives HELLO_ACK; adds chip to topology table
3. Router sends NOTIFY/NEW_DEVICE to chips subscribed to NEW_DEVICE events
      Payload: new chip_id, cell_count, cell_rows, wake_mode
4. Subscribed chips alert host; host configures logical connection if required
```

Subscriptions are written by host at startup (NOTIFY_SUBSCRIBE register).
The router sends NOTIFY only to subscribed chips, never broadcasts to all.

---

## Registers

### PCN Chip Registers

| Name | Width | Description |
|------|-------|-------------|
| MY_CHIP_ID | 8 | This chip's network identity |
| NEXT_HOP_CHIP_ID[N] | 8×N | For each output cell: destination chip |
| DEST_CELL_ID[N] | 4×N | For each output cell: destination cell on dest chip |
| SPI_CLK_DIV | 8 | SPI clock divider from system clock |
| WAKE_MODE | 2 | 0x00=push, 0x01=pull, 0x02=threshold |
| WAKE_THRESHOLD | 8 | Activation magnitude threshold (threshold mode only) |
| DORMANCY_TIMEOUT | 16 | Inactivity period in ms before sleep |
| HELLO_INTERVAL | 16 | Period between HELLO broadcasts in ms (default 1000) |
| NOTIFY_SUBSCRIBE | 8 | Bitmask of NOTIFY event types this chip receives |

N = cell count (4 for Sky130A 4cell, 8 for 28nm 8cell).

### Router Chip Registers (additional)

| Name | Width | Description |
|------|-------|-------------|
| ROUTER_ID | 8 | This router's identity; lower ROUTER_ID = ring-left master |
| MAX_HOP_COUNT | 4 | Maximum hops for REACHABILITY_AD propagation (default 16) |
| RING_LEFT_ID | 8 | ROUTER_ID of the left ring neighbour (populated by HELLO) |
| RING_RIGHT_ID | 8 | ROUTER_ID of the right ring neighbour (populated by HELLO) |
| TOPOLOGY_TABLE[256] | — | dest_chip_id → (port, hop_count, ttl) entries |
| SUBSCRIPTION_TABLE | — | notify_event → subscriber chip_id list |

PORT_COUNT_PCN = 8 (fixed). PORT_COUNT_ROUTER = 2 (fixed). These are not
runtime registers; they are hardware constants of the router chip.

---

## SPI Electrical Interface

```
Signal      Direction (PCN chip)   Description
SPI_CLK     Output                 CPOL=0, CPHA=0
SPI_MOSI    Output                 transmit (to router or peer)
SPI_MISO    Input                  receive (from router or peer)
SPI_CS_N    Output                 active low; one line per direct connection
```

Full-duplex. Two SPI ports on PCN chip: SPI_ROUTER (to router chip) and
SPI_PEER (optional; to one direct peer chip). Same electrical interface on both.

Receive has priority over transmit when a single SPI controller is used.

---

## Timing

For the temporal variant (Sky130A TDM):

- SPI slot allocated at end of TDM cycle (slot 8 in 4cell, slot 9 in 8cell)
- Packet must complete within one TDM slot period
- At 50 MHz system clock, 4× SPI divider (12.5 MHz SPI):
  - Sky130A 22-byte packet: 22 × 8 / 12.5 MHz = 14.1 µs
  - TDM slot period must be ≥ 15 µs

For the passive (non-temporal) variant:

- SPI I/O decoupled from analog cell operation
- No TDM slot constraint; packets processed as they arrive

---

## Multi-Chip Example

```
              Router chip
             ┌──────────┐
 Chip 0 ─────┤ port 0   │
 Chip 1 ─────┤ port 1   │
 Chip 2 ─────┤ port 2   │
 Chip 3 ─────┤ port 3   │
             └──────────┘
```

Chip 0 cell 0 → Chip 3 cell 2:
- Chip 0 transmits DATA [dest=3, src=0, src_cell=0, dest_cell=2, payload_length=16, activations×16]
- Router receives: dest_chip_id=3; reads payload_length=16; topology table: port 3 → chip 3; forwards verbatim
- Chip 3 receives: dest_chip_id=3 = MY_CHIP_ID; loads activations into cell 2; triggers MAC

Peer-to-peer (Chip 1 direct to Chip 2, SPI_PEER):
- Same DATA packet format; SPI_PEER used in place of SPI_ROUTER
- No router involvement; topology table on each chip records the peer's chip_id

---

## Backprojection (δ) Plane Protocol

*Spec item S1 (see `hw_multi_array_l3_fable/SPEC_ROADMAP.md`). This is the transport
for the FABLE learning signal — the "backward" δ that drives E-inject. It is the
second plane of the two-plane model (`PCN_control_and_management.md §1`): the forward
plane above moves activations; this plane moves δ. It reuses the same packet fabric,
routing tables, and framing, with three differences flagged throughout.*

### Principle — why this plane is different

Forward DATA is **store-and-forward**: the router copies a packet toward its
destination unchanged. The δ plane is **compute-and-gather**: the router is an active
node that multiplies δ by the transposed forward weights (`W.T @ δ`, the forwards-only
router-local backprojection — *not* a backward channel; see `project_pcnchip_forwards_only`),
sums contributions from many source chips onto each destination, averages (`avg_bp`),
and RMS-renormalises before the result is used. Three consequences:

1. **The router holds shadow-W** (D1): a digital copy of each local chip's forward
   weights, re-synced after every `absorb`. The `W.T @ δ` runs on this copy.
2. **Endpoints are on the control plane (WB), not SPI.** The boss (WB master, D2) seeds
   each hop's source δ into the routers over WB and reads the gathered/averaged result
   back over WB — because the per-hop `local_delta`/blend/gate (stages 5–6) run on the
   boss. **The SPI PLANE=1 packets below carry only the *cross-router partial sums*** of
   a hop, i.e. contributions whose source chip and destination chip sit on different
   routers. Same-router contributions never leave the router.
3. **Per-hop barrier.** `avg_bp` and renorm need the *complete* sum for a destination,
   so the destination router emits its result only on a frame-done barrier (like
   `irq_frame_done`), after all partial sums for that hop have arrived. δ packets are
   still fire-and-forget *into* the accumulator; a dropped partial just contributes 0.

### δ packet format (DATA subtype, PLANE = 1)

Identical to DATA, with the PLANE bit set. The router routes it by the same tables but
delivers it to the **backprojection accumulator** instead of `inp_dac`.

```
Byte  Bits   Field
0     [7:6]  packet_type   = 00 (DATA class)
      [5]    PLANE         = 1  (backprojection δ;  0 = forward activation, unchanged)
      [4]    reserved      = 0
      [3:0]  frame_seq
1            dest_chip_id  destination-LAYER chip receiving this partial sum
2            src_chip_id   source-LAYER chip whose W.T produced it (provenance/debug)
3            checksum      XOR of bytes 0–2 and payload (unchanged)
Byte 4:    [7:4] src_cell_id   source cell on src chip
           [3:0] dest_cell_id  destination cell on dest chip
Byte 5:    payload_length      = N_ROWS (16 for Sky130A)
Bytes 6…:  delta_partial       payload_length × **int8 signed** backprojected δ
```

Same 22-byte size (Sky130A). The only wire difference from forward DATA is PLANE=1 and
that the payload is signed int8 δ, not unsigned activations. The destination router
**adds** delta_partial into its accumulator for `dest_cell_id` (forward DATA
*overwrites* `inp_dac`); this add is how multi-source contributions sum before `avg_bp`.

### Backprojection routing table (the transpose connectivity)

Forward connectivity (`NEXT_HOP_CHIP_ID[c]`, `DEST_CELL_ID[c]`) says where a cell's
*activations* go. The δ plane needs the **transpose**: for each source-layer cell,
which destination-layer cells its `W.T @ δ` contributes to, plus the **fan-in count**
per destination (for `avg_bp`). Host-configured, exactly like forward connectivity.

| Register (per source chip/router) | Width | Description |
|---|---|---|
| `BP_DEST_CHIP_ID[c]` | 8×N | for source cell c: destination-layer chip(s) it backprojects to |
| `BP_DEST_CELL_ID[c]` | 4×N | destination cell on that chip |
| `BP_FANIN[dest_cell]` | 8×N | number of source cells summed onto this destination (÷ in `avg_bp`) |

For the BIG topology this encodes `L3_ROUTING`/`L2_BP_COUNT`: each L3 chip backprojects
to 2 L2 chips (fan-in per L2 chip >1 → `avg_bp` active); each L2 chip backprojects to
its 3 L1 chips one-to-one (**L1 fan-in = 1 → no averaging at L1**).

### Per-hop sequence (gather → compute → average → renorm → scatter)

One hop = one layer of backprojection (L3→L2, then L2→L1). Driven by the boss.

```
1. SEED   Boss writes the hop's source δ into each source router's δ-source buffer
          over WB (hop 1 source = δ_l3 from the boss's stage-1 direction; later hops'
          source = the boss's blended δ_used from the previous hop).
2. COMPUTE Each router, for its local source cells: δ_partial = shadow-W.T @ δ_source,
          RMS-renormed PER SOURCE BLOCK (each W.T@δ block is renormed BEFORE it is summed
          — the sim's per-block _rms_preserving_proj; see router_backproj_spec.md §1/§3).
          Same-router destinations accumulate locally; other-router destinations are
          emitted as PLANE=1 δ packets on the ring.
3. GATHER  Each destination router ADDS all incoming (already-renormed) δ_partial (local
          + packet) into its per-destination accumulator. Fire-and-forget; order-irrelevant.
4. BARRIER On frame-done, each destination router applies avg_bp (÷ BP_FANIN) to the
          gathered vector and tags a 1-bit active-mask per unit (forward act code 0 ⇒
          inactive). NO gathered-vector renorm here — renorm was per-block in step 2; the
          only gathered-vector renorm is the boss's relu_jac_dir gate (stage 6).
5. READ    Boss reads {gathered δ_dest, active-mask} back over WB.
6. PROCESS Boss (stages 5–6): local_delta (aux classifier) + β-blend + relu_jac_dir
          gate. Delivers the gated int8 δ to the destination chips as `delta_flat`
          over WB (E-inject, stage 7). Retains the pre-gate blended δ_used as the
          SOURCE for the next hop (step 1).
```

Note step 6 matches the sim: the E-write uses the *gated* δ; the *pre-gate* blended
δ_used feeds the next hop's backprojection (`_gate_renorm` never feeds the chain).

### Flow control & timing

- **Within a hop:** δ partials are fire-and-forget into accumulators (no ACK/retry, as
  forward DATA); the *emit* is barrier-synchronised (step 4). One barrier per hop.
- **Ordering vs the forward TDM:** the δ sweep is a **distinct phase** that runs after
  a forward pass has produced the L3 scores and the boss has computed the seed δ_l3. In
  the temporal (TDM) variant it occupies dedicated slots after the forward slots; in the
  passive variant it runs as its packets/WB triggers arrive. Per settle step: *forward
  → boss score → δ hop(L3→L2) → δ hop(L2→L1) → E-inject*, then next step.
- **RMS-renorm** = scale each `W.T@δ` block to preserve its input RMS (matches
  `_rms_preserving_proj`); a small-FP norm reduction **per source block** in the source
  router, applied *before* the sum. **avg_bp** = integer divide by `BP_FANIN[dest]` on the
  gathered sum (destination router, at the barrier). Both precede the boss read-back.
- **Un-gated schedule:** the δ plane issues **no weight snapshot/restore** — it only
  seeds, gathers, and delivers δ. (HW-F5.)

### Registers added (δ plane)

**PCN chip:** none on the SPI side — δ is delivered to the chip as `delta_flat` over the
existing WB path (`e_inject_ctrl`). The chip does not parse δ packets.

**Router chip (additional to the forward set):**

| Name | Width | Description |
|---|---|---|
| `BP_DEST_CHIP_ID[c]`, `BP_DEST_CELL_ID[c]`, `BP_FANIN[·]` | see above | transpose routing + fan-in |
| `SHADOW_W[chip][1024]` | 8b×1024×ports | per-chip forward-weight copy for `W.T @ δ` |
| `SHADOW_W_SYNC` | 1 | pulse: reload SHADOW_W from chips' W-SRAM (issue after each `absorb`+`save` — W-SRAM is current only post-save; see `save_load_epoch_spec.md §2`) |
| `BP_SRC_BUF` / `BP_DST_BUF` | 8b×feats | boss-written source δ / boss-read gathered δ per hop |
| `BP_HOP_TRIGGER` / `BP_HOP_DONE` | 1 / 1 | start a hop's compute / barrier-done flag |
| `BP_ACTIVE_MASK` | 1b×feats | per-unit active flag emitted with the gathered δ |

### What S1 does *not* cover (pointers)

- The **internal router datapath** (transpose-MAC microarchitecture, accumulator sizing,
  renorm arithmetic) is spec item **S3** (`router_backproj_spec.md`).
- The **boss-side** stages 1/5/6/7 (classifiers, blend, gate, int8 delivery, hop
  orchestration) are spec item **S4** (`boss_core_spec.md`).
- The **multi-chip control address map + broadcast** for `(label, push-away, bh)` is
  **S2**. The δ plane above assumes the boss can reach each router's WB slave.

### V1 simplification (δ plane)

As with the forward plane: fixed topology, host-preloaded `BP_*` tables (no discovery),
router as FPGA, `SHADOW_W` loaded from the same host weight file as the chips. For a
**single-router** v1 (≤8 chips) there are **no cross-router δ packets at all** — every
partial sum is same-router, so steps 2–3 are entirely local and the SPI δ path is
exercised only when the array grows past one router. This lets v1 validate the
backprojection compute + boss round-trip over WB before the inter-router transport is
needed.

---

## V1 Prototype Approach

The v1 prototype uses a fixed physical topology: chips and routers are placed
on a physical board according to a predetermined schema. This simplifies the
initial build without abandoning the protocol design.

**What is fixed in v1:**
- chip_ids assigned by host at startup via ASSIGN_CHIP_ID CTRL packets
- Routing tables pre-loaded from a topology config file (no ring convergence wait)
- Router implemented as FPGA; routing table held in block RAM
- RING_LEFT_ID and RING_RIGHT_ID pre-configured, not discovered

**What still runs in v1 (protocol hooks active):**
- HELLO exchange runs as a connection sanity check (confirms chips are present)
- remembered_id field in HELLO is present (set to 0xFF by all chips in v1)
- TTL fields present in routing table entries (set to max; expiry not used)
- split-horizon logic present in REACHABILITY_AD handler (even with pre-loaded tables)
- ASSIGN_CHIP_ID CTRL sub-type implemented (used by host at startup)

**Upgrade path to dynamic operation:**
The only additions needed for full dynamic operation are:
1. PCN chip stores assigned_chip_id to non-volatile memory (SRAM survives power-down,
   or a small on-chip flash cell); presents it as remembered_id on reconnect
2. Router honours remembered_id requests rather than always assigning a new ID
3. HELLO_INTERVAL-driven ring convergence used instead of pre-loaded tables

No packet format changes, no register changes, no RTL interface changes.

---

## Relationship to RTL

`router_ctrl.v` (Sky130A_16x16_4cell/temporal/rtl/) implements the PCN chip side:
- TDM FSM SPI slot: reads ADC buffer, builds DATA packet, transmits on SPI_MOSI
- Receive path: parses header, routes to inp_dac + weight_fsm, or forwards
- Dormancy controller: monitors LAST_TX_TIMER; gates clock on timeout; wakes on SPI event
- HELLO state machine: sends HELLO with remembered_id=0xFF; stores assigned_chip_id from HELLO_ACK

`router_chip.v` (router_chip/rtl/ — 14 modules, elaboration verified):
- 8-port PCN SPI switch + 2-port ring SPI switch with topology table lookup
- HELLO originator and REACHABILITY_AD propagator with split-horizon
- chip_id assignment pool; remembered_id confirmation on HELLO
- NOTIFY dispatcher to local subscriber list
- ASSIGN_CHIP_ID and SET_REGISTER forwarding to target PCN chips
