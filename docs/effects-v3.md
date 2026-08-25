# Effects wire format v3

Status: **proposed**, for discussion. Implemented behind `EFFECTS_VERSION = 3` in
the Rust engine; needs a matching C implementation before it is usable.

## Why a new version

The goal is **engine migration**: a deployment running the C engine must be able to
upgrade to Rust, and downgrade back, without dumping and reloading. RDB
loadability landed in #2459, so a Rust replica can now full-sync from a C master.
Effects are the remaining gap: after the sync, the master keeps streaming
`GRAPH.EFFECT`, and the two engines cannot read each other's buffers.

Cloning C's v2 byte-for-byte would work against shipped C, but it means adopting
the two v2 decisions that are actively hostile to a second implementation:

1. **`CREATE_NODE` carries no entity id** (`CREATE_EDGE` no edge id). The replica
   derives ids from creation order. Any disagreement between the two engines about
   what "creation order" means corrupts a replica silently, and the record itself
   carries nothing to check against.
2. **`SET_LABELS` / `REMOVE_LABELS` carry a `GxB_Vector_serialize` blob** — a
   GraphBLAS-internal serialization embedded in the replication stream. The two
   engines build against separate GraphBLAS copies (Rust pins v10.5.0 in
   `graphblas.sh`; C vendors `deps/GraphBLAS`), so the wire format inherits a
   dependency on those staying compatible. That is a poor thing to couple a
   migration path to.

Versioning is C's own established mechanism, not a new idea: C's reader accepts
any version **≤** its own (`if (*v > EFFECTS_VERSION) return false`, `effects_apply.c:797`)
and already branches per version — `if (unlikely(version == 1)) ApplyLabels(...) else ApplyLabels_V2(...)`
(`effects_apply.c:865,873`). Notably, the v1→v2 bump existed to change *the label
records* — the same record family this proposal changes again.

## The invariant

    GRAPH.EFFECT <graph-key> <payload>      one command per query, never one per record

      payload   u8 version = 3
                record · record · record …  one per (opcode, shape), in apply order

      record    u32 opcode · u32 count · <the shape, once> · <count rows>

| record | **partition key** — the `GROUP BY` | per row |
| --- | --- | --- |
| `CREATE_NODE` | `LabelSet` + `AttrSet` ids | id, values |
| `CREATE_EDGE` | `RelType` + `AttrSet` ids | id, src, dst, values |
| `UPDATE_NODE` / `UPDATE_EDGE` | `AttrSet` ids only | id, values |
| `SET_LABELS` / `REMOVE_LABELS` | `LabelSet` | id |
| `DELETE_NODE` | — nothing to hoist | id |
| `DELETE_EDGE` | `RelType` | id, src, dst |
| `ADD_SCHEMA`, `ADD_ATTRIBUTE`, index, constraint | inherently singular — no partition key, no count | — |

Four rules, and everything else follows from them:

1. **One `GRAPH.EFFECT` per query.** Batching removes per-record framing; it never
   multiplies commands. There is exactly one `ctx.replicate("GRAPH.EFFECT", args)`
   (`src/graph_core.rs:1582`), fed by one `build_effects_buffer` call at commit
   (`graph/src/runtime/ops/commit.rs:111`).
2. **One record per `(opcode, partition key)`.** Shapes are compared only within a record
   type, so a query that creates *and* updates *and* deletes groups each family
   independently.
3. **Inside a record**: the count once, the ids once, the shape once, then
   `count × n` values.
4. **Row *k* belongs to the k-th smallest id in the `IdSet`.** Ascending id order is
   the only thing binding values to entities — no per-row id is sent. This is why
   `IdSet` decode must assert cardinality: a bitmap that round-trips one id short
   would land every subsequent row on the wrong entity.

"Partition key" and "shape" are the same thing — the first is its role, and the
rest of this document uses *shape* as the short name. Rows sharing it become one
record, exactly as a `GROUP BY` would. It is deliberately not called a *sharding*
key: nothing is distributed across nodes, and all of a query's records still travel
in one command.

Shape is the attribute **ids**, never the values, and for edges it is the
relationship type exactly as labels are for nodes.

### We hoist, but deliberately do not reference

MySQL's `TABLE_MAP_EVENT` and Postgres's `Relation` message declare a shape in one
event and point at it by number from later events. That requires the decoder to
carry state between events, so a buffer is only decodable in the context of those
before it. v3 states the shape inline in the record that uses it, keeping every
record self-contained — which is what keeps AOF replay from an arbitrary offset,
and a replica that missed a buffer, both well-defined.

It also costs nothing to skip. The `AttrSet` header is `2 + 2n` bytes per record:
8 B out of 388,944 (0.002%) for a batched 3-attribute create. A declare-once table
referenced by a `u16` would be a strict loss rather than a trade-off, because
grouping makes **shapes == records by construction** — there is never a second
record to amortise a declaration over, so the reference bytes are pure addition
(1 shape: 8 → 10 B; 20 shapes: 80 → 120 B). The only regime where the header is
proportionally large is one shape per node (10.3%), and there every shape is
distinct, so a table cannot help at all.

## Design rule

**Every batchable record is `u32 opcode · u32 count · blocks…`, built from five
shared blocks.** There is no separate "batch" record type: `count == 1` and
`count == 10_000` are the same record, so nothing is left un-batchable and the
decoder has one shape per opcode. `ADD_SCHEMA` and `ADD_ATTRIBUTE` are
inherently singular and carry no count — the one exception.

The blocks are written and tested once each, then recombined:

| block | layout |
| --- | --- |
| `IdSet` | `u8 enc` · (`u8 width · uint(width) × count` \| `u32 blob_len · roaring64[blob_len]`) — unique, **ascending** |
| `IdList` | `u8 enc` · (`u8 width · uint(width) × count` \| `u32 dict_len · roaring64 · u8 w · uint(w) × count`) — ordered, duplicates kept |
| `RelType` | `i32` — one per record |
| `LabelSet` | `u16 n` · `i32 × n` — **n, not count**; hoisted, always plain |
| `AttrSet` | `u16 n` · `u16 attr_id × n` · `SIValue × (count × n)` — attribute ids stated **once**, rows row-major |

### Two encodings, not three

`ENC_PLAIN = 0` carries a width byte, so ids are written at the narrowest width
that holds the largest of them. A separate full-width encoding would be this one
with `width == 8`, so there is no reason for both. Finding the maximum is one
pass and no allocation, and FalkorDB allocates node ids densely from zero and
reuses them from a free list, so the width tracks the graph's size.

`ENC_COMPRESSED = 1` is roaring: a bitmap for `IdSet`, a dictionary plus one rank
per row for `IdList`. Roaring bitmaps are **run-optimized before sizing or
serializing** — `optimize()` is normative, not a tuning knob, because an
unoptimized bitmap serializes to different bytes. A contiguous 10k set is 8,220
bytes unoptimized and 27 optimized.

### How an encoding is chosen

Never by a count threshold. Roaring's cost tracks how ids are *spread*, not how
many there are: 1,024 ids spread 2^32 apart serialize to 22,536 bytes against the
raw form's 8,192, so no threshold is safe at any count. Both libraries report a
bitmap's serialized size without serializing it
(`RoaringTreemap::serialized_size()`, `roaring64_bitmap_portable_size_in_bytes()`),
so comparing costs nothing.

- **`IdSet`** compares `4 + roaring_size(ids)` against `1 + count × width`.
- **`IdList`** compares `5 + roaring_size(distinct) + count × rank_width` against
  `1 + count × width`, but only after the probe below.

Smaller wins; ties go to the raw form.

### The `IdList` dictionary is gated on duplication, not size

Size alone cannot decide this one, because a dictionary has to be *built* to be
measured and building it is the entire cost. Measured at 10,000 rows, taking the
dictionary for an 8% size win cost **2,455×** what those bytes are worth on the
write thread — the same trap that made compression default-off.

So the encoder first checks a necessary condition that is free (`width > 1`: if
raw ids already fit in one byte, no dictionary can beat them), then probes 128
rows strided across the list for repeats. Expected distinct in a sample of 128
drawn from *n*: n=100 gives ~72 (57%), n=1,000 gives ~120 (94%), n=10,000 gives
~127 (99%) — three quarters separates the case the dictionary wins from the ones
it does not.

**Correctness never depends on the probe.** A wrong answer either builds a
dictionary the size comparison then rejects, or skips one that would have been
slightly smaller. Both decode identically, and the decoder always supports every
encoding — so unlike the wire layout, this heuristic does **not** have to match
between the two engines, and either side can tune it without a version bump.

Measured, 10,000 rows (`cargo bench -p graph --bench effects_v3`):

| shape | picked | bytes | encode | decode |
| --- | --- | --- | --- | --- |
| 10k-node graph, 100 distinct | dictionary | 10,234 | 58.9 µs | 4.5 µs |
| 10k-node graph, 10k distinct | raw | 20,002 | 5.03 µs | 0.76 µs |
| 5M-node graph, 100 distinct | dictionary | 10,834 | 60.7 µs | 5.6 µs |
| 5M-node graph, 10k distinct | raw | 40,002 | 5.19 µs | 0.82 µs |

The last row gives up 3,020 bytes (7.5%) to save 165 µs, which is the trade the
probe exists to make.

### Why `IdSet` and `IdList` are distinct types

A roaring bitmap is a *set*: it deduplicates and it sorts. Edge endpoints do
neither — many edges share a source, and each endpoint must stay positionally
aligned with its edge id — so encoding them as a set would drop duplicates and
misalign every row after the first repeat.

`IdSet` decode **must assert cardinality == count** for the same reason: row *k*
belongs to the k-th smallest id, so a bitmap one id short would land every later
row on the wrong entity rather than fail.

### Values stay fixed-width

`SIValue` payloads are not varint-encoded. Varints would shrink small integers,
but they make every field's position depend on the values before it, so a decoder
cannot skip a record without parsing it and the two engines must agree on a
second variable-width scheme. The bytes they save are the ones compression
already removes.

## Unchanged from v2 — the primitives

Primitives (from [`effects-v2-c-wire-format.md`](effects-v2-c-wire-format.md), read from C source):

| primitive | bytes |
| --- | --- |
| `EffectType` | 4 |
| `SchemaType`, `GraphEntityType`, `ConstraintType`, `IndexFieldType` | 4 |
| `EntityID` | 8 |
| `AttributeID` | 2 |
| `LabelID` | 4 |
| `SIType` | 4 (bitmask, not an ordinal) |
| constraint attr count | 1 |

Strings are `uint64 len = strlen + 1`, then `len` bytes **including** the trailing
NUL. `SIValue` is `uint32 SIType` then payload; `T_POINT` is 2×f32, `T_ARRAY` and
`T_VECTOR_F32` use `uint32` counts.

Records are **packed** — C writes them through `#pragma pack(push, 1)` structs, so
there is no alignment padding to reproduce.

An `AttributeSet` is `ushort count`, then `count` × (`AttributeID` id, `SIValue`
value).

### What v2 looked like, for reference

Every record changes shape in v3 (see below) — a count and the shared blocks — so
nothing in the record list survives untouched. Two v2 details are worth recording
because they are easy to get wrong when reading C:

- `1`/`2 UPDATE_*` are **one record per (entity, attribute)** in v2 — not one record
  per entity with a count and N pairs, which is what Rust emitted. A structural
  difference, not a width one.
- `10 ADD_ATTRIBUTE` has **no node/relationship discriminator**. Rust used to write
  one; that was a consequence of two attribute dictionaries, which #2459 unified, so
  dropping it is a correctness alignment rather than a loss.

For the v2 byte layouts themselves see
[`effects-v2-c-wire-format.md`](effects-v2-c-wire-format.md), read from C source.

## Changed in v3

    3 CREATE_NODE
      v2   ushort n_labels · LabelID[n_labels] · AttributeSet      (one record per node, no id)
      v3   u32 count · IdSet · LabelSet · AttrSet

    4 CREATE_EDGE
      v2   ushort n_rels · RelationID · EntityID src · EntityID dest · AttributeSet
      v3   u32 count · IdSet · RelType · IdList(src) · IdList(dst) · AttrSet

    1·2 UPDATE_NODE / UPDATE_EDGE
      v2   EntityID id · AttributeID attr_id · SIValue value       (one record per (entity, attribute))
      v3   u32 count · IdSet · AttrSet

    5·6 DELETE_NODE / DELETE_EDGE
      v2   EntityID id                       | EntityID id · RelationID · src · dest
      v3   u32 count · IdSet                 | u32 count · IdSet · RelType · IdList(src) · IdList(dst)

    7·8 SET_LABELS / REMOVE_LABELS
      v2   GrB_Index blob_size · GxB_Vector_serialize(nodes)[blob_size]
      v3   u32 count · IdSet(nodes) · LabelSet

    9 ADD_SCHEMA
      v2   t · SchemaType · string name          (no id — the replica infers it)
      v3   t · SchemaType · LabelID|RelationID id · string name

    10 ADD_ATTRIBUTE
      v2   t · string name                       (no id)
      v3   t · AttributeID id · string name

    11 CREATE_INDEX   — one record per field, unchanged from v2
      t · SchemaType · LabelID id · string label · AttributeID id · string attr
        · IndexFieldType · SIValue options
    12 DROP_INDEX      — mirrors 11 without the options
    13 CREATE_CONSTRAINT
      t · ConstraintType · GraphEntityType · LabelID id · string label
        · uint8 n · (AttributeID id, string name) * n
    14 DROP_CONSTRAINT — mirrors 13

Three width traps in these, all read from C source rather than inferred:
`IndexFieldType` is a **bit flag set**, so a range index is
`NUMERIC|GEO|STR = 0x0E` and not a discriminant of its own; `GraphEntityType` is
**1-based**, because `GETYPE_UNKNOWN` takes 0, so a node is 1; and the
constraint property count is a **`uint8`**, not the `u16` used everywhere else.

`CREATE_INDEX` is deliberately not batched. Index DDL is rare, and C applies it
idempotently per field — `Index_SetLanguage` tolerates a re-set,
`Index_SetStopwords` is guarded — which is what lets a replica take the fields
of one index in any order.

### Entity ids become explicit

The replica stops inferring ids from creation order, so an id disagreement fails a
bounds/existence check on the spot instead of silently writing the right value to
the wrong entity. This is the item the earlier attempt was parked on; v3 removes
the problem rather than reimplementing it.

### Schema and attribute records carry their id

C states the invariant in its own source (`effects_apply.c`, above `VerifySchema`):
the id is authoritative *"only because every schema mutation is itself an effect,
applied in the same order on every replica — the name is a cheap cross-check that
surfaces divergence instead of silently trusting a stale/incorrect id."*

The two records that perform that assignment are the only ones carrying no id.
Audited against `origin/master` (2caafdaae):

| record | schema ids on the wire | can it detect divergence? |
| --- | --- | --- |
| `9 ADD_SCHEMA`, `10 ADD_ATTRIBUTE` | **none** — name only, id inferred from append order | only "this name already exists locally" |
| `1`/`2 UPDATE_*` | `AttributeID`, bare | no |
| `3 CREATE_NODE`, `4 CREATE_EDGE` | `LabelID`/`RelationID` + `AttributeID`, bare | **no, and cannot** — `static void ApplyCreateNode` has no return value |
| `6 DELETE_EDGE`, `7`/`8 *_LABELS` | `RelationID`/`LabelID`, bare | entity existence only |
| `11`–`14` index, constraint | id **+ the name** | yes — `VerifySchema`/`VerifyAttribute`, whose only call sites are `create_index_effect.c:113,119`, `drop_index_effect.c:91,96`, `create_constraint_effect.c:103,106`, `drop_constraint_effect.c:97,100` |

So records 9 and 10 *establish* the id spaces, 1–8 *consume* them with bare ids, and
only 11–14 check anything. A numbering disagreement is introduced by a record that
cannot report it, then silently trusted by every record that uses it.

The existing "already exists locally" check in `ApplyAddSchema` / `ApplyAddAttribute`
is necessary but not sufficient: it catches a replica that already holds the name,
but not a replica whose dictionary is a *different length*, where appending the same
new name yields a different id. That is the case that actually bit us — Rust's split
node/relationship attribute-id spaces against an RDB-seeded replica holding the
unified numbering: a property effect carried a bare `u16` and landed the value on the
wrong attribute, and healed on resync, so it hid.

v3 therefore carries the id on both records. The replica computes the id it *would*
assign, asserts it equals the id on the wire, and fails the buffer on mismatch — so
divergence surfaces where it is introduced rather than never.

The bulk records deliberately keep bare ids: a name per row would undo the batching,
and once assignment is pinned they no longer need one. Two consequences:

- **Ordering is normative.** `9`/`10` must precede any record referencing the ids
  they introduce. Rust already does this — `build_effects_buffer` emits schema
  additions (`pending.rs:1468`) before created nodes (`:1509`).
- **`ApplyCreateNode` / `ApplyCreateEdge` must return `bool`.** They are `static void`
  in C today, so they cannot report a mismatch at all. v3's explicit entity ids are
  only *enforceable* — as opposed to merely present — once those signatures change.
  This is a C-side prerequisite, not an optional hardening.

### Labels are grouped by label, not interleaved as pairs

`SET_LABELS` carries *the labels, and all their nodes* — not one
`(node_id, label_id)` pair per node. That is what makes the node ids a contiguous
`IdSet` roaring can run-encode: for 10,000 consecutive nodes gaining one label,
the pair form is 120,008 B and the grouped form is **46 B** (2,609× smaller raw,
261× after zstd-1; measured).

Grouping is sound here because label add/remove are idempotent set operations, so
order within a record carries no information — unlike edge endpoints, which is why
those stay an `IdList`.

A query that applies *different* label sets to different nodes splits into one
record per distinct label set, exactly as differing attribute shapes do. That
split is cheap in practice because a label list is **parsed, not evaluated**:
`SetItem::Label { var, labels: OrderSet<L> }` (`graph/src/parser/ast.rs:870`) is
filled by `parse_labels()` at parse time, so the number of distinct label sets a
query can produce is bounded by the query text — a handful, never one per row.
Attribute shapes have no such bound, since a map key can be computed.

### Attribute ids are stated once per record

`AttrSet` hoists the attribute ids out of the rows: `u16 n · u16 attr_id × n`,
then `count × n` values. For 10,000 nodes with 3 properties that removes
`2 × 10_000 × 3 = 60_000` bytes of repeated ids (468,936 B → 388,944 B, −17.1%).

The saving is stated as a raw byte count deliberately. Its effect *after*
compression is data-dependent and occasionally negative — the repeated ids also
acted as row delimiters that zstd-1's match finder was using — so the raw figure,
which always applies and needs no measurement, is the one the design rests on.

A row missing one of the shape's attributes encodes `T_NULL` in that slot. This is
unambiguous because FalkorDB never stores a null property value: setting a
property to null removes it, so `T_NULL` on the wire can only mean "absent".

**Rows stay row-major; columnar was measured and rejected.** Grouping values by
attribute instead of by row saves exactly zero bytes uncompressed, and its
compressed result changes sign with the data: up to −73% on compressible values,
but **+119%** on high-entropy values at 20 columns (160,314 B → 350,687 B,
zstd-1). A second wire layout that two independently written engines must match
byte-for-byte, and that gives up streaming row-at-a-time decode, is not worth a
benefit that is not reliably positive.

### What a shape split costs

A record holds one shape, so a query that varies the shape splits into many
records. Measured against the shipped codec (`the_motivating_query_end_to_end`),
10,000 nodes created by `UNWIND range(0,9999) AS i CREATE (:L {v:i})`, where v2
emitted one 34-byte record per node:

| distinct shapes | v2 | v3 |
| --- | --- | --- |
| 1 — all share one shape | 340,001 B | **120,051 B, −64.7%** |
| 20 | 340,001 B | **140,401 B, −58.7%** |
| 500 | 340,001 B | **150,001 B, −55.9%** |
| 2,000 | 340,001 B | **180,001 B, −47.1%** |
| 10,000 — every node its own | 340,001 B | **339,745 B, −0.1%** |

**There is no floor to pay for.** An earlier draft of this design budgeted +15%
on the pathological case, and a still-earlier one with a separate batch opcode
cost +99%. Merging the full-width and narrowed id encodings removed it: narrowing
the id gives back what the count field costs, so a shape carried by one node
lands at parity with v2. A single record is 33 bytes for a one-byte id, 34 for a
two-byte id (exactly v2) and 36 past 2^16. A regression test asserts the worst
case stays at or under v2's 340,001 bytes.

## Building a v3 payload

`graph/src/runtime/effects_v3_emit.rs`, the mirror of `effects_v3_apply.rs` —
and deliberately shaped like it:

    emit    Pending  --digest-->  Record  --encode-->  bytes
    apply   bytes    --decode-->  Record  --apply-->   Graph

`Record` is the pivot both directions turn on. Digesting into it rather than
writing bytes straight out of `Pending` costs nothing: the grouping already
allocated the ids, labels, attribute ids and row values a `Record` holds, so
they are moved into a value instead of passed as four loose slices. What it buys
is a testable surface for the grouping — the emit tests assert on records rather
than on decoded buffers — and a round-trip that compares values:
`digest(pending) == read_buffer(encode(digest(pending)))`.

The v2 emitter sits beside it in `effects_v2.rs`, whole and separate, so
retiring the old format is deleting a file rather than untangling one. Neither
lives in `pending.rs`: a mutation accumulator should not know what a replication
frame looks like. `Pending`'s fields are `pub(crate)` and the wire knowledge
lives in the emitters.

An earlier revision put a per-format view struct between them. Those views were
a pure projection — every field a reference to the same type in `Pending` — so
they documented the dependency surface and did nothing else, while costing a
struct and a constructor per format. `digest`'s return type states that surface
better than a mirror of the source could.

**The shapes are deliberately not pre-grouped inside `Pending`.** Attributes
accumulate incrementally — `set_node_attribute` inserts into a sorted vec — so
`CREATE (n) SET n.a=1 SET n.b=2` changes that node's shape twice, and
pre-bucketing would mean re-bucketing on every attribute write. It would also
charge every write for a grouping most of them never use, since effects are only
built when something is actually replicating. The grouping stays a view computed
at commit.

The grouping is cheap because `Pending` already holds most of it: `created_nodes`
is a `RoaringTreemap`, so ids arrive ascending for free; `created_rels_by_type`
is already partitioned by relationship type; and the attribute vectors are
already "sorted by id, unique", which is exactly the shape key.

Three rules the emitter has to enforce that the format alone does not:

**Label order must be normalized.** `set_labels` keeps caller order, so `[7,8]`
and `[8,7]` arrive for the same set. They are one shape, and `LabelSet` is
emitted ascending — which is also what makes two engines that agree on the set
produce the same bytes.

**Group iteration must be ordered.** A `HashMap` iterates arbitrarily, so the
same query could emit its records in a different order on two runs. That alone
would defeat the byte-for-byte comparison the cross-engine harness rests on, so
groups are sorted by key before emission.

**A new attribute is announced once, not once per entity kind.** #2459 unified
the two attribute dictionaries, so `get_node_attribute_names` and
`get_relationship_attribute_names` return the same store. v2 could iterate both
harmlessly because its `ADD_ATTRIBUTE` carries a node/relationship
discriminator and the replica's registration is get-or-create. v3's does not
carry one — correctly, since C has a single dictionary — so iterating both would
announce every new attribute twice under the same id.

### `DELETE_EDGE` needs the type captured at deletion time

v3 groups deleted edges by relationship type, and the type cannot be recovered
when the buffer is built: the edge is already gone. `index_remove_edge_docs` is
not a substitute — it is only populated for types that carry an index
(`graph.rs`, guarded by `has_index`).

So `delete_relationships` and `delete_implicit_edges` now return
`DeletedEdge = (RelationshipId, type_id, NodeId, NodeId)`. Both already resolved
the type internally and were discarding it.

## Applying a v3 payload

`graph/src/runtime/effects_v3_apply.rs`. It lives in the `graph` crate rather
than beside the v2 handler because it touches only graph types — no
`redis_module` — which also lets it be tested against a real `Graph`.

**The v2 path's batching machinery disappears.** v2 has to reconstruct batches it
never received: it accumulates runs of adjacent `DELETE_NODE` records and flushes
on a type change, because applying a node's edges one at a time made a replica
~40x slower than the master that produced the writes. In v3 a record already
covers every entity of its shape, so each one applies as a single bulk call. The
look-ahead, the two pending batches and the flush logic all go.

**The id checks are the point.** On `ADD_SCHEMA` and `ADD_ATTRIBUTE` the replica
computes the id it *would* assign and rejects the whole buffer if it disagrees
with the wire. C's reader has a weaker check — it refuses a name that already
exists locally — which catches a replica that is *ahead* but not one whose
dictionary is a different length, where appending the same new name yields a
different id. Index and constraint records additionally resolve label and
attribute by id and confirm the name matches, mirroring C's `VerifySchema` and
`VerifyAttribute`.

Two behaviours worth stating because they are choices, not consequences:

- **A constraint installs the master's outcome; the replica does not
  re-validate.** A replica scanning independently would do so at a different
  time against different write interleavings and could legitimately reach a
  different status.
- **Index creation uses the sync variant.** A replica must not spawn population
  threads, and must not reorder its work against the effect stream.

`OPTIONS {...}` now survives the trip. v2 dropped the map on the wire and forced
those statements to replicate as verbatim `GRAPH.QUERY`; v3 carries it, and the
replica rebuilds the same `IndexOptions` through the same
`map_to_index_options` the master used rather than approximating it.

## Migration and rollout

Both engines must speak v3 before either emits it. C accepts `≤ its own version`,
so the ordering constraint is the ordinary one — **replicas before masters**:

1. Ship a v3-capable C release (reads v2 and v3; still emits v2).
2. Ship a v3-capable Rust build (reads v3; emits v3).
3. Upgrade replicas, then masters.

### The open question

A v3-only Rust cannot be a replica of an **already shipped** C master, because
that master emits v2. Two ways to close that:

- **(a) Migrate via a v3-capable C release.** Upgrade C in place first, then swap
  the engine. No v2 code in Rust at all. Cheapest, but it forbids migrating
  directly from an older C.
- **(b) Teach Rust to decode v2 as well.** Allows attaching Rust to any shipped C,
  but it drags both v2 warts back in on the read path: deriving ids from creation
  order, and `GxB_Vector_deserialize` against a blob some other GraphBLAS wrote.

This proposal assumes **(a)** and treats (b) as a separate, optional follow-up,
justified only if migrating from pre-v3 C without an intermediate upgrade is a
hard requirement. Note (b) is decode-only: Rust would still never *emit* v2, so a
downgrade target must be v3-capable either way.

## What is implemented

In `graph/src/runtime/effects_v3.rs`, codec only — no `Pending`, no `Graph`, so
the format can be reviewed and tested byte-for-byte on its own.

| | status |
| --- | --- |
| the five blocks, encode + decode | done |
| records 1–10, encode + decode, `read_buffer` | done |
| ids on `ADD_SCHEMA` / `ADD_ATTRIBUTE` | done |
| grouping `Pending` into v3 records (`effects_v3_emit.rs`) | done, wired behind `EFFECTS_VERSION` |
| records 11–14 (index, constraint) | done |
| applying v3 on the replica (`effects_v3_apply.rs`) | done, always active — reads are version-dispatched |
| the C side | not yet — nothing can ship until this exists |

### The switch

`GRAPH.CONFIG SET EFFECTS_VERSION 2|3` picks what a node **emits**, and defaults
to **2**. Reads are always dispatched on the version byte that arrived, so a
node reads both regardless of what it writes.

That asymmetry is the whole rollout mechanism, and it is the same one C uses
(`if (*v > EFFECTS_VERSION)`, `effects_apply.c:689`): **upgrade readers, then
flip writers**. A master emitting v3 at a peer that cannot read it is silent data
loss rather than a degraded mode, which is why the default cannot move until the
C engine reads v3.

The value is range-checked to 2 or 3 at the config layer. Anything else would
stamp a version byte no peer can read, and the failure would land on the replica
rather than on the operator who typed it.

### Verified against live servers

Two module instances, one replicating from the other, 2026-08-26:

- `EFFECTS_VERSION` defaults to 2; `SET` to 1 or 99 is refused with
  `expected 2 or 3` and leaves the previous value in place.
- With the master at 3, every write reached the replica as `GRAPH.EFFECT` with a
  `\x03` version byte followed by opcode 9 (`ADD_SCHEMA`).
- Flipping the master back to 2 mid-stream produced `\x02` on the wire, and the
  same replica applied it without a restart.
- Master and replica agreed exactly afterwards: 988 nodes, 987 `:Person`, 65
  `:Young`, 1 `:Later`, and matching property values — across creates, a label
  set, a delete, and the v2 write that followed.

## Verification

Byte-level unit tests per record are necessary but not sufficient — a wrong width
does not fail on the far side, it makes C read misaligned bytes as a type and write
through the resulting pointer (the `AttributeSet_Update` segfault seen when feeding
C our v2-ish buffers).

What the unit tests do cover (`cargo test -p graph --lib effects_v3`, and
`--lib v3_tests` for the emit path):

- **Byte-pinned records**, so a width change fails here rather than on the far side.
- **`encode(decode(x)) == x`** across all ten record types. Stronger than checking
  the decoded values: it catches a writer and reader that agree with each other
  but not with the format, which value comparison cannot see.
- **Every truncation** of a buffer is an error rather than a panic, and every
  count read off the wire is checked against the bytes remaining before it
  reaches `Vec::with_capacity`, so a corrupt length cannot OOM us.
- The **cardinality assertion** on `IdSet`, and an out-of-range `IdList` rank.
- On the emit path: that label order does not split a shape, that differing
  shapes do split, that a created node's labels do not also produce a
  `SET_LABELS` record, that deleted edges group by type, and that two
  `Pending`s populated in opposite orders produce **byte-identical** buffers.
- On the apply path: a schema or attribute id disagreement aborts the buffer, a
  stale label id is caught by its name, and a **master-to-replica round trip** —
  build a real `Pending`, emit, apply to an empty graph, compare observable
  state. Encoder and decoder agreeing with each other is not the same as either
  being right.

The real gate is the cross-engine harness driving the real
`falkordb/falkordb-server:edge-c` image, in both directions, for each of: live
replication, AOF replay, and RDB full-sync followed by streamed effects. That
harness is kept out of the repo — it is investigation tooling, not a regression
test — and it cannot run at all until the C side exists.

Encoding costs are measured with criterion, `cargo bench -p graph --bench
effects_v3`. Note the benchmark profile disables LTO: release's `lto = true`
cannot link the prebuilt GraphBLAS objects, so those are not shipped-binary
numbers.
