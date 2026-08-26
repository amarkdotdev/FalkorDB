//! Turning a committed `Pending` into v3 effect records.
//!
//! The mirror of `effects_v3_apply`, and deliberately shaped like it:
//!
//! ```text
//!   emit    Pending  --digest-->  Record  --encode-->  bytes
//!   apply   bytes    --decode-->  Record  --apply-->   Graph
//! ```
//!
//! `Record` is the pivot both directions turn on. Digesting into it rather than
//! writing bytes straight out of `Pending` costs nothing — the grouping already
//! allocated the ids, labels, attribute ids and row values that a `Record`
//! holds — and it buys the grouping its own testable surface: the tests below
//! assert on records, not on decoded buffers.
//!
//! Keeping this out of `pending.rs` is the other half. A mutation accumulator
//! should not know what a replication frame looks like, so `Pending`'s fields
//! are `pub(crate)` and the wire knowledge lives here.

use atomic_refcell::AtomicRefCell;
use roaring::RoaringTreemap;
use rustc_hash::FxHashMap;

use crate::{
    effects::v3::{self as v3, EffectEncode, GraphEntity, Record, SchemaType},
    graph::graph::Graph,
    runtime::{pending::Pending, value::Value},
};

/// Build a v3 effects payload from what a query staged.
///
/// Digest, then encode. Returns the number of records written.
pub fn build_effects_buffer_v3(
    p: &Pending,
    g: &AtomicRefCell<Graph>,
    buf: &mut Vec<u8>,
) -> u64 {
    if buf.is_empty() {
        // Through new_buffer, so the flags byte cannot be forgotten here.
        buf.extend_from_slice(&v3::new_buffer());
    }
    let mut n = 0;
    for_each_record(p, g, |record| {
        record.encode(buf);
        n += 1;
    });
    n
}

/// Every record a committed `Pending` implies, in apply order, handed to `out`
/// one at a time.
///
/// Eager in the grouping — a shape's members are only known once every entity
/// has been seen — but records go out as they are built, so peak memory is one
/// stage's group map plus one record rather than every record at once. The
/// mirror of the apply path, which decodes one record and applies it rather
/// than materializing them all.
///
/// The shapes are **not** pre-grouped inside `Pending`, deliberately.
/// Attributes accumulate incrementally — `set_node_attribute` inserts into a
/// sorted vec — so `CREATE (n) SET n.a=1 SET n.b=2` changes that node's shape
/// twice, and pre-bucketing would mean re-bucketing on every attribute write.
/// It would also charge every write for a grouping most never use, since
/// effects are only built when something is actually replicating.
pub fn for_each_record(
    p: &Pending,
    g: &AtomicRefCell<Graph>,
    mut out: impl FnMut(Record),
) {
    let out = &mut out;

    // Schema and attribute additions first, each carrying its id. That ordering
    // is normative in v3, not incidental: the ids on every later record are
    // only meaningful once the replica has agreed on the numbering.
    {
        let graph = g.borrow();
        for (offset, label) in graph
            .get_labels()
            .iter()
            .enumerate()
            .skip(p.schema_label_count)
        {
            out(Record::AddSchema {
                schema_type: SchemaType::Node,
                id: offset as i32,
                name: label.to_string(),
            });
        }
        for (offset, rel_type) in graph
            .get_types()
            .iter()
            .enumerate()
            .skip(p.schema_rel_type_count)
        {
            out(Record::AddSchema {
                schema_type: SchemaType::Edge,
                id: offset as i32,
                name: rel_type.to_string(),
            });
        }
        // Once, not once per entity kind. #2459 unified the two attribute
        // dictionaries, so both getters return the same store — and unlike
        // v2's, v3's ADD_ATTRIBUTE has no node/relationship discriminator, so
        // announcing per kind would send every new attribute twice under the
        // same id.
        for (offset, attr) in graph
            .get_node_attribute_names()
            .iter()
            .enumerate()
            .skip(p.schema_node_attr_count)
        {
            out(Record::AddAttribute {
                id: offset as u16,
                name: attr.to_string(),
            });
        }
    }

    digest_created_nodes(p, out);
    digest_created_edges(p, g, out);
    digest_updates(&p.existing_nodes_attrs, GraphEntity::Node, out);
    digest_updates(&p.existing_relationships_attrs, GraphEntity::Edge, out);
    // A created node carries its labels in its CREATE_NODE record already.
    digest_labels(&p.set_labels, Some(&p.created_nodes), true, out);
    digest_labels(&p.remove_labels, None, false, out);
    // Edges before nodes, so the replica unhooks first.
    digest_deleted_edges(p, out);
    if !p.deleted_nodes.is_empty() {
        out(Record::DeleteNode {
            ids: p.deleted_nodes.iter().collect(),
        });
    }
}

/// Every record, materialized.
///
/// Convenient for tests and for comparing two payloads; the emit path streams
/// through [`for_each_record`] instead, so it never holds more than one record.
#[must_use]
pub fn digest(
    p: &Pending,
    g: &AtomicRefCell<Graph>,
) -> Vec<Record> {
    let mut out = Vec::new();
    for_each_record(p, g, |record| out.push(record));
    out
}

fn digest_created_nodes(
    p: &Pending,
    out: &mut impl FnMut(Record),
) {
    // Shape key: the node's labels and its attribute ids. Labels are sorted
    // because `set_labels` keeps caller order — [7,8] and [8,7] are the same
    // shape, and emitting them ascending is what makes two engines that agree
    // on the set produce the same bytes.
    let mut groups: FxHashMap<(Vec<i32>, Vec<u16>), Vec<u64>> = FxHashMap::default();
    for id in &p.created_nodes {
        let mut labels: Vec<i32> = p
            .set_labels
            .get(&id)
            .map(|l| l.iter().map(|&v| v as i32).collect())
            .unwrap_or_default();
        labels.sort_unstable();
        labels.dedup();
        let attr_ids: Vec<u16> = p
            .new_nodes_attrs
            .get(&id)
            .map(|a| a.iter().map(|(aid, _)| *aid).collect())
            .unwrap_or_default();
        groups.entry((labels, attr_ids)).or_default().push(id);
    }
    for ((labels, attr_ids), ids) in sorted_groups(groups) {
        let rows = gather_rows(&ids, &attr_ids, &p.new_nodes_attrs);
        out(Record::CreateNode {
            ids,
            labels,
            attr_ids,
            rows,
        });
    }
}

fn digest_created_edges(
    p: &Pending,
    g: &AtomicRefCell<Graph>,
    out: &mut impl FnMut(Record),
) {
    if p.created_rels_by_type.is_empty() {
        return;
    }
    let graph = g.borrow();
    for (type_name, entries) in &p.created_rels_by_type {
        let relation_id = graph
            .get_type_id(type_name)
            .expect("created relationship type must be registered")
            .0 as i32;
        // Already partitioned by type; split further by attribute shape.
        let mut groups: FxHashMap<Vec<u16>, Vec<(u64, u64, u64)>> = FxHashMap::default();
        for &(rel_id, from, to) in entries {
            let id = u64::from(rel_id);
            let attr_ids: Vec<u16> = p
                .new_relationships_attrs
                .get(&id)
                .map(|a| a.iter().map(|(aid, _)| *aid).collect())
                .unwrap_or_default();
            groups
                .entry(attr_ids)
                .or_default()
                .push((id, u64::from(from), u64::from(to)));
        }
        for (attr_ids, mut rows) in sorted_groups(groups) {
            // IdSet requires ascending ids, and the endpoints must stay aligned
            // with them, so sort the triples together.
            rows.sort_unstable();
            let ids: Vec<u64> = rows.iter().map(|r| r.0).collect();
            let src: Vec<u64> = rows.iter().map(|r| r.1).collect();
            let dst: Vec<u64> = rows.iter().map(|r| r.2).collect();
            let values = gather_rows(&ids, &attr_ids, &p.new_relationships_attrs);
            out(Record::CreateEdge {
                ids,
                relation_id,
                src,
                dst,
                attr_ids,
                rows: values,
            });
        }
    }
}

fn digest_deleted_edges(
    p: &Pending,
    out: &mut impl FnMut(Record),
) {
    let mut groups: FxHashMap<u64, Vec<(u64, u64, u64)>> = FxHashMap::default();
    for &(rel_id, type_id, from, to) in &p.deleted_endpoints {
        groups.entry(type_id).or_default().push((
            u64::from(rel_id),
            u64::from(from),
            u64::from(to),
        ));
    }
    for (type_id, mut rows) in sorted_groups(groups) {
        rows.sort_unstable();
        out(Record::DeleteEdge {
            ids: rows.iter().map(|r| r.0).collect(),
            relation_id: type_id as i32,
            src: rows.iter().map(|r| r.1).collect(),
            dst: rows.iter().map(|r| r.2).collect(),
        });
    }
}

/// `UPDATE_NODE` / `UPDATE_EDGE`, grouped by attribute shape.
fn digest_updates(
    attrs: &FxHashMap<u64, Vec<(u16, Value)>>,
    entity: GraphEntity,
    out: &mut impl FnMut(Record),
) {
    let mut groups: FxHashMap<Vec<u16>, Vec<u64>> = FxHashMap::default();
    for (id, pairs) in attrs {
        let shape: Vec<u16> = pairs.iter().map(|(aid, _)| *aid).collect();
        groups.entry(shape).or_default().push(*id);
    }
    for (attr_ids, mut ids) in sorted_groups(groups) {
        ids.sort_unstable();
        let rows = gather_rows(&ids, &attr_ids, attrs);
        out(Record::Update {
            entity,
            ids,
            attr_ids,
            rows,
        });
    }
}

/// `SET_LABELS` / `REMOVE_LABELS`, grouped by label set.
///
/// `skip` excludes nodes whose labels already travel inside their `CREATE_NODE`
/// record.
fn digest_labels(
    labels: &FxHashMap<u64, Vec<u64>>,
    skip: Option<&RoaringTreemap>,
    add: bool,
    out: &mut impl FnMut(Record),
) {
    let mut groups: FxHashMap<Vec<i32>, Vec<u64>> = FxHashMap::default();
    for (&id, label_ids) in labels {
        if skip.is_some_and(|s| s.contains(id)) {
            continue;
        }
        let mut shape: Vec<i32> = label_ids.iter().map(|&v| v as i32).collect();
        shape.sort_unstable();
        shape.dedup();
        groups.entry(shape).or_default().push(id);
    }
    for (labels, mut ids) in sorted_groups(groups) {
        ids.sort_unstable();
        out(Record::Labels { add, ids, labels });
    }
}

/// Groups in a deterministic order.
///
/// A `HashMap` iterates arbitrarily, so without this the same query could emit
/// its records in a different order on two runs — which would defeat the
/// byte-for-byte comparison the cross-engine harness rests on.
fn sorted_groups<K: Ord, V>(groups: FxHashMap<K, V>) -> Vec<(K, V)> {
    let mut out: Vec<(K, V)> = groups.into_iter().collect();
    out.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    out
}

/// The `count x n` row-major values for one shape.
///
/// An entity missing one of the shape's attributes gets `T_NULL` in that slot —
/// unambiguous, because FalkorDB never stores a null property value.
fn gather_rows(
    ids: &[u64],
    attr_ids: &[u16],
    attrs: &FxHashMap<u64, Vec<(u16, Value)>>,
) -> Vec<Value> {
    let mut rows = Vec::with_capacity(ids.len() * attr_ids.len());
    for id in ids {
        let entity = attrs.get(id);
        for attr_id in attr_ids {
            rows.push(
                entity
                    .and_then(|a| a.iter().find(|(aid, _)| aid == attr_id))
                    .map_or(Value::Null, |(_, v)| v.clone()),
            );
        }
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effects::v3::{GraphEntity, Record, SchemaType, read_buffer};
    use crate::graph::graphblas::test_init::ensure_init;
    use crate::runtime::pending::{EFFECTS_EMIT_VERSION, Pending, emit_v3};
    use atomic_refcell::AtomicRefCell;

    fn graph() -> AtomicRefCell<Graph> {
        // GrB_init is process-wide and may only run once.
        ensure_init();
        AtomicRefCell::new(Graph::new(64, 64, 0, 0, "t"))
    }

    /// Register `n` node attributes so the ids the tests use resolve.
    fn with_attrs(
        g: &AtomicRefCell<Graph>,
        names: &[&str],
    ) {
        let mut graph = g.borrow_mut();
        for name in names {
            graph.add_node_attribute_name(name);
        }
    }

    fn build(
        p: &Pending,
        g: &AtomicRefCell<Graph>,
    ) -> Vec<Record> {
        let mut buf = Vec::new();
        build_effects_buffer_v3(&p, g, &mut buf);
        read_buffer(&buf).expect("v3 buffer must decode")
    }

    #[test]
    fn nodes_of_one_shape_become_one_record() {
        let g = graph();
        with_attrs(&g, &["a"]);
        let mut p = Pending::default();
        p.set_schema_baseline(&g);
        for id in 0..100u64 {
            p.stage_created_node(id, &[0], &[(0, Value::Int(id as i64))]);
        }

        let records = build(&p, &g);
        assert_eq!(records.len(), 1, "100 nodes, one shape, one record");
        let Record::CreateNode {
            ids,
            labels,
            attr_ids,
            rows,
        } = &records[0]
        else {
            panic!("wrong record: {:?}", records[0]);
        };
        assert_eq!(ids.len(), 100);
        assert_eq!(labels, &[0]);
        assert_eq!(attr_ids, &[0]);
        assert_eq!(rows.len(), 100);
        // Row k belongs to the k-th smallest id.
        assert_eq!(rows[7], Value::Int(7));
    }

    #[test]
    fn label_order_does_not_split_a_shape() {
        // set_labels keeps caller order, so [0,1] and [1,0] arrive differently
        // for the same set. They must land in one record, sorted.
        let g = graph();
        let mut p = Pending::default();
        p.set_schema_baseline(&g);
        p.stage_created_node(1, &[0, 1], &[]);
        p.stage_created_node(2, &[1, 0], &[]);

        let records = build(&p, &g);
        assert_eq!(records.len(), 1, "same label set, so one record");
        let Record::CreateNode { ids, labels, .. } = &records[0] else {
            panic!("wrong record");
        };
        assert_eq!(ids, &[1, 2]);
        assert_eq!(labels, &[0, 1], "emitted ascending, so bytes are canonical");
    }

    #[test]
    fn differing_shapes_split_into_separate_records() {
        let g = graph();
        with_attrs(&g, &["a", "b"]);
        let mut p = Pending::default();
        p.set_schema_baseline(&g);
        p.stage_created_node(1, &[], &[(0, Value::Int(1))]);
        p.stage_created_node(2, &[], &[(1, Value::Int(2))]);

        let records = build(&p, &g);
        assert_eq!(
            records.len(),
            2,
            "different attribute ids are different shapes"
        );
    }

    #[test]
    fn a_created_node_carries_its_labels_and_emits_no_set_labels() {
        let g = graph();
        let mut p = Pending::default();
        p.set_schema_baseline(&g);
        p.stage_created_node(1, &[3], &[]);
        // A node that already existed does need a SET_LABELS record.
        p.stage_label_change(99, &[3]);

        let records = build(&p, &g);
        assert_eq!(records.len(), 2);
        assert!(matches!(records[0], Record::CreateNode { .. }));
        let Record::Labels { add, ids, labels } = &records[1] else {
            panic!("expected SET_LABELS for the pre-existing node");
        };
        assert!(add);
        assert_eq!(ids, &[99], "the created node is not repeated here");
        assert_eq!(labels, &[3]);
    }

    #[test]
    fn updates_group_by_attribute_shape() {
        let g = graph();
        with_attrs(&g, &["a", "b"]);
        let mut p = Pending::default();
        p.set_schema_baseline(&g);
        p.stage_updated_node(1, &[(0, Value::Int(1))]);
        p.stage_updated_node(2, &[(0, Value::Int(2))]);
        p.stage_updated_node(3, &[(0, Value::Int(3)), (1, Value::Int(4))]);

        let records = build(&p, &g);
        assert_eq!(records.len(), 2, "two shapes");
        let Record::Update { entity, ids, .. } = &records[0] else {
            panic!("wrong record");
        };
        assert_eq!(*entity, GraphEntity::Node);
        assert_eq!(ids, &[1, 2]);
    }

    #[test]
    fn schema_records_precede_the_records_that_use_them() {
        // Normative ordering: an id on the wire means nothing until the replica
        // has agreed on the numbering that produced it.
        let g = graph();
        {
            let mut graph = g.borrow_mut();
            graph.get_label_id_mut("L");
            graph.add_node_attribute_name("a");
        }
        // Default counts are already zero, which is what "the schema is new"
        // means: everything the graph holds gets announced.
        let mut p = Pending::default();
        p.stage_created_node(1, &[0], &[(0, Value::Int(1))]);

        let records = build(&p, &g);
        assert_eq!(records.len(), 3);
        assert_eq!(
            records[0],
            Record::AddSchema {
                schema_type: SchemaType::Node,
                id: 0,
                name: "L".into()
            }
        );
        assert_eq!(
            records[1],
            Record::AddAttribute {
                id: 0,
                name: "a".into()
            }
        );
        assert!(matches!(records[2], Record::CreateNode { .. }));
    }

    #[test]
    fn grouping_is_deterministic() {
        // Two Pendings built in different insertion orders must produce
        // byte-identical buffers, or the cross-engine comparison is worthless.
        let g = graph();
        with_attrs(&g, &["a", "b"]);
        let mut forward = Pending::default();
        let mut backward = Pending::default();
        forward.set_schema_baseline(&g);
        backward.set_schema_baseline(&g);
        for id in 0..50u64 {
            forward.stage_created_node(id, &[], &[((id % 2) as u16, Value::Int(id as i64))]);
        }
        for id in (0..50u64).rev() {
            backward.stage_created_node(id, &[], &[((id % 2) as u16, Value::Int(id as i64))]);
        }

        let mut a = Vec::new();
        build_effects_buffer_v3(&forward, &g, &mut a);
        let mut b = Vec::new();
        build_effects_buffer_v3(&backward, &g, &mut b);
        assert_eq!(a, b, "insertion order must not reach the wire");
    }

    #[test]
    fn a_new_attribute_is_announced_once() {
        // #2459 unified the node and relationship attribute dictionaries, so
        // the two getters return the same store. v3's ADD_ATTRIBUTE carries no
        // discriminator, so announcing per entity kind would send the same
        // attribute twice under the same id.
        let g = graph();
        {
            let mut graph = g.borrow_mut();
            graph.add_node_attribute_name("a");
            graph.add_node_attribute_name("b");
        }
        // Default counts are zero, so both attributes are new.
        let p = Pending::default();

        let records = build(&p, &g);
        assert_eq!(
            records,
            vec![
                Record::AddAttribute {
                    id: 0,
                    name: "a".into()
                },
                Record::AddAttribute {
                    id: 1,
                    name: "b".into()
                },
            ]
        );
    }

    #[test]
    fn deleted_edges_group_by_relationship_type() {
        // The type has to be captured at deletion time — the edge is gone by
        // the time effects are built, and index_remove_edge_docs only knows
        // about types that carry an index.
        let g = graph();
        let mut p = Pending::default();
        p.set_schema_baseline(&g);
        p.stage_deleted_edges(vec![
            (1.into(), 0, 10.into(), 11.into()),
            (2.into(), 1, 12.into(), 13.into()),
            (3.into(), 0, 14.into(), 15.into()),
        ]);

        let records = build(&p, &g);
        assert_eq!(records.len(), 2, "two types, two records");
        let Record::DeleteEdge {
            ids,
            relation_id,
            src,
            dst,
        } = &records[0]
        else {
            panic!("wrong record: {:?}", records[0]);
        };
        assert_eq!(*relation_id, 0);
        assert_eq!(ids, &[1, 3]);
        assert_eq!(src, &[10, 14]);
        assert_eq!(dst, &[11, 15]);
    }

    #[test]
    fn the_emit_switch_selects_the_format_and_defaults_to_v2() {
        // Default must be 2. A master emitting v3 at a peer that cannot read it
        // is silent data loss, so this is a safety property, not a preference.
        assert_eq!(
            EFFECTS_EMIT_VERSION.load(std::sync::atomic::Ordering::Relaxed),
            2,
            "v3 must not be the default until every peer can read it"
        );
        assert!(!emit_v3());

        let g = graph();
        let mut p = Pending::default();
        p.set_schema_baseline(&g);
        p.stage_created_node(1, &[], &[]);

        let mut v2 = Vec::new();
        crate::effects::v2::build_effects_buffer(&p, &g, &mut v2);
        let mut v3 = Vec::new();
        build_effects_buffer_v3(&p, &g, &mut v3);

        // The version byte is what a reader dispatches on, so it has to differ.
        assert_eq!(v2[0], 2);
        assert_eq!(v3[0], 3);
    }

    #[test]
    fn the_digest_is_what_the_bytes_decode_back_to() {
        // The check the old shape could not make. Emit is now
        // Pending -> Record -> bytes and apply is bytes -> Record -> Graph, so
        // the two Record sets must be equal — a far stronger statement than
        // "the buffer is the expected length", and it compares values rather
        // than a byte count that could be right for the wrong reason.
        let g = graph();
        {
            let mut graph = g.borrow_mut();
            graph.get_label_id_mut("Person");
            graph.add_node_attribute_name("name");
            graph.add_node_attribute_name("age");
        }
        let mut p = Pending::default();
        p.set_schema_baseline(&g);
        for id in 0..200u64 {
            p.stage_created_node(id, &[0], &[(0, Value::Int(id as i64)), (1, Value::Int(30))]);
        }
        for id in 200..250u64 {
            p.stage_created_node(id, &[0], &[(0, Value::Int(id as i64))]);
        }
        p.stage_updated_node(500, &[(1, Value::Int(41))]);
        p.stage_label_change(600, &[0]);

        let digested = digest(&p, &g);
        let mut buf = Vec::new();
        build_effects_buffer_v3(&p, &g, &mut buf);

        assert_eq!(read_buffer(&buf).unwrap(), digested);
    }

    #[test]
    fn grouping_is_testable_without_bytes() {
        // Shapes split on attribute ids, and the digest says so directly
        // instead of the test having to decode a buffer to find out.
        let g = graph();
        let mut p = Pending::default();
        p.set_schema_baseline(&g);
        p.stage_created_node(1, &[], &[(0, Value::Int(1))]);
        p.stage_created_node(2, &[], &[(1, Value::Int(2))]);

        let records = digest(&p, &g);
        assert_eq!(records.len(), 2);
        for record in &records {
            let Record::CreateNode { ids, attr_ids, .. } = record else {
                panic!("expected creates, got {record:?}");
            };
            assert_eq!(ids.len(), 1, "one node per shape here");
            assert_eq!(attr_ids.len(), 1);
        }
    }

    #[test]
    fn v3_is_smaller_than_v2_for_the_motivating_query() {
        let g = graph();
        with_attrs(&g, &["v"]);
        let mut p = Pending::default();
        p.set_schema_baseline(&g);
        for id in 0..10_000u64 {
            p.stage_created_node(id, &[0], &[(0, Value::Int(id as i64))]);
        }

        let mut v2 = Vec::new();
        crate::effects::v2::build_effects_buffer(&p, &g, &mut v2);
        let mut v3 = Vec::new();
        build_effects_buffer_v3(&p, &g, &mut v3);
        println!("\n  10,000 nodes: v2 {} B -> v3 {} B", v2.len(), v3.len());
        assert!(v3.len() * 2 < v2.len(), "v2 {} v3 {}", v2.len(), v3.len());
        assert_eq!(read_buffer(&v3).unwrap().len(), 1);
    }
}
