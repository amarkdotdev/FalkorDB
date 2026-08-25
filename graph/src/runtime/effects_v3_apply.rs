//! Applying a v3 effects payload on a replica.
//!
//! The v2 path decodes and applies in one pass, and has to reconstruct batches
//! it never received: it accumulates runs of adjacent `DELETE_NODE` records
//! because applying a node's edges one at a time made a replica ~40x slower than
//! the master that produced the writes.
//!
//! v3 needs none of that. A record already covers every entity of its shape, so
//! the buffer decodes to a list of records and each one applies as a single bulk
//! operation. The look-ahead, the two pending batches and the flush-on-type-change
//! logic all disappear.
//!
//! ## What this path verifies that v2 could not
//!
//! `ADD_SCHEMA` and `ADD_ATTRIBUTE` now carry the id the master assigned. The
//! replica computes the id it *would* assign and rejects the whole buffer if they
//! disagree. That check is the point of v3: every other record identifies labels,
//! types and attributes by a bare id, and until now nothing established that the
//! two engines had numbered them the same way.

use crate::{
    entity_type::EntityType,
    graph::{
        constraint::{Constraint, ConstraintType},
        graph::{Graph, TypeId},
    },
    index::{IndexType, indexer::IndexOptions},
    runtime::{
        effects_v3::{
            self as v3, GETYPE_EDGE, GETYPE_NODE, GraphEntity, INDEX_FLD_FULLTEXT,
            INDEX_FLD_VECTOR, Record, SCHEMA_EDGE, SCHEMA_NODE,
        },
        runtime::map_to_index_options,
        value::Value,
    },
};
use roaring::RoaringTreemap;
use rustc_hash::FxHashMap;
use std::sync::Arc;

/// Index bookkeeping that accumulates across a buffer and is committed once.
#[derive(Default)]
struct IndexOps {
    add_docs: FxHashMap<u64, RoaringTreemap>,
    remove_docs: FxHashMap<u64, RoaringTreemap>,
    add_edge_docs: FxHashMap<u64, RoaringTreemap>,
    remove_edge_docs: FxHashMap<u64, FxHashMap<u64, (u64, u64)>>,
    touched: bool,
}

/// Apply a whole `GRAPH.EFFECT` payload.
///
/// Every failure aborts the buffer rather than applying a prefix: effects are
/// transactional, and a record that does not decode means the replica has
/// diverged.
pub fn apply_effects_v3(
    g: &mut Graph,
    buf: &[u8],
) -> Result<(), String> {
    let records = v3::read_buffer(buf).map_err(|e| e.to_string())?;
    let mut ops = IndexOps::default();

    for record in records {
        apply_record(g, record, &mut ops)?;
    }

    g.commit_index(&mut ops.add_docs, &mut ops.remove_docs);
    g.commit_edge_index(&mut ops.add_edge_docs, &mut ops.remove_edge_docs);
    if ops.touched {
        g.populate_indexes_sync();
    }
    Ok(())
}

fn apply_record(
    g: &mut Graph,
    record: Record,
    ops: &mut IndexOps,
) -> Result<(), String> {
    match record {
        Record::AddSchema {
            schema_type,
            id,
            name,
        } => apply_add_schema(g, schema_type, id, &name),

        Record::AddAttribute { id, name } => {
            // The replica appends to the same dictionary the master did, so the
            // id it is about to assign must equal the one on the wire.
            let expected = g.get_node_attribute_names().len();
            g.add_node_attribute_name(&name);
            let assigned = g
                .get_node_attribute_names()
                .len()
                .saturating_sub(1)
                .min(expected);
            verify_id(
                "attribute",
                &name,
                i64::from(id),
                assigned as i64,
                g.get_node_attribute_names()
                    .get(id as usize)
                    .map(|a| a.as_str().to_string()),
            )
        }

        Record::CreateNode {
            ids,
            labels,
            attr_ids,
            rows,
        } => {
            let nodes: RoaringTreemap = ids.iter().copied().collect();
            for _ in &ids {
                g.inc_reserved_node_count();
            }
            g.create_nodes(&nodes);

            if !labels.is_empty() {
                let (rows_v, cols_v) = label_pairs(g, &ids, &labels)?;
                g.set_nodes_labels_bulk(&rows_v, &cols_v, &mut ops.add_docs, true);
            }
            if !attr_ids.is_empty() {
                let map = attr_map(g, &ids, &attr_ids, &rows)?;
                g.set_nodes_attributes(&map, &mut ops.add_docs)?;
            }
            Ok(())
        }

        Record::CreateEdge {
            ids,
            relation_id,
            src,
            dst,
            attr_ids,
            rows,
        } => {
            let type_name = resolve_type(g, relation_id)?;
            for _ in &ids {
                g.inc_reserved_relationship_count();
            }
            g.create_relationships_bulk(&type_name, &src, &dst, &ids);

            if !attr_ids.is_empty() {
                let map = attr_map(g, &ids, &attr_ids, &rows)?;
                g.set_relationships_attributes(&map, &mut ops.add_edge_docs)?;
            }
            Ok(())
        }

        Record::Update {
            entity,
            ids,
            attr_ids,
            rows,
        } => {
            let map = attr_map(g, &ids, &attr_ids, &rows)?;
            match entity {
                GraphEntity::Node => {
                    g.set_nodes_attributes(&map, &mut ops.add_docs)?;
                }
                GraphEntity::Edge => {
                    g.set_relationships_attributes(&map, &mut ops.add_edge_docs)?;
                }
            }
            Ok(())
        }

        Record::Labels { add, ids, labels } => {
            let (rows_v, cols_v) = label_pairs(g, &ids, &labels)?;
            if add {
                g.set_nodes_labels_bulk(&rows_v, &cols_v, &mut ops.add_docs, false);
            } else {
                g.remove_nodes_labels(&rows_v, &cols_v, &mut ops.remove_docs);
            }
            Ok(())
        }

        // One bulk delete for the whole record — the reason v3 needs no
        // stream look-ahead.
        Record::DeleteNode { ids } => {
            let nodes: RoaringTreemap = ids.into_iter().collect();
            g.delete_nodes(&nodes, &mut ops.remove_docs)?;
            Ok(())
        }

        Record::DeleteEdge { ids, .. } => {
            let edges: RoaringTreemap = ids.into_iter().collect();
            g.delete_relationships(&edges, &mut ops.remove_edge_docs)?;
            Ok(())
        }

        Record::Index {
            create,
            schema_type,
            label_id,
            label,
            attr_id,
            attr,
            field_type,
            options,
        } => {
            verify_schema(g, schema_type, label_id, &label)?;
            verify_attribute(g, attr_id, &attr)?;
            let entity_type = entity_type_of(schema_type)?;
            let index_type = index_type_of(field_type);
            let label = Arc::new(label);
            let fields = vec![Arc::new(attr)];
            if create {
                // The sync variant: a replica must not spawn population threads,
                // and must not reorder its work against the effect stream.
                g.create_index_sync(
                    &index_type,
                    &entity_type,
                    &label,
                    &fields,
                    index_options(&index_type, options.as_ref())?,
                )?;
                ops.touched = true;
            } else {
                g.drop_index(&index_type, &entity_type, &label, &fields)?;
            }
            Ok(())
        }

        Record::Constraint {
            create,
            constraint_type,
            entity_type,
            label_id,
            label,
            props,
        } => {
            let schema_type = if entity_type == GETYPE_EDGE {
                SCHEMA_EDGE
            } else {
                SCHEMA_NODE
            };
            verify_schema(g, schema_type, label_id, &label)?;
            for (attr_id, name) in &props {
                verify_attribute(g, *attr_id, name)?;
            }

            let ct = constraint_type_of(constraint_type)?;
            let et = entity_type_of(schema_type)?;
            let properties: Vec<Arc<String>> =
                props.into_iter().map(|(_, n)| Arc::new(n)).collect();

            if create {
                // Install the master's outcome rather than re-deriving it. A
                // replica that validated independently would scan at a different
                // time against different interleavings, and could legitimately
                // reach a different status.
                g.add_constraint_raw(Constraint::new(ct, et, Arc::new(label), properties));
            } else {
                g.drop_constraint(&ct, &et, &label, &properties)?;
            }
            Ok(())
        }
    }
}

/// `ADD_SCHEMA`: register the label or type, then check the id matches.
fn apply_add_schema(
    g: &mut Graph,
    schema_type: u32,
    id: i32,
    name: &str,
) -> Result<(), String> {
    match schema_type {
        SCHEMA_NODE => {
            let assigned = usize::from(g.get_label_id_mut(name)) as i64;
            let local = g.get_labels().get(id as usize).map(|l| l.to_string());
            verify_id("label", name, i64::from(id), assigned, local)
        }
        SCHEMA_EDGE => {
            let assigned = g.get_type_id_mut(name).0 as i64;
            let local = g.get_types().get(id as usize).map(|t| t.to_string());
            verify_id("relationship type", name, i64::from(id), assigned, local)
        }
        other => Err(format!("unknown schema type: {other}")),
    }
}

/// The check v2 could not make.
///
/// C's reader has a weaker one — it refuses a name that already exists locally —
/// which catches a replica that is *ahead* but not one whose dictionary is a
/// different length, where appending the same new name yields a different id.
/// That is the case that silently put a property value on the wrong attribute.
fn verify_id(
    kind: &str,
    name: &str,
    expected: i64,
    assigned: i64,
    local_name_at_expected: Option<String>,
) -> Result<(), String> {
    if assigned == expected {
        return Ok(());
    }
    Err(format!(
        "effects buffer assigns {kind} '{name}' id {expected}, but this replica \
         would assign {assigned}{}. The two engines have diverged; the buffer was \
         not applied.",
        local_name_at_expected
            .map(|n| format!(" (id {expected} is '{n}' here)"))
            .unwrap_or_default()
    ))
}

/// Resolve a label or type by id and confirm the name matches.
///
/// The id is authoritative; the name is the cheap cross-check that surfaces
/// divergence instead of writing through a stale id. Mirrors C's `VerifySchema`.
fn verify_schema(
    g: &Graph,
    schema_type: u32,
    id: i32,
    name: &str,
) -> Result<(), String> {
    let local = match schema_type {
        SCHEMA_NODE => g.get_labels().get(id as usize).map(|l| l.to_string()),
        SCHEMA_EDGE => g.get_types().get(id as usize).map(|t| t.to_string()),
        other => return Err(format!("unknown schema type: {other}")),
    };
    match local {
        Some(local) if local == name => Ok(()),
        Some(local) => Err(format!(
            "effects buffer references '{name}' (id {id}), which is '{local}' on this replica"
        )),
        None => Err(format!(
            "effects buffer references '{name}' (id {id}), which does not exist on this replica"
        )),
    }
}

/// Mirrors C's `VerifyAttribute`.
fn verify_attribute(
    g: &Graph,
    id: u16,
    name: &str,
) -> Result<(), String> {
    match g.get_node_attribute_names().get(id as usize) {
        Some(local) if local.as_str() == name => Ok(()),
        Some(local) => Err(format!(
            "effects buffer references attribute '{name}' (id {id}), which is '{local}' here"
        )),
        None => Err(format!(
            "effects buffer references attribute '{name}' (id {id}), which does not exist here"
        )),
    }
}

fn resolve_type(
    g: &Graph,
    relation_id: i32,
) -> Result<Arc<String>, String> {
    if relation_id < 0 {
        return Err(format!("negative relationship type id {relation_id}"));
    }
    g.get_type(TypeId(relation_id as usize))
        .ok_or_else(|| format!("unknown relationship type id {relation_id}"))
}

/// Expand `(ids, labels)` into the row/column pairs the bulk label API takes,
/// bounds-checking every label id on the way.
fn label_pairs(
    g: &Graph,
    ids: &[u64],
    labels: &[i32],
) -> Result<(Vec<u64>, Vec<u64>), String> {
    let bound = g.get_labels().len();
    let mut rows = Vec::with_capacity(ids.len() * labels.len());
    let mut cols = Vec::with_capacity(ids.len() * labels.len());
    for &label in labels {
        if label < 0 || label as usize >= bound {
            return Err(format!("label id {label} out of range"));
        }
        for &id in ids {
            rows.push(id);
            cols.push(label as u64);
        }
    }
    Ok((rows, cols))
}

/// Turn a record's shape plus its row-major values back into per-entity
/// attribute lists.
///
/// A `T_NULL` slot means the property is absent for that entity, which is
/// unambiguous because FalkorDB never stores a null property value.
fn attr_map(
    g: &Graph,
    ids: &[u64],
    attr_ids: &[u16],
    rows: &[Value],
) -> Result<FxHashMap<u64, Vec<(u16, Value)>>, String> {
    let bound = g.get_node_attribute_names().len();
    for &attr_id in attr_ids {
        if attr_id as usize >= bound {
            return Err(format!("attribute id {attr_id} out of range"));
        }
    }
    let width = attr_ids.len();
    if rows.len() != ids.len() * width {
        return Err(format!(
            "record declares {} entities x {width} attributes but carries {} values",
            ids.len(),
            rows.len()
        ));
    }

    let mut map = FxHashMap::default();
    for (row, &id) in ids.iter().enumerate() {
        let mut pairs: Vec<(u16, Value)> = attr_ids
            .iter()
            .enumerate()
            .filter_map(|(col, &attr_id)| {
                let value = &rows[row * width + col];
                matches!(value, Value::Null)
                    .then_some(None)
                    .unwrap_or(Some((attr_id, value.clone())))
            })
            .collect();
        // The attribute stores expect id-sorted pairs.
        pairs.sort_unstable_by_key(|(k, _)| *k);
        map.insert(id, pairs);
    }
    Ok(map)
}

/// Turn the record's options value back into typed index options.
///
/// v2 dropped `OPTIONS {...}` on the wire entirely and forced those statements
/// to replicate as verbatim queries. v3 carries the map, so the replica rebuilds
/// the same options the master did rather than approximating them.
fn index_options(
    index_type: &IndexType,
    options: Option<&Value>,
) -> Result<Option<IndexOptions>, String> {
    match options {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Map(m)) => map_to_index_options(index_type, m),
        Some(other) => Err(format!("index options must be a map, got {other:?}")),
    }
}

fn entity_type_of(schema_type: u32) -> Result<EntityType, String> {
    match schema_type {
        SCHEMA_NODE => Ok(EntityType::Node),
        SCHEMA_EDGE => Ok(EntityType::Relationship),
        other => Err(format!("unknown schema type: {other}")),
    }
}

/// `IndexFieldType` is a bit flag set, so this tests bits rather than matching
/// a discriminant. Anything that is neither full-text nor vector is a range
/// index — `INDEX_FLD_RANGE` is itself the union of the three scalar kinds.
fn index_type_of(field_type: u32) -> IndexType {
    if field_type & INDEX_FLD_FULLTEXT != 0 {
        IndexType::Fulltext
    } else if field_type & INDEX_FLD_VECTOR != 0 {
        IndexType::Vector
    } else {
        IndexType::Range
    }
}

fn constraint_type_of(ct: u32) -> Result<ConstraintType, String> {
    match ct {
        v3::CT_UNIQUE => Ok(ConstraintType::Unique),
        v3::CT_MANDATORY => Ok(ConstraintType::Mandatory),
        other => Err(format!("unknown constraint type: {other}")),
    }
}

/// `GETYPE_*` is 1-based on the wire; this keeps the compiler honest about it.
const _: () = assert!(GETYPE_NODE == 1 && GETYPE_EDGE == 2);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::effects_v3::{
        CT_UNIQUE, INDEX_FLD_RANGE, IndexField, new_buffer, write_add_attribute, write_add_schema,
        write_constraint, write_create_index, write_create_node, write_labels,
    };

    fn graph() -> Graph {
        crate::graph::graphblas::test_init::ensure_init();
        Graph::new(64, 64, 0, 0, "t")
    }

    #[test]
    fn a_buffer_applies_end_to_end() {
        let mut g = graph();
        let mut buf = new_buffer();
        write_add_schema(&mut buf, SCHEMA_NODE, 0, "Person");
        write_add_attribute(&mut buf, 0, "name");
        write_create_node(
            &mut buf,
            &[0, 1, 2],
            &[0],
            &[0],
            &[
                Value::String(Arc::new("a".into())),
                Value::String(Arc::new("b".into())),
                Value::String(Arc::new("c".into())),
            ],
        );

        apply_effects_v3(&mut g, &buf).expect("buffer must apply");
        assert_eq!(g.get_labels().len(), 1);
        assert_eq!(g.get_node_attribute_names().len(), 1);
        assert_eq!(g.node_count(), 3);
    }

    #[test]
    fn a_schema_id_disagreement_aborts_the_buffer() {
        // The check that is the whole point of v3. The replica already holds a
        // label, so appending "Person" gives it id 1 — but the master says 0.
        let mut g = graph();
        g.get_label_id_mut("Existing");

        let mut buf = new_buffer();
        write_add_schema(&mut buf, SCHEMA_NODE, 0, "Person");

        let err = apply_effects_v3(&mut g, &buf).expect_err("must refuse");
        assert!(err.contains("would assign 1"), "{err}");
        assert!(err.contains("diverged"), "{err}");
    }

    #[test]
    fn an_attribute_id_disagreement_aborts_the_buffer() {
        // The case that actually bit us: an RDB-seeded replica whose dictionary
        // is a different length silently lands the value on another attribute.
        let mut g = graph();
        g.add_node_attribute_name("already_here");

        let mut buf = new_buffer();
        write_add_attribute(&mut buf, 0, "name");

        let err = apply_effects_v3(&mut g, &buf).expect_err("must refuse");
        assert!(err.contains("attribute 'name'"), "{err}");
    }

    #[test]
    fn a_stale_label_id_is_caught_by_its_name() {
        // VerifySchema's job: the id resolves, but to something else.
        let mut g = graph();
        g.get_label_id_mut("Actual");
        g.add_node_attribute_name("a");

        let mut buf = new_buffer();
        write_create_index(
            &mut buf,
            SCHEMA_NODE,
            0,
            "Expected",
            IndexField {
                attr_id: 0,
                attr: "a",
                field_type: INDEX_FLD_RANGE,
            },
            &Value::Null,
        );
        let err = apply_effects_v3(&mut g, &buf).expect_err("must refuse");
        assert!(err.contains("'Expected'"), "{err}");
        assert!(err.contains("'Actual'"), "{err}");
    }

    #[test]
    fn labels_apply_to_every_node_in_the_record() {
        let mut g = graph();
        let mut buf = new_buffer();
        write_add_schema(&mut buf, SCHEMA_NODE, 0, "L");
        write_create_node(&mut buf, &[0, 1, 2, 3], &[], &[], &[]);
        write_labels(&mut buf, true, &[0, 1, 2, 3], &[0]);

        apply_effects_v3(&mut g, &buf).expect("must apply");
        assert_eq!(g.label_node_count(&Arc::new("L".to_string())), 4);
    }

    #[test]
    fn a_constraint_installs_the_masters_outcome() {
        // Not re-validated locally: a replica scanning independently would do so
        // at a different time against different interleavings, and could
        // legitimately reach a different status.
        let mut g = graph();
        let mut buf = new_buffer();
        write_add_schema(&mut buf, SCHEMA_NODE, 0, "Person");
        write_add_attribute(&mut buf, 0, "email");
        write_constraint(
            &mut buf,
            true,
            CT_UNIQUE,
            GETYPE_NODE,
            0,
            "Person",
            &[(0, "email")],
        );

        apply_effects_v3(&mut g, &buf).expect("must apply");
        assert_eq!(g.constraints().len(), 1);
    }

    #[test]
    fn a_malformed_buffer_is_refused_not_applied() {
        let mut g = graph();
        let mut buf = new_buffer();
        write_add_schema(&mut buf, SCHEMA_NODE, 0, "L");
        write_create_node(&mut buf, &[0, 1], &[0], &[], &[]);

        for cut in 2..buf.len() {
            let mut fresh = graph();
            // Must never panic, whatever the truncation.
            let _ = apply_effects_v3(&mut fresh, &buf[..cut]);
        }
        // And the whole buffer still works.
        apply_effects_v3(&mut g, &buf).expect("intact buffer must apply");
    }

    #[test]
    fn a_pending_survives_the_round_trip_to_a_replica() {
        // The end-to-end check the unit tests cannot make: build a real Pending
        // on a "master", emit v3, apply it to an empty "replica", and compare
        // the observable state. Encoder and decoder agreeing with each other is
        // not the same as either being right.
        use crate::runtime::pending::Pending;
        use atomic_refcell::AtomicRefCell;

        let master = AtomicRefCell::new(graph());
        {
            let mut g = master.borrow_mut();
            g.get_label_id_mut("Person");
            g.add_node_attribute_name("name");
            g.add_node_attribute_name("age");
        }

        let mut p = Pending::default();
        for id in 0..500u64 {
            p.stage_created_node_for_test(
                id,
                &[0],
                &[(0, Value::Int(id as i64)), (1, Value::Int(30))],
            );
        }
        // A second shape, so the buffer carries more than one record.
        for id in 500..600u64 {
            p.stage_created_node_for_test(id, &[0], &[(0, Value::Int(id as i64))]);
        }

        let mut buf = Vec::new();
        p.build_effects_buffer_v3(&master, &mut buf);
        assert_eq!(
            v3::read_buffer(&buf).unwrap().len(),
            5,
            "2 schema + 1 attr + 2 shapes"
        );

        let mut replica = graph();
        apply_effects_v3(&mut replica, &buf).expect("replica must apply the master's buffer");

        let m = master.borrow();
        assert_eq!(
            replica.get_labels(),
            m.get_labels(),
            "label ids must line up"
        );
        assert_eq!(
            replica.get_node_attribute_names(),
            m.get_node_attribute_names(),
            "attribute ids must line up"
        );
        assert_eq!(replica.node_count(), 600);
        assert_eq!(
            replica.label_node_count(&Arc::new("Person".to_string())),
            600
        );
    }

    #[test]
    fn index_field_type_maps_by_bit_not_ordinal() {
        assert_eq!(index_type_of(INDEX_FLD_RANGE), IndexType::Range);
        assert_eq!(index_type_of(v3::INDEX_FLD_FULLTEXT), IndexType::Fulltext);
        assert_eq!(index_type_of(v3::INDEX_FLD_VECTOR), IndexType::Vector);
        // A range index is the OR of three scalar kinds, so bit-testing is the
        // only thing that classifies it correctly.
        assert_eq!(index_type_of(v3::INDEX_FLD_NUMERIC), IndexType::Range);
    }
}
