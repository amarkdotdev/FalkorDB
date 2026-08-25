//! The records themselves, and reading a whole payload.

use crate::runtime::value::Value;

use super::*;

// ── records ──

/// Open a batchable record: its opcode, then how many entities it covers.
///
/// Every block that follows takes its length from this one count — it is never
/// repeated. `count == 1` and `count == 10_000` are the same record, which is
/// why no record type is left un-batchable and the decoder has one shape per
/// opcode.
fn write_header(
    buf: &mut Vec<u8>,
    opcode: u32,
    count: usize,
) {
    debug_assert!(count <= u32::MAX as usize, "record count must fit in u32");
    write_u32(buf, opcode);
    write_u32(buf, count as u32);
}

/// `3 CREATE_NODE` — `count · IdSet · LabelSet · AttrSet`.
///
/// `ids` must be ascending and unique; `rows` is `ids.len() × attr_ids.len()`
/// values in row-major order. Row *k* belongs to the k-th smallest id.
pub fn write_create_node(
    buf: &mut Vec<u8>,
    ids: &[u64],
    labels: &[i32],
    attr_ids: &[u16],
    rows: &[Value],
) {
    write_header(buf, EFFECT_CREATE_NODE, ids.len());
    write_id_set(buf, ids);
    write_label_set(buf, labels);
    write_attr_set(buf, attr_ids, rows);
}

/// `4 CREATE_EDGE` — `count · IdSet · RelType · IdList(src) · IdList(dst) · AttrSet`.
///
/// The endpoints are `IdList`s, not `IdSet`s: many edges share a source, and
/// each endpoint must stay positionally aligned with its edge id.
pub fn write_create_edge(
    buf: &mut Vec<u8>,
    ids: &[u64],
    relation_id: i32,
    src: &[u64],
    dst: &[u64],
    attr_ids: &[u16],
    rows: &[Value],
) {
    debug_assert_eq!(ids.len(), src.len(), "one source per edge");
    debug_assert_eq!(ids.len(), dst.len(), "one destination per edge");
    write_header(buf, EFFECT_CREATE_EDGE, ids.len());
    write_id_set(buf, ids);
    write_rel_type(buf, relation_id);
    write_id_list(buf, src);
    write_id_list(buf, dst);
    write_attr_set(buf, attr_ids, rows);
}

/// `1 UPDATE_NODE` / `2 UPDATE_EDGE` — `count · IdSet · AttrSet`.
///
/// v2 wrote one record per *(entity, attribute)*; v3 writes one per shape. A
/// property being removed is `T_NULL` in its slot.
pub fn write_update(
    buf: &mut Vec<u8>,
    entity: GraphEntity,
    ids: &[u64],
    attr_ids: &[u16],
    rows: &[Value],
) {
    write_header(
        buf,
        entity.pick(EFFECT_UPDATE_NODE, EFFECT_UPDATE_EDGE),
        ids.len(),
    );
    write_id_set(buf, ids);
    write_attr_set(buf, attr_ids, rows);
}

/// `5 DELETE_NODE` — `count · IdSet`.
pub fn write_delete_node(
    buf: &mut Vec<u8>,
    ids: &[u64],
) {
    write_header(buf, EFFECT_DELETE_NODE, ids.len());
    write_id_set(buf, ids);
}

/// `6 DELETE_EDGE` — `count · IdSet · RelType · IdList(src) · IdList(dst)`.
///
/// C needs the endpoints to locate the edge in its adjacency matrices, so they
/// travel even though the edge id alone identifies it.
pub fn write_delete_edge(
    buf: &mut Vec<u8>,
    ids: &[u64],
    relation_id: i32,
    src: &[u64],
    dst: &[u64],
) {
    debug_assert_eq!(ids.len(), src.len(), "one source per edge");
    debug_assert_eq!(ids.len(), dst.len(), "one destination per edge");
    write_header(buf, EFFECT_DELETE_EDGE, ids.len());
    write_id_set(buf, ids);
    write_rel_type(buf, relation_id);
    write_id_list(buf, src);
    write_id_list(buf, dst);
}

/// `7 SET_LABELS` / `8 REMOVE_LABELS` — `count · IdSet · LabelSet`.
///
/// The labels, and all their nodes — not one `(node, label)` pair per node.
/// Grouping is what lets the node ids be a contiguous `IdSet`: 10,000 nodes
/// gaining one label is 46 bytes grouped against 120,008 as pairs. It is sound
/// because label add and remove are idempotent set operations, so order within
/// the record carries nothing.
pub fn write_labels(
    buf: &mut Vec<u8>,
    add: bool,
    ids: &[u64],
    labels: &[i32],
) {
    let opcode = if add {
        EFFECT_SET_LABELS
    } else {
        EFFECT_REMOVE_LABELS
    };
    write_header(buf, opcode, ids.len());
    write_id_set(buf, ids);
    write_label_set(buf, labels);
}

/// `9 ADD_SCHEMA` — `SchemaType · id · name`. Singular, so no count.
///
/// **v3 adds the id.** v2 sent the name alone and let the replica infer the id
/// from its own append order, which is the one assumption every other record's
/// bare ids rest on — and the only one that could not be checked. The replica
/// computes the id it would assign and rejects the buffer if it disagrees.
pub fn write_add_schema(
    buf: &mut Vec<u8>,
    schema_type: u32,
    id: i32,
    name: &str,
) {
    write_u32(buf, EFFECT_ADD_SCHEMA);
    write_u32(buf, schema_type);
    write_label_id(buf, id);
    write_string(buf, name);
}

/// `10 ADD_ATTRIBUTE` — `id · name`. Singular, so no count.
///
/// **v3 adds the id**, for the reason on [`write_add_schema`]. This is the
/// record whose missing id let a property effect carrying a bare `u16` land on
/// the wrong attribute of an RDB-seeded replica.
///
/// No node/relationship discriminator: C has one attribute dictionary, and
/// #2459 unified Rust's two to match.
pub fn write_add_attribute(
    buf: &mut Vec<u8>,
    id: u16,
    name: &str,
) {
    write_u32(buf, EFFECT_ADD_ATTRIBUTE);
    write_u16(buf, id);
    write_string(buf, name);
}

/// One indexed field: which attribute, and how it is indexed.
///
/// Grouped rather than passed as three loose arguments because `label_id` and
/// `attr_id` are both small integers next to each other on the wire —
/// transposing them is precisely the kind of mistake that does not fail on the
/// far side.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexField<'a> {
    pub attr_id: u16,
    pub attr: &'a str,
    pub field_type: u32,
}

/// `11 CREATE_INDEX` — **one record per field**, matching C.
///
/// Unchanged from v2 and deliberately not batched: index DDL is rare, and C
/// applies it idempotently per field (`Index_SetLanguage` tolerates a re-set,
/// `Index_SetStopwords` is guarded). Splitting a multi-field index into one
/// record per field is what lets a replica apply the fields in any order.
///
/// Both the label and the attribute travel as `(id, name)`. The id is
/// authoritative and the name is the cross-check that turns a numbering
/// disagreement into a loud failure rather than a write to the wrong attribute.
pub fn write_create_index(
    buf: &mut Vec<u8>,
    schema_type: u32,
    label_id: i32,
    label: &str,
    field: IndexField<'_>,
    options: &Value,
) {
    write_u32(buf, EFFECT_CREATE_INDEX);
    write_u32(buf, schema_type);
    write_label_id(buf, label_id);
    write_string(buf, label);
    write_u16(buf, field.attr_id);
    write_string(buf, field.attr);
    write_u32(buf, field.field_type);
    write_value(buf, options);
}

/// `12 DROP_INDEX` — mirrors [`write_create_index`] without the options.
pub fn write_drop_index(
    buf: &mut Vec<u8>,
    schema_type: u32,
    label_id: i32,
    label: &str,
    field: IndexField<'_>,
) {
    write_u32(buf, EFFECT_DROP_INDEX);
    write_u32(buf, schema_type);
    write_label_id(buf, label_id);
    write_string(buf, label);
    write_u16(buf, field.attr_id);
    write_string(buf, field.attr);
    write_u32(buf, field.field_type);
}

/// `13 CREATE_CONSTRAINT` / `14 DROP_CONSTRAINT`.
///
/// The property count is a **`u8`**, not the `u16` used elsewhere — C's
/// `uint8_t n`. Each property travels as `(id, name)` for the same reason the
/// index records do.
pub fn write_constraint(
    buf: &mut Vec<u8>,
    create: bool,
    constraint_type: u32,
    entity_type: u32,
    label_id: i32,
    label: &str,
    props: &[(u16, &str)],
) {
    debug_assert!(
        props.len() <= u8::MAX as usize,
        "C reads the count as uint8"
    );
    write_u32(
        buf,
        if create {
            EFFECT_CREATE_CONSTRAINT
        } else {
            EFFECT_DROP_CONSTRAINT
        },
    );
    write_u32(buf, constraint_type);
    write_u32(buf, entity_type);
    write_label_id(buf, label_id);
    write_string(buf, label);
    write_u8(buf, props.len() as u8);
    for (attr_id, name) in props {
        write_u16(buf, *attr_id);
        write_string(buf, name);
    }
}

/// Which entity kind an update targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphEntity {
    Node,
    Edge,
}

impl GraphEntity {
    const fn pick(
        self,
        node: u32,
        edge: u32,
    ) -> u32 {
        match self {
            Self::Node => node,
            Self::Edge => edge,
        }
    }
}

/// One decoded record.
#[derive(Clone, Debug, PartialEq)]
pub enum Record {
    Update {
        entity: GraphEntity,
        ids: Vec<u64>,
        attr_ids: Vec<u16>,
        rows: Vec<Value>,
    },
    CreateNode {
        ids: Vec<u64>,
        labels: Vec<i32>,
        attr_ids: Vec<u16>,
        rows: Vec<Value>,
    },
    CreateEdge {
        ids: Vec<u64>,
        relation_id: i32,
        src: Vec<u64>,
        dst: Vec<u64>,
        attr_ids: Vec<u16>,
        rows: Vec<Value>,
    },
    DeleteNode {
        ids: Vec<u64>,
    },
    DeleteEdge {
        ids: Vec<u64>,
        relation_id: i32,
        src: Vec<u64>,
        dst: Vec<u64>,
    },
    Labels {
        add: bool,
        ids: Vec<u64>,
        labels: Vec<i32>,
    },
    AddSchema {
        schema_type: u32,
        id: i32,
        name: String,
    },
    AddAttribute {
        id: u16,
        name: String,
    },
    Index {
        create: bool,
        schema_type: u32,
        label_id: i32,
        label: String,
        attr_id: u16,
        attr: String,
        field_type: u32,
        /// `None` on a drop, which carries no options.
        options: Option<Value>,
    },
    Constraint {
        create: bool,
        constraint_type: u32,
        entity_type: u32,
        label_id: i32,
        label: String,
        props: Vec<(u16, String)>,
    },
}

/// Read one record, whatever its opcode.
pub fn read_record(r: &mut Reader<'_>) -> Result<Record, DecodeError> {
    let opcode = r.u32()?;
    // The singular records carry no count, so read them before asking for one.
    match opcode {
        EFFECT_ADD_SCHEMA => {
            let schema_type = r.u32()?;
            let id = r.i32()?;
            return Ok(Record::AddSchema {
                schema_type,
                id,
                name: r.string()?,
            });
        }
        EFFECT_ADD_ATTRIBUTE => {
            let id = r.u16()?;
            return Ok(Record::AddAttribute {
                id,
                name: r.string()?,
            });
        }
        EFFECT_CREATE_INDEX | EFFECT_DROP_INDEX => {
            let create = opcode == EFFECT_CREATE_INDEX;
            let schema_type = r.u32()?;
            let label_id = r.i32()?;
            let label = r.string()?;
            let attr_id = r.u16()?;
            let attr = r.string()?;
            let field_type = r.u32()?;
            let options = if create { Some(read_value(r)?) } else { None };
            return Ok(Record::Index {
                create,
                schema_type,
                label_id,
                label,
                attr_id,
                attr,
                field_type,
                options,
            });
        }
        EFFECT_CREATE_CONSTRAINT | EFFECT_DROP_CONSTRAINT => {
            let create = opcode == EFFECT_CREATE_CONSTRAINT;
            let constraint_type = r.u32()?;
            let entity_type = r.u32()?;
            let label_id = r.i32()?;
            let label = r.string()?;
            let n = r.u8()?;
            // Each pair is at least 2 bytes of id plus an 8-byte length.
            let n = r.guard_count(u64::from(n), 10)?;
            let mut props = Vec::with_capacity(n);
            for _ in 0..n {
                let attr_id = r.u16()?;
                props.push((attr_id, r.string()?));
            }
            return Ok(Record::Constraint {
                create,
                constraint_type,
                entity_type,
                label_id,
                label,
                props,
            });
        }
        _ => {}
    }

    let count = r.u32()?;
    let record = match opcode {
        EFFECT_UPDATE_NODE | EFFECT_UPDATE_EDGE => {
            let ids = read_id_set(r, count)?;
            let (attr_ids, rows) = read_attr_set(r, count)?;
            Record::Update {
                entity: if opcode == EFFECT_UPDATE_NODE {
                    GraphEntity::Node
                } else {
                    GraphEntity::Edge
                },
                ids,
                attr_ids,
                rows,
            }
        }
        EFFECT_CREATE_NODE => {
            let ids = read_id_set(r, count)?;
            let labels = read_label_set(r)?;
            let (attr_ids, rows) = read_attr_set(r, count)?;
            Record::CreateNode {
                ids,
                labels,
                attr_ids,
                rows,
            }
        }
        EFFECT_CREATE_EDGE => {
            let ids = read_id_set(r, count)?;
            let relation_id = read_rel_type(r)?;
            let src = read_id_list(r, count)?;
            let dst = read_id_list(r, count)?;
            let (attr_ids, rows) = read_attr_set(r, count)?;
            Record::CreateEdge {
                ids,
                relation_id,
                src,
                dst,
                attr_ids,
                rows,
            }
        }
        EFFECT_DELETE_NODE => Record::DeleteNode {
            ids: read_id_set(r, count)?,
        },
        EFFECT_DELETE_EDGE => {
            let ids = read_id_set(r, count)?;
            let relation_id = read_rel_type(r)?;
            let src = read_id_list(r, count)?;
            let dst = read_id_list(r, count)?;
            Record::DeleteEdge {
                ids,
                relation_id,
                src,
                dst,
            }
        }
        EFFECT_SET_LABELS | EFFECT_REMOVE_LABELS => Record::Labels {
            add: opcode == EFFECT_SET_LABELS,
            ids: read_id_set(r, count)?,
            labels: read_label_set(r)?,
        },
        other => return Err(DecodeError::BadOpcode(other)),
    };
    Ok(record)
}

/// Read a whole `GRAPH.EFFECT` payload: the version byte, then every record.
///
/// One command carries one payload carries many records — batching removes
/// per-record framing, it never multiplies commands.
pub fn read_buffer(buf: &[u8]) -> Result<Vec<Record>, DecodeError> {
    let mut r = Reader::new(buf);
    let version = r.u8()?;
    if version != EFFECTS_VERSION {
        return Err(DecodeError::UnsupportedVersion(version));
    }
    let mut records = Vec::new();
    while !r.is_empty() {
        records.push(read_record(&mut r)?);
    }
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::super::hex;
    use super::*;

    // ── records ──

    #[test]
    fn create_node_pins_its_bytes_at_count_one() {
        // One node, label 7, one int property. The smallest possible record.
        let mut buf = new_buffer();
        write_create_node(&mut buf, &[0], &[7], &[0], &[Value::Int(1)]);
        assert_eq!(
            hex(&buf),
            concat!(
                "03 ",                                 // version
                "03 00 00 00 01 00 00 00 ",            // CREATE_NODE, count = 1
                "00 01 00 ",                           // IdSet: raw, width 1, id 0
                "01 00 07 00 00 00 ",                  // LabelSet: n = 1, label 7
                "01 00 00 00 ",                        // AttrSet: n = 1, attr 0
                "00 20 00 00 01 00 00 00 00 00 00 00"  // T_INT64, 1
            )
        );
    }

    #[test]
    fn create_node_is_the_same_record_at_ten_thousand() {
        // Same opcode, same blocks, one count field apart — which is the whole
        // point of folding the count in rather than adding a batch opcode.
        let ids: Vec<u64> = (0..10_000).collect();
        let rows: Vec<Value> = (0..10_000).map(|i| Value::Int(i as i64)).collect();
        let mut buf = new_buffer();
        write_create_node(&mut buf, &ids, &[7], &[0], &rows);

        assert_eq!(&buf[1..5], &EFFECT_CREATE_NODE.to_le_bytes());
        assert_eq!(&buf[5..9], &10_000_u32.to_le_bytes());
        // 10,000 consecutive ids collapse to one roaring run.
        assert_eq!(buf[9], ENC_COMPRESSED);

        let records = read_buffer(&buf).unwrap();
        assert_eq!(records.len(), 1);
        let Record::CreateNode {
            ids: got, rows: r, ..
        } = &records[0]
        else {
            panic!("wrong record: {:?}", records[0]);
        };
        assert_eq!(got, &ids);
        assert_eq!(r.len(), 10_000);
    }

    #[test]
    fn create_edge_keeps_endpoints_aligned_with_their_edges() {
        // Three edges out of one source: an IdSet would have collapsed the
        // sources to one entry and misaligned every row after the first.
        let ids = [10_u64, 11, 12];
        let src = [7_u64, 7, 7];
        let dst = [1_u64, 2, 3];
        let mut buf = new_buffer();
        write_create_edge(&mut buf, &ids, 4, &src, &dst, &[], &[]);

        let records = read_buffer(&buf).unwrap();
        let Record::CreateEdge {
            ids: gi,
            relation_id,
            src: gs,
            dst: gd,
            ..
        } = &records[0]
        else {
            panic!("wrong record");
        };
        assert_eq!(gi, &ids);
        assert_eq!(*relation_id, 4);
        assert_eq!(gs, &src, "duplicate sources must survive");
        assert_eq!(gd, &dst);
    }

    #[test]
    fn set_labels_carries_the_labels_and_all_their_nodes() {
        // Grouped, not one (node, label) pair per node: 10,000 nodes gaining
        // one label fit in a few dozen bytes because the ids are one run.
        let ids: Vec<u64> = (0..10_000).collect();
        let mut buf = new_buffer();
        write_labels(&mut buf, true, &ids, &[5]);
        assert!(buf.len() < 100, "grouped should be tiny, got {}", buf.len());

        let records = read_buffer(&buf).unwrap();
        assert_eq!(
            records[0],
            Record::Labels {
                add: true,
                ids,
                labels: vec![5],
            }
        );
    }

    #[test]
    fn schema_records_carry_their_id() {
        // The v3 change: v2 sent the name alone and let the replica infer the
        // id from its own append order — the assumption every bare id rests on,
        // and the only one that could not be checked.
        let mut buf = new_buffer();
        write_add_schema(&mut buf, SCHEMA_NODE, 3, "Person");
        write_add_attribute(&mut buf, 9, "name");

        let records = read_buffer(&buf).unwrap();
        assert_eq!(
            records,
            vec![
                Record::AddSchema {
                    schema_type: SCHEMA_NODE,
                    id: 3,
                    name: "Person".into(),
                },
                Record::AddAttribute {
                    id: 9,
                    name: "name".into(),
                },
            ]
        );
    }

    #[test]
    fn schema_records_carry_no_count() {
        // They are inherently singular, so they are the documented exception to
        // "every record has a count".
        let mut buf = Vec::new();
        write_add_attribute(&mut buf, 9, "n");
        // opcode, then the id straight away — no 4-byte count in between.
        assert_eq!(&buf[..4], &EFFECT_ADD_ATTRIBUTE.to_le_bytes());
        assert_eq!(&buf[4..6], &9_u16.to_le_bytes());
    }

    #[test]
    fn every_record_type_round_trips() {
        let mut buf = new_buffer();
        write_add_schema(&mut buf, SCHEMA_EDGE, 1, "KNOWS");
        write_add_attribute(&mut buf, 0, "since");
        write_create_node(&mut buf, &[1, 2], &[7], &[0], &[Value::Int(1), Value::Null]);
        write_create_edge(
            &mut buf,
            &[5, 6],
            1,
            &[1, 1],
            &[2, 2],
            &[0],
            &[Value::Int(2020), Value::Int(2021)],
        );
        write_update(&mut buf, GraphEntity::Node, &[1], &[0], &[Value::Int(9)]);
        write_update(&mut buf, GraphEntity::Edge, &[5], &[0], &[Value::Null]);
        write_labels(&mut buf, true, &[1, 2], &[7, 8]);
        write_labels(&mut buf, false, &[1], &[8]);
        write_delete_edge(&mut buf, &[5, 6], 1, &[1, 1], &[2, 2]);
        write_delete_node(&mut buf, &[1, 2]);
        write_create_index(
            &mut buf,
            SCHEMA_NODE,
            7,
            "L",
            IndexField {
                attr_id: 0,
                attr: "since",
                field_type: INDEX_FLD_RANGE,
            },
            &Value::Null,
        );
        write_drop_index(
            &mut buf,
            SCHEMA_NODE,
            7,
            "L",
            IndexField {
                attr_id: 0,
                attr: "since",
                field_type: INDEX_FLD_RANGE,
            },
        );
        write_constraint(
            &mut buf,
            true,
            CT_UNIQUE,
            GETYPE_NODE,
            7,
            "L",
            &[(0, "since")],
        );
        write_constraint(&mut buf, false, CT_MANDATORY, GETYPE_EDGE, 1, "KNOWS", &[]);

        let records = read_buffer(&buf).unwrap();
        assert_eq!(records.len(), 14, "one command, fourteen records");

        // Re-encoding what was decoded must reproduce the buffer exactly.
        let mut again = new_buffer();
        for record in &records {
            match record {
                Record::AddSchema {
                    schema_type,
                    id,
                    name,
                } => {
                    write_add_schema(&mut again, *schema_type, *id, name);
                }
                Record::AddAttribute { id, name } => write_add_attribute(&mut again, *id, name),
                Record::CreateNode {
                    ids,
                    labels,
                    attr_ids,
                    rows,
                } => {
                    write_create_node(&mut again, ids, labels, attr_ids, rows);
                }
                Record::CreateEdge {
                    ids,
                    relation_id,
                    src,
                    dst,
                    attr_ids,
                    rows,
                } => {
                    write_create_edge(&mut again, ids, *relation_id, src, dst, attr_ids, rows);
                }
                Record::Update {
                    entity,
                    ids,
                    attr_ids,
                    rows,
                } => {
                    write_update(&mut again, *entity, ids, attr_ids, rows);
                }
                Record::Labels { add, ids, labels } => {
                    write_labels(&mut again, *add, ids, labels);
                }
                Record::DeleteEdge {
                    ids,
                    relation_id,
                    src,
                    dst,
                } => {
                    write_delete_edge(&mut again, ids, *relation_id, src, dst);
                }
                Record::DeleteNode { ids } => write_delete_node(&mut again, ids),
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
                    if *create {
                        write_create_index(
                            &mut again,
                            *schema_type,
                            *label_id,
                            label,
                            IndexField {
                                attr_id: *attr_id,
                                attr,
                                field_type: *field_type,
                            },
                            options.as_ref().expect("a create carries options"),
                        );
                    } else {
                        write_drop_index(
                            &mut again,
                            *schema_type,
                            *label_id,
                            label,
                            IndexField {
                                attr_id: *attr_id,
                                attr,
                                field_type: *field_type,
                            },
                        );
                    }
                }
                Record::Constraint {
                    create,
                    constraint_type,
                    entity_type,
                    label_id,
                    label,
                    props,
                } => {
                    let props: Vec<(u16, &str)> =
                        props.iter().map(|(id, n)| (*id, n.as_str())).collect();
                    write_constraint(
                        &mut again,
                        *create,
                        *constraint_type,
                        *entity_type,
                        *label_id,
                        label,
                        &props,
                    );
                }
            }
        }
        assert_eq!(hex(&again), hex(&buf), "encode(decode(x)) must equal x");
    }

    /// One `CREATE_NODE` record per distinct shape, for 10,000 nodes split
    /// `shapes` ways.
    fn create_10k_in_shapes(shapes: usize) -> Vec<u8> {
        let per = 10_000 / shapes;
        let mut buf = new_buffer();
        for s in 0..shapes {
            let ids: Vec<u64> = (0..per).map(|j| (j * shapes + s) as u64).collect();
            let rows: Vec<Value> = (0..per).map(|j| Value::Int(j as i64)).collect();
            write_create_node(&mut buf, &ids, &[7], &[s as u16], &rows);
        }
        buf
    }

    #[test]
    fn the_motivating_query_end_to_end() {
        // UNWIND range(0,9999) AS i CREATE (:L {v:i})
        // v2 emitted one record per node, 34 bytes each: 340,001 bytes.
        const V2: f64 = 340_001.0;
        println!("\n  10,000 nodes            shapes     v3 bytes    vs v2");
        for shapes in [1_usize, 20, 500, 2_000, 10_000] {
            let buf = create_10k_in_shapes(shapes);
            let n = read_buffer(&buf).unwrap().len();
            assert_eq!(n, shapes, "one record per shape");
            println!(
                "  {:<22} {shapes:6} {:12} {:+7.1}%",
                if shapes == 1 {
                    "all share one shape"
                } else if shapes == 10_000 {
                    "every node its own"
                } else {
                    "split"
                },
                buf.len(),
                100.0 * buf.len() as f64 / V2 - 100.0
            );
        }

        // The floor is now parity with v2, not the +15% the design budgeted
        // for: merging the full-width and narrowed encodings gave back what the
        // count field costs.
        let worst = create_10k_in_shapes(10_000);
        assert!(
            worst.len() <= 340_001,
            "singleton floor regressed: {}",
            worst.len()
        );

        // Two shapes, ids interleaved — the alternating-label query. Stride 2
        // breaks the single run, so each IdSet falls back to a bitset container.
        let even: Vec<u64> = (0..5_000).map(|i| i * 2).collect();
        let odd: Vec<u64> = (0..5_000).map(|i| i * 2 + 1).collect();
        let half: Vec<Value> = (0..5_000).map(|i| Value::Int(i as i64)).collect();
        let mut alt = new_buffer();
        write_create_node(&mut alt, &even, &[7], &[0], &half);
        write_create_node(&mut alt, &odd, &[8], &[0], &half);
        println!("\n  alternating labels, 2 records: {} B", alt.len());
        assert_eq!(read_buffer(&alt).unwrap().len(), 2);

        // 10,000 edges out of one supernode.
        let eids: Vec<u64> = (0..10_000).collect();
        let src = vec![4_000_000_000_u64; 10_000];
        let dst: Vec<u64> = (0..10_000).collect();
        let mut edges = new_buffer();
        write_create_edge(&mut edges, &eids, 1, &src, &dst, &[], &[]);
        println!("  10,000 supernode edges:        {} B", edges.len());
        assert_eq!(read_buffer(&edges).unwrap().len(), 1);
    }

    #[test]
    fn a_singleton_costs_about_what_v2_charged() {
        // The price of one uniform record shape is a count field and an
        // encoding byte — but narrowing the id gives most of it back, so the
        // "+15% on singletons" the design was budgeted for did not materialize.
        // v2 spent 34 bytes on this record.
        //
        //   4 opcode + 4 count + IdSet + 6 LabelSet + 16 AttrSet
        let mut small = Vec::new();
        write_create_node(&mut small, &[0], &[7], &[0], &[Value::Int(1)]);
        assert_eq!(small.len(), 33, "id 0 narrows to one byte");

        let mut mid = Vec::new();
        write_create_node(&mut mid, &[5_000], &[7], &[0], &[Value::Int(1)]);
        assert_eq!(mid.len(), 34, "a mid-size graph's id takes two: exactly v2");

        let mut large = Vec::new();
        write_create_node(&mut large, &[5_000_000], &[7], &[0], &[Value::Int(1)]);
        assert_eq!(large.len(), 36, "past 2^16 the id takes four");
    }

    // ── index and constraint records ──

    #[test]
    fn index_field_types_are_c_bit_flags() {
        // A range index is the union of the three scalar kinds, not a
        // discriminant of its own — 0x0E, which reads as garbage if treated as
        // an ordinal.
        assert_eq!(INDEX_FLD_RANGE, 0x0E);
        assert_eq!(INDEX_FLD_FULLTEXT, 0x01);
        assert_eq!(INDEX_FLD_VECTOR, 0x10);
        // And entity types are 1-based, because GETYPE_UNKNOWN takes 0.
        assert_eq!(GETYPE_NODE, 1);
        assert_eq!(GETYPE_EDGE, 2);
    }

    #[test]
    fn create_index_keeps_the_name_beside_each_id() {
        let mut buf = new_buffer();
        write_create_index(
            &mut buf,
            SCHEMA_NODE,
            3,
            "Person",
            IndexField {
                attr_id: 9,
                attr: "name",
                field_type: INDEX_FLD_RANGE,
            },
            &Value::Null,
        );
        let records = read_buffer(&buf).unwrap();
        assert_eq!(
            records[0],
            Record::Index {
                create: true,
                schema_type: SCHEMA_NODE,
                label_id: 3,
                label: "Person".into(),
                attr_id: 9,
                attr: "name".into(),
                field_type: INDEX_FLD_RANGE,
                options: Some(Value::Null),
            }
        );
    }

    #[test]
    fn a_multi_field_index_is_one_record_per_field() {
        // Not batched, matching C: index DDL is rare, and C applies it
        // idempotently per field so a replica can take them in any order.
        let mut buf = new_buffer();
        for (attr_id, attr) in [(0_u16, "a"), (1, "b"), (2, "c")] {
            write_create_index(
                &mut buf,
                SCHEMA_NODE,
                1,
                "L",
                IndexField {
                    attr_id,
                    attr,
                    field_type: INDEX_FLD_RANGE,
                },
                &Value::Null,
            );
        }
        assert_eq!(read_buffer(&buf).unwrap().len(), 3);
    }

    #[test]
    fn drop_index_carries_no_options() {
        let mut buf = new_buffer();
        write_drop_index(
            &mut buf,
            SCHEMA_EDGE,
            1,
            "KNOWS",
            IndexField {
                attr_id: 0,
                attr: "since",
                field_type: INDEX_FLD_RANGE,
            },
        );
        let records = read_buffer(&buf).unwrap();
        let Record::Index {
            create, options, ..
        } = &records[0]
        else {
            panic!("wrong record");
        };
        assert!(!create);
        assert_eq!(*options, None);
    }

    #[test]
    fn constraint_property_count_is_one_byte() {
        // C reads it as uint8_t, not the u16 used elsewhere in the format.
        let mut buf = Vec::new();
        write_constraint(
            &mut buf,
            true,
            CT_UNIQUE,
            GETYPE_NODE,
            3,
            "Person",
            &[(0, "first"), (1, "last")],
        );
        // opcode, ct, et, label_id, then the string, then a single count byte.
        let after_label = 4 + 4 + 4 + 4 + (8 + "Person".len() + 1);
        assert_eq!(buf[after_label], 2, "the count is one byte wide");
    }

    #[test]
    fn constraints_round_trip_with_their_property_names() {
        let mut buf = new_buffer();
        write_constraint(
            &mut buf,
            true,
            CT_UNIQUE,
            GETYPE_NODE,
            3,
            "Person",
            &[(0, "first"), (1, "last")],
        );
        write_constraint(
            &mut buf,
            false,
            CT_MANDATORY,
            GETYPE_EDGE,
            1,
            "KNOWS",
            &[(2, "since")],
        );

        let records = read_buffer(&buf).unwrap();
        assert_eq!(
            records[0],
            Record::Constraint {
                create: true,
                constraint_type: CT_UNIQUE,
                entity_type: GETYPE_NODE,
                label_id: 3,
                label: "Person".into(),
                props: vec![(0, "first".into()), (1, "last".into())],
            }
        );
        let Record::Constraint {
            create,
            entity_type,
            props,
            ..
        } = &records[1]
        else {
            panic!("wrong record");
        };
        assert!(!create);
        assert_eq!(*entity_type, GETYPE_EDGE);
        assert_eq!(props, &vec![(2_u16, "since".to_string())]);
    }

    #[test]
    fn ddl_records_carry_no_count() {
        // Like the schema records, these are singular.
        let mut buf = Vec::new();
        write_drop_index(
            &mut buf,
            SCHEMA_NODE,
            1,
            "L",
            IndexField {
                attr_id: 0,
                attr: "a",
                field_type: INDEX_FLD_RANGE,
            },
        );
        assert_eq!(&buf[..4], &EFFECT_DROP_INDEX.to_le_bytes());
        assert_eq!(
            &buf[4..8],
            &SCHEMA_NODE.to_le_bytes(),
            "schema type, not a count"
        );
    }

    #[test]
    fn a_foreign_version_is_refused() {
        let mut buf = new_buffer();
        write_delete_node(&mut buf, &[1]);
        buf[0] = 2;
        assert_eq!(read_buffer(&buf), Err(DecodeError::UnsupportedVersion(2)));
    }

    #[test]
    fn an_unknown_opcode_is_refused() {
        let mut buf = new_buffer();
        write_u32(&mut buf, 99);
        write_u32(&mut buf, 0);
        assert_eq!(read_buffer(&buf), Err(DecodeError::BadOpcode(99)));
    }

    #[test]
    fn a_truncated_buffer_is_an_error_not_a_panic() {
        let mut buf = new_buffer();
        write_create_edge(
            &mut buf,
            &[5, 6],
            1,
            &[1, 1],
            &[2, 2],
            &[0],
            &[Value::Int(1), Value::Int(2)],
        );
        // From 2: a buffer of nothing but the version byte carries zero
        // records, which is valid rather than truncated.
        for cut in 2..buf.len() {
            assert!(read_buffer(&buf[..cut]).is_err(), "cut at {cut}");
        }
    }

    #[test]
    fn buffer_header_is_the_version() {
        assert_eq!(new_buffer(), vec![3_u8]);
    }
}
