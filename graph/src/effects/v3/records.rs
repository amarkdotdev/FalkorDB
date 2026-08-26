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
    opcode: Opcode,
    count: usize,
) {
    debug_assert!(opcode.is_batchable(), "{opcode:?} carries no count");
    debug_assert!(count <= u32::MAX as usize, "record count must fit in u32");
    write_u32(buf, opcode.as_u32());
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
    write_header(buf, Opcode::CreateNode, ids.len());
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
    write_header(buf, Opcode::CreateEdge, ids.len());
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
    write_header(buf, entity.update_opcode(), ids.len());
    write_id_set(buf, ids);
    write_attr_set(buf, attr_ids, rows);
}

/// `5 DELETE_NODE` — `count · IdSet`.
pub fn write_delete_node(
    buf: &mut Vec<u8>,
    ids: &[u64],
) {
    write_header(buf, Opcode::DeleteNode, ids.len());
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
    write_header(buf, Opcode::DeleteEdge, ids.len());
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
        Opcode::SetLabels
    } else {
        Opcode::RemoveLabels
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
    schema_type: SchemaType,
    id: i32,
    name: &str,
) {
    write_u32(buf, Opcode::AddSchema.as_u32());
    write_u32(buf, schema_type.as_u32());
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
    write_u32(buf, Opcode::AddAttribute.as_u32());
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
    schema_type: SchemaType,
    label_id: i32,
    label: &str,
    field: IndexField<'_>,
    options: &Value,
) {
    write_u32(buf, Opcode::CreateIndex.as_u32());
    write_u32(buf, schema_type.as_u32());
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
    schema_type: SchemaType,
    label_id: i32,
    label: &str,
    field: IndexField<'_>,
) {
    write_u32(buf, Opcode::DropIndex.as_u32());
    write_u32(buf, schema_type.as_u32());
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
    constraint_type: ConstraintKind,
    entity_type: GraphEntity,
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
            Opcode::CreateConstraint.as_u32()
        } else {
            Opcode::DropConstraint.as_u32()
        },
    );
    write_u32(buf, constraint_type.as_u32());
    write_u32(buf, entity_type.as_u32());
    write_label_id(buf, label_id);
    write_string(buf, label);
    write_u8(buf, props.len() as u8);
    for (attr_id, name) in props {
        write_u16(buf, *attr_id);
        write_string(buf, name);
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
        schema_type: SchemaType,
        id: i32,
        name: String,
    },
    AddAttribute {
        id: u16,
        name: String,
    },
    Index {
        create: bool,
        schema_type: SchemaType,
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
        constraint_type: ConstraintKind,
        entity_type: GraphEntity,
        label_id: i32,
        label: String,
        props: Vec<(u16, String)>,
    },
}

/// Read one record, whatever its opcode.
pub fn read_record(r: &mut Reader<'_>) -> Result<Record, DecodeError> {
    let opcode = Opcode::try_from(r.u32()?)?;

    // The singular records carry no count, so they are read before one is asked
    // for. `is_batchable` is the same predicate the writer asserts on.
    if !opcode.is_batchable() {
        return read_singular(r, opcode);
    }

    let count = r.u32()?;
    let record = match opcode {
        Opcode::UpdateNode | Opcode::UpdateEdge => {
            let ids = read_id_set(r, count)?;
            let (attr_ids, rows) = read_attr_set(r, count)?;
            Record::Update {
                entity: if opcode == Opcode::UpdateNode {
                    GraphEntity::Node
                } else {
                    GraphEntity::Edge
                },
                ids,
                attr_ids,
                rows,
            }
        }
        Opcode::CreateNode => {
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
        Opcode::CreateEdge => {
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
        Opcode::DeleteNode => Record::DeleteNode {
            ids: read_id_set(r, count)?,
        },
        Opcode::DeleteEdge => {
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
        Opcode::SetLabels | Opcode::RemoveLabels => Record::Labels {
            add: opcode == Opcode::SetLabels,
            ids: read_id_set(r, count)?,
            labels: read_label_set(r)?,
        },
        // Every remaining opcode answered false to `is_batchable` and was
        // handled above. Listing them keeps the match exhaustive, so a new
        // record type stops compiling here until it is decided which it is.
        Opcode::AddSchema
        | Opcode::AddAttribute
        | Opcode::CreateIndex
        | Opcode::DropIndex
        | Opcode::CreateConstraint
        | Opcode::DropConstraint => unreachable!("handled by read_singular"),
    };
    Ok(record)
}

/// The records that describe one thing and so carry no count.
fn read_singular(
    r: &mut Reader<'_>,
    opcode: Opcode,
) -> Result<Record, DecodeError> {
    Ok(match opcode {
        Opcode::AddSchema => {
            let schema_type = SchemaType::try_from(r.u32()?)?;
            let id = r.i32()?;
            Record::AddSchema {
                schema_type,
                id,
                name: r.string()?,
            }
        }
        Opcode::AddAttribute => {
            let id = r.u16()?;
            Record::AddAttribute {
                id,
                name: r.string()?,
            }
        }
        Opcode::CreateIndex | Opcode::DropIndex => {
            let create = opcode == Opcode::CreateIndex;
            let schema_type = SchemaType::try_from(r.u32()?)?;
            let label_id = r.i32()?;
            let label = r.string()?;
            let attr_id = r.u16()?;
            let attr = r.string()?;
            let field_type = r.u32()?;
            let options = if create { Some(read_value(r)?) } else { None };
            Record::Index {
                create,
                schema_type,
                label_id,
                label,
                attr_id,
                attr,
                field_type,
                options,
            }
        }
        Opcode::CreateConstraint | Opcode::DropConstraint => {
            let create = opcode == Opcode::CreateConstraint;
            let constraint_type = ConstraintKind::try_from(r.u32()?)?;
            let entity_type = GraphEntity::try_from(r.u32()?)?;
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
            Record::Constraint {
                create,
                constraint_type,
                entity_type,
                label_id,
                label,
                props,
            }
        }
        other => unreachable!("{other:?} is batchable"),
    })
}

/// Write one record, whatever its opcode.
fn write_record(
    buf: &mut Vec<u8>,
    record: &Record,
) {
    match record {
        Record::AddSchema {
            schema_type,
            id,
            name,
        } => write_add_schema(buf, *schema_type, *id, name),
        Record::AddAttribute { id, name } => write_add_attribute(buf, *id, name),
        Record::CreateNode {
            ids,
            labels,
            attr_ids,
            rows,
        } => write_create_node(buf, ids, labels, attr_ids, rows),
        Record::CreateEdge {
            ids,
            relation_id,
            src,
            dst,
            attr_ids,
            rows,
        } => write_create_edge(buf, ids, *relation_id, src, dst, attr_ids, rows),
        Record::Update {
            entity,
            ids,
            attr_ids,
            rows,
        } => write_update(buf, *entity, ids, attr_ids, rows),
        Record::Labels { add, ids, labels } => write_labels(buf, *add, ids, labels),
        Record::DeleteNode { ids } => write_delete_node(buf, ids),
        Record::DeleteEdge {
            ids,
            relation_id,
            src,
            dst,
        } => write_delete_edge(buf, ids, *relation_id, src, dst),
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
            let field = IndexField {
                attr_id: *attr_id,
                attr,
                field_type: *field_type,
            };
            if *create {
                write_create_index(
                    buf,
                    *schema_type,
                    *label_id,
                    label,
                    field,
                    options.as_ref().unwrap_or(&Value::Null),
                );
            } else {
                write_drop_index(buf, *schema_type, *label_id, label, field);
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
            let props: Vec<(u16, &str)> = props.iter().map(|(i, n)| (*i, n.as_str())).collect();
            write_constraint(
                buf,
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

/// The effects encoding of one record.
///
/// `encode` is total — every variant has a shape — while `decode` reads the
/// opcode first and dispatches, so the pair is not symmetric in signature even
/// though it is in effect.
impl EffectEncode<3> for Record {
    fn encode(
        &self,
        buf: &mut Vec<u8>,
    ) {
        write_record(buf, self);
    }
}

impl EffectDecode<3> for Record {
    fn decode(r: &mut Reader<'_>) -> Result<Self, DecodeError> {
        read_record(r)
    }
}

/// Read a whole `GRAPH.EFFECT` payload: the header, then every record.
///
/// One command carries one payload carries many records — batching removes
/// per-record framing, it never multiplies commands.
pub fn read_buffer(buf: &[u8]) -> Result<Vec<Record>, DecodeError> {
    let mut r = Reader::new(buf);
    let version = r.u8()?;
    if version != EFFECTS_VERSION {
        return Err(DecodeError::UnsupportedVersion(version));
    }
    let flags = r.u8()?;
    if flags & !KNOWN_FLAGS != 0 {
        return Err(DecodeError::UnknownFlags(flags));
    }

    if flags & FLAG_COMPRESSED != 0 {
        let plain_len = r.u32()? as usize;
        let frame = r.take(r.remaining())?;
        let plain = zstd::stream::decode_all(frame)
            .map_err(|e| DecodeError::BadCompression(e.to_string()))?;
        // The declared length is a cross-check, not a hint: a frame that
        // expands to something else means the header and the payload disagree,
        // and the records after it cannot be trusted.
        if plain.len() != plain_len {
            return Err(DecodeError::CompressedLengthMismatch {
                declared: plain_len,
                actual: plain.len(),
            });
        }
        return read_records(&plain);
    }

    read_records(r.rest())
}

/// Every record in an already-plaintext record stream.
fn read_records(buf: &[u8]) -> Result<Vec<Record>, DecodeError> {
    let mut r = Reader::new(buf);
    let mut records = Vec::new();
    while !r.is_empty() {
        records.push(<Record as EffectDecode<3>>::decode(&mut r)?);
    }
    Ok(records)
}

/// Compress a finished payload in place, if that makes it smaller.
///
/// **Default off.** Compression is a bandwidth trade, not a CPU one: measured
/// at 3.245 cycles/byte to compress against 0.067 to copy into the replica
/// output buffer, so on a fast link it spends far more than the bytes are
/// worth. `min_bytes` of 0 disables it entirely, which is what
/// `EFFECTS_COMPRESSION` defaults to.
///
/// Even when enabled the smaller form wins: zstd inflates an already-minimal
/// buffer, and the batched records are already most of the way there.
///
/// Returns whether the payload ended up compressed.
pub fn maybe_compress(
    buf: &mut Vec<u8>,
    min_bytes: usize,
    level: i32,
) -> bool {
    const HEADER: usize = 2;
    if min_bytes == 0 || buf.len() < HEADER || buf.len() - HEADER < min_bytes {
        return false;
    }
    let Ok(frame) = zstd::stream::encode_all(&buf[HEADER..], level) else {
        // Compression failing is not a reason to fail the write; the
        // uncompressed payload is still correct.
        return false;
    };
    // 4 bytes of declared length ride along, so they count.
    if frame.len() + 4 >= buf.len() - HEADER {
        return false;
    }

    let plain_len = (buf.len() - HEADER) as u32;
    buf.truncate(HEADER);
    buf[1] |= FLAG_COMPRESSED;
    buf.extend_from_slice(&plain_len.to_le_bytes());
    buf.extend_from_slice(&frame);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effects::hex;

    // ── records ──

    #[test]
    fn create_node_pins_its_bytes_at_count_one() {
        // One node, label 7, one int property. The smallest possible record.
        let mut buf = new_buffer();
        write_create_node(&mut buf, &[0], &[7], &[0], &[Value::Int(1)]);
        assert_eq!(
            hex(&buf),
            concat!(
                "03 00 ",                              // version, flags
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

        assert_eq!(&buf[2..6], &Opcode::CreateNode.as_u32().to_le_bytes());
        assert_eq!(&buf[6..10], &10_000_u32.to_le_bytes());
        // 10,000 consecutive ids collapse to one roaring run.
        assert_eq!(buf[10], BlockEncoding::Compressed.as_u8());

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
        write_add_schema(&mut buf, SchemaType::Node, 3, "Person");
        write_add_attribute(&mut buf, 9, "name");

        let records = read_buffer(&buf).unwrap();
        assert_eq!(
            records,
            vec![
                Record::AddSchema {
                    schema_type: SchemaType::Node,
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
        assert_eq!(&buf[..4], &Opcode::AddAttribute.as_u32().to_le_bytes());
        assert_eq!(&buf[4..6], &9_u16.to_le_bytes());
    }

    #[test]
    fn every_record_type_round_trips() {
        let mut buf = new_buffer();
        write_add_schema(&mut buf, SchemaType::Edge, 1, "KNOWS");
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
            SchemaType::Node,
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
            SchemaType::Node,
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
            ConstraintKind::Unique,
            GraphEntity::Node,
            7,
            "L",
            &[(0, "since")],
        );
        write_constraint(
            &mut buf,
            false,
            ConstraintKind::Mandatory,
            GraphEntity::Edge,
            1,
            "KNOWS",
            &[],
        );

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
        assert_eq!(GraphEntity::Node.as_u32(), 1);
        assert_eq!(GraphEntity::Edge.as_u32(), 2);
    }

    #[test]
    fn create_index_keeps_the_name_beside_each_id() {
        let mut buf = new_buffer();
        write_create_index(
            &mut buf,
            SchemaType::Node,
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
                schema_type: SchemaType::Node,
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
                SchemaType::Node,
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
            SchemaType::Edge,
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
            ConstraintKind::Unique,
            GraphEntity::Node,
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
            ConstraintKind::Unique,
            GraphEntity::Node,
            3,
            "Person",
            &[(0, "first"), (1, "last")],
        );
        write_constraint(
            &mut buf,
            false,
            ConstraintKind::Mandatory,
            GraphEntity::Edge,
            1,
            "KNOWS",
            &[(2, "since")],
        );

        let records = read_buffer(&buf).unwrap();
        assert_eq!(
            records[0],
            Record::Constraint {
                create: true,
                constraint_type: ConstraintKind::Unique,
                entity_type: GraphEntity::Node,
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
        assert_eq!(*entity_type, GraphEntity::Edge);
        assert_eq!(props, &vec![(2_u16, "since".to_string())]);
    }

    #[test]
    fn ddl_records_carry_no_count() {
        // Like the schema records, these are singular.
        let mut buf = Vec::new();
        write_drop_index(
            &mut buf,
            SchemaType::Node,
            1,
            "L",
            IndexField {
                attr_id: 0,
                attr: "a",
                field_type: INDEX_FLD_RANGE,
            },
        );
        assert_eq!(&buf[..4], &Opcode::DropIndex.as_u32().to_le_bytes());
        assert_eq!(
            &buf[4..8],
            &SchemaType::Node.as_u32().to_le_bytes(),
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
        // From 3: a header with no records after it is a valid empty payload,
        // not a truncated one.
        for cut in 3..buf.len() {
            assert!(read_buffer(&buf[..cut]).is_err(), "cut at {cut}");
        }
    }

    // ── the payload header ──

    #[test]
    fn the_flags_byte_is_reserved_even_though_nothing_sets_it() {
        // Reserving it now is the point: adding a byte to the header later
        // would cost another version bump for something compression needs.
        let mut buf = new_buffer();
        write_delete_node(&mut buf, &[1]);
        assert_eq!(buf[0], EFFECTS_VERSION);
        assert_eq!(buf[1], 0, "no flags by default");
        assert_eq!(read_buffer(&buf).unwrap().len(), 1);
    }

    #[test]
    fn compression_round_trips_and_shrinks() {
        // 10,000 nodes of one shape: highly repetitive, which is the case worth
        // compressing at all.
        let ids: Vec<u64> = (0..10_000).collect();
        let rows: Vec<Value> = (0..10_000).map(|i| Value::Int(i as i64)).collect();
        let mut buf = new_buffer();
        write_create_node(&mut buf, &ids, &[7], &[0], &rows);
        let plain = buf.clone();

        assert!(maybe_compress(&mut buf, 1024, 1), "should compress");
        assert_eq!(buf[0], EFFECTS_VERSION, "the header stays plaintext");
        assert_eq!(buf[1], FLAG_COMPRESSED);
        assert!(
            buf.len() < plain.len() / 4,
            "{} vs {}",
            buf.len(),
            plain.len()
        );

        // And it decodes to exactly what the uncompressed payload does.
        assert_eq!(read_buffer(&buf).unwrap(), read_buffer(&plain).unwrap());
    }

    #[test]
    fn compression_is_off_at_zero_and_declines_when_it_would_not_help() {
        let mut buf = new_buffer();
        write_create_node(&mut buf, &[0], &[7], &[0], &[Value::Int(1)]);
        let before = buf.clone();

        assert!(!maybe_compress(&mut buf, 0, 1), "0 disables it");
        assert_eq!(buf, before);

        // Enabled, but a 33-byte record is below any sane floor — and even
        // forced, zstd inflates a payload this small, so the smaller form wins.
        assert!(!maybe_compress(&mut buf, 1, 1));
        assert_eq!(
            buf, before,
            "left uncompressed because compressing is bigger"
        );
    }

    #[test]
    fn an_unknown_flag_is_refused_rather_than_ignored() {
        // An old node meeting a future payload must fail loudly. Decoding the
        // records anyway would apply a prefix of something whose shape it does
        // not know.
        let mut buf = new_buffer();
        write_delete_node(&mut buf, &[1]);
        buf[1] = 0x80;
        assert_eq!(read_buffer(&buf), Err(DecodeError::UnknownFlags(0x80)));
    }

    #[test]
    fn a_corrupt_frame_is_an_error_not_a_panic() {
        let ids: Vec<u64> = (0..2_000).collect();
        let rows: Vec<Value> = (0..2_000).map(|i| Value::Int(i as i64)).collect();
        let mut buf = new_buffer();
        write_create_node(&mut buf, &ids, &[7], &[0], &rows);
        assert!(maybe_compress(&mut buf, 16, 1));

        // Truncating anywhere inside the frame must error, never panic.
        for cut in 7..buf.len() {
            assert!(read_buffer(&buf[..cut]).is_err(), "cut at {cut}");
        }
        // And a frame that expands to the wrong size is caught by the length.
        let mut lied = buf.clone();
        lied[2] = lied[2].wrapping_add(1);
        assert!(matches!(
            read_buffer(&lied),
            Err(DecodeError::CompressedLengthMismatch { .. })
        ));
    }

    #[test]
    fn buffer_header_is_the_version() {
        assert_eq!(
            new_buffer(),
            vec![3_u8, 0],
            "version, then the reserved flags byte"
        );
    }
}
