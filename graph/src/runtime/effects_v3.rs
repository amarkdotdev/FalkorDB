//! Effects wire format v3 — see `docs/effects-v3.md`.
//!
//! Every record is `u32 opcode · u32 count · blocks…`, built from five shared
//! blocks. There is no separate "batch" record type: a record with `count == 1`
//! and one with `count == 10_000` are the same record, so no record type is left
//! un-batchable and the decoder has one shape per opcode.
//!
//! This module is the codec only. It is deliberately free of `Pending` and
//! `Graph` so the format can be reviewed, and tested byte-for-byte, on its own.
//!
//! ## The blocks
//!
//! | block | layout |
//! | --- | --- |
//! | `IdSet` | `u8 enc` · (`u64 × count` \| `u32 blob_len · roaring64[blob_len]`) |
//! | `IdList` | `u8 enc` · (`u64 × count` \| `u32 dict_len · roaring64 · u8 w · uint(w) × count`) |
//! | `RelType` | `i32` |
//! | `LabelSet` | `u16 n` · `i32 × n` |
//! | `AttrSet` | `u16 n` · `u16 attr_id × n` · `SIValue × (count × n)` |
//!
//! `IdSet` and `IdList` are deliberately distinct types. A roaring bitmap is a
//! *set*: it deduplicates and it sorts. Edge endpoints do neither — many edges
//! share a source, and each endpoint must stay positionally aligned with its edge
//! id — so encoding them as a set would drop duplicates and misalign every row
//! after the first repeat.
//!
//! ## Row order is id order
//!
//! Row *k* belongs to the k-th smallest id in the record's `IdSet`. No per-row id
//! is sent. That is why [`read_id_set`] asserts the decoded cardinality equals
//! `count`: a bitmap that round-tripped one id short would land every later row
//! on the wrong entity, silently.
//!
//! ## Why the widths matter more than usual
//!
//! A wrong width does not produce a decode error on the far side. C reads the
//! misaligned bytes as a type tag and writes through the resulting pointer — the
//! `AttributeSet_Update` segfault observed when feeding C a Rust buffer. Every
//! width here came from C source (`src/effects/effects.c`, `effects.h`), not from
//! inference, and the tests below pin the bytes rather than round-tripping only
//! against ourselves.
//!
//! ## Records
//!
//! Batchable records are `u32 opcode · u32 count · blocks…`. `ADD_SCHEMA` and
//! `ADD_ATTRIBUTE` are inherently singular — one schema, one name — so they
//! carry an opcode but no count. Index and constraint records (11–14) are not
//! in this module yet.

use std::{fmt, sync::Arc};

use roaring::RoaringTreemap;
use thin_vec::ThinVec;

use crate::runtime::{
    ordermap::OrderMap,
    value::{Point, Value},
};

/// Buffer header. C accepts any version `<= EFFECTS_VERSION` and branches per
/// version, so raising this is its established mechanism rather than a break —
/// but C must be raised to 3 as well before it can read what we write.
pub const EFFECTS_VERSION: u8 = 3;

// ── effect types ──
//
// C's `EffectType` is an enum, so 4 bytes on the wire, and `EFFECT_UNKNOWN = 0`
// shifts every discriminant down by one relative to a naive 1-based list.

pub const EFFECT_UPDATE_NODE: u32 = 1;
pub const EFFECT_UPDATE_EDGE: u32 = 2;
pub const EFFECT_CREATE_NODE: u32 = 3;
pub const EFFECT_CREATE_EDGE: u32 = 4;
pub const EFFECT_DELETE_NODE: u32 = 5;
pub const EFFECT_DELETE_EDGE: u32 = 6;
pub const EFFECT_SET_LABELS: u32 = 7;
pub const EFFECT_REMOVE_LABELS: u32 = 8;
pub const EFFECT_ADD_SCHEMA: u32 = 9;
pub const EFFECT_ADD_ATTRIBUTE: u32 = 10;
pub const EFFECT_CREATE_INDEX: u32 = 11;
pub const EFFECT_DROP_INDEX: u32 = 12;
pub const EFFECT_CREATE_CONSTRAINT: u32 = 13;
pub const EFFECT_DROP_CONSTRAINT: u32 = 14;

/// C's `SchemaType`.
pub const SCHEMA_NODE: u32 = 0;
pub const SCHEMA_EDGE: u32 = 1;

// ── block encodings ──

/// Raw ids at the narrowest width that holds the largest of them:
/// `u8 width` then `uint(width) x count`.
///
/// A separate full-width encoding would be this one with `width == 8`, so there
/// is no reason for both. Finding the maximum is one pass and no allocation.
pub const ENC_PLAIN: u8 = 0;
/// `IdSet`: a run-optimized roaring64 bitmap. `IdList`: a roaring64 dictionary
/// plus one rank per row.
pub const ENC_COMPRESSED: u8 = 1;

// ── SIValue type tags ──
//
// C's `SIType` is a **bitmask**, not an ordinal: each type is a distinct bit.
// Rust's own effects codec used sequential 0..12 tags, which collide with these
// almost everywhere and are the single most dangerous divergence in the format.

pub const T_MAP: u32 = 1 << 0;
pub const T_ARRAY: u32 = 1 << 3;
pub const T_DATETIME: u32 = 1 << 5;
pub const T_DATE: u32 = 1 << 7;
pub const T_TIME: u32 = 1 << 8;
pub const T_DURATION: u32 = 1 << 10;
pub const T_STRING: u32 = 1 << 11;
pub const T_BOOL: u32 = 1 << 12;
pub const T_INT64: u32 = 1 << 13;
pub const T_DOUBLE: u32 = 1 << 14;
pub const T_NULL: u32 = 1 << 15;
pub const T_POINT: u32 = 1 << 17;
pub const T_VECTOR_F32: u32 = 1 << 18;

// ── decode errors ──

/// Why a buffer could not be decoded.
///
/// Every variant is fatal for the whole buffer: effects are applied
/// transactionally, so a record that does not decode means the replica has
/// diverged and must not apply a partial prefix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecodeError {
    /// Fewer bytes remain than the next field needs.
    UnexpectedEof { want: usize, have: usize },
    /// A count would require more bytes than the buffer can possibly hold.
    /// Checked *before* allocating, so a corrupt length cannot OOM us.
    ImplausibleCount { count: u64, remaining: usize },
    /// An encoding discriminator that is neither [`ENC_PLAIN`] nor
    /// [`ENC_COMPRESSED`].
    BadEncoding(u8),
    /// The roaring blob did not deserialize.
    BadRoaring(String),
    /// The decoded bitmap holds a different number of ids than the record's
    /// count — the guard that keeps row *k* bound to the right entity.
    CardinalityMismatch { expected: u64, actual: u64 },
    /// An `IdList` rank that does not address the dictionary.
    RankOutOfRange { rank: u64, cardinality: u64 },
    /// An `idx_width` other than 1, 2, 4 or 8.
    BadIndexWidth(u8),
    /// An `SIType` bit pattern with no corresponding [`Value`].
    BadValueType(u32),
    /// A string field that is not valid UTF-8, or is missing its NUL.
    BadString,
    /// A buffer whose version byte this build cannot read.
    UnsupportedVersion(u8),
    /// An opcode with no record shape.
    BadOpcode(u32),
}

impl fmt::Display for DecodeError {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::UnexpectedEof { want, have } => {
                write!(
                    f,
                    "truncated effects buffer: want {want} bytes, have {have}"
                )
            }
            Self::ImplausibleCount { count, remaining } => write!(
                f,
                "effects record claims {count} entries but only {remaining} bytes remain"
            ),
            Self::BadEncoding(e) => write!(f, "unknown block encoding {e}"),
            Self::BadRoaring(e) => write!(f, "malformed roaring bitmap: {e}"),
            Self::CardinalityMismatch { expected, actual } => write!(
                f,
                "effects record declares {expected} ids but its bitmap holds {actual}"
            ),
            Self::RankOutOfRange { rank, cardinality } => write!(
                f,
                "dictionary rank {rank} out of range for {cardinality} entries"
            ),
            Self::BadIndexWidth(w) => write!(f, "unsupported index width {w}"),
            Self::BadValueType(t) => write!(f, "unknown SIValue type {t:#x}"),
            Self::BadString => write!(f, "malformed string in effects buffer"),
            Self::UnsupportedVersion(v) => write!(
                f,
                "effects buffer is version {v}, this build reads {EFFECTS_VERSION}"
            ),
            Self::BadOpcode(o) => write!(f, "unknown effect opcode {o}"),
        }
    }
}

impl std::error::Error for DecodeError {}

// ── reader ──

/// A bounds-checked cursor over an effects buffer.
///
/// Every read is checked. Trusting a length off the wire is how a malformed
/// buffer becomes a segfault rather than an error, which is what C does today.
pub struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    #[must_use]
    pub const fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    /// Bytes not yet consumed.
    #[must_use]
    pub const fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    /// True once every byte has been consumed.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.remaining() == 0
    }

    fn take(
        &mut self,
        n: usize,
    ) -> Result<&'a [u8], DecodeError> {
        if self.remaining() < n {
            return Err(DecodeError::UnexpectedEof {
                want: n,
                have: self.remaining(),
            });
        }
        let out = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }

    /// Reject a count the remaining bytes could not possibly satisfy, *before*
    /// it reaches `Vec::with_capacity`.
    ///
    /// `min_bytes_each` is the smallest number of bytes one entry can occupy —
    /// for a variable-width entry that is its fixed prefix, which bounds the
    /// allocation without having to predict the payload.
    fn guard_count(
        &self,
        count: u64,
        min_bytes_each: usize,
    ) -> Result<usize, DecodeError> {
        let need = count.saturating_mul(min_bytes_each as u64);
        if need > self.remaining() as u64 {
            return Err(DecodeError::ImplausibleCount {
                count,
                remaining: self.remaining(),
            });
        }
        Ok(count as usize)
    }

    pub fn u8(&mut self) -> Result<u8, DecodeError> {
        Ok(self.take(1)?[0])
    }

    pub fn u16(&mut self) -> Result<u16, DecodeError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    pub fn u32(&mut self) -> Result<u32, DecodeError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    pub fn i32(&mut self) -> Result<i32, DecodeError> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    pub fn u64(&mut self) -> Result<u64, DecodeError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    pub fn i64(&mut self) -> Result<i64, DecodeError> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    pub fn f32(&mut self) -> Result<f32, DecodeError> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    pub fn f64(&mut self) -> Result<f64, DecodeError> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    /// A C string: `u64` length **including** the NUL, then the bytes.
    pub fn string(&mut self) -> Result<String, DecodeError> {
        let len = self.u64()?;
        if len == 0 {
            return Err(DecodeError::BadString);
        }
        let n = self.guard_count(len, 1)?;
        let raw = self.take(n)?;
        // The last byte is the terminator C requires; it is not part of the value.
        let (body, nul) = raw.split_at(n - 1);
        if nul != [0] {
            return Err(DecodeError::BadString);
        }
        String::from_utf8(body.to_vec()).map_err(|_| DecodeError::BadString)
    }
}

// ── primitive writers ──

pub fn write_u8(
    buf: &mut Vec<u8>,
    v: u8,
) {
    buf.push(v);
}

pub fn write_u16(
    buf: &mut Vec<u8>,
    v: u16,
) {
    buf.extend_from_slice(&v.to_le_bytes());
}

pub fn write_u32(
    buf: &mut Vec<u8>,
    v: u32,
) {
    buf.extend_from_slice(&v.to_le_bytes());
}

pub fn write_u64(
    buf: &mut Vec<u8>,
    v: u64,
) {
    buf.extend_from_slice(&v.to_le_bytes());
}

pub fn write_i64(
    buf: &mut Vec<u8>,
    v: i64,
) {
    buf.extend_from_slice(&v.to_le_bytes());
}

pub fn write_f64(
    buf: &mut Vec<u8>,
    v: f64,
) {
    buf.extend_from_slice(&v.to_le_bytes());
}

/// C's `LabelID` and `RelationID` are `int`, so signed and 4 bytes.
pub fn write_label_id(
    buf: &mut Vec<u8>,
    v: i32,
) {
    buf.extend_from_slice(&v.to_le_bytes());
}

/// A C string: length **including** the NUL terminator, then the bytes and the
/// terminator. C reads these straight into a `char[]` and treats them as C
/// strings, so omitting the NUL makes it read the name plus whatever follows.
pub fn write_string(
    buf: &mut Vec<u8>,
    s: &str,
) {
    debug_assert!(
        !s.as_bytes().contains(&0),
        "an interior NUL would truncate this string on the C side"
    );
    write_u64(buf, s.len() as u64 + 1);
    buf.extend_from_slice(s.as_bytes());
    buf.push(0);
}

// ── SIValue ──

/// A `SIValue`: 4-byte type bitmask, then the payload.
///
/// `T_NULL` has no payload at all — not a zero byte.
pub fn write_value(
    buf: &mut Vec<u8>,
    value: &Value,
) {
    match value {
        Value::Null => write_u32(buf, T_NULL),
        Value::Bool(b) => {
            write_u32(buf, T_BOOL);
            write_u8(buf, u8::from(*b));
        }
        Value::Int(i) => {
            write_u32(buf, T_INT64);
            write_i64(buf, *i);
        }
        Value::Float(f) => {
            write_u32(buf, T_DOUBLE);
            write_f64(buf, *f);
        }
        Value::String(s) => {
            write_u32(buf, T_STRING);
            write_string(buf, s);
        }
        Value::List(items) => {
            write_u32(buf, T_ARRAY);
            // u32, not u64: C reads the count as `uint32`.
            write_u32(buf, items.len() as u32);
            for item in items.iter() {
                write_value(buf, item);
            }
        }
        Value::Map(m) => {
            write_u32(buf, T_MAP);
            write_u32(buf, m.len() as u32);
            for (k, v) in m.iter() {
                // Keys are strings on both sides; C writes the key as a bare
                // string, not as a nested SIValue.
                write_string(buf, k);
                write_value(buf, v);
            }
        }
        Value::Point(p) => {
            write_u32(buf, T_POINT);
            // 2 x f32. Rust's own format used f64 here, which silently doubles
            // the payload and desyncs everything after it.
            buf.extend_from_slice(&p.latitude.to_le_bytes());
            buf.extend_from_slice(&p.longitude.to_le_bytes());
        }
        Value::VecF32(v) => {
            write_u32(buf, T_VECTOR_F32);
            write_u32(buf, v.len() as u32);
            for f in v.iter() {
                buf.extend_from_slice(&f.to_le_bytes());
            }
        }
        Value::Datetime(ts) => {
            write_u32(buf, T_DATETIME);
            write_i64(buf, *ts);
        }
        Value::Date(ts) => {
            write_u32(buf, T_DATE);
            write_i64(buf, *ts);
        }
        Value::Time(ts) => {
            write_u32(buf, T_TIME);
            write_i64(buf, *ts);
        }
        Value::Duration(d) => {
            write_u32(buf, T_DURATION);
            write_i64(buf, *d);
        }
        // Nodes, edges and paths are never property values, so they cannot reach
        // an effect. Encoding one as NULL would corrupt the stream silently.
        other => panic!("value cannot appear in an effect: {other:?}"),
    }
}

/// Inverse of [`write_value`].
pub fn read_value(r: &mut Reader<'_>) -> Result<Value, DecodeError> {
    let t = r.u32()?;
    let v = match t {
        T_NULL => Value::Null,
        T_BOOL => Value::Bool(r.u8()? != 0),
        T_INT64 => Value::Int(r.i64()?),
        T_DOUBLE => Value::Float(r.f64()?),
        T_STRING => Value::String(Arc::new(r.string()?)),
        T_ARRAY => {
            let n = r.u32()?;
            // A list entry is at least a 4-byte type tag.
            let n = r.guard_count(u64::from(n), 4)?;
            let mut items = ThinVec::with_capacity(n);
            for _ in 0..n {
                items.push(read_value(r)?);
            }
            Value::List(Arc::new(items))
        }
        T_MAP => {
            let n = r.u32()?;
            // Each pair is at least an 8-byte length plus a 4-byte type tag.
            let n = r.guard_count(u64::from(n), 12)?;
            let mut m = OrderMap::default();
            for _ in 0..n {
                let k = Arc::new(r.string()?);
                m.insert(k, read_value(r)?);
            }
            Value::Map(Arc::new(m))
        }
        T_POINT => {
            let latitude = r.f32()?;
            let longitude = r.f32()?;
            Value::Point(Point {
                latitude,
                longitude,
            })
        }
        T_VECTOR_F32 => {
            let n = r.u32()?;
            let n = r.guard_count(u64::from(n), 4)?;
            let mut v = ThinVec::with_capacity(n);
            for _ in 0..n {
                v.push(r.f32()?);
            }
            Value::VecF32(Arc::new(v))
        }
        T_DATETIME => Value::Datetime(r.i64()?),
        T_DATE => Value::Date(r.i64()?),
        T_TIME => Value::Time(r.i64()?),
        T_DURATION => Value::Duration(r.i64()?),
        other => return Err(DecodeError::BadValueType(other)),
    };
    Ok(v)
}

// ── IdSet ──

/// Build a run-optimized bitmap over `ids`.
///
/// `optimize()` is **normative**, not a tuning knob: an unoptimized bitmap
/// serializes to different bytes, so two engines that disagree about calling it
/// produce different buffers for the same set. A contiguous 10k set is 8,220
/// bytes unoptimized and 27 optimized.
fn build_bitmap(ids: &[u64]) -> RoaringTreemap {
    let mut bitmap = RoaringTreemap::new();
    for &id in ids {
        bitmap.insert(id);
    }
    bitmap.optimize();
    bitmap
}

fn serialize_bitmap(bitmap: &RoaringTreemap) -> Vec<u8> {
    let mut blob = Vec::with_capacity(bitmap.serialized_size());
    bitmap
        .serialize_into(&mut blob)
        .expect("writing to a Vec cannot fail");
    blob
}

/// `IdSet` — unique, order-free ids: `u8 enc` · either `u64 × count` or
/// `u32 blob_len · roaring64[blob_len]`.
///
/// `ids` must be **ascending and unique**. Ascending is not cosmetic: the roaring
/// encoding can only ever yield ascending order, so accepting unsorted input
/// would make the two encodings disagree about which row belongs to which entity.
///
/// Whichever encoding is smaller wins, ties going to the plain form. There is no
/// count threshold, because roaring's cost tracks how ids are *spread*, not how
/// many there are — 1,024 ids spread 2^32 apart serialize to 22,536 bytes against
/// the plain form's 8,192. Both libraries report a bitmap's serialized size
/// without serializing it, so choosing costs nothing.
pub fn write_id_set(
    buf: &mut Vec<u8>,
    ids: &[u64],
) {
    debug_assert!(
        ids.windows(2).all(|w| w[0] < w[1]),
        "IdSet requires ascending, unique ids"
    );

    // Ascending, so the maximum is the last element.
    let width = value_width_for(ids.last().copied().unwrap_or(0));
    let plain_bytes = 1 + ids.len() * width as usize;

    let bitmap = build_bitmap(ids);
    let roaring_bytes = 4 + bitmap.serialized_size();

    if roaring_bytes < plain_bytes {
        write_u8(buf, ENC_COMPRESSED);
        let blob = serialize_bitmap(&bitmap);
        write_u32(buf, blob.len() as u32);
        buf.extend_from_slice(&blob);
    } else {
        write_u8(buf, ENC_PLAIN);
        write_u8(buf, width);
        write_widths(buf, ids, width);
    }
}

/// Inverse of [`write_id_set`], returning exactly `count` ascending ids.
///
/// The cardinality assertion is the guard that keeps row *k* bound to the k-th
/// smallest id: a set silently deduplicates, so a bitmap one id short would shift
/// every subsequent row onto the wrong entity rather than fail.
pub fn read_id_set(
    r: &mut Reader<'_>,
    count: u32,
) -> Result<Vec<u64>, DecodeError> {
    match r.u8()? {
        ENC_PLAIN => {
            let width = r.u8()?;
            if !matches!(width, 1 | 2 | 4 | 8) {
                return Err(DecodeError::BadIndexWidth(width));
            }
            let n = r.guard_count(u64::from(count), width as usize)?;
            read_widths(r, n, width)
        }
        ENC_COMPRESSED => {
            let blob_len = r.u32()?;
            let blob = r.take(blob_len as usize)?;
            let bitmap = RoaringTreemap::deserialize_from(blob)
                .map_err(|e| DecodeError::BadRoaring(e.to_string()))?;
            if bitmap.len() != u64::from(count) {
                return Err(DecodeError::CardinalityMismatch {
                    expected: u64::from(count),
                    actual: bitmap.len(),
                });
            }
            Ok(bitmap.iter().collect())
        }
        other => Err(DecodeError::BadEncoding(other)),
    }
}

// ── IdList ──

/// The narrowest unsigned width that can address `cardinality` entries.
#[must_use]
pub const fn idx_width_for(cardinality: u64) -> u8 {
    if cardinality <= 0x100 {
        1
    } else if cardinality <= 0x1_0000 {
        2
    } else if cardinality <= 0x1_0000_0000 {
        4
    } else {
        8
    }
}

/// The narrowest unsigned width that can hold `max` itself.
///
/// Distinct from [`idx_width_for`], which sizes a *rank* into a dictionary:
/// this sizes the value.
#[must_use]
pub const fn value_width_for(max: u64) -> u8 {
    if max <= 0xFF {
        1
    } else if max <= 0xFFFF {
        2
    } else if max <= 0xFFFF_FFFF {
        4
    } else {
        8
    }
}

/// Write `values` at a fixed width.
///
/// The width is matched **once**, not per value: a branch inside the loop stops
/// the compiler vectorizing what is otherwise a straight truncate-and-store.
fn write_widths(
    buf: &mut Vec<u8>,
    values: &[u64],
    width: u8,
) {
    match width {
        1 => buf.extend(values.iter().map(|&v| v as u8)),
        2 => {
            buf.reserve(values.len() * 2);
            for &v in values {
                buf.extend_from_slice(&(v as u16).to_le_bytes());
            }
        }
        4 => {
            buf.reserve(values.len() * 4);
            for &v in values {
                buf.extend_from_slice(&(v as u32).to_le_bytes());
            }
        }
        _ => {
            buf.reserve(values.len() * 8);
            for &v in values {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
}

/// Read `count` values at a fixed width.
///
/// One bounds check for the whole run and one width match for the whole loop,
/// so each arm is a tight widening map the compiler can vectorize. Doing it per
/// value instead measured 3.5x slower than reading raw `u64`s.
fn read_widths(
    r: &mut Reader<'_>,
    count: usize,
    width: u8,
) -> Result<Vec<u64>, DecodeError> {
    let bytes = r.take(count * width as usize)?;
    let mut out = Vec::with_capacity(count);
    match width {
        1 => out.extend(bytes.iter().map(|&b| u64::from(b))),
        2 => out.extend(
            bytes
                .chunks_exact(2)
                .map(|c| u64::from(u16::from_le_bytes(c.try_into().unwrap()))),
        ),
        4 => out.extend(
            bytes
                .chunks_exact(4)
                .map(|c| u64::from(u32::from_le_bytes(c.try_into().unwrap()))),
        ),
        _ => out.extend(
            bytes
                .chunks_exact(8)
                .map(|c| u64::from_le_bytes(c.try_into().unwrap())),
        ),
    }
    Ok(out)
}

/// `IdList` — ordered ids that may repeat: `u8 enc` · either `u64 × count` or
/// `u32 dict_len · roaring64 · u8 idx_width · uint(idx_width) × count`.
///
/// Used for edge endpoints, where both properties matter: many edges share a
/// source, and each endpoint must stay positionally aligned with its edge id.
///
/// The dictionary holds the **distinct** ids as a run-optimized roaring bitmap;
/// each row then carries its **rank in that bitmap's ascending order**. Order and
/// duplication live entirely in the rank array, which is what lets the ids
/// themselves be stored as a set.
pub fn write_id_list(
    buf: &mut Vec<u8>,
    ids: &[u64],
) {
    // Narrowing needs one pass for the maximum and nothing else — no sort, no
    // bitmap, no ranks. Priced first because it is the cheap candidate.
    let max = ids.iter().copied().max().unwrap_or(0);
    let raw_width = value_width_for(max);
    let raw_bytes = 1 + ids.len() * raw_width as usize;

    // The dictionary can only beat raw ids when its ranks are strictly narrower
    // than they are, which takes real duplication. Deciding by size alone cannot
    // see that: it has to build the dictionary to measure it, and building it is
    // the whole cost. Measured at 10,000 rows, choosing the dictionary for an 8%
    // size win cost 2,455x what those bytes are worth on the write thread. So
    // probe for duplication *before* paying the sort.
    if raw_width > 1 && duplication_probe_passes(ids) {
        let ascending = distinct_ascending(ids);
        let dict = build_bitmap(&ascending);
        let rank_width = idx_width_for(dict.len());
        let dict_bytes = 5 + dict.serialized_size() + ids.len() * rank_width as usize;
        if dict_bytes < raw_bytes {
            write_u8(buf, ENC_COMPRESSED);
            let blob = serialize_bitmap(&dict);
            write_u32(buf, blob.len() as u32);
            buf.extend_from_slice(&blob);
            write_u8(buf, rank_width);
            write_ranks(buf, ids, &ascending, rank_width);
            return;
        }
    }

    write_u8(buf, ENC_PLAIN);
    write_u8(buf, raw_width);
    write_widths(buf, ids, raw_width);
}

/// The distinct ids in ascending order.
///
/// Deduplicating *before* building the bitmap matters more than it looks:
/// inserting every row inserts `count` times to produce a `distinct`-element
/// bitmap, and roaring insertion dominates this block. It also yields the
/// ascending order the ranks are taken against, so the bitmap never has to be
/// iterated back out.
fn distinct_ascending(ids: &[u64]) -> Vec<u64> {
    let mut out = ids.to_vec();
    out.sort_unstable();
    out.dedup();
    out
}

/// Each row's rank in the dictionary's ascending order.
///
/// Binary search over the materialized order. Measured against a HashMap of
/// id -> rank, which loses at every cardinality tried: the dictionary is small
/// and contiguous, so the search stays in cache while hashing does not.
fn write_ranks(
    buf: &mut Vec<u8>,
    ids: &[u64],
    ascending: &[u64],
    width: u8,
) {
    let ranks: Vec<u64> = ids
        .iter()
        .map(|id| {
            ascending
                .binary_search(id)
                .expect("every id is in the dictionary built from those same ids")
                as u64
        })
        .collect();
    write_widths(buf, &ranks, width);
}

/// How many rows the duplication probe looks at.
const DUP_PROBE_SAMPLE: usize = 128;

/// Whether a cheap sample suggests enough duplication for the dictionary to be
/// worth building.
///
/// **Correctness never depends on this.** A wrong answer either builds a
/// dictionary the size comparison then rejects (wasted work) or skips one that
/// would have been slightly smaller (a bigger buffer). Both decode identically,
/// and the decoder always supports every encoding — so unlike the wire layout,
/// this heuristic does **not** have to match between the two engines, and either
/// side can tune it without a version bump.
///
/// Expected distinct values in a sample of 128 drawn from *n*, uniform:
/// n=100 gives ~72 (57%), n=1,000 gives ~120 (94%), n=10,000 gives ~127 (99%).
/// Three quarters separates the case the dictionary wins from the ones it does
/// not.
fn duplication_probe_passes(ids: &[u64]) -> bool {
    let sampled = ids.len().min(DUP_PROBE_SAMPLE);
    if sampled == 0 {
        return false;
    }

    // Stride across the whole list rather than taking a prefix: endpoints
    // often arrive grouped by source, and a prefix would see one group.
    let step = (ids.len() / sampled).max(1);
    let mut sample = [0_u64; DUP_PROBE_SAMPLE];
    for (slot, &id) in sample[..sampled].iter_mut().zip(ids.iter().step_by(step)) {
        *slot = id;
    }

    // Sort and count runs: 128 elements, so this beats the quadratic scan a
    // linear "have I seen it" search would do.
    let sample = &mut sample[..sampled];
    sample.sort_unstable();
    let distinct = 1 + sample.windows(2).filter(|w| w[0] != w[1]).count();

    distinct * 4 < sampled * 3
}

/// Encode with a chosen encoding rather than the one the rule picks.
///
/// Exposed only so benchmarks can price both paths on identical input —
/// production code calls [`write_id_list`]. A forced [`ENC_COMPRESSED`] can be
/// *larger* than the raw form, and bypasses the duplication probe entirely.
#[doc(hidden)]
pub fn write_id_list_forced(
    buf: &mut Vec<u8>,
    ids: &[u64],
    encoding: u8,
) {
    if encoding == ENC_COMPRESSED {
        let ascending = distinct_ascending(ids);
        let dict = build_bitmap(&ascending);
        let width = idx_width_for(dict.len());
        write_u8(buf, ENC_COMPRESSED);
        let blob = serialize_bitmap(&dict);
        write_u32(buf, blob.len() as u32);
        buf.extend_from_slice(&blob);
        write_u8(buf, width);
        write_ranks(buf, ids, &ascending, width);
    } else {
        let max = ids.iter().copied().max().unwrap_or(0);
        let width = value_width_for(max);
        write_u8(buf, ENC_PLAIN);
        write_u8(buf, width);
        write_widths(buf, ids, width);
    }
}

/// Inverse of [`write_id_list`], returning exactly `count` ids in row order.
pub fn read_id_list(
    r: &mut Reader<'_>,
    count: u32,
) -> Result<Vec<u64>, DecodeError> {
    match r.u8()? {
        ENC_PLAIN => {
            let width = r.u8()?;
            if !matches!(width, 1 | 2 | 4 | 8) {
                return Err(DecodeError::BadIndexWidth(width));
            }
            let n = r.guard_count(u64::from(count), width as usize)?;
            read_widths(r, n, width)
        }
        ENC_COMPRESSED => {
            let dict_len = r.u32()?;
            let blob = r.take(dict_len as usize)?;
            let dict = RoaringTreemap::deserialize_from(blob)
                .map_err(|e| DecodeError::BadRoaring(e.to_string()))?;

            let width = r.u8()?;
            if !matches!(width, 1 | 2 | 4 | 8) {
                return Err(DecodeError::BadIndexWidth(width));
            }

            // Materialize the ascending order once, then every row is one index.
            let ascending: Vec<u64> = dict.iter().collect();
            let cardinality = ascending.len() as u64;

            let n = r.guard_count(u64::from(count), width as usize)?;
            let ranks = read_widths(r, n, width)?;
            ranks
                .into_iter()
                .map(|rank| {
                    ascending
                        .get(rank as usize)
                        .copied()
                        .ok_or(DecodeError::RankOutOfRange { rank, cardinality })
                })
                .collect()
        }
        other => Err(DecodeError::BadEncoding(other)),
    }
}

// ── RelType ──

/// `RelType` — one relationship id per record, part of the partition key.
pub fn write_rel_type(
    buf: &mut Vec<u8>,
    relation_id: i32,
) {
    write_label_id(buf, relation_id);
}

/// Inverse of [`write_rel_type`].
pub fn read_rel_type(r: &mut Reader<'_>) -> Result<i32, DecodeError> {
    r.i32()
}

// ── LabelSet ──

/// `LabelSet` — `u16 n` · `i32 × n`, stated once per record.
///
/// Always plain, never roaring: roaring's 27–30 byte floor exceeds the whole
/// block until roughly seven labels, and the block is already amortised over
/// every row in the record.
pub fn write_label_set(
    buf: &mut Vec<u8>,
    labels: &[i32],
) {
    write_u16(buf, labels.len() as u16);
    for &label in labels {
        write_label_id(buf, label);
    }
}

/// Inverse of [`write_label_set`].
pub fn read_label_set(r: &mut Reader<'_>) -> Result<Vec<i32>, DecodeError> {
    let n = r.u16()?;
    let n = r.guard_count(u64::from(n), 4)?;
    let mut labels = Vec::with_capacity(n);
    for _ in 0..n {
        labels.push(r.i32()?);
    }
    Ok(labels)
}

// ── AttrSet ──

/// `AttrSet` — `u16 n` · `u16 attr_id × n` · `SIValue × (count × n)`, row-major.
///
/// The attribute ids are stated **once**, not per row: that removes exactly
/// `2 × count × n` bytes, which is provable arithmetic rather than a measured
/// effect. A row missing one of the shape's attributes encodes `T_NULL` in that
/// slot, which is unambiguous because FalkorDB never stores a null property —
/// setting one to null removes it.
///
/// Rows stay row-major. Grouping values by attribute instead saves zero bytes
/// uncompressed and changes sign with the data once compressed, which does not
/// justify a second layout two engines must match byte-for-byte.
///
/// `rows` is `count × attr_ids.len()` values in row-major order.
pub fn write_attr_set(
    buf: &mut Vec<u8>,
    attr_ids: &[u16],
    rows: &[Value],
) {
    debug_assert!(
        attr_ids.is_empty() || rows.len().is_multiple_of(attr_ids.len()),
        "rows must be a whole number of records"
    );
    write_u16(buf, attr_ids.len() as u16);
    for &attr_id in attr_ids {
        write_u16(buf, attr_id);
    }
    for value in rows {
        write_value(buf, value);
    }
}

/// Inverse of [`write_attr_set`]: the shape, then `count × n` values row-major.
pub fn read_attr_set(
    r: &mut Reader<'_>,
    count: u32,
) -> Result<(Vec<u16>, Vec<Value>), DecodeError> {
    let n = r.u16()?;
    let n = r.guard_count(u64::from(n), 2)?;
    let mut attr_ids = Vec::with_capacity(n);
    for _ in 0..n {
        attr_ids.push(r.u16()?);
    }

    let total = u64::from(count).saturating_mul(n as u64);
    // Every value is at least its 4-byte type tag.
    let total = r.guard_count(total, 4)?;
    let mut rows = Vec::with_capacity(total);
    for _ in 0..total {
        rows.push(read_value(r)?);
    }
    Ok((attr_ids, rows))
}

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

// ── buffer ──

/// Start a fresh buffer, stamped with the version header.
#[must_use]
pub fn new_buffer() -> Vec<u8> {
    vec![EFFECTS_VERSION]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(buf: &[u8]) -> String {
        buf.iter()
            .map(|b| format!("{b:02x}"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    // ── IdSet ──

    #[test]
    fn id_set_singleton_stays_plain() {
        // Roaring has a ~27 byte floor, so one id is always cheaper raw. Merging
        // the full-width and narrowed encodings shrinks this from 9 bytes to 3:
        // encoding, width, value.
        let mut buf = Vec::new();
        write_id_set(&mut buf, &[0]);
        assert_eq!(hex(&buf), "00 01 00");
    }

    #[test]
    fn id_set_consecutive_run_collapses() {
        // 10,000 consecutive ids are one run container: 27 bytes of bitmap
        // regardless of the count, plus discriminator and length prefix.
        let ids: Vec<u64> = (0..10_000).collect();
        let mut buf = Vec::new();
        write_id_set(&mut buf, &ids);
        assert_eq!(buf[0], ENC_COMPRESSED);
        assert_eq!(buf.len(), 32, "1 enc + 4 blob_len + 27 bitmap");

        let mut r = Reader::new(&buf);
        assert_eq!(read_id_set(&mut r, 10_000).unwrap(), ids);
        assert!(r.is_empty());
    }

    #[test]
    fn id_set_wide_spread_stays_plain() {
        // The case that makes a count threshold unsound: at stride 2^32 roaring
        // costs ~22 B per id, so the plain form wins at *every* count.
        let ids: Vec<u64> = (0..1_024).map(|i| i << 32).collect();
        let mut buf = Vec::new();
        write_id_set(&mut buf, &ids);
        assert_eq!(buf[0], ENC_PLAIN);
        assert_eq!(buf[1], 8, "ids past 2^32 need the full width");
        assert_eq!(buf.len(), 2 + 1_024 * 8);

        let mut r = Reader::new(&buf);
        assert_eq!(read_id_set(&mut r, 1_024).unwrap(), ids);
    }

    #[test]
    fn id_set_roundtrips_across_the_crossover() {
        for n in [1_u64, 2, 4, 8, 63, 64, 65, 1_000, 4_096] {
            let ids: Vec<u64> = (0..n).collect();
            let mut buf = Vec::new();
            write_id_set(&mut buf, &ids);
            let mut r = Reader::new(&buf);
            assert_eq!(read_id_set(&mut r, n as u32).unwrap(), ids, "n = {n}");
            assert!(r.is_empty(), "n = {n} left {} bytes", r.remaining());
        }
    }

    #[test]
    fn id_set_rejects_a_cardinality_lie() {
        // The guard that stops a short bitmap from shifting every row onto the
        // wrong entity.
        let ids: Vec<u64> = (0..100).collect();
        let mut buf = Vec::new();
        write_id_set(&mut buf, &ids);
        let mut r = Reader::new(&buf);
        assert_eq!(
            read_id_set(&mut r, 99),
            Err(DecodeError::CardinalityMismatch {
                expected: 99,
                actual: 100,
            })
        );
    }

    // ── IdList ──

    #[test]
    fn id_list_supernode_uses_the_dictionary() {
        // 10,000 edges from 3 sources in a graph big enough that raw ids need 4
        // bytes each. Heavy duplication plus wide ids is the dictionary's case:
        // the ranks collapse to one byte and the dictionary holds three values.
        let sources = [4_000_000_000_u64, 7, 2_100_000_000];
        let src: Vec<u64> = (0..10_000).map(|i| sources[i % 3]).collect();

        let mut buf = Vec::new();
        write_id_list(&mut buf, &src);
        assert_eq!(buf[0], ENC_COMPRESSED, "the dictionary should win here");

        let mut narrow = Vec::new();
        write_id_list_forced(&mut narrow, &src, ENC_PLAIN);
        assert!(
            buf.len() < narrow.len(),
            "dict {} should beat narrow {}",
            buf.len(),
            narrow.len()
        );

        let mut r = Reader::new(&buf);
        assert_eq!(read_id_list(&mut r, 10_000).unwrap(), src);
        assert!(r.is_empty());
    }

    #[test]
    fn id_list_raw_beats_the_dictionary_without_duplication() {
        // The case the dictionary was assumed to own and does not: 10,000
        // distinct dense ids. Narrowing to 2 bytes costs one pass for the
        // maximum, where the dictionary costs a sort, a bitmap and a rank
        // search per row — for more bytes. The probe skips it entirely.
        let src: Vec<u64> = (0..10_000).map(|i| (i * 6) % 60_000).collect();
        let mut buf = Vec::new();
        write_id_list(&mut buf, &src);
        assert_eq!(buf[0], ENC_PLAIN);

        let mut dict = Vec::new();
        write_id_list_forced(&mut dict, &src, ENC_COMPRESSED);
        assert!(buf.len() < dict.len(), "{} vs {}", buf.len(), dict.len());

        let mut r = Reader::new(&buf);
        assert_eq!(read_id_list(&mut r, 10_000).unwrap(), src);
    }

    #[test]
    fn id_list_wide_spread_stays_plain() {
        // Where the dictionary does lose: ids spread a bucket apart make the
        // bitmap itself larger than the ids it replaces, exactly as for IdSet.
        let src: Vec<u64> = (0..1_000).map(|i| i << 32).collect();
        let mut buf = Vec::new();
        write_id_list(&mut buf, &src);
        assert_eq!(buf[0], ENC_PLAIN);
        assert_eq!(buf[1], 8, "ids past 2^32 need the full width");
        assert_eq!(buf.len(), 2 + 1_000 * 8);

        let mut r = Reader::new(&buf);
        assert_eq!(read_id_list(&mut r, 1_000).unwrap(), src);
    }

    #[test]
    fn id_list_preserves_duplicates_and_order() {
        // The property that separates IdList from IdSet.
        let src = [9_u64, 1, 9, 9, 1, 5];
        let mut buf = Vec::new();
        write_id_list(&mut buf, &src);
        let mut r = Reader::new(&buf);
        let out = read_id_list(&mut r, 6).unwrap();
        assert_eq!(out, src);
        assert_ne!(out, vec![1, 5, 9], "an IdSet would have produced this");
    }

    /// `rows` endpoints drawn from `distinct` sources inside a graph of
    /// `nodes` nodes. FalkorDB allocates node ids densely from zero and reuses
    /// them from a free list, so the largest id tracks the graph's size — which
    /// is what decides whether narrowing alone already captures the saving.
    fn endpoints(
        rows: usize,
        distinct: usize,
        nodes: u64,
    ) -> Vec<u64> {
        (0..rows)
            .map(|i| {
                let h = (i as u64).wrapping_mul(0x9E37_7997_9F4A_7C15);
                (h % distinct as u64) * (nodes / distinct as u64).max(1)
            })
            .collect()
    }

    #[test]
    fn id_list_selection_crossover() {
        // Which of the three encodings the size rule picks, and why. The
        // dictionary only wins where duplication is heavy *and* the ids are
        // wide enough that narrowing has not already taken the saving.
        println!("\n   nodes   rows distinct      raw     dict  picked");
        for (nodes, distinct) in [
            (10_000_u64, 1_usize),
            (10_000, 100),
            (10_000, 1_000),
            (10_000, 10_000),
            (5_000_000, 100),
            (5_000_000, 10_000),
        ] {
            let rows = 10_000;
            let ids = endpoints(rows, distinct, nodes);
            let mut p = Vec::new();
            write_id_list_forced(&mut p, &ids, ENC_PLAIN);
            let mut d = Vec::new();
            write_id_list_forced(&mut d, &ids, ENC_COMPRESSED);
            let mut auto = Vec::new();
            write_id_list(&mut auto, &ids);

            let picked = if auto[0] == ENC_COMPRESSED {
                "dict"
            } else {
                "raw"
            };
            let given_up = auto.len() as i64 - d.len() as i64;
            println!(
                "{nodes:8} {rows:6} {distinct:8} {:8} {:8}  {picked:<5}{}",
                p.len(),
                d.len(),
                if given_up > 0 {
                    format!("  (+{given_up} B, dictionary declined by the probe)")
                } else {
                    String::new()
                }
            );

            // The rule is *not* "smallest of the three wins": the duplication
            // probe declines to even build a dictionary that would be only
            // marginally smaller, because building it is the expensive part.
            // What must hold is that the two cheap encodings are always beaten.
            assert!(auto.len() <= p.len());
            // And when the dictionary is chosen, it is genuinely the smallest.
            if auto[0] == ENC_COMPRESSED {
                assert_eq!(auto.len(), d.len().min(p.len()));
            }
            // and whatever was picked must round-trip
            let mut r = Reader::new(&auto);
            assert_eq!(read_id_list(&mut r, rows as u32).unwrap(), ids);
        }
    }

    #[test]
    fn id_list_index_width_boundaries() {
        assert_eq!(idx_width_for(1), 1);
        assert_eq!(idx_width_for(0x100), 1);
        assert_eq!(idx_width_for(0x101), 2);
        assert_eq!(idx_width_for(0x1_0000), 2);
        assert_eq!(idx_width_for(0x1_0001), 4);
        assert_eq!(idx_width_for(0x1_0000_0000), 4);
        assert_eq!(idx_width_for(0x1_0000_0001), 8);
    }

    #[test]
    fn id_list_rejects_an_out_of_range_rank() {
        let src = [5_u64; 10];
        let mut buf = Vec::new();
        write_id_list_forced(&mut buf, &src, ENC_COMPRESSED);
        assert_eq!(buf[0], ENC_COMPRESSED);
        // The dictionary holds one entry, so rank 1 is out of range.
        *buf.last_mut().unwrap() = 1;
        let mut r = Reader::new(&buf);
        assert_eq!(
            read_id_list(&mut r, 10),
            Err(DecodeError::RankOutOfRange {
                rank: 1,
                cardinality: 1,
            })
        );
    }

    // ── the other three blocks ──

    #[test]
    fn rel_type_is_four_signed_bytes() {
        let mut buf = Vec::new();
        write_rel_type(&mut buf, 7);
        assert_eq!(hex(&buf), "07 00 00 00");
        let mut r = Reader::new(&buf);
        assert_eq!(read_rel_type(&mut r).unwrap(), 7);
    }

    #[test]
    fn label_set_pins_its_bytes() {
        let mut buf = Vec::new();
        write_label_set(&mut buf, &[7, 9]);
        assert_eq!(hex(&buf), "02 00 07 00 00 00 09 00 00 00");
        let mut r = Reader::new(&buf);
        assert_eq!(read_label_set(&mut r).unwrap(), vec![7, 9]);
    }

    #[test]
    fn attr_set_states_ids_once() {
        // Two rows, one attribute: the id appears once, the values twice.
        let mut buf = Vec::new();
        write_attr_set(&mut buf, &[0], &[Value::Int(1), Value::Int(2)]);
        assert_eq!(
            hex(&buf),
            "01 00 00 00 \
             00 20 00 00 01 00 00 00 00 00 00 00 \
             00 20 00 00 02 00 00 00 00 00 00 00"
        );

        let mut r = Reader::new(&buf);
        let (attr_ids, rows) = read_attr_set(&mut r, 2).unwrap();
        assert_eq!(attr_ids, vec![0]);
        assert_eq!(rows, vec![Value::Int(1), Value::Int(2)]);
        assert!(r.is_empty());
    }

    #[test]
    fn attr_set_null_marks_an_absent_property() {
        // T_NULL is unambiguous: FalkorDB never stores a null property value.
        let mut buf = Vec::new();
        write_attr_set(
            &mut buf,
            &[0, 1],
            &[Value::Int(1), Value::Null, Value::Null, Value::Int(2)],
        );
        let mut r = Reader::new(&buf);
        let (attr_ids, rows) = read_attr_set(&mut r, 2).unwrap();
        assert_eq!(attr_ids, vec![0, 1]);
        assert_eq!(rows[1], Value::Null);
        assert_eq!(rows[3], Value::Int(2));
    }

    // ── SIValue ──

    #[test]
    fn value_tags_are_c_bitmasks_not_ordinals() {
        // The single most dangerous divergence: Rust's own codec used sequential
        // 0..12 tags, which collide with C's bitmask almost everywhere.
        let mut buf = Vec::new();
        write_value(&mut buf, &Value::Int(1));
        assert_eq!(&buf[..4], &[0x00, 0x20, 0x00, 0x00], "T_INT64 = 1 << 13");

        buf.clear();
        write_value(&mut buf, &Value::Null);
        assert_eq!(hex(&buf), "00 80 00 00", "T_NULL = 1 << 15, no payload");
    }

    #[test]
    fn value_point_is_two_f32() {
        let mut buf = Vec::new();
        write_value(
            &mut buf,
            &Value::Point(Point {
                latitude: 1.0,
                longitude: 2.0,
            }),
        );
        assert_eq!(buf.len(), 4 + 8, "type tag plus 2 x f32, not 2 x f64");
    }

    #[test]
    fn value_string_carries_its_nul() {
        let mut buf = Vec::new();
        write_value(&mut buf, &Value::String(Arc::new("ab".into())));
        // type tag, then len = 3 (including NUL), then "ab\0"
        assert_eq!(hex(&buf), "00 08 00 00 03 00 00 00 00 00 00 00 61 62 00");
    }

    #[test]
    fn value_roundtrips_every_encodable_variant() {
        let cases = vec![
            Value::Null,
            Value::Bool(true),
            Value::Bool(false),
            Value::Int(-9_007_199_254_740_993),
            Value::Float(0.1),
            Value::String(Arc::new("hello".into())),
            Value::List(Arc::new([Value::Int(1), Value::Null].into_iter().collect())),
            Value::Point(Point {
                latitude: 32.07,
                longitude: 34.79,
            }),
            Value::VecF32(Arc::new([1.5_f32, -2.5].into_iter().collect())),
            Value::Datetime(1_700_000_000),
            Value::Date(19_000),
            Value::Time(3_600),
            Value::Duration(90),
        ];
        for case in cases {
            let mut buf = Vec::new();
            write_value(&mut buf, &case);
            let mut r = Reader::new(&buf);
            assert_eq!(read_value(&mut r).unwrap(), case);
            assert!(r.is_empty(), "{case:?} left {} bytes", r.remaining());
        }
    }

    // ── malformed input ──

    #[test]
    fn truncation_is_an_error_not_a_panic() {
        let ids: Vec<u64> = (0..10).collect();
        let mut buf = Vec::new();
        write_id_set(&mut buf, &ids);
        for cut in 0..buf.len() {
            let mut r = Reader::new(&buf[..cut]);
            // Must not panic, and must not claim success.
            assert!(read_id_set(&mut r, 10).is_err(), "cut at {cut}");
        }
    }

    #[test]
    fn an_absurd_count_is_rejected_before_allocating() {
        // A short buffer claiming four billion entries must not try to reserve
        // for them.
        let buf = [ENC_PLAIN, 8, 0, 0];
        let mut r = Reader::new(&buf);
        assert!(matches!(
            read_id_set(&mut r, u32::MAX),
            Err(DecodeError::ImplausibleCount { .. })
        ));
    }

    #[test]
    fn a_bad_encoding_byte_is_rejected() {
        let buf = [0xEE_u8];
        let mut r = Reader::new(&buf);
        assert_eq!(read_id_set(&mut r, 1), Err(DecodeError::BadEncoding(0xEE)));
    }

    #[test]
    fn an_unknown_value_tag_is_rejected() {
        let buf = 0x0000_0002_u32.to_le_bytes();
        let mut r = Reader::new(&buf);
        assert_eq!(read_value(&mut r), Err(DecodeError::BadValueType(2)));
    }

    #[test]
    fn a_string_without_its_nul_is_rejected() {
        let mut buf = Vec::new();
        write_u64(&mut buf, 3);
        buf.extend_from_slice(b"abc"); // no terminator
        let mut r = Reader::new(&buf);
        assert_eq!(r.string(), Err(DecodeError::BadString));
    }

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

        let records = read_buffer(&buf).unwrap();
        assert_eq!(records.len(), 10, "one command, ten records");

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
            }
        }
        assert_eq!(hex(&again), hex(&buf), "encode(decode(x)) must equal x");
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
