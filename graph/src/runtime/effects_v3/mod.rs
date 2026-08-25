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

use crate::graph::graphblas::serialization::{index_field_type, si_type};

/// Buffer header. C accepts any version `<= EFFECTS_VERSION` and branches per
/// version, so raising this is its established mechanism rather than a break —
/// but C must be raised to 3 as well before it can read what we write.
pub const EFFECTS_VERSION: u8 = 3;

mod blocks;
mod error;
mod reader;
mod records;
mod value;

pub use blocks::*;
pub use error::DecodeError;
pub use reader::*;
pub use records::*;
pub use value::*;

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

// ── index field types ──
//
// Also derived from the RDB module's copy of C's `index_field.h`. This is a
// **bit flag set**, not an ordinal — several can be OR'd — which is why
// `INDEX_FLD_RANGE` is the union of the three scalar kinds and reads as `0x0E`
// rather than having a discriminant of its own. A bit set is the one shape an
// enum genuinely cannot model, so these stay constants.

macro_rules! index_flags {
    ($($name:ident),* $(,)?) => {
        $(
            pub const $name: u32 = index_field_type::$name as u32;
        )*
    };
}

index_flags!(
    INDEX_FLD_FULLTEXT,
    INDEX_FLD_NUMERIC,
    INDEX_FLD_GEO,
    INDEX_FLD_STR,
    INDEX_FLD_VECTOR,
);

pub const INDEX_FLD_UNKNOWN: u32 = 0x00;
/// `INDEX_FLD_NUMERIC | INDEX_FLD_GEO | INDEX_FLD_STR` = `0x0E`.
pub const INDEX_FLD_RANGE: u32 = INDEX_FLD_NUMERIC | INDEX_FLD_GEO | INDEX_FLD_STR;

// ── constraint types ──

pub const CT_UNIQUE: u32 = 0;
pub const CT_MANDATORY: u32 = 1;

// ── graph entity types ──
//
// **1-based**: `GETYPE_UNKNOWN = 0` comes first, so a node is 1 and an edge 2.
// A 0-based encoding here would silently make every node record read as
// "unknown" on the C side.

pub const GETYPE_UNKNOWN: u32 = 0;
pub const GETYPE_NODE: u32 = 1;
pub const GETYPE_EDGE: u32 = 2;

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
// Derived from `serialization::si_type` rather than restated: the RDB encoder
// already carries C's tags, and two copies of a bitmask would drift silently —
// nothing fails when a tag is wrong, the far side just reads the next field as
// a type. They are `u64` there because RDB writes them through a
// variable-width `write_unsigned`; the effects wire is a fixed 4 bytes, so
// these narrow.
//
// C's `SIType` is a **bitmask**, not an ordinal: each type is a distinct bit.
// Rust's own v2 codec used sequential 0..12 tags, which collide with these
// almost everywhere and are the single most dangerous divergence in the format.

macro_rules! si_tags {
    ($($name:ident),* $(,)?) => {
        $(
            pub const $name: u32 = {
                assert!(si_type::$name <= u32::MAX as u64, "SIType must fit the 4 bytes C reads");
                si_type::$name as u32
            };
        )*
    };
}

si_tags!(
    T_ARRAY,
    T_DATETIME,
    T_DATE,
    T_TIME,
    T_DURATION,
    T_STRING,
    T_BOOL,
    T_INT64,
    T_DOUBLE,
    T_NULL,
    T_POINT,
    T_VECTOR_F32,
);

/// `T_MAP` has no RDB counterpart — the RDB path never stores a bare map — so
/// unlike the rest it is stated here.
pub const T_MAP: u32 = 1 << 0;

/// Start a fresh buffer, stamped with the version header.
#[must_use]
pub fn new_buffer() -> Vec<u8> {
    vec![EFFECTS_VERSION]
}

/// Render bytes for the byte-pinning tests, which live beside what they pin.
#[cfg(test)]
pub(crate) fn hex(buf: &[u8]) -> String {
    buf.iter()
        .map(|b| format!("{b:02x}"))
        .collect::<Vec<_>>()
        .join(" ")
}
