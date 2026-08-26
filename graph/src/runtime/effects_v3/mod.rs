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

/// What a record is. C's `EffectType`, so 4 bytes on the wire, and
/// `EFFECT_UNKNOWN = 0` shifts every discriminant down by one relative to a
/// naive 1-based list.
///
/// An enum rather than constants because the set is closed and the decoder
/// matches it exhaustively: adding a record type stops compiling until every
/// match handles it, which is the failure mode worth having. The bit-flag sets
/// below stay constants for the opposite reason — several of those OR together,
/// which no enum can express.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u32)]
pub enum Opcode {
    UpdateNode = 1,
    UpdateEdge = 2,
    CreateNode = 3,
    CreateEdge = 4,
    DeleteNode = 5,
    DeleteEdge = 6,
    SetLabels = 7,
    RemoveLabels = 8,
    AddSchema = 9,
    AddAttribute = 10,
    CreateIndex = 11,
    DropIndex = 12,
    CreateConstraint = 13,
    DropConstraint = 14,
}

impl Opcode {
    /// The wire discriminant.
    #[must_use]
    pub const fn as_u32(self) -> u32 {
        self as u32
    }

    /// Whether the record carries a `count` and blocks sized by it.
    ///
    /// The schema and DDL records are inherently singular — one schema, one
    /// name, one index field — so they carry an opcode and no count.
    #[must_use]
    pub const fn is_batchable(self) -> bool {
        matches!(
            self,
            Self::UpdateNode
                | Self::UpdateEdge
                | Self::CreateNode
                | Self::CreateEdge
                | Self::DeleteNode
                | Self::DeleteEdge
                | Self::SetLabels
                | Self::RemoveLabels
        )
    }
}

impl TryFrom<u32> for Opcode {
    type Error = DecodeError;

    fn try_from(v: u32) -> Result<Self, Self::Error> {
        Ok(match v {
            1 => Self::UpdateNode,
            2 => Self::UpdateEdge,
            3 => Self::CreateNode,
            4 => Self::CreateEdge,
            5 => Self::DeleteNode,
            6 => Self::DeleteEdge,
            7 => Self::SetLabels,
            8 => Self::RemoveLabels,
            9 => Self::AddSchema,
            10 => Self::AddAttribute,
            11 => Self::CreateIndex,
            12 => Self::DropIndex,
            13 => Self::CreateConstraint,
            14 => Self::DropConstraint,
            other => return Err(DecodeError::BadOpcode(other)),
        })
    }
}

/// Which dictionary a schema record names: C's `SchemaType`.
///
/// **Zero-based**, unlike [`GraphEntity`] on the constraint records, which C
/// numbers from 1. The two are the same distinction with different wire
/// encodings, and getting them the same way round is not optional.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum SchemaType {
    Node = 0,
    Edge = 1,
}

impl SchemaType {
    #[must_use]
    pub const fn as_u32(self) -> u32 {
        self as u32
    }
}

impl TryFrom<u32> for SchemaType {
    type Error = DecodeError;

    fn try_from(v: u32) -> Result<Self, Self::Error> {
        match v {
            0 => Ok(Self::Node),
            1 => Ok(Self::Edge),
            other => Err(DecodeError::BadSchemaType(other)),
        }
    }
}

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

// ── constraint and entity types ──

/// C's `ConstraintType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum ConstraintKind {
    Unique = 0,
    Mandatory = 1,
}

impl ConstraintKind {
    #[must_use]
    pub const fn as_u32(self) -> u32 {
        self as u32
    }
}

impl TryFrom<u32> for ConstraintKind {
    type Error = DecodeError;

    fn try_from(v: u32) -> Result<Self, Self::Error> {
        match v {
            0 => Ok(Self::Unique),
            1 => Ok(Self::Mandatory),
            other => Err(DecodeError::BadConstraintType(other)),
        }
    }
}

/// C's `GraphEntityType`, **1-based**: `GETYPE_UNKNOWN` takes 0, so a node is 1
/// and an edge 2. A 0-based encoding here would silently make every node record
/// read as "unknown" to C.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum GraphEntity {
    Node = 1,
    Edge = 2,
}

impl GraphEntity {
    #[must_use]
    pub const fn as_u32(self) -> u32 {
        self as u32
    }

    /// Which `UPDATE_*` opcode targets this entity kind.
    #[must_use]
    pub const fn update_opcode(self) -> Opcode {
        match self {
            Self::Node => Opcode::UpdateNode,
            Self::Edge => Opcode::UpdateEdge,
        }
    }

    /// The schema dictionary this entity kind is named in.
    #[must_use]
    pub const fn schema_type(self) -> SchemaType {
        match self {
            Self::Node => SchemaType::Node,
            Self::Edge => SchemaType::Edge,
        }
    }
}

impl TryFrom<u32> for GraphEntity {
    type Error = DecodeError;

    fn try_from(v: u32) -> Result<Self, Self::Error> {
        match v {
            1 => Ok(Self::Node),
            2 => Ok(Self::Edge),
            other => Err(DecodeError::BadEntityType(other)),
        }
    }
}

// ── block encodings ──

/// How an `IdSet` or `IdList` stores its ids.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum BlockEncoding {
    /// Raw ids at the narrowest width that holds the largest of them:
    /// `u8 width` then `uint(width) x count`.
    ///
    /// A separate full-width encoding would be this one with `width == 8`, so
    /// there is no reason for both. Finding the maximum is one pass and no
    /// allocation.
    Plain = 0,
    /// `IdSet`: a run-optimized roaring64 bitmap. `IdList`: a roaring64
    /// dictionary plus one rank per row.
    Compressed = 1,
}

impl BlockEncoding {
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }
}

impl TryFrom<u8> for BlockEncoding {
    type Error = DecodeError;

    fn try_from(v: u8) -> Result<Self, Self::Error> {
        match v {
            0 => Ok(Self::Plain),
            1 => Ok(Self::Compressed),
            other => Err(DecodeError::BadEncoding(other)),
        }
    }
}

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

// ── the codec traits ──

/// Types that can write themselves into an effects payload of a given version.
///
/// Mirrors `serialization::Encode<VERSION>` on the RDB side, and for the same
/// reason: the version belongs in the type system rather than in a module name,
/// so a v4 lands as `impl EffectEncode<4> for Record` beside this one and the
/// compiler picks between them.
///
/// The unit is a whole [`Record`]. Blocks stay free functions because they are
/// sized by the record's `count`, which a `&self` encoder has no way to carry —
/// the same reason the RDB traits grow `encode_with_range` / `decode_with_count`
/// rather than folding counts into `encode`.
pub trait EffectEncode<const VERSION: u8> {
    fn encode(
        &self,
        buf: &mut Vec<u8>,
    );
}

/// Types that can read themselves out of an effects payload of a given version.
pub trait EffectDecode<const VERSION: u8>: Sized {
    /// # Errors
    ///
    /// Returns [`DecodeError`] if the bytes are malformed, truncated, or
    /// describe a shape this version does not have.
    fn decode(r: &mut Reader<'_>) -> Result<Self, DecodeError>;
}

// ── payload header ──

/// The payload is a zstd frame, not records.
///
/// When set, a `u32` uncompressed length follows the flags byte and the frame
/// follows that. The header itself is never compressed, so a reader always
/// knows what it is holding before it commits to decoding anything.
pub const FLAG_COMPRESSED: u8 = 1 << 0;

/// Every flag this build understands. A reader that meets a bit outside this
/// mask **rejects the buffer** rather than guessing: an old node reading a
/// future payload must fail loudly, not decode the first record it recognises
/// and corrupt itself with the rest.
pub const KNOWN_FLAGS: u8 = FLAG_COMPRESSED;

/// Start a fresh buffer: `u8 version · u8 flags`.
///
/// The flags byte is reserved from day one even though nothing sets it by
/// default. Adding it later would have meant another version bump for a single
/// byte, which is the whole reason it is here now rather than when compression
/// is first switched on.
#[must_use]
pub fn new_buffer() -> Vec<u8> {
    vec![EFFECTS_VERSION, 0]
}

/// Render bytes for the byte-pinning tests, which live beside what they pin.
#[cfg(test)]
pub(crate) fn hex(buf: &[u8]) -> String {
    buf.iter()
        .map(|b| format!("{b:02x}"))
        .collect::<Vec<_>>()
        .join(" ")
}
