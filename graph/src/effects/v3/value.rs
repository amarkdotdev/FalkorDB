//! The `SIValue` codec: a 4-byte type bitmask, then the payload.

use std::sync::Arc;

use thin_vec::ThinVec;

use crate::runtime::{
    ordermap::OrderMap,
    value::{Point, Value},
};

use super::{
    DecodeError, EffectDecode, EffectEncode, Reader, T_ARRAY, T_BOOL, T_DATE, T_DATETIME, T_DOUBLE,
    T_DURATION, T_INT64, T_MAP, T_NULL, T_POINT, T_STRING, T_TIME, T_VECTOR_F32, write_f64,
    write_i64, write_string, write_u8, write_u32,
};

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effects::hex;

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

    #[test]
    fn an_unknown_value_tag_is_rejected() {
        let buf = 0x0000_0002_u32.to_le_bytes();
        let mut r = Reader::new(&buf);
        assert_eq!(read_value(&mut r), Err(DecodeError::BadValueType(2)));
    }
}

/// The effects encoding of a `SIValue`.
///
/// `Value` already carries `Encode<19>` for the RDB path; this is the same type
/// on a different wire, and the two disagree in every way that matters — RDB
/// writes tags through a variable-width `write_unsigned`, effects write a fixed
/// 4 bytes. Keeping them as separate trait impls is what stops one being
/// mistaken for the other.
impl EffectEncode<3> for Value {
    fn encode(
        &self,
        buf: &mut Vec<u8>,
    ) {
        write_value(buf, self);
    }
}

impl EffectDecode<3> for Value {
    fn decode(r: &mut Reader<'_>) -> Result<Self, DecodeError> {
        read_value(r)
    }
}
