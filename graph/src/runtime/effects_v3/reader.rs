//! The bounds-checked cursor, and the fixed-width primitive writers.

use super::DecodeError;

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

    pub(super) fn take(
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
    /// `pub(super)` rather than public: the blocks need it, but nothing outside
    /// the codec should be inventing its own bounds checks.
    ///
    /// `min_bytes_each` is the smallest number of bytes one entry can occupy —
    /// for a variable-width entry that is its fixed prefix, which bounds the
    /// allocation without having to predict the payload.
    pub(super) fn guard_count(
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

#[cfg(test)]
mod tests {
    use super::super::{BlockEncoding, read_id_set, write_id_set};
    use super::*;

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
        let buf = [BlockEncoding::Plain.as_u8(), 8, 0, 0];
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
    fn a_string_without_its_nul_is_rejected() {
        let mut buf = Vec::new();
        write_u64(&mut buf, 3);
        buf.extend_from_slice(b"abc"); // no terminator
        let mut r = Reader::new(&buf);
        assert_eq!(r.string(), Err(DecodeError::BadString));
    }
}
