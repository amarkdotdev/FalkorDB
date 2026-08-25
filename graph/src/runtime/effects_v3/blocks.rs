//! The five shared blocks every record composes from.

use roaring::RoaringTreemap;

use crate::runtime::value::Value;

use super::{
    DecodeError, ENC_COMPRESSED, ENC_PLAIN, Reader, read_value, write_label_id, write_u8,
    write_u16, write_u32, write_value,
};

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

#[cfg(test)]
mod tests {
    use super::super::hex;
    use super::*;

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
}
