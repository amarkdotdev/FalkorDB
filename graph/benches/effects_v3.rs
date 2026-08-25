//! What the `IdList` dictionary costs in CPU, against the plain form.
//!
//! Size already favours the dictionary by 2x-8x uncompressed, and the encoder
//! picks by size. The open question this answers is the other half: whether
//! building the dictionary (a rank per row) and resolving it (a lookup per row)
//! costs more than the bytes it saves are worth.
//!
//! It matters where the work lands. Encoding runs on the write thread while it
//! holds the GIL, and decoding runs on the replica's main thread — neither is a
//! background cost.
//!
//! Both paths are measured on **identical input** via `write_id_list_forced`, so
//! the comparison is the encoding rather than the data. `endpoints(rows,
//! distinct)` models edge endpoints: `distinct` sources drawn pseudo-randomly,
//! which is the shape a supernode produces.
//!
//! ```bash
//! cargo bench -p graph --bench effects_v3
//! ```

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use graph::runtime::effects_v3::{
    ENC_COMPRESSED, ENC_PLAIN, Reader, read_id_list, read_id_set, write_id_list,
    write_id_list_forced, write_id_set,
};

/// `rows` endpoints drawn from `distinct` sources inside a graph of `nodes`
/// nodes.
///
/// The draw is a fixed multiplicative hash rather than a cycle: cycling makes
/// the rank array perfectly periodic, which flatters the dictionary in a way
/// real endpoint order does not. `nodes` matters because FalkorDB allocates ids
/// densely from zero, so graph size sets the largest id — and that is what
/// decides whether narrowing alone already captures the saving.
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

const ENCODINGS: [(&str, u8); 2] = [("raw", ENC_PLAIN), ("dictionary", ENC_COMPRESSED)];

/// The shapes the selection table shows a real winner for.
const SHAPES: [(u64, usize); 4] = [
    (10_000, 100),
    (10_000, 10_000),
    (5_000_000, 100),
    (5_000_000, 10_000),
];

fn encode(c: &mut Criterion) {
    let mut g = c.benchmark_group("id_list/encode");
    for (nodes, distinct) in SHAPES {
        let ids = endpoints(10_000, distinct, nodes);
        let label = format!("{nodes}nodes/{distinct}distinct");
        g.throughput(Throughput::Elements(10_000));
        for (name, enc) in ENCODINGS {
            g.bench_with_input(BenchmarkId::new(name, &label), &ids, |b, ids| {
                b.iter(|| {
                    let mut buf = Vec::new();
                    write_id_list_forced(&mut buf, black_box(ids), enc);
                    black_box(buf.len())
                });
            });
        }
    }
    g.finish();
}

fn decode(c: &mut Criterion) {
    let mut g = c.benchmark_group("id_list/decode");
    for (nodes, distinct) in SHAPES {
        let ids = endpoints(10_000, distinct, nodes);
        let label = format!("{nodes}nodes/{distinct}distinct");
        g.throughput(Throughput::Elements(10_000));
        for (name, enc) in ENCODINGS {
            let mut buf = Vec::new();
            write_id_list_forced(&mut buf, &ids, enc);
            g.bench_with_input(BenchmarkId::new(name, &label), &buf, |b, buf| {
                b.iter(|| {
                    let mut r = Reader::new(black_box(buf));
                    black_box(read_id_list(&mut r, 10_000).unwrap().len())
                });
            });
        }
    }
    g.finish();
}

/// `IdSet` on the shape `CREATE_NODE` actually produces: consecutive ids, which
/// roaring collapses to a single run. This is a different regime from `IdList`
/// — the saving is three orders of magnitude, not a percentage — so it is worth
/// pricing separately rather than assuming the IdList conclusion carries.
fn id_set(c: &mut Criterion) {
    let mut g = c.benchmark_group("id_set");
    let ids: Vec<u64> = (0..10_000).collect();
    g.throughput(Throughput::Elements(10_000));

    g.bench_function("encode/consecutive", |b| {
        b.iter(|| {
            let mut buf = Vec::new();
            write_id_set(&mut buf, black_box(&ids));
            black_box(buf.len())
        });
    });

    let mut buf = Vec::new();
    write_id_set(&mut buf, &ids);
    let bytes = buf.len();
    g.bench_function("decode/consecutive", |b| {
        b.iter(|| {
            let mut r = Reader::new(black_box(&buf));
            black_box(read_id_set(&mut r, 10_000).unwrap().len())
        });
    });
    println!("    [id_set: 10,000 consecutive ids -> {bytes} B, vs 80,001 plain]");
    g.finish();
}

/// What the shipped rule costs, as opposed to each encoding in isolation.
/// This is the number that matters: it includes the duplication probe, and the
/// dictionary build only where the probe lets it happen.
fn selection(c: &mut Criterion) {
    let mut g = c.benchmark_group("id_list/selection");
    for (nodes, distinct) in SHAPES {
        let ids = endpoints(10_000, distinct, nodes);
        g.throughput(Throughput::Elements(10_000));
        g.bench_with_input(
            BenchmarkId::from_parameter(format!("{nodes}nodes/{distinct}distinct")),
            &ids,
            |b, ids| {
                b.iter(|| {
                    let mut buf = Vec::new();
                    write_id_list(&mut buf, black_box(ids));
                    black_box(buf.len())
                });
            },
        );
    }
    g.finish();
}

criterion_group!(benches, encode, decode, id_set, selection);
criterion_main!(benches);
