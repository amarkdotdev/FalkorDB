//! v2 against v3, on the same staged writes, across five orders of magnitude.
//!
//! The query modelled throughout is the motivating one:
//!
//! ```cypher
//! UNWIND range(0, N-1) AS i CREATE (:Person {name: 'n' + i, age: i % 80})
//! ```
//!
//! One shape, so v3 emits a single record and v2 emits N. That is the case v3
//! was designed for, and the ratios below are its best case rather than a
//! typical one — a query whose shapes all differ lands at parity, which
//! `the_motivating_query_end_to_end` pins.
//!
//! **What is not here.** v2 has no standalone decoder: its bytes are read and
//! applied to a `Graph` in one pass, in the root crate, so there is nothing to
//! time against `read_buffer` from inside this crate. Decode below is v3 only.
//! End-to-end query time for both is measured against a live server instead,
//! which is the number that decides anything anyway.
//!
//! ```bash
//! cargo bench -p graph --bench effects_v2_vs_v3
//! ```

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use atomic_refcell::AtomicRefCell;
use graph::{
    graph::graph::Graph,
    runtime::{effects_v2, effects_v3, effects_v3_emit, pending::Pending, value::Value},
};

/// Node counts spanning the range worth caring about: a single write, a small
/// batch, and up to a bulk load.
const SCALES: [usize; 6] = [1, 100, 1_000, 10_000, 100_000, 1_000_000];

/// Criterion's default 100 samples is far too many once a single iteration
/// encodes a million records.
fn samples_for(n: usize) -> usize {
    match n {
        0..=10_000 => 50,
        0..=100_000 => 20,
        _ => 10,
    }
}

/// `GxB_init` is process-wide and may only run once.
fn ensure_graphblas() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    unsafe extern "C" {
        fn malloc(size: usize) -> *mut std::ffi::c_void;
        fn calloc(
            n: usize,
            size: usize,
        ) -> *mut std::ffi::c_void;
        fn realloc(
            p: *mut std::ffi::c_void,
            size: usize,
        ) -> *mut std::ffi::c_void;
        fn free(p: *mut std::ffi::c_void);
    }
    INIT.call_once(|| {
        graph::graph::graphblas::matrix::init(
            Some(malloc),
            Some(calloc),
            Some(realloc),
            Some(free),
        )
        .expect("GraphBLAS must initialize");
    });
}

fn fixture(n: usize) -> (AtomicRefCell<Graph>, Pending) {
    ensure_graphblas();
    let g = AtomicRefCell::new(Graph::new(64, 64, 0, 0, "bench"));
    {
        let mut graph = g.borrow_mut();
        graph.get_label_id_mut("Person");
        graph.add_node_attribute_name("name");
        graph.add_node_attribute_name("age");
    }
    let mut p = Pending::default();
    p.set_schema_baseline(&g);
    for id in 0..n as u64 {
        p.stage_created_node(
            id,
            &[0],
            &[
                (0, Value::String(std::sync::Arc::new(format!("n{id}")))),
                (1, Value::Int((id % 80) as i64)),
            ],
        );
    }
    (g, p)
}

/// Payload size is exact, so it is printed once rather than sampled.
fn report_sizes() {
    println!("\n  nodes        v2 bytes      v3 bytes   v3/v2   v3+zstd-1   zstd/v2");
    for n in SCALES {
        let (g, p) = fixture(n);
        let mut v2 = Vec::new();
        effects_v2::build_effects_buffer(&p, &g, &mut v2);
        let mut v3 = Vec::new();
        effects_v3_emit::build_effects_buffer_v3(&p, &g, &mut v3);
        let mut z = v3.clone();
        effects_v3::maybe_compress(&mut z, 1, 1);
        println!(
            "  {n:>9} {:>13} {:>13} {:>6.1}% {:>11} {:>8.1}%",
            v2.len(),
            v3.len(),
            100.0 * v3.len() as f64 / v2.len() as f64,
            z.len(),
            100.0 * z.len() as f64 / v2.len() as f64,
        );
    }
    println!();
}

fn encode(c: &mut Criterion) {
    report_sizes();
    let mut group = c.benchmark_group("encode");
    for n in SCALES {
        let (g, p) = fixture(n);
        group.sample_size(samples_for(n));
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("v2", n), &n, |b, _| {
            b.iter(|| {
                let mut buf = Vec::new();
                effects_v2::build_effects_buffer(black_box(&p), &g, &mut buf);
                black_box(buf.len())
            });
        });
        group.bench_with_input(BenchmarkId::new("v3", n), &n, |b, _| {
            b.iter(|| {
                let mut buf = Vec::new();
                effects_v3_emit::build_effects_buffer_v3(black_box(&p), &g, &mut buf);
                black_box(buf.len())
            });
        });
        // What compression adds on top of the v3 encode it follows.
        group.bench_with_input(BenchmarkId::new("v3+zstd-1", n), &n, |b, _| {
            b.iter(|| {
                let mut buf = Vec::new();
                effects_v3_emit::build_effects_buffer_v3(black_box(&p), &g, &mut buf);
                effects_v3::maybe_compress(&mut buf, 1, 1);
                black_box(buf.len())
            });
        });
    }
    group.finish();
}

fn decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode");
    for n in SCALES {
        let (g, p) = fixture(n);
        let mut plain = Vec::new();
        effects_v3_emit::build_effects_buffer_v3(&p, &g, &mut plain);
        let mut compressed = plain.clone();
        effects_v3::maybe_compress(&mut compressed, 1, 1);

        group.sample_size(samples_for(n));
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("v3", n), &n, |b, _| {
            b.iter(|| black_box(effects_v3::read_buffer(black_box(&plain)).unwrap().len()));
        });
        group.bench_with_input(BenchmarkId::new("v3+zstd-1", n), &n, |b, _| {
            b.iter(|| {
                black_box(
                    effects_v3::read_buffer(black_box(&compressed))
                        .unwrap()
                        .len(),
                )
            });
        });
    }
    group.finish();
}

criterion_group!(benches, encode, decode);
criterion_main!(benches);
