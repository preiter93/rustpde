use criterion::Criterion;
use criterion::{criterion_group, criterion_main};
use rustpde::*;

const SIZES: [usize; 4] = [128, 264, 512, 1024];

pub fn bench_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("Transform");
    group.significance_level(0.1).sample_size(10);
    for n in SIZES.iter() {
        let cdx = cheb_dirichlet::<f64>(*n);
        let cdy = cheb_dirichlet::<f64>(*n);
        let space = Space2::new(&cdx, &cdy);
        let mut field = Field2::new(&space);
        for (i, v) in field.v.iter_mut().enumerate() {
            *v = i as f64;
        }
        let name = format!("Size: {}", *n);
        group.bench_function(&name, |b| b.iter(|| field.forward()));
    }
    group.finish();
}

criterion_group!(benches, bench_transform);
criterion_main!(benches);
