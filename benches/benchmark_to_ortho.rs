use criterion::Criterion;
use criterion::{criterion_group, criterion_main};
use rustpde::*;

const SIZES: [usize; 3] = [128, 264, 512];

pub fn bench_to_ortho(c: &mut Criterion) {
    let mut group = c.benchmark_group("ToOrtho");
    group.significance_level(0.1).sample_size(10);
    for n in SIZES.iter() {
        let cdx = cheb_dirichlet::<f64>(*n);
        let cdy = cheb_dirichlet::<f64>(*n);
        let space = Space2::new(&cdx, &cdy);
        let mut field = Field2::new(&space);
        for (i, v) in field.vhat.iter_mut().enumerate() {
            *v = i as f64;
        }
        let name = format!("Size: {}", *n);

        group.bench_function(&name, |b| b.iter(|| field.to_ortho()));
    }
    group.finish();
}

pub fn bench_from_ortho(c: &mut Criterion) {
    let mut group = c.benchmark_group("FromOrtho");
    group.significance_level(0.1).sample_size(10);
    for n in SIZES.iter() {
        let cdx = cheb_dirichlet::<f64>(*n);
        let cdy = cheb_dirichlet::<f64>(*n);
        let space = Space2::new(&cdx, &cdy);
        let mut field = Field2::new(&space);
        for (i, v) in field.v.iter_mut().enumerate() {
            *v = i as f64;
        }
        let array = field.v.to_owned();
        let name = format!("Size: {}", *n);

        group.bench_function(&name, |b| b.iter(|| field.from_ortho(&array)));
    }
    group.finish();
}

criterion_group!(benches, bench_to_ortho, bench_from_ortho);
criterion_main!(benches);
