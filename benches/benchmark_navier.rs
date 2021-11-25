use criterion::Criterion;
use criterion::{criterion_group, criterion_main};
use rustpde::navier::navier::Navier2D;
use rustpde::Integrate;

const EVEN: [usize; 3] = [128, 264, 512];
const ODD: [usize; 3] = [129, 265, 513];

/// Iterate pde
pub fn integrate<T: Integrate>(pde: &mut T) {
    let mut timestep: usize = 0;
    loop {
        // Update
        pde.update();
        timestep += 1;

        if timestep >= 10 {
            break;
        }
    }
}

pub fn bench_diffusion(c: &mut Criterion) {
    let mut group = c.benchmark_group("Navier2D");
    group.significance_level(0.1).sample_size(10);
    let ra = 1e5;
    let pr = 1.;
    let adiabatic = true;
    let aspect = 1.0;
    let dt = 0.01;
    for (n1, n2) in EVEN.iter().zip(ODD.iter()) {
        let mut navier = Navier2D::new(*n1, *n1, ra, pr, dt, aspect, adiabatic);
        let name = format!("Size: {}", *n1);
        group.bench_function(&name, |b| b.iter(|| navier.update()));
        let mut navier = Navier2D::new(*n2, *n2, ra, pr, dt, aspect, adiabatic);
        let name = format!("Size: {}", *n2);
        group.bench_function(&name, |b| b.iter(|| navier.update()));
    }
    group.finish();
}

criterion_group!(benches, bench_diffusion);
criterion_main!(benches);
