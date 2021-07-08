use criterion::Criterion;
use criterion::{criterion_group, criterion_main};
use ndspectral::cheb_dirichlet;
use ndspectral::integrate::diffusion::Diffusion2D;
use ndspectral::Integrate;
use ndspectral::{Field2, Space2};

const SIZES: [usize; 4] = [128, 264, 512, 1024];

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
    let mut group = c.benchmark_group("Diffusion2D");
    group.significance_level(0.1).sample_size(10);
    for n in SIZES.iter() {
        let cdx = cheb_dirichlet(*n);
        let cdy = cheb_dirichlet(*n);
        let space = Space2::new([cdx, cdy]);
        let field = Field2::new(space);
        let mut diff = Diffusion2D::new(field, 1.0, 0.1);
        diff.impulse();

        let name = format!("Size: {}", *n);
        group.bench_function(&name, |b| b.iter(|| integrate(&mut diff)));
    }
    group.finish();
}

criterion_group!(benches, bench_diffusion);
criterion_main!(benches);
