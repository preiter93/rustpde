use rustpde::cheb_dirichlet;
use rustpde::field::{Field2, Space2};
use rustpde::solver::{Hholtz, Solve};

fn main() {
    // Init
    let (nx, ny) = (64, 64);
    let space = Space2::new(&cheb_dirichlet(nx), &cheb_dirichlet(ny));
    let mut field = Field2::new(&space);
    let alpha = 1e-1;
    let hholtz = Hholtz::new2(&field, [1.0, 1.0], 1./alpha);
    let x = &field.x[0];
    let y = &field.x[1];

    // Analytical field and solution
    let n = std::f64::consts::PI / 2.;
    let mut expected = field.v.clone();
    for (i, xi) in x.iter().enumerate() {
        for (j, yi) in y.iter().enumerate() {
            field.v[[i, j]] = (n * xi).cos() * (n * yi).cos();
            expected[[i, j]] = alpha / (1. + alpha * n * n * 2.) * field.v[[i, j]];
        }
    }

    // Solve
    field.forward();
    hholtz.solve(&field.to_ortho(), &mut field.vhat, 0);
    field.backward();

    // Compare
    approx_eq(&field.v, &expected);
}

fn approx_eq<S, D>(result: &ndarray::ArrayBase<S, D>, expected: &ndarray::ArrayBase<S, D>)
where
    S: ndarray::Data<Elem = f64>,
    D: ndarray::Dimension,
{
    let dif = 1e-3;
    for (a, b) in expected.iter().zip(result.iter()) {
        if (a - b).abs() > dif {
            panic!("Large difference of values, got {} expected {}.", b, a)
        }
    }
}
