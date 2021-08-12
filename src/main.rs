//#![allow(unused_imports)]
// use ndarray::Array2;
// use ndarray::ArrayBase;
// use ndarray::Data;
// use ndarray::Dimension;
// use num_complex::Complex;
// use rustpde::cheb_dirichlet;
// use rustpde::chebyshev;
// use rustpde::fourier_c2c;
// use rustpde::fourier_r2c;
// use rustpde::solver::Solve;
// use std::f64::consts::PI;
//use rustpde::Field2;
//use rustpde::examples::diffusion::Diffusion2D;
//use rustpde::examples::solid_masks::solid_cylinder_inner;
use rustpde::examples::Navier2D;
//use rustpde::hdf5::write_to_hdf5;
use rustpde::integrate;
// use rustpde::solver::Hholtz;
// use rustpde::solver::Poisson;
// use rustpde::Field2;
use rustpde::Integrate;

#[allow(dead_code)]
fn main() {
    // Parameters
    let (nx, ny) = (64, 64);
    let ra = 1e4;
    let pr = 1.;
    let adiabatic = true;
    let aspect = 1.0;
    let dt = 0.01;
    //let mut navier = Navier2D::new(nx, ny, ra, pr, dt, adiabatic, aspect);
    let mut navier = Navier2D::new(nx, ny, ra, pr, dt, adiabatic, aspect);
    navier.callback();
    integrate(&mut navier, 2.0, Some(0.50));
    // // Set mask
    // let mut mask = solid_cylinder_inner(&navier.temp.x[0], &navier.temp.x[1], 0.5, 0.0, 0.2);
    // mask = mask + solid_cylinder_inner(&navier.temp.x[0], &navier.temp.x[1], 1.5, 0.0, 0.2);
    // mask = mask + solid_cylinder_inner(&navier.temp.x[0], &navier.temp.x[1], 2.5, 0.0, 0.2);
    // mask = mask + solid_cylinder_inner(&navier.temp.x[0], &navier.temp.x[1], 3.5, 0.0, 0.2);
    // mask = mask + solid_cylinder_inner(&navier.temp.x[0], &navier.temp.x[1], 4.5, 0.0, 0.2);
    // mask = mask + solid_cylinder_inner(&navier.temp.x[0], &navier.temp.x[1], 5.5, 0.0, 0.2);
    // navier.solid = Some(mask);
    //navier.read("data/flow200.000.h5");
    //navier.write();
    // write_to_hdf5("data/solid.h5", "v", None, &navier.solid.as_ref().unwrap()).unwrap();
    // write_to_hdf5("data/solid.h5", "x", None, &navier.temp.x[0]).unwrap();
    // write_to_hdf5("data/solid.h5", "y", None, &navier.temp.x[1]).unwrap();
    // Solve
    //integrate(&mut navier, 2.0, Some(0.50));
}

// let (nx, ny) = (26, 26);
// let mut navier_1 = Navier2D::new(nx, ny, ra, pr, dt, adiabatic, aspect);
// let shape_0 = navier.temp.shape();
// let shape_1 = navier_1.temp.shape();
// }
// pub fn main() {
//     println!("Hello World.");
// }
/*
pub fn main() {
    let n = 1024;
    let cdx = cheb_dirichlet::<f64>(n);
    let cdy = cheb_dirichlet::<f64>(n);
    let mut field = Field2::new(&[cdx, cdy]);
    let alpha = 1e-5;
    let hholtz: Hholtz<f64, 2_usize> = Hholtz::from_space(&field.space, [alpha, alpha]);
    let x = &field.x[0];
    let y = &field.x[1];

    // Analytical field and solution
    let n = std::f64::consts::PI / 2.;
    for (i, xi) in x.iter().enumerate() {
        for (j, yi) in y.iter().enumerate() {
            field.v[[i, j]] = (n * xi).cos() * (n * yi).cos();
        }
    }
    field.forward();

    for _ in 0..100 {
        hholtz.solve(&field.to_ortho(), &mut field.vhat, 0);
    }
}


pub fn main() {
    let n = 1024;
    let cdx = cheb_dirichlet(n);
    let cdy = cheb_dirichlet(n);
    let field = Field2::new(&[cdx, cdy]);
    let mut diff = Diffusion2D::new(field, 1.0, 0.1);
    diff.impulse();
    for _ in 0..100 {
        diff.update();
    }
}

pub fn main() {
    let n = 1024;
    let cdx = cheb_dirichlet(n);
    let cdy = cheb_dirichlet(n);
    let mut field = Field2::new(&[cdx, cdy]);
    for (i, v) in field.vhat.iter_mut().enumerate() {
        *v = i as f64;
    }

    for _ in 0..100 {
        field.to_ortho();
    }
}


*/
