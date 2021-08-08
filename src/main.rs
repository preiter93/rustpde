#![allow(unused_imports)]
use ndarray::Array2;
use ndarray::ArrayBase;
use ndarray::Data;
use ndarray::Dimension;
use num_complex::Complex;
use rustpde::bases::BaseBasics;
use rustpde::cheb_dirichlet;
use rustpde::chebyshev;
use rustpde::fourier_c2c;
use rustpde::fourier_r2c;
use rustpde::solver::Solve;
use std::f64::consts::PI;
//use rustpde::Field2;
use rustpde::examples::solid_masks::solid_cylinder_inner;
use rustpde::examples::Navier2D;
use rustpde::examples::Navier2DPeriodic;
use rustpde::field::Field;
use rustpde::field::Field1Complex;
use rustpde::field::Field2Complex;
use rustpde::hdf5::write_to_hdf5;
use rustpde::integrate;
use rustpde::solver::Poisson;
use rustpde::Integrate;
// fn main() {
//     // Parameters
//     let (nx, ny) = (14, 14);
//     let ra = 1e3;
//     let pr = 1.;
//     let adiabatic = true;
//     let aspect = 1.0;
//     let dt = 0.01;
//     let mut navier = Navier2D::new(nx, ny, ra, pr, dt, adiabatic, aspect);
//     // Set initial conditions
//     navier.set_velocity(0.2, 1., 1.);
//     // // Want to restart?
//     navier.read("restart.h5");
//     // Write first field
//     //navier.write();
//     integrate(navier, 3., Some(1.0));
// }
// fn approx_eq<S, D>(result: &ArrayBase<S, D>, expected: &ArrayBase<S, D>)
// where
//     S: Data<Elem = f64>,
//     D: Dimension,
// {
//     let dif = 1e-3;
//     for (a, b) in expected.iter().zip(result.iter()) {
//         if (a - b).abs() > dif {
//             panic!("Large difference of values, got {} expected {}.", b, a)
//         }
//     }
// }

// #[allow(unused_variables)]
// fn fo() {
//     let nx = 14;
//     let bases = [fourier_r2c(nx)];
//     let mut field = Field1Complex::new(&bases.clone());
//     let x = bases[0].coords();
//     for (i, xi) in x.iter().enumerate() {
//         println!("{:?}", xi);
//         field.v[i] = (1. * xi).sin();
//     }
//     println!("{:?}", field.v);
//     field.forward();

//     // Solve poisson
//     let poisson = Poisson::from_space(&field.space, [1.0]);
//     let input = field.to_ortho();
//     poisson.solve(&input, &mut field.vhat, 0);
//     println!("{:?}", field.vhat);
//     field.vhat[0];
//     field.backward();
//     printl

// #[allow(unused_variables)]
// fn main() {
//     let (nx, ny) = (8, 6);
//     let bases = [fourier_r2c(nx), cheb_dirichlet(ny)];
//     let mut field = Field2Complex::new(&bases.clone());
//     //let poisson = Poisson::from_field(&field, [1.0, 1.0]);
//     let x = bases[0].coords();
//     let y = bases[1].coords();

//     let n = PI / 2.;
//     let mut expected = field.v.clone();
//     let ax = 1.;
//     let ay = PI / 2.;
//     for (i, xi) in x.iter().enumerate() {
//         for (j, yi) in y.iter().enumerate() {
//             field.v[[i, j]] = (ax * xi).sin() * (ay * yi).cos();
//             expected[[i, j]] = -1. / (ax * ax + ay * ay) * field.v[[i, j]];
//         }
//     }
//     //println!("{:?}", field.v);
//     println!("");
//     field.forward();
//     println!("{:?}", field.vhat);
//     println!("");
//     // Solve poisson
//     let poisson = Poisson::from_space(&field.space, [1.0, 1.0]);
//     //println!("{:?}", field.vhat);
//     let input = field.to_ortho();
//     //let mut result = Array2::<Complex<f64>>::zeros(field.vhat.raw_dim());
//     poisson.solve(&input, &mut field.vhat, 0);
//     field.backward();
//     println!("{:?}", field.v);
//     println!("{:?}", expected);
//     approx_eq(&field.v, &expected);

//     // println!("{:?}", field.v);
//     // println!("");
//     // //println!("{:?}", field.v.slice(ndarray::s![.., 1]));
//     // // field.forward();
//     // println!("");
//     // field.forward();
//     // field.backward();
//     // //println!("{:?}", field.v.slice(ndarray::s![.., 1]));
//     // println!("");
//     // println!("{:?}", field.v);
//     // let poisson = Poisson::from_space(&field.space, [1.0, 1.0]);
//     // //println!("{:?}", field.vhat);
//     // let input = field.to_ortho();
//     // let mut result = Array2::<Complex<f64>>::zeros(field.vhat.raw_dim());
//     // poisson.solve(&input, &mut result, 0);
//     // println!("{:?}", result);
//     // // println!("{:?}", field.vhat);
//     // let input = field.to_ortho();
//     // let mut result = Array2::<f64>::zeros(field.vhat.raw_dim());
//     // poisson.solve(&input, &mut result, 0);
//     // field.vhat.assign(&result);
//     // field.backward();
//     // println!("{:?}", field.v);
//     // println!("{:?}", expected);
//     // approx_eq(&field.v, &expected);
// }n!("{:?}", field.v);
// }

// #[allow(unused_variables, dead_code)]
// fn cdcd() {
//     let (nx, ny) = (8, 7);
//     let bases = [cheb_dirichlet(nx), cheb_dirichlet(ny)];
//     let mut field = Field2::new(Space2::new(bases.clone()));
//     let poisson = Poisson::from_field(&field, [1.0, 1.0]);
//     let x = bases[0].coords();
//     let y = bases[1].coords();

//     let n = PI/2.;
//     let mut expected = field.v.clone();
//     for (i,xi) in x.iter().enumerate() {
//         for (j,yi) in y.iter().enumerate() {
//             field.v[[i,j]] = (n*xi).cos() * (n*yi).cos();
//             expected[[i,j]] = -1./(n*n*2.)*field.v[[i,j]];
//         }
//     }
//     field.forward();
//     // println!("{:?}", field.v);
//     // println!("{:?}", field.vhat);
//     let input = field.to_parent();
//     let mut result = Array2::<f64>::zeros(field.vhat.raw_dim());
//     poisson.solve(&input, &mut result, 0);
//     field.vhat.assign(&result);
//     field.backward();
//     println!("{:?}", field.v);
//     println!("{:?}", expected);
//     approx_eq(&field.v, &expected);
// }

#[allow(dead_code)]
fn main() {
    // Parameters
    let (nx, ny) = (64, 32);
    let ra = 1e4;
    let pr = 1.;
    let adiabatic = true;
    let aspect = 1.0;
    let dt = 0.01;
    //let mut navier = Navier2D::new(nx, ny, ra, pr, dt, adiabatic, aspect);
    let mut navier = Navier2DPeriodic::new(nx, ny, ra, pr, dt, aspect);
    // Set mask
    //let mask = solid_cylinder_inner(&navier.temp.x[0], &navier.temp.x[1], 0.2, 0.0, 0.2);
    //navier.solid = Some(mask);
    //navier.read("restart.h5");
    navier.write();
    // write_to_hdf5("data/solid.h5", "v", None, &navier.solid.as_ref().unwrap()).unwrap();
    // write_to_hdf5("data/solid.h5", "x", None, &navier.temp.x[0]).unwrap();
    // write_to_hdf5("data/solid.h5", "y", None, &navier.temp.x[1]).unwrap();
    // Solve
    integrate(&mut navier, 100.0, Some(0.50));

    // let (nx, ny) = (26, 26);
    // let mut navier_1 = Navier2D::new(nx, ny, ra, pr, dt, adiabatic, aspect);
    // let shape_0 = navier.temp.shape();
    // let shape_1 = navier_1.temp.shape();
}
