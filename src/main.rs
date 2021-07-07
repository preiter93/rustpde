#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]
use ndarray::prelude::*;
use ndarray::{Data, Ix, Ix1};
use ndspectral::bases::{Base, ChebDirichlet, ChebNeumann, Chebyshev};
use ndspectral::bases::{Differentiate, Mass, Size, Transform};
use ndspectral::*;
// use ndspectral::solver::utils::eig;
// use ndspectral::solver::Fdma;
// use ndspectral::solver::Solve;
// use ndspectral::solver::{MatVec, MatVecDot};
// use ndspectral::Real;
// use ndspectral::{Base, Transform};

fn main() {
    println!("Hello, world!");
    let (nx, ny) = (6, 8);
    let cdx = cheb_dirichlet(nx);
    let cdy = cheb_dirichlet(ny);
    let space = Space::new([cdx, cdy]);
    let mut field = Field2::new(space);
    // // let mut cheby = Chebyshev::new(ny);
    // let mut data = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, ny));
    // let mut vhat = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, ny));
    for (i, v) in field.v.iter_mut().enumerate() {
        *v = i as f64;
    }
    println!("{:?}", field.v);
    println!("{:?}", field.vhat);
    field.forward();
    println!("{:?}", field.vhat);
    field.backward();
    println!("{:?}", field.v);
    // // cheby.differentiate(&data, &mut vhat, 2, 1);
    // // println!("{:?}", vhat);
    //
    // let mut cd = ChebNeumann::new(ny + 2);
    // //let mut cd = Base::ChebNeumann(cd);
    // let mut vhat = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, ny + 2));
    // for (i, v) in data.iter_mut().enumerate() {
    //     *v = i as f64;
    // }
    // //cd.differentiate(&data, &mut vhat, 2, 1);
    // cd.backward(&mut data, &mut vhat, 1);
    // cd.forward(&mut vhat, &mut data, 1);
    // println!("{:?}", data);
    // println!("{:?}", cd.mass::<f64>());
    // println!("{:?}", cd.len_phys());

    //
    // // let stencil = StencilChebyshev::dirichlet(5);
    // // // dot
    // // let mut composite = Array::from_vec(vec![2., 0.70710678, 1.]);
    // // let mut parent = Array1::zeros(5);
    // // stencil.dot(&composite, &mut parent, 0);
    // // //let expected: Array1<f64> = Array::from_vec(vec![2., 0.7071, -1., -0.7071, -1.]);
    // // // solve
    // // //let expected = composite.clone();
    // // stencil.solve(&parent, &mut composite, 0);
    //
    // // let cd = ChebDirichlet::new(6);
    // // println!("{:?}", cd.coords());

    // let nx = 6;
    // let mut data = Array::<f64, Dim<[Ix; 1]>>::zeros(nx);
    // let mut result = Array::<f64, Dim<[Ix; 1]>>::zeros(nx);
    // let mut matrix = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, nx));
    // for (i, v) in data.iter_mut().enumerate() {
    //     *v = i as f64;
    // }
    // for i in 0..nx {
    //     let j = (i + 1) as f64;
    //     matrix[[i, i]] = 0.5 * j;
    //     if i > 1 {
    //         matrix[[i, i - 2]] = 10. * j;
    //     }
    //     if i < nx - 2 {
    //         matrix[[i, i + 2]] = 1.5 * j;
    //     }
    //     if i < nx - 4 {
    //         matrix[[i, i + 4]] = 2.5 * j;
    //     }
    // }
    //
    // let (lam, q, qi) = eig(&matrix);

    //let nal = ndarray_to_nalgebra(matrix);

    // let matvec = MatVec::MatVecDot(MatVecDot::new(&matrix));
    // matvec.solve(&data, &mut result, 0);
    // println!("{:?}", result);
    // println!("{:?}", matrix.dot(&data));

    // println!("{:?}", matrix);
    // let solver = Fdma::from_matrix(&matrix);
    // solver.solve(&data, &mut result, 0);
    // let recover = matrix.dot(&result);
    // println!("{:?}", result);
    // println!("{:?}", recover);
    // println!("{:?}", data);
}
