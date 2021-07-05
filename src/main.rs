use ndarray::prelude::*;
use ndarray::{Data, Ix, Ix1};
use ndspectral::bases::{ChebDirichlet, ChebNeumann, Chebyshev};
use ndspectral::Real;

fn main() {
    println!("Hello, world!");
    let (nx, ny) = (6, 4);
    let mut cheby = Chebyshev::new(ny);
    let mut data = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, ny));
    let mut vhat = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, ny));
    for (i, v) in data.iter_mut().enumerate() {
        *v = i as f64;
    }
    cheby.differentiate(&data, &mut vhat, 2, 1);
    println!("{:?}", vhat);

    let mut cd = ChebNeumann::new(ny + 2);
    let mut vhat = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, ny + 2));
    for (i, v) in data.iter_mut().enumerate() {
        *v = i as f64;
    }
    //cd.differentiate(&data, &mut vhat, 2, 1);
    cd.backward(&mut data, &mut vhat, 1);
    cd.forward(&mut vhat, &mut data, 1);
    println!("{:?}", data);

    // let stencil = StencilChebyshev::dirichlet(5);
    // // dot
    // let mut composite = Array::from_vec(vec![2., 0.70710678, 1.]);
    // let mut parent = Array1::zeros(5);
    // stencil.dot(&composite, &mut parent, 0);
    // //let expected: Array1<f64> = Array::from_vec(vec![2., 0.7071, -1., -0.7071, -1.]);
    // // solve
    // //let expected = composite.clone();
    // stencil.solve(&parent, &mut composite, 0);

    // let cd = ChebDirichlet::new(6);
    // println!("{:?}", cd.coords());
}
