//! Collection of (sparse) matrix/vector products
use super::{Solve, SolverScalar};
use ndarray::prelude::*;
use ndarray::{Data, DataMut, RemoveAxis, Zip};

/// Collection of Matrix-Vector Product Solver
//#[derive(Debug)]
#[enum_dispatch(Solver)]
pub enum MatVec<T> {
    /// Ndarrays Matrix Vector Product
    MatVecDot(MatVecDot<T>),
}

// // Don't know how to use enum_dispatch with
// // traits...
// impl Solve<Array1<f64>> for MatVec {
//     fn solve(&self, input: &Array1<f64>) -> Array1<f64> {
//         match self {
//             MatVec::MatVecDot(ref t) => t.solve(input),
//         }
//     }
// }
//
// impl Solve<Array2<f64>> for MatVec {
//     fn solve(&self, input: &Array2<f64>) -> Array2<f64> {
//         match self {
//             MatVec::MatVecDot(ref t) => t.solve(input),
//         }
//     }
// }

/// Simple class to multiply n-dimensional vector
/// with a matrix along the first Axis.
///
/// Uses ndarrays 'dot' for matrix multiplication.
#[derive(Debug)]
pub struct MatVecDot<T> {
    mat: Array2<T>,
}

impl<T: SolverScalar> MatVecDot<T> {
    /// Return new MatVecDot (wrapper around ndarray)
    pub fn new(mat: &Array2<T>) -> Self {
        MatVecDot {
            mat: mat.to_owned(),
        }
    }
}

impl<T: SolverScalar> Solve<T> for MatVecDot<T> {
    /// Solve Matrix Vector product
    fn solve<S, D>(&self, input: &ArrayBase<S, D>, output: &mut ArrayBase<S, D>, axis: usize)
    where
        S: Data<Elem = T> + DataMut,
        D: Dimension + RemoveAxis,
    {
        Zip::from(input.lanes(Axis(axis)))
            .and(output.lanes_mut(Axis(axis)))
            .for_each(|inp, mut out| {
                out.assign(&self.mat.dot(&inp));
            });
    }
}
