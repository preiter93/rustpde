//! Collection of (sparse) matrix/vector products
use super::{Solve, SolverScalar};
use ndarray::prelude::*;
use ndarray::{Data, DataMut};
use num_traits::Zero;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul};

/// Collection of Matrix-Vector Product Solver
#[enum_dispatch(Solve<A, D>)]
pub enum MatVec<T, A, D>
where
    T: SolverScalar,
    A: SolverScalar
        + std::ops::Div<T, Output = A>
        + std::ops::Mul<T, Output = A>
        + std::ops::Add<T, Output = A>
        + From<T>,
    D: ndarray::Dimension + ndarray::RemoveAxis,
{
    /// Ndarrays Matrix Vector Product
    MatVecDot(MatVecDot<T, A, D>),
}

/// Simple class to multiply n-dimensional vector
/// with a matrix along the first Axis.
///
/// Uses ndarrays 'dot' for matrix multiplication.
#[derive(Debug)]
pub struct MatVecDot<T, A, D> {
    mat: Array2<T>,
    marker: PhantomData<(A, D)>,
}

impl<T: SolverScalar, A, D> MatVecDot<T, A, D> {
    /// Return new MatVecDot (wrapper around ndarray)
    pub fn new(mat: &Array2<T>) -> Self {
        MatVecDot {
            mat: mat.to_owned(),
            marker: Default::default(),
        }
    }
}

impl<T, A, D> MatVecDot<T, A, D>
where
    T: SolverScalar,
    A: From<T> + Zero + Clone,
{
    /// convert mat to new type, to satisfy constraints on dot product
    fn mat_into(&self) -> Array2<A> {
        let mut mat_newtype: Array2<A> = Array2::zeros(self.mat.raw_dim());
        for (new, old) in mat_newtype.iter_mut().zip(self.mat.iter()) {
            *new = A::from(*old);
        }
        mat_newtype
    }
}

impl<T, A, D> Solve<A, D> for MatVecDot<T, A, D>
where
    T: SolverScalar,
    A: SolverScalar + Div<T, Output = A> + Mul<T, Output = A> + Add<T, Output = A> + From<T>,
    D: ndarray::Dimension + ndarray::RemoveAxis,
{
    /// Solve Matrix Vector product
    fn solve<S1: Data<Elem = A>, S2: Data<Elem = A> + DataMut>(
        &self,
        input: &ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) {
        // Convert to type A
        let mat_new = self.mat_into();

        ndarray::Zip::from(input.lanes(Axis(axis)))
            .and(output.lanes_mut(Axis(axis)))
            .for_each(|inp, mut out| {
                out.assign(&mat_new.dot(&inp));
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::Solve;
    use ndarray::{Array, Dim, Ix};

    fn approx_eq<S, D>(result: &ArrayBase<S, D>, expected: &ArrayBase<S, D>)
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

    #[test]
    fn test_matvecdot_dim1() {
        let nx = 6;
        let mut data = Array::<f64, Dim<[Ix; 1]>>::zeros(nx);
        let mut result = Array::<f64, Dim<[Ix; 1]>>::zeros(nx);
        let mut matrix = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, nx));
        for (i, v) in data.iter_mut().enumerate() {
            *v = i as f64;
        }
        for i in 0..nx {
            let j = (i + 1) as f64;
            matrix[[i, i]] = 0.5 * j;
            if i > 1 {
                matrix[[i, i - 2]] = 10. * j;
            }
            if i < nx - 2 {
                matrix[[i, i + 2]] = 1.5 * j;
            }
            if i < nx - 4 {
                matrix[[i, i + 4]] = 2.5 * j;
            }
        }

        let matvec = MatVec::MatVecDot(MatVecDot::new(&matrix));
        matvec.solve(&data, &mut result, 0);
        let expected = matrix.dot(&data);
        approx_eq(&result, &expected);
    }
}
