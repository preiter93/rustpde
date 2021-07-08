//! Collection of (sparse) matrix/vector products
use super::{SolveReturn, SolverScalar};
use ndarray::prelude::*;
use ndarray::Data;
use num_traits::Zero;
use std::ops::{Add, Div, Mul};

/// Collection of Matrix-Vector Product Solver
//#[enum_dispatch(SolveReturn<A, D>)]
pub enum MatVec<T> {
    /// Ndarrays Matrix Vector Product
    MatVecDot(MatVecDot<T>),
}

// Don't know how to use enum_dispatch with
// traits...
impl<T, A> SolveReturn<A, Ix1> for MatVec<T>
where
    T: SolverScalar,
    A: SolverScalar + Div<T, Output = A> + Mul<T, Output = A> + Add<T, Output = A> + From<T>,
{
    fn solve<S1>(&self, input: &ArrayBase<S1, Ix1>, axis: usize) -> Array<A, Ix1>
    where
        S1: Data<Elem = A>,
    {
        match self {
            MatVec::MatVecDot(ref t) => t.solve(input, axis),
        }
    }
}

impl<T, A> SolveReturn<A, Ix2> for MatVec<T>
where
    T: SolverScalar,
    A: SolverScalar + Div<T, Output = A> + Mul<T, Output = A> + Add<T, Output = A> + From<T>,
{
    fn solve<S1>(&self, input: &ArrayBase<S1, Ix2>, axis: usize) -> Array<A, Ix2>
    where
        S1: Data<Elem = A>,
    {
        match self {
            MatVec::MatVecDot(ref t) => t.solve(input, axis),
        }
    }
}

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

impl<T> MatVecDot<T>
where
    T: SolverScalar,
{
    /// convert mat to new type, to satisfy constraints on dot product
    fn mat_into<A>(&self) -> Array2<A>
    where
        A: From<T> + Zero + Clone,
    {
        let mut mat_newtype: Array2<A> = Array2::zeros(self.mat.raw_dim());
        for (new, old) in mat_newtype.iter_mut().zip(self.mat.iter()) {
            *new = A::from(*old);
        }
        mat_newtype
    }
}

#[allow(unused_variables)]
impl<T, A> SolveReturn<A, Ix1> for MatVecDot<T>
where
    T: SolverScalar,
    A: SolverScalar + Div<T, Output = A> + Mul<T, Output = A> + Add<T, Output = A> + From<T>,
{
    fn solve<S1>(&self, input: &ArrayBase<S1, Ix1>, axis: usize) -> Array<A, Ix1>
    where
        S1: Data<Elem = A>,
    {
        let mat_new = self.mat_into();
        mat_new.dot(input)
    }
}

#[allow(unused_variables)]
impl<T, A> SolveReturn<A, Ix2> for MatVecDot<T>
where
    T: SolverScalar,
    A: SolverScalar + Div<T, Output = A> + Mul<T, Output = A> + Add<T, Output = A> + From<T>,
{
    fn solve<S1>(&self, input: &ArrayBase<S1, Ix2>, axis: usize) -> Array<A, Ix2>
    where
        S1: Data<Elem = A>,
    {
        let mat_new = self.mat_into();
        if axis == 0 {
            mat_new.dot(input)
        } else {
            let rv: Array<A, Ix2> = mat_new.dot(&input.t());
            rv.t().to_owned()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        let result = matvec.solve(&data, 0);
        let expected = matrix.dot(&data);
        approx_eq(&result, &expected);
    }

    #[test]
    fn test_matvecdot_dim2() {
        let nx = 6;
        let mut data = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, nx));
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
        let result = matvec.solve(&data, 0);
        let expected = matrix.dot(&data);
        approx_eq(&result, &expected);

        let result = matvec.solve(&data, 1);
        let expected = matrix.dot(&data.t());
        approx_eq(&result, &expected.t().to_owned());
    }
}
