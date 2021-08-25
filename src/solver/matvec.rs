//! Collection of (sparse) matrix/vector products
use super::{SolveReturn, SolverScalar};
use ndarray::Data;
use ndarray::Zip;
use ndarray::{prelude::*, DataMut};
use num_traits::Zero;
use std::ops::{Add, Div, Mul};

/// Collection of Matrix-Vector Product Solver
//#[enum_dispatch(SolveReturn<A, D>)]
#[derive(Clone)]
pub enum MatVec<T> {
    /// Ndarrays Matrix Vector Product
    MatVecDot(MatVecDot<T>),
    /// Banded Matrix Vector Product with offsets -2, 0, 2, 4
    MatVecFdma(MatVecFdma<T>),
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
            MatVec::MatVecFdma(ref t) => t.solve(input, axis),
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
            MatVec::MatVecFdma(ref t) => t.solve(input, axis),
        }
    }
}

/// Simple class to multiply n-dimensional vector
/// with a matrix along the first Axis.
///
/// Uses ndarrays 'dot' for matrix multiplication.
#[derive(Debug, Clone)]
pub struct MatVecDot<T> {
    mat: Array2<T>,
}

impl<T: SolverScalar + std::fmt::Debug> MatVecDot<T> {
    /// Return new `MatVecDot` (wrapper around ndarray)
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

/// Use if Matrix is banded with offets -2, 0, 2, 4
#[derive(Debug, Clone)]
pub struct MatVecFdma<T> {
    /// Number of matrix rows
    pub m: usize,
    /// Number of matrix columns
    pub n: usize,
    /// Lower diagonal (-2)
    pub low: Array1<T>,
    /// Main diagonal
    pub dia: Array1<T>,
    /// Upper diagonal (+2)
    pub up1: Array1<T>,
    /// Upper diagonal (+4)
    pub up2: Array1<T>,
}

impl<T: SolverScalar> MatVecFdma<T> {
    /// Initialize Fdma from matrix.
    pub fn new(a: &Array2<T>) -> Self {
        let m = a.shape()[0];
        let n = a.shape()[1];
        let mut low: Array1<T> = Array1::zeros(m);
        let mut dia: Array1<T> = Array1::zeros(m);
        let mut up1: Array1<T> = Array1::zeros(m);
        let mut up2: Array1<T> = Array1::zeros(m);
        for i in 0..m {
            dia[i] = a[[i, i]];
            if i > 1 {
                low[i] = a[[i, i - 2]];
            }
            if i < m - 2 {
                up1[i] = a[[i, i + 2]];
            }
            if i < m - 4 {
                up2[i] = a[[i, i + 4]];
            }
        }

        Self {
            m,
            n,
            low,
            dia,
            up1,
            up2,
        }
    }

    fn solve_lane<S1, S2, A>(&self, input: &ArrayBase<S1, Ix1>, output: &mut ArrayBase<S2, Ix1>)
    where
        S1: Data<Elem = A>,
        S2: Data<Elem = A> + DataMut,
        A: SolverScalar + Div<T, Output = A> + Mul<T, Output = A> + Add<T, Output = A>,
    {
        //self.fdma(input);
        let n = output.len();

        for i in 0..n {
            output[i] = input[i] * self.dia[i];
            if i > 1 {
                output[i] = output[i] + input[i - 2] * self.low[i];
            }
            if i < n - 2 {
                output[i] = output[i] + input[i + 2] * self.up1[i];
            }
            if i < n - 4 {
                output[i] = output[i] + input[i + 4] * self.up2[i];
            }
        }
    }
}

#[allow(unused_variables)]
impl<T, A> SolveReturn<A, Ix1> for MatVecFdma<T>
where
    T: SolverScalar,
    A: SolverScalar + Div<T, Output = A> + Mul<T, Output = A> + Add<T, Output = A> + From<T>,
{
    fn solve<S1>(&self, input: &ArrayBase<S1, Ix1>, axis: usize) -> Array<A, Ix1>
    where
        S1: Data<Elem = A>,
    {
        let mut output = Array1::zeros(self.m);
        self.solve_lane(input, &mut output);
        output
    }
}

impl<T, A> SolveReturn<A, Ix2> for MatVecFdma<T>
where
    T: SolverScalar,
    A: SolverScalar + Div<T, Output = A> + Mul<T, Output = A> + Add<T, Output = A> + From<T>,
{
    fn solve<S1>(&self, input: &ArrayBase<S1, Ix2>, axis: usize) -> Array<A, Ix2>
    where
        S1: Data<Elem = A>,
    {
        let (m, n) = if axis == 0 {
            (self.m, input.shape()[1])
        } else {
            (input.shape()[0], self.m)
        };
        let mut output = Array2::zeros((m, n));
        Zip::from(output.lanes_mut(Axis(axis)))
            .and(input.lanes(Axis(axis)))
            .par_for_each(|mut out, inp| self.solve_lane(&inp, &mut out));
        output

        // if axis == 0 {
        //     let (m, n) = (self.m, input.shape()[1]);
        //     let mut output = Array2::zeros((m, n));
        //     for i in 0..n {
        //         self.solve_lane(&input.slice(s![.., i]), &mut output.slice_mut(s![.., i]));
        //     }
        //     output
        // } else {
        //     let (n, m) = (self.m, input.shape()[0]);
        //     let mut output = Array2::zeros((m, n));
        //     for i in 0..m {
        //         self.solve_lane(&input.slice(s![i, ..]), &mut output.slice_mut(s![i, ..]));
        //     }
        //     output
        // }
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
        let mut data = Array::<f64, Dim<[Ix; 2]>>::zeros((nx + 2, nx + 2));
        let mut matrix = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, nx + 2));
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

    #[test]
    fn test_matvecfdma_dim2() {
        let nx = 6;
        let mut data = Array::<f64, Dim<[Ix; 2]>>::zeros((nx + 2, nx + 2));
        let mut matrix = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, nx + 2));
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

        let matvecfdma = MatVec::MatVecFdma(MatVecFdma::new(&matrix));
        let result = matvecfdma.solve(&data, 0);
        let expected = matrix.dot(&data);
        approx_eq(&result, &expected);

        let result = matvecfdma.solve(&data, 1);
        let expected = matrix.dot(&data.t());

        approx_eq(&result, &expected.t().to_owned());
    }
}
