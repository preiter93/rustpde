//! Tri-diagonal matrix solver
use super::diag;
use super::{Solve, SolverScalar};
use ndarray::prelude::*;
use ndarray::{Data, DataMut, RemoveAxis, Zip};
use std::fmt::Debug;
use std::ops::{Add, Div, Mul};

/// Solve tridiagonal system with diagonals-offsets: -2, 0, 2
#[derive(Debug, Clone)]
pub struct Tdma<T> {
    /// Size of matrix (= size of main diagonal)
    pub n: usize,
    /// Lower diagonal (-2)
    pub low: Array1<T>,
    /// Main diagonal
    pub dia: Array1<T>,
    /// Upper diagonal (+2)
    pub upp: Array1<T>,
}

impl<T: SolverScalar> Tdma<T> {
    /// Initialize Tdma from matrix.
    /// Extracts the diagonals
    pub fn from_matrix(a: &Array2<T>) -> Self {
        Tdma {
            n: a.shape()[0],
            low: diag(a, -2),
            dia: diag(a, 0),
            upp: diag(a, 2),
        }
    }

    fn solve_lane<A>(&self, input: &mut ArrayViewMut1<A>)
    where
        A: SolverScalar + Div<T, Output = A> + Mul<T, Output = A> + Add<T, Output = A>,
    {
        self.tdma(input);
    }

    /// Tridiagonal matrix solver
    ///     Ax = d
    /// where A is banded with diagonals in offsets -2, 0, 2
    ///
    /// a: sub-diagonal (-2)
    /// b: main-diagonal (0)
    /// c: sub-diagonal (+2)
    #[allow(clippy::many_single_char_names)]
    fn tdma<A>(&self, d: &mut ArrayViewMut1<A>)
    where
        A: SolverScalar + Div<T, Output = A> + Mul<T, Output = A> + Add<T, Output = A>,
    {
        let n = d.len();
        //let mut x = Array1::from(vec![T::zero(); n]);
        let a = self.low.view();
        let b = self.dia.view();
        let c = self.upp.view();
        let mut w = vec![T::zero(); n - 2];
        let mut g = vec![A::zero(); n];

        // Forward sweep
        w[0] = c[0] / b[0];
        g[0] = d[0] / b[0];
        if c.len() > 1 {
            w[1] = c[1] / b[1];
        }
        g[1] = d[1] / b[1];

        for i in 2..n - 2 {
            w[i] = c[i] / (b[i] - a[i - 2] * w[i - 2]);
        }
        for i in 2..n {
            g[i] = (d[i] - g[i - 2] * a[i - 2]) / (b[i] - a[i - 2] * w[i - 2]);
        }

        // Back substitution
        d[n - 1] = g[n - 1];
        d[n - 2] = g[n - 2];
        for i in (1..n - 1).rev() {
            d[i - 1] = g[i - 1] - d[i + 1] * w[i - 1];
        }
    }
}

impl<T, A, D> Solve<A, D> for Tdma<T>
where
    T: SolverScalar,
    A: SolverScalar + Div<T, Output = A> + Mul<T, Output = A> + Add<T, Output = A> + From<T>,
    D: Dimension + RemoveAxis,
{
    /// # Example
    ///```
    /// use rustpde::solver::Tdma;
    /// use rustpde::solver::Solve;
    /// use ndarray::prelude::*;
    /// let nx =  6;
    /// let mut data = Array1::<f64>::zeros(nx);
    /// let mut result = Array1::<f64>::zeros(nx);
    /// let mut matrix = Array2::<f64>::zeros((nx,nx));
    /// for (i, v) in data.iter_mut().enumerate() {
    ///     *v = i as f64;
    /// }
    /// for i in 0..nx {
    ///     let j = (i+1) as f64;
    ///     matrix[[i,i]] = 0.5*j;
    ///     if i>1 {
    ///         matrix[[i,i-2]] = 10.*j;
    ///     }
    ///     if i<nx-2 {
    ///         matrix[[i,i+2]] = 1.5*j;
    ///     }
    /// }
    /// let solver = Tdma::from_matrix(&matrix);
    /// solver.solve(&data, &mut result,0);
    /// let recover = matrix.dot(&result);
    /// for (a, b) in recover.iter().zip(data.iter()) {
    ///     if (a - b).abs() > 1e-4 {
    ///         panic!("Large difference of values, got {} expected {}.", b, a)
    ///     }
    /// }
    ///```
    fn solve<S1: Data<Elem = A>, S2: Data<Elem = A> + DataMut>(
        &self,
        input: &ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) {
        output.assign(input);
        Zip::from(output.lanes_mut(Axis(axis))).for_each(|mut out| {
            self.solve_lane(&mut out);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};
    use num_complex::Complex;

    fn approx_eq<S, D>(result: &ArrayBase<S, D>, expected: &ArrayBase<S, D>)
    where
        S: Data<Elem = f64>,
        D: Dimension,
    {
        let dif = 1e-3;
        for (a, b) in expected.iter().zip(result.iter()) {
            if (a - b).abs() > dif {
                panic!("Large difference of values, got {} expected {}.", b, a)
            }
        }
    }

    fn approx_eq_complex<S, D>(result: &ArrayBase<S, D>, expected: &ArrayBase<S, D>)
    where
        S: Data<Elem = Complex<f64>>,
        D: Dimension,
    {
        let dif = 1e-3;
        for (a, b) in expected.iter().zip(result.iter()) {
            if (a.re - b.re).abs() > dif || (a.im - b.im).abs() > dif {
                panic!("Large difference of values, got {} expected {}.", b, a)
            }
        }
    }

    #[test]
    fn test_tdma_dim1() {
        let nx = 6;
        let mut data = Array1::<f64>::zeros(nx);
        let mut result = Array1::<f64>::zeros(nx);
        let mut matrix = Array2::<f64>::zeros((nx, nx));
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
        }
        let solver = Tdma::from_matrix(&matrix);
        solver.solve(&data, &mut result, 0);
        let recover: Array1<f64> = matrix.dot(&result);
        approx_eq(&recover, &data);
    }

    #[test]
    fn test_tdma_dim1_complex() {
        let nx = 6;
        let mut data = Array1::<Complex<f64>>::zeros(nx);
        let mut result = Array1::<Complex<f64>>::zeros(nx);
        let mut matrix = Array2::<Complex<f64>>::zeros((nx, nx));
        for (i, v) in data.iter_mut().enumerate() {
            v.re = (i + 0) as f64;
            v.im = (i + 1) as f64;
        }
        for i in 0..nx {
            let j = (i + 1) as f64;
            matrix[[i, i]].re = 0.5 * j;
            matrix[[i, i]].im = 0.5 * j;
            if i > 1 {
                matrix[[i, i - 2]].re = 10. * j;
                matrix[[i, i - 2]].im = 10. * j;
            }
            if i < nx - 2 {
                matrix[[i, i + 2]].re = 1.5 * j;
                matrix[[i, i + 2]].im = 1.5 * j;
            }
        }
        let solver = Tdma::<Complex<f64>>::from_matrix(&matrix);
        solver.solve(&data, &mut result, 0);
        let recover: Array1<Complex<f64>> = matrix.dot(&result);
        approx_eq_complex(&recover, &data);
    }
}
