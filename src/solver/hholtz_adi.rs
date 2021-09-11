//! # Helmoltz Solver
//!  Solve equations of the form:
//!
//!  (I-c*D2) vhat = f
//!
//! where D2 is the second derivative.
//! Alternatively, if defined, multiply rhs
//! before the solve step, i.e.
//!
//!  (I-c*D2) vhat = A f
//!
//! For multidimensional equations, apply
//! alternating-direction implicit method (ADI)
//! to solve each dimension individually. But take
//! in mind that this method introduces a numerical
//! error, which is large if *c* is large.
//!
//! Chebyshev bases: The equation becomes
//! banded after multiplication with the pseudoinverse
//! of D2 (B2). In this case, the second equation is
//! solved, with A = B2.
use super::{MatVec, MatVecFdma, Solver, SolverScalar};
use crate::bases::BaseSpace;
use crate::field::FieldBase;
use crate::solver::{Fdma, Solve, SolveReturn};
use ndarray::prelude::*;
use ndarray::{Data, DataMut};
use std::ops::{Add, Div, Mul};

/// Container for `HholtzAdi`
#[derive(Clone)]
pub struct HholtzAdi<T, const N: usize>
// where
//     T: SolverScalar,
{
    solver: Vec<Solver<T>>,
    matvec: Vec<Option<MatVec<T>>>,
}

impl<const N: usize> HholtzAdi<f64, N> {
    /// Construct Helmholtz solver from field:
    ///
    ///  (I-c*D2) vhat = A f
    pub fn new<T2, S>(field: &FieldBase<f64, f64, T2, S, N>, c: [f64; N]) -> Self
    where
        S: BaseSpace<f64, N, Physical = f64, Spectral = T2>,
    {
        // Gather matrices and preconditioner
        let mut solver: Vec<Solver<f64>> = Vec::new();
        let mut matvec: Vec<Option<MatVec<f64>>> = Vec::new();
        for (axis, ci) in c.iter().enumerate() {
            // Matrices and preconditioner
            let (mat_a, mat_b, precond) = field.ingredients_for_hholtz(axis);
            let mat: Array2<f64> = mat_a - mat_b * *ci;
            let solver_axis = Solver::Fdma(Fdma::from_matrix(&mat));
            let matvec_axis = precond.map(|x| MatVec::MatVecFdma(MatVecFdma::new(&x)));

            solver.push(solver_axis);
            matvec.push(matvec_axis);
        }

        Self { solver, matvec }
    }
}

#[allow(unused_variables)]
impl<T, A> Solve<A, ndarray::Ix1> for HholtzAdi<T, 1>
where
    T: SolverScalar,
    A: SolverScalar + Div<T, Output = A> + Mul<T, Output = A> + Add<T, Output = A> + From<T>,
{
    /// # Example
    fn solve<S1, S2>(
        &self,
        input: &ArrayBase<S1, Ix1>,
        output: &mut ArrayBase<S2, Ix1>,
        axis: usize,
    ) where
        S1: Data<Elem = A>,
        S2: Data<Elem = A> + DataMut,
    {
        if let Some(matvec) = &self.matvec[0] {
            let buffer = matvec.solve(input, 0);
            self.solver[0].solve(&buffer, output, 0);
        } else {
            self.solver[0].solve(input, output, 0);
        }
    }
}

#[allow(unused_variables)]
impl<T, A> Solve<A, ndarray::Ix2> for HholtzAdi<T, 2>
where
    T: SolverScalar,
    A: SolverScalar + Div<T, Output = A> + Mul<T, Output = A> + Add<T, Output = A> + From<T>,
{
    /// # Example
    fn solve<S1, S2>(
        &self,
        input: &ArrayBase<S1, Ix2>,
        output: &mut ArrayBase<S2, Ix2>,
        axis: usize,
    ) where
        S1: Data<Elem = A>,
        S2: Data<Elem = A> + DataMut,
    {
        // Matvec
        let mut rhs = self.matvec[0]
            .as_ref()
            .map_or_else(|| input.to_owned(), |x| x.solve(input, 0));
        if let Some(x) = &self.matvec[1] {
            rhs = x.solve(&rhs, 1);
        }

        // // Matvec
        // let rhs = if let Some(x) = &self.matvec[0] {
        //     x.solve(&input, 0)
        // } else {
        //     input.to_owned()
        // };
        // let rhs = if let Some(x) = &self.matvec[1] {
        //     x.solve(&rhs, 1)
        // } else {
        //     rhs
        // };

        // Solve
        self.solver[0].solve(&rhs, output, 0);
        self.solver[1].solve(&output.to_owned(), output, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{Field1, Field2, Space1, Space2};
    use crate::{cheb_dirichlet, fourier_r2c};
    use ndarray::array;

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

    #[test]
    fn test_hholtz_adi() {
        let nx = 7;
        let space = Space1::new(&cheb_dirichlet(nx));
        let field = Field1::new(&space);
        let hholtz = HholtzAdi::new(&field, [1.0]);
        let b: Array1<f64> = array![1., 2., 3., 4., 5., 6., 7.];
        let mut x = Array1::<f64>::zeros(nx - 2);
        // Solve Hholtz
        hholtz.solve(&b, &mut x, 0);
        // Python's (pypde) solution
        let y = array![
            -0.08214845,
            -0.10466761,
            -0.06042153,
            0.04809052,
            0.04082296
        ];

        approx_eq(&x, &y);
    }

    #[test]
    fn test_hholtz2d_adi() {
        let nx = 7;

        let space = Space2::new(&cheb_dirichlet(nx), &cheb_dirichlet(nx));
        let field = Field2::new(&space);

        let hholtz = HholtzAdi::new(&field, [1.0, 1.0]);
        let b: Array2<f64> = array![
            [1., 2., 3., 4., 5., 6., 7.],
            [1., 2., 3., 4., 5., 6., 7.],
            [1., 2., 3., 4., 5., 6., 7.],
            [1., 2., 3., 4., 5., 6., 7.],
            [1., 2., 3., 4., 5., 6., 7.],
            [1., 2., 3., 4., 5., 6., 7.],
            [1., 2., 3., 4., 5., 6., 7.],
        ];
        let mut x = Array2::<f64>::zeros((nx - 2, nx - 2));
        // Solve Hholtz
        hholtz.solve(&b, &mut x, 0);

        // Python's (pypde) solution
        let y = array![
            [-7.083e-03, -9.025e-03, -5.210e-03, 4.146e-03, 3.520e-03],
            [5.809e-04, 7.402e-04, 4.273e-04, -3.401e-04, -2.887e-04],
            [1.699e-04, 2.165e-04, 1.250e-04, -9.951e-05, -8.447e-05],
            [-1.007e-03, -1.283e-03, -7.406e-04, 5.895e-04, 5.004e-04],
            [-6.775e-04, -8.632e-04, -4.983e-04, 3.966e-04, 3.366e-04],
        ];

        // Assert
        approx_eq(&x, &y);
    }

    #[test]
    fn test_hholtz2d_cd_cd_adi() {
        // Init
        let (nx, ny) = (16, 7);
        let space = Space2::new(&cheb_dirichlet(nx), &cheb_dirichlet(ny));
        let mut field = Field2::new(&space);
        let alpha = 1e-5;
        let hholtz = HholtzAdi::new(&field, [alpha, alpha]);
        let x = &field.x[0];
        let y = &field.x[1];

        // Analytical field and solution
        let n = std::f64::consts::PI / 2.;
        let mut expected = field.v.clone();
        for (i, xi) in x.iter().enumerate() {
            for (j, yi) in y.iter().enumerate() {
                field.v[[i, j]] = (n * xi).cos() * (n * yi).cos();
                expected[[i, j]] = 1. / (1. + alpha * n * n * 2.) * field.v[[i, j]];
            }
        }

        // Solve
        field.forward();
        hholtz.solve(&field.to_ortho(), &mut field.vhat, 0);
        //field.vhat.assign(&result);
        field.backward();

        // Compare
        approx_eq(&field.v, &expected);
    }

    #[test]
    fn test_hholtz2d_fo_cd_adi() {
        // Init
        let (nx, ny) = (16, 7);
        let space = Space2::new(&fourier_r2c(nx), &cheb_dirichlet(ny));
        let mut field = Field2::new(&space);
        let alpha = 1e-5;
        let hholtz = HholtzAdi::new(&field, [alpha, alpha]);
        let x = &field.x[0];
        let y = &field.x[1];

        // Analytical field and solution
        let n = std::f64::consts::PI / 2.;
        let mut expected = field.v.clone();
        for (i, xi) in x.iter().enumerate() {
            for (j, yi) in y.iter().enumerate() {
                field.v[[i, j]] = xi.cos() * (n * yi).cos();
                expected[[i, j]] = 1. / (1. + alpha * n * n + alpha) * field.v[[i, j]];
            }
        }

        // Solve
        field.forward();
        hholtz.solve(&field.to_ortho(), &mut field.vhat, 0);
        //field.vhat.assign(&result);
        field.backward();

        // Compare
        approx_eq(&field.v, &expected);
    }
}
