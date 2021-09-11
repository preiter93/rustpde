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
//! For multidimensional equations, an eigendecomposition
//! is applied to to (n-1) dimensions. See [`crate::solver::FdmaTensor`]
//!
//! Chebyshev bases: The equation becomes
//! banded after multiplication with the pseudoinverse
//! of D2 (B2). In this case, the second equation is
//! solved, with A = B2.
use super::{MatVec, MatVecFdma, SolverScalar};
use crate::bases::BaseSpace;
use crate::field::FieldBase;
use crate::solver::utils::vec_to_array;
use crate::solver::{FdmaTensor, Solve, SolveReturn};
use ndarray::prelude::*;
use std::ops::{Add, Div, Mul};

/// Container for Hholtz
#[derive(Clone)]
pub struct Hholtz<T, const N: usize>
// where
//     T: SolverScalar,
{
    // solver: Vec<Solver<T>>,
    solver: Box<FdmaTensor<T, N>>,
    matvec: Vec<Option<MatVec<T>>>,
}

impl<const N: usize> Hholtz<f64, N> {
    // /// Construct Helmholtz solver from field:
    // ///
    // ///  (I-c*D2) vhat = A f
    // pub fn new<T2, S>(field: &FieldBase<f64, f64, T2, S, N>, c: [f64; N]) -> Self
    // where
    //     S: BaseSpace<f64, N, Physical = f64, Spectral = T2>,
    // {
    //     // Gather matrices and preconditioner
    //     let mut solver: Vec<Solver<f64>> = Vec::new();
    //     let mut matvec: Vec<Option<MatVec<f64>>> = Vec::new();
    //     for (axis, ci) in c.iter().enumerate() {
    //         // Matrices and preconditioner
    //         let (mat_a, mat_b, precond) = field.ingredients_for_hholtz(axis);
    //         let mat: Array2<f64> = mat_a - mat_b * *ci;
    //         let solver_axis = Solver::Fdma(Fdma::from_matrix(&mat));
    //         let matvec_axis = precond.map(|x| MatVec::MatVecFdma(MatVecFdma::new(&x)));
    //
    //         solver.push(solver_axis);
    //         matvec.push(matvec_axis);
    //     }
    //
    //     Self { solver, matvec }
    // }

    /// Construct Helmholtz solver from field:
    ///
    ///  (I-c*D2) vhat = A f
    pub fn new<T2, S>(field: &FieldBase<f64, f64, T2, S, N>, c: [f64; N]) -> Self
    where
        S: BaseSpace<f64, N, Physical = f64, Spectral = T2>,
    {
        // Gather matrices and preconditioner
        let mut laplacians: Vec<Array2<f64>> = Vec::new();
        let mut masses: Vec<Array2<f64>> = Vec::new();
        let mut is_diags: Vec<bool> = Vec::new();
        let mut matvec: Vec<Option<MatVec<f64>>> = Vec::new();
        for (axis, ci) in c.iter().enumerate() {
            // Matrices and preconditioner
            let (mat_a, mat_b, precond, is_diag) = field.ingredients_for_poisson(axis);
            let mass = mat_a;
            let laplacian = -1.0 * mat_b * *ci;
            let matvec_axis = precond.map(|x| MatVec::MatVecFdma(MatVecFdma::new(&x)));

            laplacians.push(laplacian);
            masses.push(mass);
            matvec.push(matvec_axis);
            is_diags.push(is_diag);
        }

        // Vectors -> Arrays
        let laplacians = vec_to_array::<&Array2<f64>, N>(laplacians.iter().collect());
        let masses = vec_to_array::<&Array2<f64>, N>(masses.iter().collect());
        let is_diag = vec_to_array::<&bool, N>(is_diags.iter().collect());

        // Solver
        let solver = FdmaTensor::from_matrix(laplacians, masses, is_diag, 1.0);

        Self {
            solver: Box::new(solver),
            matvec,
        }
    }
}

#[allow(unused_variables)]
impl<A> Solve<A, ndarray::Ix1> for Hholtz<f64, 1>
where
    A: SolverScalar
        + Div<f64, Output = A>
        + Mul<f64, Output = A>
        + Add<f64, Output = A>
        + From<f64>,
{
    /// # Example
    fn solve<S1, S2>(
        &self,
        input: &ArrayBase<S1, Ix1>,
        output: &mut ArrayBase<S2, Ix1>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = A>,
        S2: ndarray::Data<Elem = A> + ndarray::DataMut,
    {
        if let Some(matvec) = &self.matvec[0] {
            let buffer = matvec.solve(input, 0);
            self.solver.solve(&buffer, output, 0);
        } else {
            self.solver.solve(input, output, 0);
        }
    }
}

#[allow(unused_variables)]
impl<A> Solve<A, ndarray::Ix2> for Hholtz<f64, 2>
where
    A: SolverScalar
        + Div<f64, Output = A>
        + Mul<f64, Output = A>
        + Add<f64, Output = A>
        + From<f64>,
{
    /// # Example
    fn solve<S1, S2>(
        &self,
        input: &ArrayBase<S1, Ix2>,
        output: &mut ArrayBase<S2, Ix2>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = A>,
        S2: ndarray::Data<Elem = A> + ndarray::DataMut,
    {
        // Matvec
        let mut rhs = self.matvec[0]
            .as_ref()
            .map_or_else(|| input.to_owned(), |x| x.solve(input, 0));
        if let Some(x) = &self.matvec[1] {
            rhs = x.solve(&rhs, 1);
        };
        // Solve fdma-tensor
        self.solver.solve(&rhs, output, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{Field2, Space2};
    use crate::{cheb_dirichlet, fourier_r2c};

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
    fn test_hholtz2d_cd_cd() {
        // Init
        let (nx, ny) = (64, 64);
        let space = Space2::new(&cheb_dirichlet(nx), &cheb_dirichlet(ny));
        let mut field = Field2::new(&space);
        let alpha = 1e-0;
        let hholtz = Hholtz::new(&field, [alpha, alpha]);
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
    fn test_hholtz2d_fo_cd() {
        // Init
        let (nx, ny) = (16, 7);
        let space = Space2::new(&fourier_r2c(nx), &cheb_dirichlet(ny));
        let mut field = Field2::new(&space);
        let alpha = 1e-5;
        let hholtz = Hholtz::new(&field, [alpha, alpha]);
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
