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
//! the alternating-direction implicit method (ADI)
//! to solve each dimension individually.
//!
//! Chebyshev bases: The equation becomes
//! banded after multiplication with the pseudoinverse
//! of D2 (B2). In this case, the second equation is
//! solved, with A = B2.
use super::{MatVec, Solver, SolverScalar};
use crate::field::Field;
use crate::solver::{Solve, SolveReturn};
use crate::space::Spaced;
use crate::Base;
use ndarray::prelude::*;
use ndarray::{Data, DataMut};

/// Container for Hholtz
#[derive(Clone)]
pub struct Hholtz<T, const N: usize>
// where
//     T: SolverScalar,
{
    solver: Vec<Solver<T>>,
    matvec: Vec<Option<MatVec<T>>>,
}

impl<T, const N: usize> Hholtz<T, N>
where
    T: SolverScalar + ndarray::ScalarOperand,
    f64: Into<T>,
{
    /// Construct Helmholtz solver from field
    pub fn from_field<S>(field: &Field<S, T, N>, c: [f64; N]) -> Self
    where
        S: Spaced<T, N>,
    {
        Self::from_space(&field.space, c)
    }

    /// Construct Helmholtz solver from space
    pub fn from_space<S>(space: &S, c: [f64; N]) -> Self
    where
        S: Spaced<T, N>,
    {
        let solver: Vec<Solver<T>> = space
            .get_bases()
            .iter()
            .enumerate()
            .map(|(i, base)| Self::solver_from_base(base, c[i]))
            .collect();

        let matvec: Vec<Option<MatVec<T>>> = space
            .get_bases()
            .iter()
            .map(|base| Self::matvec_from_base(base))
            .collect();

        Hholtz { solver, matvec }
    }

    /// Returns the solver for the lhs, depending on the base
    fn solver_from_base(base: &Base<f64>, c: f64) -> Solver<T> {
        use crate::bases::LaplacianInverse;
        use crate::bases::Mass;
        let mass = base.mass();
        match base {
            Base::Chebyshev(ref b) => {
                let pinv = b.laplace_inv();
                let eye = b.laplace_inv_eye();
                let mat = eye.dot(&pinv).dot(&mass.slice(ndarray::s![.., 2..]))
                    - eye.dot(&mass.slice(ndarray::s![.., 2..])) * c;
                let mat = mat.mapv(std::convert::Into::into);
                Solver::Fdma(crate::solver::Fdma::from_matrix(&mat))
            }
            Base::CompositeChebyshev(ref b) => {
                let pinv = b.laplace_inv();
                let eye = b.laplace_inv_eye();
                let mat = eye.dot(&pinv).dot(&mass) - eye.dot(&mass) * c;
                let mat = mat.mapv(std::convert::Into::into);
                Solver::Fdma(crate::solver::Fdma::from_matrix(&mat))
            } //_ => todo!(),
        }
    }

    /// Returns the solver for the rhs, depending on the base
    #[allow(clippy::unnecessary_wraps)]
    fn matvec_from_base(base: &Base<f64>) -> Option<MatVec<T>> {
        use crate::bases::LaplacianInverse;
        use crate::solver::MatVecDot;
        let pinv = base.laplace_inv();
        let mat = pinv.slice(ndarray::s![2.., ..]).to_owned();
        let mat = mat.mapv(std::convert::Into::into);
        let matvec = MatVec::MatVecDot(MatVecDot::new(&mat));
        Some(matvec)
    }
}

#[allow(unused_variables)]
impl<T, A> Solve<A, ndarray::Ix1> for Hholtz<T, 1>
where
    T: SolverScalar,
    A: SolverScalar
        + std::ops::Div<T, Output = A>
        + std::ops::Mul<T, Output = A>
        + std::ops::Add<T, Output = A>
        + From<T>,
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
            let buffer = matvec.solve(&input, 0);
            self.solver[0].solve(&buffer, output, 0);
        } else {
            self.solver[0].solve(input, output, 0);
        }
    }
}

#[allow(unused_variables)]
impl<T, A> Solve<A, ndarray::Ix2> for Hholtz<T, 2>
where
    T: SolverScalar,
    A: SolverScalar
        + std::ops::Div<T, Output = A>
        + std::ops::Mul<T, Output = A>
        + std::ops::Add<T, Output = A>
        + From<T>,
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
            .map_or_else(|| input.to_owned(), |x| x.solve(&input, 0));
        if let Some(x) = &self.matvec[1] {
            rhs = x.solve(&rhs, 1);
        }

        // Solve
        self.solver[0].solve(&rhs, output, 0);
        self.solver[1].solve(&output.to_owned(), output, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cheb_dirichlet;
    //use crate::cheb_neumann;
    use crate::{Field1, Space1};
    use crate::{Field2, Space2};
    use ndarray::array;
    //use std::f64::consts::PI;

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
    fn test_hholtz() {
        let nx = 7;
        let bases = [cheb_dirichlet(nx)];
        let field = Field1::new(Space1::new(bases));
        let hholtz = Hholtz::from_field(&field, [1.0]);
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
    fn test_hholtz2d() {
        let nx = 7;
        let bases = [cheb_dirichlet(nx), cheb_dirichlet(nx)];
        let field = Field2::new(Space2::new(bases));
        let hholtz = Hholtz::from_field(&field, [1.0, 1.0]);
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
}
