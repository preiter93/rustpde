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
use super::{MatVec, Solve, Solver, SolverScalar};
use crate::Base;

/// Container for Hholtz Solver
pub struct Hholtz<T, A, D, const N: usize>
where
    T: SolverScalar,
    A: SolverScalar
        + std::ops::Div<T, Output = A>
        + std::ops::Mul<T, Output = A>
        + std::ops::Add<T, Output = A>
        + From<T>,
    D: ndarray::Dimension + ndarray::RemoveAxis,
{
    solver: Vec<Solver<T, N>>,
    matvec: Vec<Option<MatVec<T, A, D>>>,
}

impl<T, A, D, const N: usize> Hholtz<T, A, D, N>
where
    T: SolverScalar + ndarray::ScalarOperand,
    A: SolverScalar
        + std::ops::Div<T, Output = A>
        + std::ops::Mul<T, Output = A>
        + std::ops::Add<T, Output = A>
        + From<T>,
    D: ndarray::Dimension + ndarray::RemoveAxis,
    f64: Into<T>,
{
    /// Returns the solver for the lhs, depending on the base
    fn solver_from_base(base: &Base, c: f64) -> Solver<T, N> {
        use crate::bases::LaplacianInverse;
        use crate::bases::Mass;
        let mass = base.mass::<T>();
        match base {
            Base::Chebyshev(ref b) => {
                let pinv = b.pinv::<T>();
                let eye = b.pinv_eye::<T>();
                let mat = eye.dot(&pinv).dot(&mass.slice(ndarray::s![.., 2..]))
                    - eye.dot(&mass.slice(ndarray::s![.., 2..])) * c.into();
                Solver::Fdma(crate::solver::Fdma::from_matrix(&mat))
            }
            Base::ChebDirichlet(ref b) => {
                let pinv = b.pinv();
                let eye = b.pinv_eye();
                let mat = eye.dot(&pinv).dot(&mass) - eye.dot(&mass) * c.into();
                Solver::Fdma(crate::solver::Fdma::from_matrix(&mat))
            }
            Base::ChebNeumann(ref b) => {
                let pinv = b.pinv();
                let eye = b.pinv_eye();
                let mat = eye.dot(&pinv).dot(&mass) - eye.dot(&mass) * c.into();
                Solver::Fdma(crate::solver::Fdma::from_matrix(&mat))
            } //Base::Fourier(_) => panic!("Not Implemented"),
        }
    }

    /// Returns the solver for the rhs, depending on the base
    #[allow(clippy::unnecessary_wraps)]
    fn matvec_from_base(base: &Base) -> Option<MatVec<T, A, D>> {
        use crate::bases::LaplacianInverse;
        use crate::solver::MatVecDot;
        let pinv = base.pinv();
        let mat = pinv.slice(ndarray::s![2.., ..]).to_owned();
        let matvec = MatVec::MatVecDot(MatVecDot::new(&mat));
        Some(matvec)
    }
}
