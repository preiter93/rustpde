//! # Collection of linear algebra Solver
pub mod fdma;
pub mod fdma_tensor;
pub mod hholtz;
pub mod matvec;
pub mod tdma;
pub mod utils;
pub use fdma::Fdma;
pub use tdma::Tdma;
//pub use hholtz::Hholtz;
pub use fdma_tensor::FdmaTensor;
pub use matvec::{MatVec, MatVecDot};
use ndarray::{ArrayBase, Data, DataMut};
use utils::diag;

/// Combination of linear algebra traits
pub trait SolverScalar:
    ndarray::LinalgScalar
    + std::ops::SubAssign
    + std::ops::DivAssign
    + From<f64>
    + num_traits::Zero
    + num_traits::Zero
    + std::marker::Copy
    + std::ops::Div
    + std::ops::Sub
{
}
impl<T> SolverScalar for T where
    T: ndarray::LinalgScalar
        + std::ops::SubAssign
        + std::ops::DivAssign
        + From<f64>
        + num_traits::Zero
        + num_traits::One
        + std::marker::Copy
        + std::ops::Div
        + std::ops::Sub
{
}

/// Solve linear algebraix systems of the form: M x = b.
#[enum_dispatch]
pub trait Solve<A, D> {
    /// Solves M x = b, returns x, which is of type A
    /// Output (x) matches input (b) in type and size.
    fn solve<S1, S2>(&self, input: &ArrayBase<S1, D>, output: &mut ArrayBase<S2, D>, axis: usize)
    where
        A: SolverScalar,
        S1: Data<Elem = A>,
        S2: Data<Elem = A> + DataMut;
}

/// Collection of Linalg Solver
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum Solver<T, const N: usize> {
    /// Two-diagonal Solver
    Tdma(Tdma<T>),
    /// Four-diagonal Solver
    Fdma(Fdma<T>),
    //Hholtz(Hholtz<T, N>),
    //Poisson(Poisson<N>),
    /// Multidimensional four-diagonal Solver
    FdmaTensor(FdmaTensor<N>),
}
