//! # Collection of linear algebra Solver
pub mod fdma;
pub mod fdma_tensor;
pub mod matvec;
pub mod tdma;
pub mod utils;
pub use fdma::Fdma;
pub use fdma_tensor::FdmaTensor;
pub use matvec::{MatVec, MatVecDot};
use ndarray::{ArrayBase, Data, DataMut};
pub use tdma::Tdma;
use utils::diag;

/// Combination of linear algebra traits
pub trait SolverScalar: ndarray::LinalgScalar + std::ops::SubAssign + std::ops::DivAssign {}
impl<T: ndarray::LinalgScalar + std::ops::SubAssign + std::ops::DivAssign> SolverScalar for T {}

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
