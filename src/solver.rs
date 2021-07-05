//! # Collection of linear algebra Solver
pub mod fdma;
pub mod matvec;
pub mod tdma;
pub mod utils;
pub use fdma::Fdma;
use ndarray::{ArrayBase, Data, DataMut, Dimension, LinalgScalar, RemoveAxis};
pub use tdma::Tdma;
use utils::diag;

// /// Solve linear algebraix systems of the form: A x = b.
// /// Output (x) matches input (b) in type and size.
// //#[enum_dispatch]
// pub trait Solve<T> {
//     /// Solves A x = b, returns x
//     fn solve(&self, input: &T) -> T;
// }

/// Combination of linear algebra traits
pub trait SolverScalar: LinalgScalar + std::ops::SubAssign + std::ops::DivAssign {}
impl<T: LinalgScalar + std::ops::SubAssign + std::ops::DivAssign> SolverScalar for T {}

/// Solve linear algebraix systems of the form: A x = b.
#[enum_dispatch]
pub trait Solve<T> {
    /// Solves A x = b, returns x
    /// Output (x) matches input (b) in type and size.
    fn solve<S, D>(&self, input: &ArrayBase<S, D>, output: &mut ArrayBase<S, D>, axis: usize)
    where
        S: Data<Elem = T> + DataMut,
        D: Dimension + RemoveAxis;
}
