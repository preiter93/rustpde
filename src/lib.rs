//! # ndspectral: *n*-dimensional transforms of various basis functions
//!
//! This library is intended for simulation softwares which solve the
//! partial differential equations using spectral methods.
//!
//! Currently ndspectral implements transforms from physical to spectral space
//! for the following basis functions:
//! - Chebyshev (Orthonormal)
//! - ChebDirichlet (Composite)
//! - ChebNeumann (Composite)
//!
//! Composite basis combine several basis functions of its parent space to
//! satisfy the needed boundary conditions, this is often called a Galerkin method.
#![warn(missing_docs)]
#![warn(missing_doc_code_examples)]
#![allow(clippy::unnecessary_cast)]
#[macro_use]
extern crate enum_dispatch;
pub mod bases;
pub mod field;
pub mod hdf5;
pub mod integrate;
pub mod solver;
pub mod space;
pub use bases::{cheb_dirichlet, cheb_neumann, chebyshev};
pub use bases::{Base, Differentiate, Transform};
pub use field::{Field, Field1, Field2};
pub use integrate::{integrate, Integrate};
pub use solver::{Solver, SolverField, SolverScalar};
pub use space::{Space, Space1, Space2, Spaced};

/// Real type
pub type Real = f64;
// Complex type
// pub type Complex = Cmplx<f64>;
