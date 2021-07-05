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
pub mod bases;
pub use bases::Transform;
pub use bases::{ChebDirichlet, ChebNeumann, Chebyshev};
use ndrustfft::Complex as Cmplx;

/// Real type
pub type Real = f64;
/// Complex type
pub type Complex = Cmplx<f64>;
