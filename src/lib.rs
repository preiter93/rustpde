#![warn(missing_docs)]
#![warn(missing_doc_code_examples)]
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
pub mod bases;
pub use bases::{ChebDirichlet, ChebNeumann, Chebyshev};

/// Real type
pub type Real = f64;
