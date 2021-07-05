#![warn(missing_docs)]
#![warn(missing_doc_code_examples)]
//! # ndspectral: *n*-dimensional transforms of various basis functions (spectral method)
pub mod bases;
pub use bases::{ChebDirichlet, ChebNeumann, Chebyshev};

/// Real type
pub type Real = f64;
