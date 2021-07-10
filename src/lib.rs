//! # rustpde: Spectral method solver for Navier-Stokes equations
//!
//! This library is intended for simulation softwares which solve the
//! partial differential equations using spectral methods.
//!
//! Currently rustpde implements transforms from physical to spectral space
//! for the following basis functions:
//! - Chebyshev (Orthonormal), see [`chebyshev()`]
//! - ChebDirichlet (Composite), see [`cheb_dirichlet()`]
//! - ChebNeumann (Composite), see [`cheb_neumann()`]
//!
//! Composite basis combine several basis functions of its parent space to
//! satisfy the needed boundary conditions, this is often called a Galerkin method.
//!
//! # Example
//! 2-D Rayleigh Benard Convection
//!```
//! use rustpde::integrate;
//! use rustpde::integrate::navier::Navier2D;
//! let (nx, ny) = (33, 33);
//! let ra = 1e5;
//! let pr = 1.;
//! let adiabatic = true;
//! let aspect = 1.0;
//! let dt = 0.01;
//! let mut navier = Navier2D::new(nx, ny, ra, pr, dt, adiabatic, aspect);
//! // You want to read from an existing file?
//! // navier.read("data/flow1.000.h5");
//! // To generate output every 1 TU use Some(1.) instead of None
//! integrate(navier, 2.0, None);
//!```
//!
//! ## Postprocess the output
//!
//! *rustpde* contains a python folder with examples scripts.
//! After the above example has been finished, and you specified
//! to save snapshot ( replace *None* with Some(1.) or any
//! other value), *hdf5* files are generated in a *data* folder.
//!
//! You can create an animation using python using
//!
//! *python3 python/anim.py*
//!
//! Provided python has all librarys installed, you should now
//! see an animation.
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
