//! # `rustpde`: Spectral method solver for Navier-Stokes equations
//!
//! # Dependencies
//! - cargo >= v1.49
//! - `hd5` (sudo apt-get install -y libhdf5-dev)
//!
//! # Details
//!
//! This library is intended for simulation softwares which solve the
//! partial differential equations using spectral methods.
//!
//! Currently `rustpde` implements transforms from physical to spectral space
//! for the following basis functions:
//! - `Chebyshev` (Orthonormal), see [`chebyshev()`]
//! - `ChebDirichlet` (Composite), see [`cheb_dirichlet()`]
//! - `ChebNeumann` (Composite), see [`cheb_neumann()`]
//!
//! Composite basis combine several basis functions of its parent space to
//! satisfy the needed boundary conditions, this is often called a Galerkin method.
//!
//! ## Implemented solver
//!
//! - `2-D Rayleigh Benard Convection: Direct numerical simulation`,
//! see [`integrate::navier`]
//! - `2-D Rayleigh Benard Convection: Steady state solver`,
//! see [`integrate::navier_adjoint`]
//!
//! # Example
//! Solve 2-D Rayleigh Benard Convection ( Run with `cargo run --release` )
//! ```ignore
//! use rustpde::integrate;
//! use rustpde::integrate::Navier2D;
//! use rustpde::Integrate;
//!
//! fn main() {
//!     // Parameters
//!     let (nx, ny) = (64, 64);
//!     let ra = 1e5;
//!     let pr = 1.;
//!     let adiabatic = true;
//!     let aspect = 1.0;
//!     let dt = 0.02;
//!     let mut navier = Navier2D::new(nx, ny, ra, pr, dt, adiabatic, aspect);
//!     // Set initial conditions
//!     navier.set_velocity(0.2, 1., 1.);
//!     // // Want to restart?
//!     // navier.read("data/flow100.000.h5");
//!     // Write first field
//!     navier.write();
//!     integrate(navier, 100., Some(1.0));
//! }
//! ```
//!
//! ## Postprocess the output
//!
//! `rustpde` contains a `python` folder with some scripts.
//! If you have run the above example, and you specified
//! to save snapshots ( replace *None* with Some(1.) or any
//! other value), you will see `hdf5` in the `data` folder.
//!
//! You can create an animation with python's matplotlib by typing
//!
//! `python3 python/anim2d.py`
//!
//! Or just plot a single snapshot
//!
//! `python3 python/plot2d.py`
//!
//! Provided python has all librarys installed, you should now
//! see an animation.
//!
//! ## Documentation
//!
//! Download and run:
//!
//! `cargo doc --open`
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
