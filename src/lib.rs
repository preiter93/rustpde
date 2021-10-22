//! # `rustpde`: Spectral method solver for Navier-Stokes equations
//!<img align="right" src="https://rustacean.net/assets/cuddlyferris.png" width="80">
//!
//! # Dependencies
//! - cargo >= v1.49
//! - `hdf5` (sudo apt-get install -y libhdf5-dev)
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
//! - `FourierR2c` (Orthonormal), see [`fourier_r2c()`]
//!
//! Composite basis combine several basis functions of its parent space to
//! satisfy the needed boundary conditions, this is often called a Galerkin method.
//!
//! ## Implemented solver
//!
//! - `2-D Rayleigh Benard Convection: Direct numerical simulation`,
//! see [`navier::navier`]
//! - `2-D Rayleigh Benard Convection: Steady state solver`,
//! see [`navier::navier_adjoint`]
//!
//! # Example
//! Solve 2-D Rayleigh Benard Convection ( Run with `cargo run --release` )
//! ```ignore
//! use rustpde::integrate;
//! use rustpde::navier::Navier2D;
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
//!     let mut navier = Navier2D::new(nx, ny, ra, pr, dt, aspect, adiabatic);
//!     // Set initial conditions
//!     navier.set_velocity(0.2, 1., 1.);
//!     // // Want to restart?
//!     // navier.read("data/flow100.000.h5");
//!     // Write first field
//!     navier.callback();
//!     integrate(&mut navier, 100., Some(1.0));
//! }
//! ```
//! Solve 2-D Rayleigh Benard Convection with periodic sidewall
//! ```ignore
//! use rustpde::integrate;
//! use rustpde::navier::Navier2D;
//! use rustpde::Integrate;
//!
//! fn main() {
//!     // Parameters
//!     let (nx, ny) = (64, 64);
//!     let ra = 1e5;
//!     let pr = 1.;
//!     let aspect = 1.0;
//!     let dt = 0.02;
//!     let mut navier = Navier2D::new_periodic(nx, ny, ra, pr, dt, aspect);
//!     integrate(&mut navier, 100., Some(1.0));
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
//! ### Paraview
//!
//! The xmf files, corresponding to the h5 files can be created
//! by the script
//!
//! `./bin/create_xmf`.
//!
//! This script works only for fields from the `Navier2D`
//! solver with the attributes temp, ux, uy and pres.
//! The bin folder contains also the full crate `create_xmf`, which
//! can be adapted for specific usecases.
//!
//! ## Documentation
//!
//! Download and run:
//!
//! `cargo doc --open`
#![warn(missing_docs)]
#![warn(missing_doc_code_examples)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::module_name_repetitions)]
#[macro_use]
extern crate enum_dispatch;
pub mod bases;
pub mod navier;
pub mod field;
pub mod hdf5;
pub mod solver;
pub mod types;
pub use bases::{cheb_dirichlet, cheb_neumann, chebyshev, fourier_c2c, fourier_r2c};
pub use field::{BaseSpace, Field1, Field2, FieldBase, ReadField, Space1, Space2, WriteField};
pub use solver::{Solver, SolverField, SolverScalar};

/// Real type (not active)
//pub type Real = f64;

const MAX_TIMESTEP: usize = 10_000_000;

/// Integrate trait, step forward in time, and write results
pub trait Integrate {
    /// Update solution
    fn update(&mut self);
    /// Receive current time
    fn get_time(&self) -> f64;
    /// Get timestep
    fn get_dt(&self) -> f64;
    /// Callback function (can be used for i/o)
    fn callback(&mut self);
    /// Additional break criteria
    fn exit(&mut self) -> bool;
}

/// Integrade pde, that implements the Integrate trait.
///
/// Specify `save_intervall` to force writing an output.
///
/// Stop Criteria:
/// 1. Timestep limit
/// 2. Time limit
pub fn integrate<T: Integrate>(pde: &mut T, max_time: f64, save_intervall: Option<f64>) {
    let mut timestep: usize = 0;
    let eps_dt = pde.get_dt() * 1e-4;
    loop {
        // Update
        pde.update();
        timestep += 1;

        // Save
        if let Some(dt_save) = &save_intervall {
            if (pde.get_time() % dt_save) < pde.get_dt() / 2.
                || (pde.get_time() % dt_save) > dt_save - pde.get_dt() / 2.
            {
                //println!("Save at time: {:4.3}", pde.get_time());
                pde.callback();
            }
        }

        // Break
        if pde.get_time() + eps_dt >= max_time {
            println!("time limit reached: {:?}", pde.get_time());
            break;
        }
        if timestep >= MAX_TIMESTEP {
            println!("timestep limit reached: {:?}", timestep);
            break;
        }
        if pde.exit() {
            println!("break criteria triggered");
            break;
        }
    }
}
