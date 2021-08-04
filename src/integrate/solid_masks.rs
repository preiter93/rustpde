//! Collection of mask functions for solid-fluid interaction
//! # Example
//! Solve 2-D Rayleigh Benard Convection with cylindrical obstacle
//! ```ignore
//! use rustpde::integrate;
//! use rustpde::integrate::Navier2D;
//! use rustpde::Integrate;
//! use rustpde::integrate::solid_masks::solid_cylinder_inner;
//!
//! fn main() {
//!     // Parameters
//!     let (nx, ny) = (64, 64);
//!     let ra = 1e5;
//!     let pr = 1.;
//!     let adiabatic = true;
//!     let aspect = 1.0;
//!     let dt = 0.01;
//!     let mut navier = Navier2D::new(nx, ny, ra, pr, dt, adiabatic, aspect);
//!     // Set initial conditions
//!     navier.set_velocity(0.2, 1., 1.);
//!     // Set mask
//!     let mask = solid_cylinder_inner(&navier.temp.x[0], &navier.temp.x[1], 0.2, 0.0, 0.2);
//!     navier.solid = Some(mask);
//!     // Write first field
//!     navier.write();
//!     // Solve
//!     integrate(&mut navier, 100., Some(1.0));
//! }
//! ```
use ndarray::{Array1, Array2};

/// Return mask for solid cylinder (everything with r < radius is solid)
pub fn solid_cylinder_inner(
    x: &Array1<f64>,
    y: &Array1<f64>,
    x0: f64,
    y0: f64,
    radius: f64,
) -> Array2<f64> {
    let mut mask = Array2::<f64>::zeros((x.len(), y.len()));
    let layer_thickness = radius/10.;
    for (i, xi) in x.iter().enumerate() {
        for (j, yi) in y.iter().enumerate() {
            let r = ((x0 - xi).powf(2.0) + (y0 - yi).powf(2.0)).sqrt();
            if r < radius - layer_thickness {
                mask[[i, j]] = 1.0;
            } else if r < radius + layer_thickness {
                // Smoothin layer, see https://arxiv.org/pdf/1903.11914.pdf ( eq. 12 )
                mask[[i, j]] = 0.5*(1. - (2.*(r-radius)/layer_thickness).tanh());
            }
        }
    }
    mask
}

/// Return mask for solid cylinder (everything with r > radius is solid)
pub fn solid_cylinder_outer(
    x: &Array1<f64>,
    y: &Array1<f64>,
    x0: f64,
    y0: f64,
    radius: f64,
) -> Array2<f64> {
    1. - solid_cylinder_inner(x, y, x0, y0, radius)
}
