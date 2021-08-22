//! Collection of mask functions for solid-fluid interaction
//! # Example
//! Solve 2-D Rayleigh Benard Convection with cylindrical obstacle
//! ```ignore
//! use rustpde::integrate;
//! use rustpde::examples::Navier2D;
//! use rustpde::Integrate;
//! use rustpde::examples::solid_masks::solid_cylinder_inner;
//!
//! fn main() {
//!     // Parameters
//!     let (nx, ny) = (64, 64);
//!     let ra = 1e5;
//!     let pr = 1.;
//!     let adiabatic = true;
//!     let aspect = 1.0;
//!     let dt = 0.01;
//!     let mut navier = Navier2D::new(nx, ny, ra, pr, dt, aspect, adiabatic);
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
) -> [Array2<f64>; 2]{
    let mut mask = Array2::<f64>::zeros((x.len(), y.len()));
    let mut value = Array2::<f64>::zeros((x.len(), y.len()));
    let layer_thickness = radius / 10.;
    for (i, xi) in x.iter().enumerate() {
        for (j, yi) in y.iter().enumerate() {
            let r = ((x0 - xi).powf(2.0) + (y0 - yi).powf(2.0)).sqrt();
            if r < radius - layer_thickness {
                mask[[i, j]] = 1.0;
            } else if r < radius + layer_thickness {
                // Smoothin layer, see https://arxiv.org/pdf/1903.11914.pdf ( eq. 12 )
                mask[[i, j]] = 0.5 * (1. - (2. * (r - radius) / layer_thickness).tanh());
            }
        }
    }
    [mask, value]
}

/// Return mask for sinusoidal roughness elements
pub fn solid_roughness_sinusoid(
    x: &Array1<f64>,
    y: &Array1<f64>,
    height: f64,
    wavenumber: f64,
) -> [Array2<f64>; 2] {
    let mut mask = Array2::<f64>::zeros((x.len(), y.len()));
    let mut value = Array2::<f64>::zeros((x.len(), y.len()));
    let bottom = y[0];
    let top = y[y.len() - 1];
    let layer_thickness = height / 10.;
    for (i, xi) in x.iter().enumerate() {
        for (j, yi) in y.iter().enumerate() {
            let y_rough = height * (top - bottom) / 2. * ((wavenumber * xi).sin() + 0.5);
            let mut y_dist = yi - bottom;
            if y_dist <= y_rough {
                mask[[i, j]] = 1.0;
                value[[i, j]] = 0.5;
            } else if y_dist <= y_rough + layer_thickness {
                // Smoothin layer, see https://arxiv.org/pdf/1903.11914.pdf ( eq. 12 )
                mask[[i, j]] = 0.5 * (1. - (2. * (y_dist - y_rough) / layer_thickness).tanh());
                value[[i, j]] = 0.5;
            }
            y_dist = top - yi;
            if y_dist <= y_rough {
                mask[[i, j]] = 1.0;
                value[[i, j]] = -0.5;
            } else if y_dist <= y_rough + layer_thickness {
                // Smoothin layer, see https://arxiv.org/pdf/1903.11914.pdf ( eq. 12 )
                mask[[i, j]] = 0.5 * (1. - (2. * (y_dist - y_rough) / layer_thickness).tanh());
                value[[i, j]] = -0.5;
            }
        }
    }
    [mask, value]
}
