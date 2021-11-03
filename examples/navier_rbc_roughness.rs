//! Simulate Rayleigh-Benard Convection two dimensional
//! in a bounded domain with roughness elements
//!
//! cargo run --release --example navier_rbc_roughness
use rustpde::integrate;
use rustpde::navier::solid_masks::solid_roughness_sinusoid;
use rustpde::navier::Navier2D;
// use rustpde::Integrate;

fn main() {
    // Parameters
    let (nx, ny) = (129, 129);
    let ra = 1e5;
    let pr = 1.;
    let adiabatic = true;
    let aspect = 1.0;
    let dt = 0.01;
    let mut navier = Navier2D::new(nx, ny, ra, pr, dt, aspect, adiabatic);
    // Add roughness elements
    let mask = solid_roughness_sinusoid(&navier.temp.x[0], &navier.temp.x[1], 0.1, 10.);
    navier.solid = Some(mask);
    //navier.read("restart.h5");
    //navier.reset_time();
    // Set initial conditions
    navier.set_velocity(0.2, 1., 1.);
    navier.set_temperature(0.2, 1., 1.);
    integrate(&mut navier, 10., Some(1.0));
}
