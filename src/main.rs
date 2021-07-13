#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]
use rustpde::integrate;
use rustpde::integrate::navier::Navier2D;
use rustpde::Integrate;

fn main() {
    let (nx, ny) = (64, 64);
    let ra = 6e6;
    let pr = 1.;
    let adiabatic = true;
    let aspect = 1.0;
    let dt = 0.005;
    let mut navier = Navier2D::new(nx, ny, ra, pr, dt, adiabatic, aspect);
    navier.set_velocity(0.2, 1., 1.);
    navier.write();
    // navier.read("data/flow1500.000.h5");
    integrate(navier, 0.1, Some(0.1));
}
