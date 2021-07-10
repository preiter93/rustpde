#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]
use rustpde::integrate;
use rustpde::integrate::navier::Navier2D;

fn main() {
    let (nx, ny) = (129, 64);
    let ra = 1e5;
    let pr = 1.;
    let adiabatic = true;
    let aspect = 3.0;
    let dt = 0.01;
    let mut navier = Navier2D::new(nx, ny, ra, pr, dt, adiabatic, aspect);
    //navier.read("data/flow40.000.h5");
    integrate(navier, 100.0, Some(1.0));
}
