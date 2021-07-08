#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]
use ndspectral::integrate;
use ndspectral::integrate::navier::Navier2D;

fn main() {
    let (nx, ny) = (64, 64);
    let ra = 1e5;
    let pr = 1.;
    let adiabatic = true;
    let aspect = 1.0;
    let dt = 0.01;
    let mut navier = Navier2D::new(nx, ny, ra, pr, dt, adiabatic, aspect);
    integrate(navier, 240.0, Some(1.0));
}
