use rustpde::integrate;
use rustpde::integrate::Navier2D;
use rustpde::Integrate;

fn main() {
    // Parameters
    let (nx, ny) = (64, 64);
    let ra = 1e5;
    let pr = 1.;
    let adiabatic = false;
    let aspect = 1.0;
    let dt = 0.01;
    let mut navier = Navier2D::new(nx, ny, ra, pr, dt, adiabatic, aspect);
    // Set initial conditions
    navier.set_velocity(0.2, 1., 1.);
    // // Want to restart?
    navier.read("data/flow1500.000.h5");
    // Write first field
    //navier.write();
    integrate(navier, 2500., Some(1.0));
}
