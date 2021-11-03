use rustpde::integrate;
use rustpde::navier::statistics::Statistics;
use rustpde::navier::Navier2D;

fn main() {
    // Parameters
    let (nx, ny) = (65, 65);
    let ra = 1e4;
    let pr = 1.;
    let adiabatic = true;
    let aspect = 1.0;
    let dt = 0.01;
    let mut navier = Navier2D::new(nx, ny, ra, pr, dt, aspect, adiabatic);
    navier.statistics = Some(Statistics::new(&navier, 0.1, 10.0));
    // navier.read("restart.h5");
    // navier.reset_time();
    // // Set initial conditions
    // navier.set_velocity(0.2, 1., 1.);
    // navier.set_temperature(0.2, 1., 1.);
    navier.random_disturbance(1e-4);
    integrate(&mut navier, 100., Some(1.0));
}
