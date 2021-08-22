use rustpde::examples::Navier2D;
use rustpde::examples::Navier2DAdjoint;
use rustpde::integrate;
//use rustpde::ReadField;

fn main() {
    // Parameters
    let (nx, ny) = (64, 64);
    let ra = 2e4;
    let pr = 1.;
    let aspect = 1.0;
    let adiabatic = true;
    let dt = 0.01;
    let mut navier = Navier2D::new(nx, ny, ra, pr, dt, aspect, adiabatic);
    integrate(&mut navier, 100.0, Some(2.0));
    navier.write("restart.h5");

    // Steady state analysis
    let mut navier = Navier2DAdjoint::new(nx, ny, ra, pr, dt, aspect, adiabatic);
    navier.read("restart.h5");
    navier.reset_time();
    integrate(&mut navier, 20.0, Some(2.0));
}

// fn main() {
//     // Parameters
//     let (nx, ny) = (64, 64);
//     let ra = 2e4;
//     let pr = 1.;
//     let aspect = 2.0 / std::f64::consts::PI;
//     let dt = 0.01;
//     let mut navier = Navier2D::new_periodic(nx, ny, ra, pr, dt, aspect);
//     //let mut navier = Navier2DAdjoint::new_periodic(nx, ny, ra, pr, dt, aspect);
//     //navier.read("restart.h5");
//     navier.reset_time();
//     //navier.callback();
//     integrate(&mut navier, 100.0, Some(2.0));
// }
