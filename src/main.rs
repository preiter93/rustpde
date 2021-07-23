use rustpde::integrate;
use rustpde::integrate::Navier2D;
use rustpde::Integrate;

// fn main() {
//     // Parameters
//     let (nx, ny) = (14, 14);
//     let ra = 1e3;
//     let pr = 1.;
//     let adiabatic = true;
//     let aspect = 1.0;
//     let dt = 0.01;
//     let mut navier = Navier2D::new(nx, ny, ra, pr, dt, adiabatic, aspect);
//     // Set initial conditions
//     navier.set_velocity(0.2, 1., 1.);
//     // // Want to restart?
//     navier.read("restart.h5");
//     // Write first field
//     //navier.write();
//     integrate(navier, 3., Some(1.0));
// }

fn main() {
    // Parameters
    let (nx, ny) = (10, 10);
    let ra = 1e03;
    let pr = 1.;
    let adiabatic = true;
    let aspect = 1.0;
    let dt = 0.01;
    let mut navier = Navier2D::new(nx, ny, ra, pr, dt, adiabatic, aspect);
    //navier.read("restart.h5");
    navier.write();
    integrate(&mut navier, 50., Some(1.0));

    // let (nx, ny) = (26, 26);
    // let mut navier_1 = Navier2D::new(nx, ny, ra, pr, dt, adiabatic, aspect);
    // let shape_0 = navier.temp.shape();
    // let shape_1 = navier_1.temp.shape();
}
