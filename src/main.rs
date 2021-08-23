use rustpde::examples::Navier2D;
use rustpde::examples::Navier2DAdjoint;
use rustpde::integrate;
use rustpde::Integrate;
//use rustpde::ReadField;
use rustpde::examples::solid_masks::solid_roughness_sinusoid;

fn main() {
    // Parameters
    let (nx, ny) = (512, 257);
    let ra = 3e8;
    let pr = 1.;
    let aspect = 1.0;
    let dt = 0.0002;
    let mut navier = Navier2D::new_periodic(nx, ny, ra, pr, dt, aspect);
    // Add roughness elements
    let mask = solid_roughness_sinusoid(&navier.temp.x[0], &navier.temp.x[1], 0.1, 10.);
    navier.solid = Some(mask);
    // Write first flow field;
    navier.callback();
    // navier.read("data/flow00078.00.h5");
    // Set how often flow fields are written
    navier.write_intervall = Some(0.5);
    // Integrate over 200 TU, save info every 0.1 TU
    integrate(&mut navier, 200.0, Some(0.1));
}

// fn main() {
//     // Parameters
//     let (nx, ny) = (64, 64);
//     let ra = 2e4;
//     let pr = 1.;
//     let aspect = 2.0 / std::f64::consts::PI;
//     llet adiabatic = true;
//     let dt = 0.01;
//     let mut navier = Navier2D::new(nx, ny, ra, pr, dt, aspect, adiabatic);
//     //navier.read("restart.h5");
//     navier.reset_time();
//     //navier.callback();
//     integrate(&mut navier, 100.0, Some(2.0));
//     //    // Steady state analysis
//     // let mut navier = Navier2DAdjoint::new(nx, ny, ra, pr, dt, aspect, adiabatic);
//     // navier.read("restart.h5");
//     // navier.reset_time();
//     // integrate(&mut navier, 20.0, Some(2.0));
// }
