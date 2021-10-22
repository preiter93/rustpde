use rustpde::navier::vorticity_from_file;
use rustpde::navier::Navier2D;
use rustpde::navier::Navier2DAdjoint;
use rustpde::integrate;
use rustpde::Integrate;

fn direct() {
    // Parameters
    let (nx, ny) = (257, 257);
    let ra = 7e5;
    let pr = 1.;
    let adiabatic = true;
    let aspect = 1.0;
    let dt = 0.01;
    let mut navier = Navier2D::new(nx, ny, ra, pr, dt, aspect, adiabatic);
    navier.read("restart.h5");
    navier.reset_time();
    // // Set initial conditions
    // navier.set_velocity(0.2, 1., 1.);
    // navier.set_temperature(0.2, 1., 1.);
    integrate(&mut navier, 10., Some(1.0));
}

fn adjoint() {
    // Parameters
    let (nx, ny) = (257, 257);
    let ra = 7e5;
    let pr = 1.;
    let adiabatic = true;
    let aspect = 1.0;
    let dt = 2.0;
    let timesteps = 2000.;
    let mut navier = Navier2DAdjoint::new(nx, ny, ra, pr, dt, aspect, adiabatic);
    navier.write_intervall = Some(10. * dt);
    // Want to restart?
    navier.read("restart.h5");
    navier.reset_time();
    /*    navier.set_velocity(0.5, 1., 1.);
    navier.set_temperature(0.2, 1., 1.);
    navier.callback();*/
    //Integrate
    integrate(&mut navier, timesteps * dt, Some(dt));
}

/// Append vorticity field
fn main() {
    use std::path::PathBuf;
    let root = "data/";
    let files: Vec<PathBuf> = std::fs::read_dir(root)
        .unwrap()
        .into_iter()
        .filter(|r| r.is_ok())
        .map(|r| r.unwrap().path())
        .filter(|r| r.extension() == Some(std::ffi::OsStr::new("h5")))
        .collect();
    for f in files.iter() {
        let fname = f.to_str().unwrap();
        vorticity_from_file(&fname);
    }
}

// fn periodic() {
//     // Parameters
//     let (nx, ny) = (129, 129);
//     let ra = 1e6;
//     let pr = 1.;
//     let aspect = 1.0;
//     let dt = 0.01;
//     let mut navier = Navier2D::new_periodic(nx, ny, ra, pr, dt, aspect);
//     integrate(&mut navier, 10., Some(1.0));
// }

// fn roughness() {
//     use rustpde::navier::solid_masks::solid_roughness_sinusoid;
//     // Parameters
//     let (nx, ny) = (512, 257);
//     let ra = 3e8;
//     let pr = 1.;
//     let aspect = 1.0;
//     let dt = 0.0002;
//     let mut navier = Navier2D::new_periodic(nx, ny, ra, pr, dt, aspect);
//     // Add roughness elements
//     let mask = solid_roughness_sinusoid(&navier.temp.x[0], &navier.temp.x[1], 0.1, 10.);
//     navier.solid = Some(mask);
//     // Write first flow field;
//     navier.callback();
//     // navier.read("data/flow00078.00.h5");
//     // Set how often flow fields are written
//     navier.write_intervall = Some(0.5);
//     // Integrate over 200 TU, save info every 0.1 TU
//     integrate(&mut navier, 200.0, Some(0.1));
// }
