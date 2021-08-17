use rustpde::examples::Navier2D;
use rustpde::examples::Navier2DAdjoint;
use rustpde::integrate;
//use rustpde::ReadField;

fn a() {
    // Parameters
    let (nx, ny) = (64, 64);
    let ra = 2e4;
    let pr = 1.;
    let aspect = 2.0 / std::f64::consts::PI;
    let dt = 0.01;
    let mut navier = Navier2D::new_periodic(nx, ny, ra, pr, dt, aspect);
    //let mut navier = Navier2DAdjoint::new_periodic(nx, ny, ra, pr, dt, aspect);
    //navier.read("restart.h5");
    navier.reset_time();
    //navier.callback();
    integrate(&mut navier, 300.0, Some(2.0));
}

fn main() {
    // Parameters
    let (nx, ny) = (64, 64);
    let ra = 2e4;
    let pr = 1.;
    let dt = 0.01;
    // let aspect = 2.0 / std::f64::consts::PI;
    let aspects = [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2];
    for a in aspects {
        let aspect = a / std::f64::consts::PI;
        let hdffile = format!("aspect{:4.2e}.h5", aspect);
        //let mut navier = Navier2D::new_periodic(nx, ny, ra, pr, dt, aspect);
        let mut navier = Navier2DAdjoint::new_periodic(nx, ny, ra, pr, dt, aspect);
        navier.read("restart.h5");
        navier.reset_time();
        //navier.callback();
        integrate(&mut navier, 300.0, Some(2.0));
        navier.write(&hdffile);
    }
}

// fn vof() {
//     // Parameters
//     let (nx, ny) = (129, 129);
//     let ra = 7e6;
//     let pr = 1.;
//     let adiabatic = true;
//     let aspect = 2.0;
//     let dt = 0.001;
//     let mut navier = Navier2D::new(nx, ny, ra, pr, dt, adiabatic, aspect);
//     // Set mask
//     let mut mask = solid_cylinder_inner(&navier.temp.x[0], &navier.temp.x[1], -2.0, 0.0, 0.3);
//     mask = mask + solid_cylinder_inner(&navier.temp.x[0], &navier.temp.x[1], -1.0, 0.0, 0.3);
//     mask = mask + solid_cylinder_inner(&navier.temp.x[0], &navier.temp.x[1], 0.0, 0.0, 0.3);
//     mask = mask + solid_cylinder_inner(&navier.temp.x[0], &navier.temp.x[1], 1.0, 0.0, 0.3);
//     mask = mask + solid_cylinder_inner(&navier.temp.x[0], &navier.temp.x[1], 2.0, 0.0, 0.3);
//     navier.solid = Some(mask);
//     write_to_hdf5("data/solid.h5", "v", None, &navier.solid.as_ref().unwrap()).unwrap();
//     write_to_hdf5("data/solid.h5", "x", None, &navier.temp.x[0]).unwrap();
//     write_to_hdf5("data/solid.h5", "y", None, &navier.temp.x[1]).unwrap();

//     //let mut navier = Navier2D::new_periodic(nx, ny, ra, pr, dt, aspect);
//     //navier.callback();
//     navier.read("data/flow00520.00.h5");
//     integrate(&mut navier, 720.0, Some(0.50));
// }

// fn main() {
//     // Parameters
//     let (nx, ny) = (64, 64);
//     let ra = 3e4;
//     let pr = 1.;
//     //let adiabatic = true;
//     let aspect = 2.0;
//     let dt = 0.01;
//     let mut navier = Navier2D::new(nx, ny, ra, pr, dt, true, aspect);
//     integrate(&mut navier, 1.0, Some(0.1));
// }
