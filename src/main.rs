//use rustpde::examples::solid_cylinder_inner;
use rustpde::examples::Navier2D;
//
//use rustpde::hdf5::write_to_hdf5;
use rustpde::integrate;
use rustpde::Integrate;
//use rustpde::ReadField;

// fn main() {
//     // Parameters
//     let (nx, ny) = (64, 64);
//     let ra = 1e4;
//     let pr = 1.;
//     let adiabatic = true;
//     let aspect = 1.0;
//     let dt = 0.01;
//     //let mut navier = Navier2D::new(nx, ny, ra, pr, dt, adiabatic, aspect);
//     let mut navier = Navier2DAdjoint::new(nx, ny, ra, pr, dt, adiabatic, aspect);
//     navier.callback();
//     integrate(&mut navier, 2.0, Some(0.50));
// }

// fn main() {
//     // Parameters
//     let (nx, ny) = (129, 129);
//     let ra = 3e5;
//     let pr = 1.;
//     let adiabatic = true;
//     let aspect = 2.0;
//     let dt = 0.002;
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
//     navier.read("data/flow218.000.h5");
//     integrate(&mut navier, 320.0, Some(0.50));
// }

fn main() {
    // Parameters
    let (nx, ny) = (1024, 513);
    let ra = 3e4;
    let pr = 1.;
    //let adiabatic = true;
    let aspect = 2.0;
    let dt = 0.01;
    let mut navier = Navier2D::new_periodic(nx, ny, ra, pr, dt, aspect);
    integrate(&mut navier, 1.0, None);
}
