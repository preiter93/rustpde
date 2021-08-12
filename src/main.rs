//use rustpde::examples::Navier2D;
use rustpde::examples::Navier2DAdjoint;
use rustpde::integrate;
use rustpde::Integrate;
//use rustpde::ReadField;

fn main() {
    // Parameters
    let (nx, ny) = (64, 64);
    let ra = 1e4;
    let pr = 1.;
    let adiabatic = true;
    let aspect = 1.0;
    let dt = 0.01;
    //let mut navier = Navier2D::new(nx, ny, ra, pr, dt, adiabatic, aspect);
    let mut navier = Navier2DAdjoint::new(nx, ny, ra, pr, dt, adiabatic, aspect);
    navier.callback();
    integrate(&mut navier, 2.0, Some(0.50));
}

/*
fn navier() {
    // Parameters
    let (nx, ny) = (64, 64);
    let ra = 1e4;
    let pr = 1.;
    let adiabatic = true;
    let aspect = 1.0;
    let dt = 0.01;
    //let mut navier = Navier2D::new(nx, ny, ra, pr, dt, adiabatic, aspect);
    let mut navier = Navier2D::new_periodic(nx, ny, ra, pr, dt, aspect);
    navier.callback();
    integrate(&mut navier, 2.0, Some(0.50));
}
*/
