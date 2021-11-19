//! Calculate vorticity from a given file
//!
//! # Example
//!
//! ```ignore
//! fn main() {
//!    use std::path::PathBuf;
//!    use rustpde::navier::vorticity::vorticity_from_file;
//!
//!    let root = "data/";
//!    let files: Vec<PathBuf> = std::fs::read_dir(root)
//!        .unwrap()
//!        .into_iter()
//!        .filter(|r| r.is_ok())
//!        .map(|r| r.unwrap().path())
//!        .filter(|r| r.extension() == Some(std::ffi::OsStr::new("h5")))
//!        .collect();
//!    for f in files.iter() {
//!        let fname = f.to_str().unwrap();
//!        vorticity_from_file(&fname);
//! }
//! ```
use crate::BaseSpace;
use crate::ReadField;
use crate::WriteField;
use crate::{cheb_dirichlet, fourier_r2c, hebyshev, Field2, Space2};
use hdf5_interface::hdf5_get_size_dimension;
use num_traits::Zero;

/// Read velocities from file,
/// calculate dudy - dvdx and append vortictiy
///
/// # Panics
/// If reading dimension from file fails
pub fn vorticity_from_file(fname: &str) {
    let nx = hdf5_get_size_dimension(&fname, "x").unwrap();
    let ny = hdf5_get_size_dimension(&fname, "y").unwrap();
    let mut ux = Field2::new(&Space2::new(&cheb_dirichlet(nx), &cheb_dirichlet(ny)));
    let mut uy = Field2::new(&Space2::new(&cheb_dirichlet(nx), &cheb_dirichlet(ny)));
    let mut vorticity = Field2::new(&Space2::new(&chebyshev(nx), &chebyshev(ny)));
    ux.read(&fname, Some("ux"));
    uy.read(&fname, Some("uy"));
    let dudz = ux.gradient([0, 1], Some([1.0, 1.0]));
    let dvdx = uy.gradient([1, 0], Some([1.0, 1.0]));
    vorticity.vhat.assign(&(dvdx - dudz));
    dealias(&mut vorticity);
    vorticity.backward();
    vorticity.write(&fname, Some("vorticity"));
}

/// Read velocities from file,
/// calculate dudy - dvdx and append vortictiy
///
/// x-direction is periodic
pub fn vorticity_from_file_periodic(fname: &str) {
    let nx = hdf5_get_size_dimension(&fname, "x").unwrap();
    let ny = hdf5_get_size_dimension(&fname, "y").unwrap();
    let mut ux = Field2::new(&Space2::new(&fourier_r2c(nx), &cheb_dirichlet(ny)));
    let mut uy = Field2::new(&Space2::new(&fourier_r2c(nx), &cheb_dirichlet(ny)));
    let mut vorticity = Field2::new(&Space2::new(&fourier_r2c(nx), &chebyshev(ny)));
    ux.read(&fname, Some("ux"));
    uy.read(&fname, Some("uy"));
    let dudz = ux.gradient([0, 1], Some([1.0, 1.0]));
    let dvdx = uy.gradient([1, 0], Some([1.0, 1.0]));
    vorticity.vhat.assign(&(dvdx - dudz));
    dealias(&mut vorticity);
    vorticity.backward();
    vorticity.write(&fname, Some("vorticity"));
}

/// Dealias field (2/3 rule)
fn dealias<S, T2>(field: &mut Field2<T2, S>)
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T2>,
    T2: Zero + Clone + Copy,
{
    let zero = T2::zero();
    let n_x: usize = field.vhat.shape()[0] * 2 / 3;
    let n_y: usize = field.vhat.shape()[1] * 2 / 3;
    field.vhat.slice_mut(ndarray::s![n_x.., ..]).fill(zero);
    field.vhat.slice_mut(ndarray::s![.., n_y..]).fill(zero);
}
