use failure::Error;
use ndarray::prelude::*;
use std::path::Path;

/// Retrieve size of dimenion from an hdf5 file
pub fn hdf5_get_size_dimension<P: AsRef<Path>>(filename: P, name: &str) -> Result<usize, Error> {
    let file = hdf5::File::open(filename)?;
    let dset = file.dataset(name)?;

    assert!(
        dset.shape().len() == 1,
        "Dimension must be of size 1, but is of size {}",
        dset.shape().len()
    );

    Ok(dset.shape()[0])
}

/// Retrieve size of dimenion from an hdf5 file
pub fn hdf5_get_scalar<P: AsRef<Path>>(filename: P, name: &str) -> Result<f64, Error> {
    let file = hdf5::File::open(filename)?;
    let dset = file.dataset(name)?;

    assert!(
        dset.shape().len() == 1,
        "Dimension must be of size 1, but is of size {}",
        dset.shape().len()
    );

    let scalar: Array1<f64> = dset.read()?;

    Ok(scalar[0])
}
