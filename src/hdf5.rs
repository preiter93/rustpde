//! Read / Write with hdf5
//!
//! Currently the hdf5 library was build with an older version
//! of ndarray. This is why the data must first be transferred
//! to arrays from the older ndarray version, before they can be
//! written (or read).
use ndarray::{Array, Array0, Array1, Array2, ArrayD, Dimension, IxDyn};
use std::path::Path;

/// Collection of methods for reading of and writing to hdf5.
///
/// Supported:
///
/// - Array0 (Scalar),
/// - Array1,
/// - Array2,

pub enum Hdf5<'a> {
    /// Read/Write Scalar
    Array0(&'a mut Array0<f64>),
    /// Read/Write Vector
    Array1(&'a mut Array1<f64>),
    /// Read/Write Matrix
    Array2(&'a mut Array2<f64>),
}

impl Hdf5<'_> {
    /// Returns shape
    fn shape(&self) -> &[usize] {
        match self {
            Hdf5::Array0(ref x) => x.shape(),
            Hdf5::Array1(ref x) => x.shape(),
            Hdf5::Array2(ref x) => x.shape(),
        }
    }
}

///////////////////////////////////////////////////////////////
//                      Read
///////////////////////////////////////////////////////////////

/// Read dataset from hdf5 file
///
pub fn read_from_hdf5(
    filename: &str,
    name: &str,
    group: Option<&str>,
    mut array: Hdf5,
) -> hdf5::Result<()> {
    // Open file
    let file = hdf5::File::open(filename)?;

    //Read dataset
    let name_path = gen_name_path(name, group);
    let data = file.dataset(&name_path)?;

    match array {
        Hdf5::Array0(ref mut x) => {
            let y: ndarray_02::Array0<f64> =
                data.read::<f64, ndarray_02::Dim<[ndarray_02::Ix; 0]>>()?;
            // Transfer to new ndarray
            for (xi, yi) in x.iter_mut().zip(y.iter()) {
                *xi = *yi
            }
        }
        //
        Hdf5::Array1(ref mut x) => {
            let y: ndarray_02::Array1<f64> =
                data.read::<f64, ndarray_02::Dim<[ndarray_02::Ix; 1]>>()?;
            // Transfer to new ndarray
            for (xi, yi) in x.iter_mut().zip(y.iter()) {
                *xi = *yi
            }
        }
        Hdf5::Array2(ref mut x) => {
            let y: ndarray_02::Array2<f64> =
                data.read::<f64, ndarray_02::Dim<[ndarray_02::Ix; 2]>>()?;
            // Transfer to new ndarray
            for (xi, yi) in x.iter_mut().zip(y.iter()) {
                *xi = *yi
            }
        }
    }
    Ok(())
}

/// Read dataset from hdf5 file
pub fn read_from_hdf5_2<D: Dimension, const N: usize>(
    filename: &str,
    name: &str,
    group: Option<&str>,
    //mut array: Hdf5,
) -> hdf5::Result<Array<f64, D>> {
    // Open file
    let file = hdf5::File::open(filename)?;

    //Read dataset
    let name_path = gen_name_path(name, group);
    let data = file.dataset(&name_path)?;
    let y: ndarray_02::ArrayD<f64> = data.read_dyn::<f64>()?;
    assert!(y.ndim() == N, "Dimension mismatch.");

    // Construct dimension
    let mut dim = [0; N];
    for (i, d) in y.shape().iter().zip(dim.iter_mut()) {
        *d = *i;
    }

    // Transfer data to new ndarray
    let mut x = ArrayD::<f64>::zeros(IxDyn(&dim));
    for (xi, yi) in x.iter_mut().zip(y.iter()) {
        *xi = *yi
    }
    let x = x.into_dimensionality::<D>().unwrap();
    Ok(x)
}

/// Write dataset to hdf5 file
///
/// write_to_hdf5(filename, "x", group, Hdf5::Array1(&mut x))?;
pub fn write_to_hdf5(
    filename: &str,
    name: &str,
    group: Option<&str>,
    array: Hdf5,
) -> hdf5::Result<()> {
    // Open file
    let file = if Path::new(filename).exists() {
        hdf5::File::append(filename)?
    } else {
        hdf5::File::create(filename)?
    };

    //Write dataset
    let name_path = gen_name_path(name, group);
    let variable_exists = variable_exists(&file, name, group);

    let dset = if variable_exists? {
        file.dataset(&name_path)?
    } else {
        file.new_dataset::<f64>()
            .create(&name_path, array.shape())?
    };
    match array {
        Hdf5::Array0(x) => {
            let mut data = ndarray_02::Array0::<f64>::zeros(());
            for (xi, yi) in data.iter_mut().zip(x.iter()) {
                *xi = *yi
            }
            dset.write(&data)?;
        }
        Hdf5::Array1(x) => {
            let mut data = ndarray_02::Array1::<f64>::zeros(x.len());
            for (xi, yi) in data.iter_mut().zip(x.iter()) {
                *xi = *yi
            }
            dset.write(&data)?;
        }
        Hdf5::Array2(x) => {
            let (n, m) = (x.shape()[0], x.shape()[1]);
            let mut data = ndarray_02::Array2::<f64>::zeros((n, m));
            for (xi, yi) in data.iter_mut().zip(x.iter()) {
                *xi = *yi
            }
            dset.write(&data)?;
        }
    }

    Ok(())
}

/// Generate full variable path inside hdf5 file from name
/// of the variable and name of the group (optional)
fn gen_name_path(name: &str, group: Option<&str>) -> String {
    if let Some(g) = group {
        if g.chars().last().unwrap().to_string() == "/" {
            g.to_owned() + name
        } else {
            g.to_owned() + "/" + name
        }
    } else {
        name.to_owned()
    }
}

/// Check if a variable exists in a hdf5 file
fn variable_exists(file: &hdf5::File, name: &str, group: Option<&str>) -> hdf5::Result<bool> {
    if let Some(g) = group {
        if file
            .member_names()?
            .iter()
            .any(|i| i == g || i.to_owned() + "/" == g)
        {
            let group = file.group(g)?;
            Ok(group.member_names()?.iter().any(|i| i == name))
        } else {
            Ok(false)
        }
    } else {
        Ok(file.member_names()?.iter().any(|i| i == name))
    }
}
