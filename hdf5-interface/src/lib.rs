//! Interface between ndarray and hdf5 for
//! easier reading/writing of scalars and multidimensional arrays.
pub use hdf5::H5Type;
pub use hdf5::Result;
use ndarray::{Array, Array1, ArrayBase, ArrayD, Dimension, Zip};
use num_complex::Complex;
use num_traits::Num;
use std::path::Path;

/// Read dataset from hdf5 file into array
pub fn read_from_hdf5_into<T, S, D>(
    filename: &str,
    name: &str,
    group: Option<&str>,
    mut array: ArrayBase<S, D>,
) where
    T: H5Type + Copy,
    S: ndarray::Data<Elem = T> + ndarray::DataMut,
    D: ndarray::Dimension,
{
    let result = read_from_hdf5::<T, D>(filename, name, group);
    match result {
        Ok(x) => array.assign(&x),
        Err(_) => println!("Error while reading file {:?}.", filename),
    }
}

/// Read dataset from hdf5 file into array
pub fn read_from_hdf5_into_complex<T, S, D>(
    filename: &str,
    name: &str,
    group: Option<&str>,
    mut array: ArrayBase<S, D>,
) where
    T: H5Type + Copy + Num,
    S: ndarray::Data<Elem = Complex<T>> + ndarray::DataMut,
    D: ndarray::Dimension,
{
    let result = read_from_hdf5_complex::<T, D>(filename, name, group);
    match result {
        Ok(x) => array.assign(&x),
        Err(_) => println!("Error while reading file {:?}.", filename),
    }
}

/// Read dataset from hdf5 file, return array
///
/// # Errors
/// Errors when file/variable does not exist and
/// when array is not supported by ndarrays
///
/// # Panics
/// Panics when  array is not supported by ndarrays
/// `into_dimensionality`.
///
/// # Example
/// ```
/// use hdf5_interface::read_from_hdf5;
/// use hdf5_interface::write_to_hdf5;
/// use ndarray::prelude::*;
/// let x = Array1::<f64>::zeros(6);
/// write_to_hdf5("test.h5", "x", None, &x).unwrap();
/// let x: Array1<f64> = read_from_hdf5("test.h5", "x", None).unwrap();
/// ```
pub fn read_from_hdf5<T, D>(
    filename: &str,
    name: &str,
    group: Option<&str>,
) -> hdf5::Result<Array<T, D>>
where
    T: H5Type + Copy,
    D: Dimension,
{
    // Open file
    let file = hdf5::File::open(filename)?;

    //Read dataset
    let name_path = gen_name_path(name, group);
    let data = file.dataset(&name_path)?;
    let y: ArrayD<T> = data.read_dyn::<T>()?;

    // Dyn to static
    let x = y.into_dimensionality::<D>().unwrap();
    Ok(x)
}

/// Read complex dataset from hdf5 file, return array
///
/// # Errors
/// Errors when file/variable does not exist and
/// when array is not supported by ndarrays
///
/// # Panics
/// Panics when  array is not supported by ndarrays
/// `into_dimensionality`.
///
/// # Example
/// ```
/// use hdf5_interface::read_from_hdf5_complex;
/// use hdf5_interface::write_to_hdf5_complex;
/// use num_complex::Complex;
/// use ndarray::prelude::*;
/// let x = Array1::<Complex<f64>>::zeros(6);
/// write_to_hdf5_complex("test.h5", "x", None, &x).unwrap();
/// let x: Array1<Complex<f64>> = read_from_hdf5_complex("test.h5", "x", None).unwrap();
/// ```
pub fn read_from_hdf5_complex<T, D>(
    filename: &str,
    name: &str,
    group: Option<&str>,
) -> hdf5::Result<Array<Complex<T>, D>>
where
    T: H5Type + Copy + Num,
    D: Dimension,
{
    // Read real part
    let name_re = format!("{}_re", name);
    let re = read_from_hdf5::<T, D>(filename, &name_re, group)?;
    // Read imag part
    let name_im = format!("{}_im", name);
    let im = read_from_hdf5::<T, D>(filename, &name_im, group)?;

    let mut x = Array::<Complex<T>, D>::zeros(re.raw_dim());
    Zip::from(&mut x).and(&re).and(&im).for_each(|w, &r, &i| {
        w.re = r;
        w.im = i;
    });
    Ok(x)
}

/// Write dataset to hdf5 file
///
/// # Errors
/// When file does not exist or when file and
/// variable exists, but variable has different
/// shape than input array (assign new value will fail).
///
/// # Example
/// ```
/// use hdf5_interface::write_to_hdf5;
/// use ndarray::prelude::*;
/// let x = Array1::<f64>::zeros(6);
/// write_to_hdf5("test.h5", "x", None, &x).unwrap();
/// ```
pub fn write_to_hdf5<T, S, D>(
    filename: &str,
    name: &str,
    group: Option<&str>,
    array: &ArrayBase<S, D>,
) -> hdf5::Result<()>
where
    T: H5Type + Copy,
    S: ndarray::Data<Elem = T>,
    D: ndarray::Dimension,
{
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
        file.new_dataset::<T>()
            .no_chunk()
            .shape(array.shape())
            .create(&name_path[..])?
    };
    dset.write(&array.view())?;

    Ok(())
}

/// Write complex valued dataset to hdf5 file
///
/// # Errors
/// When file does not exist or when file and
/// variable exists, but variable has different
/// shape than input array (assign new value will fail).
///
/// # Example
/// ```
/// use hdf5_interface::write_to_hdf5_complex;
/// use num_complex::Complex;
/// use ndarray::prelude::*;
/// let x = Array1::<Complex<f64>>::zeros(6);
/// write_to_hdf5_complex("test_complex.h5", "x", None, &x).unwrap();
/// ```
pub fn write_to_hdf5_complex<T, S, D>(
    filename: &str,
    name: &str,
    group: Option<&str>,
    array: &ArrayBase<S, D>,
) -> hdf5::Result<()>
where
    T: H5Type + Copy,
    S: ndarray::Data<Elem = Complex<T>>,
    D: ndarray::Dimension,
{
    // Write real part
    let name_re = format!("{}_re", name);
    write_to_hdf5(filename, &name_re, group, &array.mapv(|x| x.re))?;
    // Write imag part
    let name_im = format!("{}_im", name);
    write_to_hdf5(filename, &name_im, group, &array.mapv(|x| x.im))?;
    Ok(())
}

/// Read scalar from hdf5
///
/// # Errors
/// When file or variable does not exists.
///
/// # Panics
/// Panics when requested field is not of dimensionality
/// 1 (i.e. not a scalar).
///
/// # Example
/// ```
/// use hdf5_interface::write_scalar_to_hdf5;
/// use hdf5_interface::read_scalar_from_hdf5;
/// use ndarray::prelude::*;
/// let x = 5.;
/// write_scalar_to_hdf5("test.h5", "scalar", None, x).unwrap();
/// let x_read: f64 = read_scalar_from_hdf5("test.h5", "scalar", None).unwrap();
/// assert!(x == x_read);
/// ```
pub fn read_scalar_from_hdf5<T>(filename: &str, name: &str, group: Option<&str>) -> hdf5::Result<T>
where
    T: H5Type + Clone + Copy,
{
    let name_path = gen_name_path(name, group);
    let file = hdf5::File::open(filename)?;
    let dset = file.dataset(&name_path)?;

    assert!(
        dset.shape().len() == 1,
        "Dimension must be of size 1, but is of size {}",
        dset.shape().len()
    );

    let scalar: Array1<T> = dset.read()?;

    Ok(scalar[0])
}

/// Write scalar to hdf5 file
///
/// # Errors
/// When file does not exist.
pub fn write_scalar_to_hdf5<T>(
    filename: &str,
    name: &str,
    group: Option<&str>,
    scalar: T,
) -> hdf5::Result<()>
where
    T: H5Type + Copy,
{
    let x = Array1::<T>::from_elem(1, scalar);
    write_to_hdf5(filename, name, group, &x)?;
    Ok(())
}

/// Retrieve size of dimension from an hdf5 file
///
/// # Errors
/// When file or variable does not exists.
///
/// # Panics
/// Panics when requested dimension is not of dimensionality
/// 1 (i.e. not a scalar).
pub fn hdf5_get_size_dimension<P: AsRef<Path>>(filename: P, name: &str) -> hdf5::Result<usize> {
    let file = hdf5::File::open(filename)?;
    let dset = file.dataset(name)?;

    assert!(
        dset.shape().len() == 1,
        "Dimension must be of size 1, but is of size {}",
        dset.shape().len()
    );

    Ok(dset.shape()[0])
}

/// Retrieve scalar from an hdf5 file
pub fn hdf5_get_scalar<P: AsRef<Path>>(filename: P, name: &str) -> hdf5::Result<f64> {
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

/// Check if a variable exists in a hdf5 file
///
/// # Errors
/// When file does not exists.
pub fn variable_exists(file: &hdf5::File, name: &str, group: Option<&str>) -> hdf5::Result<bool> {
    if let Some(g) = group {
        if file
            .member_names()?
            .iter()
            .any(|i| i == g || i.clone() + "/" == g)
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

/// Generate full variable path inside hdf5 file from name
/// of the variable and name of the group (optional)
fn gen_name_path(name: &str, group: Option<&str>) -> String {
    group.map_or_else(
        || name.to_owned(),
        |g| {
            if g.chars().last().unwrap().to_string() == "/" {
                g.to_owned() + name
            } else {
                g.to_owned() + "/" + name
            }
        },
    )
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    /// Read & Write 2-D data
    fn test_read_write() {
        use ndarray::Array2;
        let fname = "test.h5";
        let array = Array2::<f64>::from_elem((10, 10), 5.);
        write_to_hdf5(&fname, "var", None, &array).unwrap();
        let array_read: Array2<f64> = read_from_hdf5(&fname, "var", None).unwrap();
        assert_eq!(array, array_read);
    }
}
