//! Implement reading from hdf5 file for struct Field
use super::{Field, FieldBase};
use crate::hdf5::read_from_hdf5;
use crate::hdf5::read_from_hdf5_complex;
use crate::hdf5::H5Type;
use crate::types::FloatNum;
use ndarray::prelude::*;
use ndarray::ScalarOperand;
use num_complex::Complex;
use std::clone::Clone;

/// Write field to hdf5 file
pub trait ReadField<T1, T2> {
    /// Read Field data from hdf5 file
    fn read(&mut self, filename: &str, group: Option<&str>);
    // Broadcast shape from hdf5 file to field. Only
    // used when both fields mismatch. This is equivalent
    // to an interpolation.
    //fn interpolate<T: Clone>(old: Array2<T>, new: &mut Array2<T>);
}

/// Implement for 1-D field, which has a real valued spectral space
impl<T> ReadField<T, T> for FieldBase<T, T, 1>
where
    T: FloatNum + H5Type + std::ops::DivAssign,
    Complex<T>: ScalarOperand,
{
    fn read(&mut self, filename: &str, group: Option<&str>) {
        let result = read_from_hdf5::<T, Ix1>(filename, "vhat", group);
        match result {
            Ok(x) => {
                self.vhat.assign(&x);
                self.backward();
                println!("Reading file {:?} was successfull.", filename);
            }
            Err(_) => println!("Error while reading file {:?}.", filename),
        }
    }
}

/// Implement for 1-D field, which has a complex valued spectral space
impl<T> ReadField<T, Complex<T>> for FieldBase<T, Complex<T>, 1>
where
    T: FloatNum + H5Type,
    Complex<T>: ScalarOperand + std::ops::DivAssign,
{
    fn read(&mut self, filename: &str, group: Option<&str>) {
        let result = read_from_hdf5_complex::<T, Ix1>(filename, "vhat", group);
        match result {
            Ok(x) => {
                self.vhat.assign(&x);
                self.backward();
                println!("Reading file {:?} was successfull.", filename);
            }
            Err(_) => println!("Error while reading file {:?}.", filename),
        }
    }
}

/// Implement for 2-D field, which has a real valued spectral space
impl<T> ReadField<T, T> for FieldBase<T, T, 2>
where
    T: FloatNum + H5Type + std::ops::DivAssign,
    Complex<T>: ScalarOperand,
{
    fn read(&mut self, filename: &str, group: Option<&str>) {
        let result = read_from_hdf5::<T, Ix2>(filename, "vhat", group);
        match result {
            Ok(x) => {
                if x.shape() == self.vhat.shape() {
                    self.vhat.assign(&x);
                } else {
                    println!(
                        "Attention! Broadcast from shape {:?} to shape {:?}.",
                        x.shape(),
                        self.vhat.shape()
                    );
                    broadcast_2d(&x, &mut self.vhat);
                }
                self.backward();
                println!("Reading file {:?} was successfull.", filename);
            }
            Err(_) => println!("Error while reading file {:?}.", filename),
        }
    }
}

/// Implement for 2-D field, which has a complex valued spectral space
impl<T> ReadField<T, Complex<T>> for FieldBase<T, Complex<T>, 2>
where
    T: FloatNum + H5Type,
    Complex<T>: ScalarOperand + std::ops::DivAssign,
{
    fn read(&mut self, filename: &str, group: Option<&str>) {
        let result = read_from_hdf5_complex::<T, Ix2>(filename, "vhat", group);
        match result {
            Ok(x) => {
                if x.shape() == self.vhat.shape() {
                    self.vhat.assign(&x);
                } else {
                    println!(
                        "Attention! Broadcast from shape {:?} to shape {:?}.",
                        x.shape(),
                        self.vhat.shape()
                    );
                    broadcast_2d(&x, &mut self.vhat);
                }
                self.backward();
                println!("Reading file {:?} was successfull.", filename);
            }
            Err(_) => println!("Error while reading file {:?}.", filename),
        }
    }
}

/// Broadcast 2d array
fn broadcast_2d<T: Clone>(old: &Array2<T>, new: &mut Array2<T>) {
    let sh: Vec<usize> = old
        .shape()
        .iter()
        .zip(new.shape().iter())
        .map(|(i, j)| *std::cmp::min(i, j))
        .collect();
    new.slice_mut(s![..sh[0], ..sh[1]])
        .assign(&old.slice(s![..sh[0], ..sh[1]]));
}

// /// Read hdf5 file and store results in Field
// pub fn read(&mut self, filename: &str, group: Option<&str>) {
//     let result = self.read_hdf5(filename, group);
//     match result {
//         Ok(_) => println!("Reading file {:?} was successfull.", filename),
//         Err(_) => println!("Error while reading file {:?}.", filename),
//     }
// }

// fn read_hdf5(&mut self, filename: &str, group: Option<&str>) -> hdf5::Result<()> {
//     read_from_hdf5(filename, "v", group, Hdf5::Array2(&mut self.v))?;
//     read_from_hdf5(filename, "vhat", group, Hdf5::Array2(&mut self.vhat))?;
//     Ok(())
// }
