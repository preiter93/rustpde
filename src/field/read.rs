//! Implement reading from hdf5 file for struct Field
use super::{BaseSpace, FieldBase};
use crate::hdf5::read_from_hdf5;
use crate::hdf5::read_from_hdf5_complex;
use crate::hdf5::H5Type;
use crate::types::FloatNum;
use ndarray::prelude::*;
use ndarray::ScalarOperand;
use num_complex::Complex;
use std::clone::Clone;

/// Write field to hdf5 file
pub trait ReadField {
    /// Read Field data from hdf5 file
    fn read(&mut self, filename: &str, group: Option<&str>);
}

impl<A, S> ReadField for FieldBase<A, A, A, S, 1>
where
    A: FloatNum + H5Type,
    Complex<A>: ScalarOperand,
    S: BaseSpace<A, 1, Physical = A, Spectral = A>,
{
    fn read(&mut self, filename: &str, group: Option<&str>) {
        let result = read_from_hdf5::<A, Ix1>(filename, "vhat", group);
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

impl<A, S> ReadField for FieldBase<A, A, Complex<A>, S, 1>
where
    A: FloatNum + H5Type,
    Complex<A>: ScalarOperand,
    S: BaseSpace<A, 1, Physical = A, Spectral = Complex<A>>,
{
    fn read(&mut self, filename: &str, group: Option<&str>) {
        let result = read_from_hdf5_complex::<A, Ix1>(filename, "vhat", group);
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

impl<A, S> ReadField for FieldBase<A, A, A, S, 2>
where
    A: FloatNum + H5Type,
    Complex<A>: ScalarOperand,
    S: BaseSpace<A, 2, Physical = A, Spectral = A>,
{
    fn read(&mut self, filename: &str, group: Option<&str>) {
        let result = read_from_hdf5::<A, Ix2>(filename, "vhat", group);
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

impl<A, S> ReadField for FieldBase<A, A, Complex<A>, S, 2>
where
    A: FloatNum + H5Type,
    Complex<A>: ScalarOperand,
    S: BaseSpace<A, 2, Physical = A, Spectral = Complex<A>>,
{
    fn read(&mut self, filename: &str, group: Option<&str>) {
        let result = read_from_hdf5_complex::<A, Ix2>(filename, "vhat", group);
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
