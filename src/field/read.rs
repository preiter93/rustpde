//! Implement reading from hdf5 file for struct Field
use super::Field;
use crate::hdf5::{read_from_hdf5, read_from_hdf5_2, Hdf5};

/// 1-D
impl<S> Field<S, f64, 1> 
where
    S: crate::Spaced<f64, 1_usize>,
{
    /// Read hdf5 file and store results in Field
    pub fn read(&mut self, filename: &str, group: Option<&str>) {
        let result = read_from_hdf5_2::<ndarray::Ix1, 1>(filename, "vhat", group);
        match result {
            Ok(x) => {
                self.vhat.assign(&x);
                self.backward();
                println!("Reading file {:?} was successfull.", filename);
            }
            Err(_) => println!("Error while reading file {:?}.", filename),
        }
    }

    // fn read_hdf5(&mut self, filename: &str, group: Option<&str>) -> hdf5::Result<()> {
    //     read_from_hdf5(filename, "v", group, Hdf5::Array1(&mut self.v))?;
    //     read_from_hdf5(filename, "vhat", group, Hdf5::Array1(&mut self.vhat))?;
    //     Ok(())
    // }
}

/// 2-D
impl<S> Field<S, f64, 2>
where
    S: crate::Spaced<f64, 2_usize>,
{
    /// Read hdf5 file and store results in Field
    ///
    /// Reads spectral coefficients only and restores
    /// physical field by a backtransform.
    /// Supports reading of same shape but different
    /// size arrays.
    pub fn read(&mut self, filename: &str, group: Option<&str>) {
        let result = read_from_hdf5_2::<ndarray::Ix2, 2>(filename, "vhat", group);
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
fn broadcast_2d<T: std::clone::Clone>(old: &ndarray::Array2<T>, new: &mut ndarray::Array2<T>) {
    use ndarray::s;
    let sh: Vec<usize> = old
        .shape()
        .iter()
        .zip(new.shape().iter())
        .map(|(i, j)| std::cmp::min(i, j).clone())
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
