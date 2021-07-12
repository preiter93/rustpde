//! Implement reading from hdf5 file for struct Field
use super::Field;
use crate::hdf5::{read_from_hdf5, Hdf5};

/// 1-D
impl<S> Field<S, f64, 1> {
    /// Read hdf5 file and store results in Field
    pub fn read(&mut self, filename: &str, group: Option<&str>) {
        let result = self.read_hdf5(filename, group);
        match result {
            Ok(_) => println!("Reading file {:?} was successfull.", filename),
            Err(_) => println!("Error while reading file {:?}.", filename),
        }
    }

    fn read_hdf5(&mut self, filename: &str, group: Option<&str>) -> hdf5::Result<()> {
        read_from_hdf5(filename, "v", group, Hdf5::Array1(&mut self.v))?;
        read_from_hdf5(filename, "vhat", group, Hdf5::Array1(&mut self.vhat))?;
        read_from_hdf5(filename, "x", None, Hdf5::Array1(&mut self.x[0]))?;
        read_from_hdf5(filename, "dx", None, Hdf5::Array1(&mut self.dx[0]))?;
        Ok(())
    }
}

/// 2-D
impl<S> Field<S, f64, 2> {
    /// Read hdf5 file and store results in Field
    pub fn read(&mut self, filename: &str, group: Option<&str>) {
        let result = self.read_hdf5(filename, group);
        match result {
            Ok(_) => println!("Reading file {:?} was successfull.", filename),
            Err(_) => println!("Error while reading file {:?}.", filename),
        }
    }

    fn read_hdf5(&mut self, filename: &str, group: Option<&str>) -> hdf5::Result<()> {
        read_from_hdf5(filename, "v", group, Hdf5::Array2(&mut self.v))?;
        read_from_hdf5(filename, "vhat", group, Hdf5::Array2(&mut self.vhat))?;
        read_from_hdf5(filename, "x", None, Hdf5::Array1(&mut self.x[0]))?;
        read_from_hdf5(filename, "y", None, Hdf5::Array1(&mut self.x[1]))?;
        read_from_hdf5(filename, "dx", None, Hdf5::Array1(&mut self.dx[0]))?;
        read_from_hdf5(filename, "dy", None, Hdf5::Array1(&mut self.dx[1]))?;
        Ok(())
    }
}
