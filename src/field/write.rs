//! Implement writing to hdf5 file for struct Field
use super::Field;
use crate::hdf5::{write_to_hdf5, Hdf5};

/// 1-D
impl<S> Field<S, f64, 1> {
    /// Write Field data to hdf5 file
    pub fn write(&mut self, filename: &str, group: Option<&str>) {
        let result = self.write_hdf5(filename, group);
        match result {
            Ok(_) => (), //println!(" ==> {:?}", filename),
            Err(_) => println!("Error while writing file {:?}.", filename),
        }
    }

    fn write_hdf5(&mut self, filename: &str, group: Option<&str>) -> hdf5::Result<()> {
        write_to_hdf5(filename, "v", group, Hdf5::Array1(&mut self.v))?;
        write_to_hdf5(filename, "vhat", group, Hdf5::Array1(&mut self.vhat))?;
        write_to_hdf5(filename, "x", None, Hdf5::Array1(&mut self.x[0]))?;
        Ok(())
    }
}

/// 2-D
impl<S> Field<S, f64, 2> {
    /// Write Field data to hdf5 file
    pub fn write(&mut self, filename: &str, group: Option<&str>) {
        let result = self.write_hdf5(filename, group);
        match result {
            Ok(_) => (), //println!(" ==> {:?}", filename),
            Err(_) => println!("Error while writing file {:?}.", filename),
        }
    }

    fn write_hdf5(&mut self, filename: &str, group: Option<&str>) -> hdf5::Result<()> {
        write_to_hdf5(filename, "v", group, Hdf5::Array2(&mut self.v))?;
        write_to_hdf5(filename, "vhat", group, Hdf5::Array2(&mut self.vhat))?;
        write_to_hdf5(filename, "x", None, Hdf5::Array1(&mut self.x[0]))?;
        write_to_hdf5(filename, "y", None, Hdf5::Array1(&mut self.x[1]))?;
        Ok(())
    }
}
