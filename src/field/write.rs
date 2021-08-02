//! Implement writing to hdf5 file for struct Field
use super::Field;
use crate::hdf5::write_to_hdf5;
use crate::hdf5::Result;

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

    fn write_hdf5(&mut self, filename: &str, group: Option<&str>) -> Result<()> {
        write_to_hdf5(filename, "v", group, &self.v)?;
        write_to_hdf5(filename, "vhat", group, &self.vhat)?;
        write_to_hdf5(filename, "x", None, &self.x[0])?;
        write_to_hdf5(filename, "dx", None, &self.dx[0])?;
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

    fn write_hdf5(&mut self, filename: &str, group: Option<&str>) -> Result<()> {
        write_to_hdf5(filename, "v", group, &self.v)?;
        write_to_hdf5(filename, "vhat", group, &self.vhat)?;
        write_to_hdf5(filename, "x", None, &self.x[0])?;
        write_to_hdf5(filename, "dx", None, &self.dx[0])?;
        write_to_hdf5(filename, "y", None, &self.x[1])?;
        write_to_hdf5(filename, "dy", None, &self.dx[1])?;
        Ok(())
    }
}
