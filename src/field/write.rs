//! Implement writing to hdf5 file for struct Field
use super::FieldBase;
use crate::hdf5::write_to_hdf5;
use crate::hdf5::write_to_hdf5_complex;
use crate::hdf5::H5Type;
use crate::hdf5::Result;
use crate::types::FloatNum;
use num_complex::Complex;

/// Write field to hdf5 file
pub trait WriteField<T1, T2> {
    /// Write Field data to hdf5 file
    fn write(&mut self, filename: &str, group: Option<&str>);
    /// Write Field and return result
    ///
    /// ## Errors
    /// **Errors** when file with fields exists and the fields
    /// in the file mismatch with the current fields.
    fn write_return_result(&mut self, filename: &str, group: Option<&str>) -> Result<()>;
}

/// Implement for 1-D field, which has a real valued spectral space
impl<T> WriteField<T, T> for FieldBase<T, T, 1>
where
    T: FloatNum + H5Type,
{
    /// Write Field data to hdf5 file
    fn write(&mut self, filename: &str, group: Option<&str>) {
        let result = self.write_return_result(filename, group);
        match result {
            Ok(_) => (),
            Err(_) => println!("Error while writing file {:?}.", filename),
        }
    }

    fn write_return_result(&mut self, filename: &str, group: Option<&str>) -> Result<()> {
        write_to_hdf5(filename, "v", group, &self.v)?;
        write_to_hdf5(filename, "vhat", group, &self.vhat)?;
        write_to_hdf5(filename, "x", None, &self.x[0])?;
        write_to_hdf5(filename, "dx", None, &self.dx[0])?;
        Ok(())
    }
}

/// Implement for 1-D field, which has a complex valued spectral space
impl<T> WriteField<T, Complex<T>> for FieldBase<T, Complex<T>, 1>
where
    T: FloatNum + H5Type,
{
    /// Write Field data to hdf5 file
    fn write(&mut self, filename: &str, group: Option<&str>) {
        let result = self.write_return_result(filename, group);
        match result {
            Ok(_) => (),
            Err(_) => println!("Error while writing file {:?}.", filename),
        }
    }

    fn write_return_result(&mut self, filename: &str, group: Option<&str>) -> Result<()> {
        write_to_hdf5(filename, "v", group, &self.v)?;
        write_to_hdf5_complex(filename, "vhat", group, &self.vhat)?;
        write_to_hdf5(filename, "x", None, &self.x[0])?;
        write_to_hdf5(filename, "dx", None, &self.dx[0])?;
        Ok(())
    }
}

/// Implement for 2-D field, which has a real valued spectral space
impl<T> WriteField<T, T> for FieldBase<T, T, 2>
where
    T: FloatNum + H5Type,
{
    /// Write Field data to hdf5 file
    fn write(&mut self, filename: &str, group: Option<&str>) {
        let result = self.write_return_result(filename, group);
        match result {
            Ok(_) => (),
            Err(_) => println!("Error while writing file {:?}.", filename),
        }
    }

    fn write_return_result(&mut self, filename: &str, group: Option<&str>) -> Result<()> {
        write_to_hdf5(filename, "v", group, &self.v)?;
        write_to_hdf5(filename, "vhat", group, &self.vhat)?;
        write_to_hdf5(filename, "x", None, &self.x[0])?;
        write_to_hdf5(filename, "dx", None, &self.dx[0])?;
        write_to_hdf5(filename, "y", None, &self.x[1])?;
        write_to_hdf5(filename, "dy", None, &self.dx[1])?;
        Ok(())
    }
}

/// Implement for 2-D field, which has a complex valued spectral space
impl<T> WriteField<T, Complex<T>> for FieldBase<T, Complex<T>, 2>
where
    T: FloatNum + H5Type,
{
    /// Write Field data to hdf5 file
    fn write(&mut self, filename: &str, group: Option<&str>) {
        let result = self.write_return_result(filename, group);
        match result {
            Ok(_) => (),
            Err(_) => println!("Error while writing file {:?}.", filename),
        }
    }

    fn write_return_result(&mut self, filename: &str, group: Option<&str>) -> Result<()> {
        write_to_hdf5(filename, "v", group, &self.v)?;
        write_to_hdf5_complex(filename, "vhat", group, &self.vhat)?;
        write_to_hdf5(filename, "x", None, &self.x[0])?;
        write_to_hdf5(filename, "dx", None, &self.dx[0])?;
        write_to_hdf5(filename, "y", None, &self.x[1])?;
        write_to_hdf5(filename, "dy", None, &self.dx[1])?;
        Ok(())
    }
}

// /// 2-D
// impl<S> Field<S, f64, 2> {
//     /// Write Field data to hdf5 file
//     pub fn write(&mut self, filename: &str, group: Option<&str>) {
//         let result = self.write_hdf5(filename, group);
//         match result {
//             Ok(_) => (), //println!(" ==> {:?}", filename),
//             Err(_) => println!("Error while writing file {:?}.", filename),
//         }
//     }

//     fn write_hdf5(&mut self, filename: &str, group: Option<&str>) -> Result<()> {
//         write_to_hdf5(filename, "v", group, &self.v)?;
//         write_to_hdf5(filename, "vhat", group, &self.vhat)?;
//         write_to_hdf5(filename, "x", None, &self.x[0])?;
//         write_to_hdf5(filename, "dx", None, &self.dx[0])?;
//         write_to_hdf5(filename, "y", None, &self.x[1])?;
//         write_to_hdf5(filename, "dy", None, &self.dx[1])?;
//         Ok(())
//     }
// }
