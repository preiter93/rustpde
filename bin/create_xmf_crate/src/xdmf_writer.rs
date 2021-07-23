use crate::utils::{hdf5_get_scalar, hdf5_get_size_dimension};
use failure::Error;
use ndarray::prelude::*;
use std::fs::File;
use std::io::LineWriter;
use std::io::Write;
use std::path::PathBuf;

/// Create xmf file from hdf5 file for paraview
#[derive(Debug)]
pub struct XdmfWriter {
    /// Number of points in x
    nx: usize,
    /// Number of points in y
    ny: usize,
    /// Filename, which contains the field
    fname: String,
    /// Filename, which contains mesh coordinates (2d version of x, y)
    /// Create with self.create_cartesian.
    cname: String,
    /// Attribute name
    aname: Vec<String>,
    /// variables name in file
    vname: Vec<String>,
    /// x-dimension name, default "x"
    xname: String,
    /// x-dimension name, default "y"
    yname: String,
    /// xmf-filename
    xmfname: String,
    /// Time info
    time: Option<f64>,
}

impl XdmfWriter {
    /// Return xdmfwriter from filename, attribute name and variable name
    pub fn default(fname: &str, aname: &[&str], vname: &[&str], xmfname: Option<&str>) -> Self {
        // Get dimensions
        let nx = hdf5_get_size_dimension(&fname, "x").unwrap();
        let ny = hdf5_get_size_dimension(&fname, "y").unwrap();

        // cartesian.nc (stores 2d grid points)
        let path = PathBuf::from(fname);
        let parent = path.parent().unwrap().to_str().unwrap();
        let cname = if parent.trim().is_empty() {
            String::from("cartesian.nc")
        } else {
            parent.to_owned() + "/" + "cartesian.nc"
        };

        // Check for time
        let time = match hdf5_get_scalar(&fname, "time") {
            Ok(number) => Some(number),
            Err(_) => None,
        };

        // Filename
        let fname = String::from(fname);

        // Get xmf name from filename
        let xmfname = if let Some(x) = xmfname {
            String::from(x)
        } else if fname.ends_with(".h5") {
            fname.replace(".h5", ".xmf")
        } else {
            println!(
                "Warning! File {:?} doesnt end with \".h5\", used \"default.xmf\" instead",
                fname
            );
            String::from("default.xmf")
        };

        // Return
        Self {
            nx,
            ny,
            fname,
            cname,
            aname: aname.iter().map(|s| s.to_string()).collect(),
            vname: vname.iter().map(|s| s.to_string()).collect(),
            xname: String::from("x"),
            yname: String::from("y"),
            xmfname,
            time,
        }
    }

    /// Create cartesian.nc file, which converts the
    /// 1d array coordinates "x" and "y" to a 2-dimensional
    /// meshgrid and stores it in "cartesian.nc".
    pub fn create_cartesian(&self, overwrite: bool) -> Result<(), Error> {
        // Return if file exists and overwrite is false
        if !overwrite && std::path::Path::new(&self.cname).exists() {
            return Ok(());
        }

        // Open file
        let file = hdf5::File::open(&self.fname)?;

        // Get "x" and "y" coordinates
        let data = file.dataset(&self.xname)?;
        let x: Array1<f64> = data.read()?;
        let data = file.dataset(&self.yname)?;
        let y: Array1<f64> = data.read()?;

        // Create meshgrid
        let mut xx = Array2::<f64>::zeros((self.nx, self.ny));
        let mut yy = Array2::<f64>::zeros((self.nx, self.ny));

        for (i, _x) in x.iter().enumerate() {
            xx.slice_mut(s![i, ..]).fill(*_x);
        }
        for (i, _y) in y.iter().enumerate() {
            yy.slice_mut(s![.., i]).fill(*_y);
        }

        let file = hdf5::File::create(&self.cname)?;
        let dset = file.new_dataset::<f64>().create(&self.xname, xx.shape())?;
        dset.write(&xx)?;
        let dset = file.new_dataset::<f64>().create(&self.yname, yy.shape())?;
        dset.write(&yy)?;
        println!("Created {:?}", self.cname);
        Ok(())
    }

    /// Return string, which defines the geometry attributes
    fn _geometry_string(&self) -> String {
        // Strip filename from cname
        let path = PathBuf::from(&self.cname);
        let cname = path.file_name().unwrap().to_str().unwrap();

        let mut string = String::from("");
        string += "<Geometry GeometryType=\"X_Y\">\n";
        string += &format!("<DataItem Dimensions=\"{:6}{:6}\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">{}:/{}</DataItem>\n",
            self.nx, self.ny, cname, self.xname);
        string += &format!("<DataItem Dimensions=\"{:6}{:6}\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">{}:/{}</DataItem>\n",
            self.nx, self.ny, cname, self.yname);
        string += "</Geometry>\n";
        string
    }

    /// Return string, which defines the data attributes
    fn _data_string(&self, aname: &str, vname: &str) -> String {
        // Strip filename from fname
        let path = PathBuf::from(&self.fname);
        let fname = path.file_name().unwrap().to_str().unwrap();

        let mut string = self._geometry_string();
        string += &format!(
            "<Attribute Name=\"{}\" AttributeType=\"Scalar\" Center=\"Node\">\n",
            aname
        );
        string += &format!("<DataItem Dimensions=\"{:6}{:6}\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">{}:/{}</DataItem>\n",
            self.nx, self.ny, fname, vname);
        string += "</Attribute>\n";
        string
    }

    /// Update filename
    pub fn update_fname(&mut self, fname: &str) {
        self.fname = String::from(fname);
    }

    /// Write xdfm file, belong to hdf5 file (filename)
    /// Supply vector of Data2D, which contains information
    /// about grid size, destionation, variable name, etc..
    pub fn write(&self) -> Result<(), Error> {
        // Create file
        let file = File::create(&self.xmfname)?;
        let mut file = LineWriter::new(file);

        // Write header
        file.write_all(b"<?xml version=\"1.0\" ?>\n")?;
        file.write_all(b"<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n")?;
        file.write_all(b"<Xdmf Version=\"2.0\">\n")?;
        file.write_all(b"<Domain>\n")?;

        let mut string = String::from("<Grid Name=\"Box\" GridType=\"Uniform\">\n");
        string += &format!(
            "<Topology TopologyType=\"3DSMesh\" NumberOfElements=\"{:6}{:6}\"/>\n",
            self.nx, self.ny
        );
        file.write_all(string.as_bytes())?;

        // Write datasets
        for (aname, vname) in self.aname.iter().zip(self.vname.iter()) {
            file.write_all(self._data_string(aname, vname).as_bytes())?;
        }

        // Write time
        let time = if let Some(x) = &self.time {
            format!("<Time Value=\" {:12.10}\" />\n", x)
        } else {
            String::from("<Time Value=\" 0.0\" />\n")
        };
        file.write_all(time.as_bytes())?;

        // Write closer
        file.write_all(b"</Grid>\n")?;
        file.write_all(b"</Domain>\n")?;
        file.write_all(b"</Xdmf>\n")?;
        Ok(())
    }
}
