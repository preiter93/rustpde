use hdf5_interface::read_from_hdf5;
use ndarray::prelude::*;
use particle_tracer::Particle;
use std::path::Path;
use std::path::PathBuf;

fn main() {
    let (nx, ny) = (51, 51);
    let x = Array1::linspace(-1., 1., nx);
    let y = Array1::linspace(-1., 1., nx);
    let mut ux: Array2<f64> = Array2::zeros((nx, ny));
    let mut uy: Array2<f64> = Array2::zeros((nx, ny));

    // Circular veclocity field
    for (i, xi) in x.iter().enumerate() {
        for (j, yi) in y.iter().enumerate() {
            ux[[i, j]] = -yi;
            uy[[i, j]] = *xi;
        }
    }

    let mut particle = Particle::init(0.2, 0., x.as_slice().unwrap(), y.as_slice().unwrap());
    particle.set_intervall(0.1);
    particle.set_timestep(0.1);
    loop {
        particle.update(&[&ux.view(), &uy.view()]).unwrap();
        if particle.time > 10.0 {
            break;
        }
    }
    particle.write("test_trajectory.txt").unwrap();

    let mut particle = Particle::init(0.2, 0., x.as_slice().unwrap(), y.as_slice().unwrap());
    particle.set_intervall(0.1);
    particle.set_timestep(0.1);
    loop {
        particle.update_rk2(&[&ux.view(), &uy.view()]).unwrap();
        if particle.time > 10.0 {
            break;
        }
    }
    particle.write("test_trajectory2.txt").unwrap();
}

/// Return list of files in root with specific ending
fn list_of_files_of_type<P: AsRef<Path>>(root: P, ending: &str) -> Vec<PathBuf> {
    std::fs::read_dir(root)
        .unwrap()
        .into_iter()
        .filter(std::result::Result::is_ok)
        .map(|r| r.unwrap().path())
        .filter(|r| r.extension() == Some(std::ffi::OsStr::new(ending)))
        .collect()
}

#[allow(dead_code)]
fn loop_through_files() {
    // List files
    let files = list_of_files_of_type("roll1/", "h5");
    let (x0, y0) = (0.95, 0.0);
    for file in files {
        let fname = &file.into_os_string().into_string().unwrap();
        println!("Process {:?} ...", fname);
        let ux: Array2<f64> = read_from_hdf5(&fname, "v", Some("ux")).unwrap();
        let uy: Array2<f64> = read_from_hdf5(&fname, "v", Some("uy")).unwrap();
        let x: Array1<f64> = read_from_hdf5(&fname, "x", None).unwrap();
        let y: Array1<f64> = read_from_hdf5(&fname, "y", None).unwrap();
        let mut particle = Particle::init(x0, y0, x.as_slice().unwrap(), y.as_slice().unwrap());
        particle.set_intervall(0.5);
        particle.set_timestep(0.001);
        particle.push();
        loop {
            particle.update_rk4(&[&ux.view(), &uy.view()]).unwrap();
            if particle.time > 1000. {
                break;
            }
        }
        particle.write(&fname.replace(".h5", "_trajectory.txt")).unwrap();
    }
}


#[allow(dead_code)]
fn single_file() {
    println!("Hello, world!");
    let fname = "data/test.h5";
    let ux: Array2<f64> = read_from_hdf5(&fname, "v", Some("ux")).unwrap();
    let uy: Array2<f64> = read_from_hdf5(&fname, "v", Some("uy")).unwrap();
    let x: Array1<f64> = read_from_hdf5(&fname, "x", None).unwrap();
    let y: Array1<f64> = read_from_hdf5(&fname, "y", None).unwrap();

    let (x0, y0) = (0.0, 0.001);
    let mut particle = Particle::init(x0, y0, x.as_slice().unwrap(), y.as_slice().unwrap());
    particle.set_intervall(0.5);
    particle.set_timestep(0.0001);
    loop {
        particle.update_rk4(&[&ux.view(), &uy.view()]).unwrap();
        if particle.time > 1000. {
            break;
        }
    }
    particle.write(&fname.replace(".h5", "_trajectory.txt")).unwrap();
}
