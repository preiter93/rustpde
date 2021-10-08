use hdf5_interface::read_from_hdf5;
use ndarray::prelude::*;
use particle_tracer::Particle;
use rand::{self, Rng};
use std::path::Path;
use std::path::PathBuf;

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

fn particle_swarm(x0: f64, y0: f64, range: f64, n: usize) -> Vec<(f64, f64)> {
    let mut rng = rand::thread_rng();
    let mut swarm: Vec<(f64, f64)> = vec![];
    for _ in 0..n {
        let x = x0 + rng.gen_range(-range..range);
        let y = y0 + rng.gen_range(-range..range);
        swarm.push((x, y))
    }
    swarm
}

#[allow(dead_code)]
fn main() {
    // List files
    let mut files = list_of_files_of_type("data/", "h5");
    files.sort();
    //files.sort_by_key(|dir| &dir.path());
    let x0 = 0.7;
    let y0 = -0.7;
    let dt = 0.1;
    let swarm = particle_swarm(x0, y0, 0.25, 5000);
    let mut particles: Vec<Particle> = vec![];
    // init
    let fname = &files[0].as_path().display().to_string();
    let x: Array1<f64> = read_from_hdf5(&fname, "x", None).unwrap();
    let y: Array1<f64> = read_from_hdf5(&fname, "y", None).unwrap();
    for pos in swarm {
        let mut particle =
            Particle::init(pos.0, pos.1, x.as_slice().unwrap(), y.as_slice().unwrap());
        // particle.set_intervall(None);
        particle.set_timestep(0.001);
        particles.push(particle);
    }
    // iterate over flow fields
    for file in files {
        let fname = &file.into_os_string().into_string().unwrap();
        println!("Process {:?} ...", fname);
        let ux: Array2<f64> = read_from_hdf5(&fname, "v", Some("ux")).unwrap();
        let uy: Array2<f64> = read_from_hdf5(&fname, "v", Some("uy")).unwrap();
        // iterate over particles
        for particle in &mut particles {
            let t_end = particle.time + dt;
            loop {
                particle.update_rk4(&[&ux.view(), &uy.view()]).unwrap();
                if particle.time >= t_end {
                    break;
                }
            }
            particle.history.clear();
            particle.push();
            particle
                .write(&fname.replace(".h5", "_trajectory.txt"))
                .unwrap();
            particle.history.clear();
        }
    }
}
