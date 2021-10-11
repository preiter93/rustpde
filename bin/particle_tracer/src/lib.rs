//! Calculate trajectories of passive particles for Lagrangian statistics
//! and vizualization.
//!
//! # Example
//! Circular velcocity field
//! ```ignore
//! use ndarray::prelude::*;
//! use particle_tracer::Particle;
//! fn main() {
//!     let (nx, ny) = (51, 51);
//!     let x = Array1::linspace(-1., 1., nx);
//!     let y = Array1::linspace(-1., 1., nx);
//!     let mut ux: Array2<f64> = Array2::zeros((nx, ny));
//!     let mut uy: Array2<f64> = Array2::zeros((nx, ny));
//!
//!     // Circular veclocity field
//!     for (i, xi) in x.iter().enumerate() {
//!         for (j, yi) in y.iter().enumerate() {
//!             ux[[i,j]] = -yi;
//!             uy[[i,j]] = *xi;
//!         }
//!     }
//!
//!     let mut particle = Particle::init(0.2, 0., x.as_slice().unwrap(), y.as_slice().unwrap());
//!     particle.set_intervall(0.1);
//!     particle.set_timestep(0.001);
//!     loop {
//!         particle.update(&[&ux.view(), &uy.view()]).unwrap();
//!         if particle.time > 10.0 {
//!             break;
//!         }
//!     }
//!     particle.write("test_trajectory.txt").unwrap();
//! }
//! ```
#![allow(dead_code)]
extern crate hdf5_interface;
pub use hdf5_interface::read_from_hdf5;
pub use hdf5_interface::read_scalar_from_hdf5;
pub use hdf5_interface::write_scalar_to_hdf5;
pub use hdf5_interface::write_to_hdf5;
pub use hdf5_interface::Result as Hdf5Result;
use ndarray::prelude::*;
use rand::{self, Rng};
use std::fmt;

type Result<T> = std::result::Result<T, TracerError>;

// Define own error tpyes
#[derive(Debug, Clone)]
pub struct TracerError;

// Generation of an error is completely separate from how it is displayed.
// There's no need to be concerned about cluttering complex logic with the display style.
impl fmt::Display for TracerError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "particle went out of bounds.")
    }
}

pub struct ParticleSwarm<'a> {
    /// Array of Particles
    pub particles: Vec<Particle<'a>>,
    /// Reference to x-coordinate
    pub x: &'a [f64],
    /// Reference to y-coordinate
    pub y: &'a [f64],
    /// Timestep size
    pub timestep: f64,
    /// Current time
    pub time: f64,
}

impl<'a> ParticleSwarm<'a> {
    /// Initialize new Particle
    #[must_use]
    pub fn init(position: Vec<(f64, f64)>, x: &'a [f64], y: &'a [f64], timestep: f64) -> Self {
        let mut particles: Vec<Particle> = vec![];
        for pos in position {
            particles.push(Particle::init(pos.0, pos.1, x, y, timestep));
        }
        Self {
            particles,
            x,
            y,
            timestep,
            time: 0.,
        }
    }

    /// Initialize rectangular particle swarm
    #[must_use]
    pub fn from_rectangle(
        x0: f64,
        y0: f64,
        range: f64,
        n: usize,
        x: &'a [f64],
        y: &'a [f64],
        timestep: f64,
    ) -> Self {
        let mut rng = rand::thread_rng();
        let mut swarm: Vec<(f64, f64)> = vec![];
        for _ in 0..n {
            let x = x0 + rng.gen_range(-range..range);
            let y = y0 + rng.gen_range(-range..range);
            swarm.push((x, y))
        }
        Self::init(swarm, x, y, timestep)
    }

    /// Read particle coordinates from file
    ///
    /// # Panics
    /// Unable to read file
    #[must_use]
    pub fn from_file(fname: &str, x: &'a [f64], y: &'a [f64], timestep: f64) -> Self {
        use std::fs::File;
        use std::io::{BufRead, BufReader};
        // Open the file in read-only mode (ignoring errors).
        let file = File::open(fname).unwrap();
        let reader = BufReader::new(file);
        // Read the file
        let mut swarm: Vec<(f64, f64)> = vec![];
        for line in reader.lines() {
            let line = line.unwrap(); // Ignore errors.
            let particle: Vec<f64> = line
                .split(' ')
                .map(|x| x.parse().expect("Not a float!"))
                .collect();
            let x_pos = particle[1];
            let y_pos = particle[2];
            swarm.push((x_pos, y_pos));
        }
        Self::init(swarm, x, y, timestep)
    }

    /// Update position using rk4
    pub fn update(&mut self, u: &[&ArrayView2<f64>]) {
        for particle in &mut self.particles {
            // Ignore the error
            let _ = particle.update_rk4(u);
        }
        self.time += self.timestep;
    }

    /// Write to file
    ///
    /// # Errors
    /// Unable to create file
    #[allow(clippy::write_with_newline)]
    pub fn write(&mut self, fname: &str) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::Write;
        let mut f = File::create(fname).expect("Unable to create file");
        for particle in &mut self.particles {
            write!(
                f,
                "{} {} {}\n",
                particle.time, particle.x_pos, particle.y_pos
            )?;
        }
        Ok(())
    }
}

pub struct Particle<'a> {
    /// Current x-position
    pub x_pos: f64,
    /// Current y-position
    pub y_pos: f64,
    /// Current time
    pub time: f64,
    /// Timestep size
    pub timestep: f64,
    /// Reference to x-coordinate
    pub x: &'a [f64],
    /// Reference to y-coordinate
    pub y: &'a [f64],
    /// History of coordinates (Time | x_pos | y_pos)
    pub history: Vec<(f64, f64, f64)>,
    /// Save every x time units. If none, save never.
    pub save_intervall: Option<f64>,
}

impl<'a> Particle<'a> {
    /// Initialize new Particle
    #[must_use]
    pub fn init(x_pos: f64, y_pos: f64, x: &'a [f64], y: &'a [f64], timestep: f64) -> Self {
        let time = 0.;
        Self {
            x_pos,
            y_pos,
            time,
            timestep,
            x,
            y,
            history: Vec::new(),
            save_intervall: None,
        }
    }

    /// Set intervall for saving trajectory ( default is every timestep )
    pub fn set_save_intervall(&mut self, save_intervall: f64) {
        self.save_intervall = Some(save_intervall);
    }

    /// Set timestep size ( defaul = 0.1 )
    pub fn set_timestep(&mut self, timestep: f64) {
        self.timestep = timestep;
    }

    /// Update position using euler method
    ///
    /// # Errors
    /// Errors when particle goes out of bounce
    pub fn update(&mut self, u: &[&ArrayView2<f64>]) -> Result<()> {
        // Get interpolated velocity
        let u_i = self.bilinear_interpolation(u, self.x_pos, self.y_pos)?;
        // Update position (euler step)
        self.x_pos += self.timestep * u_i[0];
        self.y_pos += self.timestep * u_i[1];
        self.time += self.timestep;
        // Save history
        self.push();
        Ok(())
    }

    /// Update position using runge kutta 2
    ///
    /// # Errors
    /// Errors when particle goes out of bounce
    ///
    /// # References
    /// <http://web.cse.ohio-state.edu/~shen.94/788/Site/Slides_files/vectorViz.pdf>
    #[allow(clippy::similar_names)]
    pub fn update_rk2(&mut self, u: &[&ArrayView2<f64>]) -> Result<()> {
        // Step 1
        let u1 = self.bilinear_interpolation(u, self.x_pos, self.y_pos)?;
        let k_x1 = self.timestep * u1[0];
        let k_y1 = self.timestep * u1[1];
        // Step 2
        let u2 = self.bilinear_interpolation(u, self.x_pos + k_x1, self.y_pos + k_y1)?;
        let k_x2 = self.timestep * u2[0];
        let k_y2 = self.timestep * u2[1];
        // Update position (euler step)
        self.x_pos += 0.5 * (k_x1 + k_x2);
        self.y_pos += 0.5 * (k_y1 + k_y2);
        self.time += self.timestep;
        // Save history
        self.push();
        Ok(())
    }

    /// Update position using runge kutta 4
    ///
    /// # Errors
    /// Errors when particle goes out of bounce
    ///
    /// # References
    /// <http://web.cse.ohio-state.edu/~shen.94/788/Site/Slides_files/vectorViz.pdf>
    pub fn update_rk4(&mut self, u: &[&ArrayView2<f64>]) -> Result<()> {
        // Step 1
        let k1 = self.bilinear_interpolation(u, self.x_pos, self.y_pos)?;
        // Step 2
        let mut x_pos_new = self.x_pos + self.timestep / 2. * k1[0];
        let mut y_pos_new = self.y_pos + self.timestep / 2. * k1[1];
        let k2 = self.bilinear_interpolation(u, x_pos_new, y_pos_new)?;
        // Step 3
        x_pos_new = self.x_pos + self.timestep / 2. * k2[0];
        y_pos_new = self.y_pos + self.timestep / 2. * k2[1];
        let k3 = self.bilinear_interpolation(u, x_pos_new, y_pos_new)?;
        // Step 4
        x_pos_new = self.x_pos + self.timestep * k3[0];
        y_pos_new = self.y_pos + self.timestep * k3[1];
        let k4 = self.bilinear_interpolation(u, x_pos_new, y_pos_new)?;
        // Update position (euler step)
        self.x_pos += self.timestep / 6. * (k1[0] + 2. * k2[0] + 2. * k3[0] + k4[0]);
        self.y_pos += self.timestep / 6. * (k1[1] + 2. * k2[1] + 2. * k3[1] + k4[1]);
        self.time += self.timestep;
        // Save history
        self.push();
        Ok(())
    }

    /// Bilinear interpolation
    ///
    /// # Errors
    /// Errors when particle is out of bounce
    pub fn bilinear_interpolation(
        &self,
        fields: &[&ArrayView2<f64>],
        x_pos: f64,
        y_pos: f64,
    ) -> Result<Vec<f64>> {
        // Get neareast points
        let (x_low, x_upp, y_low, y_upp) = self.find_nearest_index(x_pos, y_pos)?;
        // Bilinear weight
        let dxdy = (self.x[x_upp] - self.x[x_low]) * (self.y[y_upp] - self.y[y_low]);
        let w1 = (self.x[x_upp] - self.x_pos) * (self.y[y_upp] - self.y_pos) / dxdy;
        let w2 = (self.x[x_upp] - self.x_pos) * (self.y_pos - self.y[y_low]) / dxdy;
        let w3 = (self.x_pos - self.x[x_low]) * (self.y[y_upp] - self.y_pos) / dxdy;
        let w4 = (self.x_pos - self.x[x_low]) * (self.y_pos - self.y[y_low]) / dxdy;
        Ok(fields
            .iter()
            .map(|f| {
                w1 * f[[x_low, y_low]]
                    + w2 * f[[x_low, y_upp]]
                    + w3 * f[[x_upp, y_low]]
                    + w4 * f[[x_upp, y_upp]]
            })
            .collect())
    }

    /// Save history
    pub fn push(&mut self) {
        if let Some(x) = self.save_intervall {
            if (self.time % x) < self.timestep / 2. || (self.time % x) > x - self.timestep / 2. {
                self.history.push((self.time, self.x_pos, self.y_pos));
            }
        } //else {
          //    self.history.push((self.time, self.x_pos, self.y_pos));
          //}
    }

    /// Write to file
    ///
    /// # Errors
    /// Unable to create file
    #[allow(clippy::write_with_newline)]
    pub fn write(&mut self, fname: &str) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::Write;
        let mut f = File::create(fname).expect("Unable to create file");
        for (t, x, y) in &self.history {
            write!(f, "{} {} {}\n", t, x, y)?;
        }
        Ok(())
    }

    // Reset particle
    pub fn reset(&mut self, x_pos: f64, y_pos: f64) {
        self.x_pos = x_pos;
        self.y_pos = y_pos;
        self.time = 0.;
        self.history = Vec::new();
    }

    /// Return index of nearest grid points (xlow, xupp, ylow, yupp)
    #[allow(clippy::shadow_unrelated)]
    fn find_nearest_index(&self, x_pos: f64, y_pos: f64) -> Result<(usize, usize, usize, usize)> {
        self.check_bounds()?;
        let first_index_above = self.x.iter().position(|&x| x > x_pos).unwrap();
        let x_low = first_index_above - 1;
        let x_upp = first_index_above;
        let first_index_above = self.y.iter().position(|&x| x > y_pos).unwrap();
        let y_low = first_index_above - 1;
        let y_upp = first_index_above;
        Ok((x_low, x_upp, y_low, y_upp))
    }

    /// Check if position of particle is not out or bounce
    fn check_bounds(&self) -> Result<()> {
        if self.x_pos > self.x[0]
            || self.x_pos < self.x[&self.x.len() - 1]
            || self.y_pos > self.y[0]
            || self.y_pos < self.y[&self.y.len() - 1]
        {
            Ok(())
        } else {
            Err(TracerError)
        }
    }
}
