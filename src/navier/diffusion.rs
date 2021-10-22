//! # Diffusion equation
#![allow(dead_code)]
use crate::field::{BaseSpace, Field1, Field2, WriteField};
use crate::solver::{Hholtz, Solve, SolverField};
use crate::Integrate;
use ndarray::{Array1, Array2};
//use num_complex::Complex;

/// Solve 1-dimensional diffusion equation.
///
/// Struct must be mutable, to perform the
/// update step, which advances the solution
/// by 1 timestep.
///
/// Fully implicit
///..math:
///    (1-kappa*dt*D2) unew = uold
///
///```
/// use rustpde::*;
/// use rustpde::navier::diffusion::Diffusion1D;
/// let mut field = Field1::new(&Space1::new(&cheb_dirichlet(7)));
/// let mut diff = Diffusion1D::new(field, 1.0, 0.1);
/// diff.impulse();
/// diff.update();
/// diff.callback();
///```
pub struct Diffusion1D<S> {
    field: Field1<f64, S>,
    solver: SolverField<f64, 1>,
    force: Option<Array1<f64>>,
    time: f64,
    dt: f64,
}

impl<S> Diffusion1D<S>
where
    S: BaseSpace<f64, 1, Physical = f64, Spectral = f64>,
{
    /// Create new instance
    pub fn new(field: Field1<f64, S>, kappa: f64, dt: f64) -> Self {
        let solver = SolverField::Hholtz(Hholtz::new(&field, [dt * kappa]));
        Self {
            field,
            solver,
            force: None,
            time: 0.0,
            dt,
        }
    }

    /// Apply impulse
    pub fn impulse(&mut self) {
        let n = self.field.v.shape()[0] as usize;
        self.field.v.assign(&Array1::zeros(n));
        self.field.v[n / 2] = 1.;
        self.field.forward();
        self.field.backward();
    }

    /// Add constant force
    /// ## Panics
    /// Panics when shapes of fields do not match.
    pub fn add_force(&mut self, force: &Array1<f64>) {
        assert!(force.len() == self.field.v.len());
        self.force = Some(force.to_owned());
    }
}

impl<S> Integrate for Diffusion1D<S>
where
    S: BaseSpace<f64, 1, Physical = f64, Spectral = f64>,
{
    fn update(&mut self) {
        // rhs: uold -> parent space
        self.field.v.assign(&self.field.to_ortho());
        // add forcing
        if let Some(x) = &self.force {
            self.field.v = &self.field.v + &(self.dt * x);
        }
        // lhs: update unew
        self.solver.solve(&self.field.v, &mut self.field.vhat, 0);
        // update time
        self.time += self.dt;
    }

    fn get_time(&self) -> f64 {
        self.time
    }

    fn get_dt(&self) -> f64 {
        self.dt
    }

    fn callback(&mut self) {
        std::fs::create_dir_all("data").unwrap();
        let fname = format!("data/diffusion1d_{:.*}.h5", 3, self.time);
        self.field.backward();
        self.field.write(&fname, None);
    }

    fn exit(&mut self) -> bool {
        false
    }
}

/// Solve 2-dimensional diffusion equation.
///
/// Struct must be mutable, to perform the
/// update step, which advances the solution
/// by 1 timestep.
///
/// Fully implicit
///.. math:
///    [(1-kappa*dt*D2x) + (1-kappa*dt*D2y)] unew = uold
///
///```
/// use rustpde::*;
/// use rustpde::navier::diffusion::Diffusion2D;
/// let space = Space2::new(&cheb_dirichlet(7), &cheb_dirichlet(7));
/// let mut field = Field2::new(&space);
/// let mut diff = Diffusion2D::new(field, 1.0, 0.1);
/// diff.impulse();
/// diff.update();
/// diff.callback();
/// integrate(&mut diff,0.1,None);
///```
pub struct Diffusion2D<S> {
    field: Field2<f64, S>,
    solver: SolverField<f64, 2>,
    rhs: Array2<f64>,
    force: Option<Array2<f64>>,
    fieldbc: Option<Field2<f64, S>>,
    kappa: f64,
    time: f64,
    dt: f64,
}

impl<S> Diffusion2D<S>
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = f64>,
{
    /// Return instance
    pub fn new(field: Field2<f64, S>, kappa: f64, dt: f64) -> Self {
        let solver = SolverField::Hholtz(Hholtz::new(&field, [dt * kappa, dt * kappa]));
        let rhs = Array2::zeros(field.v.raw_dim());
        Self {
            field,
            solver,
            rhs,
            force: None,
            fieldbc: None,
            kappa,
            time: 0.0,
            dt,
        }
    }

    /// Add impulse
    pub fn impulse(&mut self) {
        let n = self.field.v.shape()[0] as usize;
        let m = self.field.v.shape()[1] as usize;
        self.field.v.assign(&Array2::zeros((n, m)));
        self.field.v[[n / 2, m / 2]] = 1.;
        self.field.forward();
        self.field.backward();
    }

    /// Add external force
    /// ## Panics
    /// Panics when shapes of fields do not match.
    pub fn add_force(&mut self, force: &Array2<f64>) {
        assert!(force.shape() == self.field.v.shape());
        self.force = Some(force.to_owned());
    }

    /// Add force from inhomogeneous bc's
    /// ## Panics
    /// Panics when shapes of fields do not match.
    pub fn add_fieldbc(&mut self, fieldbc: Field2<f64, S>) {
        assert!(fieldbc.v.shape() == self.field.v.shape());
        self.fieldbc = Some(fieldbc);
    }
}

impl<S> Integrate for Diffusion2D<S>
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = f64>,
{
    fn update(&mut self) {
        // rhs: uold -> parent space
        for r in self.rhs.iter_mut() {
            *r = 0.;
        }
        self.rhs = self.field.to_ortho();

        // add forcing
        if let Some(x) = &self.force {
            self.rhs += &(self.dt * x);
        }
        // add fieldbc
        if let Some(x) = &self.fieldbc {
            self.rhs += &(self.dt * self.kappa * x.gradient([2, 0], None));
            self.rhs += &(self.dt * self.kappa * x.gradient([0, 2], None));
        }
        // lhs: update unew
        self.solver.solve(&self.rhs, &mut self.field.vhat, 0);
        // update time
        self.time += self.dt;
    }

    fn get_time(&self) -> f64 {
        self.time
    }

    fn get_dt(&self) -> f64 {
        self.dt
    }

    fn callback(&mut self) {
        std::fs::create_dir_all("data").unwrap();
        let fname = format!("data/diffusion2d_{:.*}.h5", 3, self.time);
        self.field.backward();
        // Add boundary contribution
        if let Some(x) = &self.fieldbc {
            self.field.v = &self.field.v + &x.v;
        }
        self.field.write(&fname, None);
        // Undo addition of bc
        if self.fieldbc.is_some() {
            self.field.backward();
        }
    }

    fn exit(&mut self) -> bool {
        false
    }
}
