//! # Diffusion equation
#![allow(dead_code)]
use crate::solver::{Hholtz, Solve, SolverField};
use crate::Integrate;
use crate::{Field, Field1, Field1Complex, Field2, WriteField};
use ndarray::{Array1, Array2};
use num_complex::Complex;

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
/// use rustpde::examples::diffusion::Diffusion1D;
/// let mut field = Field1::new(&[cheb_dirichlet(7)]);
/// let mut diff = Diffusion1D::new(field, 1.0, 0.1);
/// diff.impulse();
/// diff.update();
/// diff.write();
///```
pub struct Diffusion1D {
    field: Field1,
    solver: SolverField<f64, 1>,
    force: Option<Array1<f64>>,
    time: f64,
    dt: f64,
}

impl Diffusion1D {
    /// Create new instance
    pub fn new(field: Field1, kappa: f64, dt: f64) -> Self {
        let solver = SolverField::Hholtz(Hholtz::from_space(&field.space, [dt * kappa]));
        Diffusion1D {
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

impl Integrate for Diffusion1D {
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

    fn write(&mut self) {
        std::fs::create_dir_all("data").unwrap();
        let fname = format!("data/diffusion1d_{:.*}.h5", 3, self.time);
        self.field.backward();
        self.field.write(&fname, None);
    }

    fn exit(&mut self) -> bool {
        false
    }
}

/// Diffusion equation for Complex spectral space
///```
/// use rustpde::*;
/// use rustpde::examples::diffusion::Diffusion1DComplex;
/// let mut field = Field1Complex::new(&[fourier_r2c(7)]);
/// let mut diff = Diffusion1DComplex::new(field, 1.0, 0.1);
/// diff.impulse();
/// diff.update();
/// diff.write();
///```
pub struct Diffusion1DComplex {
    field: Field1Complex,
    solver: SolverField<f64, 1>,
    force: Option<Array1<Complex<f64>>>,
    time: f64,
    dt: f64,
    rhs: Array1<Complex<f64>>,
}

impl Diffusion1DComplex {
    /// Create new instance
    pub fn new(field: Field1Complex, kappa: f64, dt: f64) -> Self {
        let solver = SolverField::Hholtz(Hholtz::from_space(&field.space, [dt * kappa]));
        let rhs = Array1::<Complex<f64>>::zeros(field.vhat.raw_dim());
        Self {
            field,
            solver,
            force: None,
            time: 0.0,
            dt,
            rhs,
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
    pub fn add_force(&mut self, force: &Array1<Complex<f64>>) {
        assert!(force.len() == self.field.v.len());
        self.force = Some(force.to_owned());
    }
}

impl Integrate for Diffusion1DComplex {
    fn update(&mut self) {
        // rhs: uold -> parent space
        self.rhs.assign(&self.field.to_ortho());
        // add forcing
        if let Some(x) = &self.force {
            self.rhs = &self.rhs + &(x * self.dt);
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

    fn write(&mut self) {
        std::fs::create_dir_all("data").unwrap();
        let fname = format!("data/diffusion1d_complex_{:.*}.h5", 3, self.time);
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
/// use rustpde::examples::diffusion::Diffusion2D;
/// let bases = [cheb_dirichlet(7), cheb_dirichlet(7)];
/// let mut field = Field2::new(&bases);
/// let mut diff = Diffusion2D::new(field, 1.0, 0.1);
/// diff.impulse();
/// diff.update();
/// diff.write();
/// integrate(&mut diff,0.1,None);
///```
pub struct Diffusion2D {
    field: Field2,
    solver: SolverField<f64, 2>,
    rhs: Array2<f64>,
    force: Option<Array2<f64>>,
    fieldbc: Option<Field2>,
    kappa: f64,
    time: f64,
    dt: f64,
}

impl Diffusion2D {
    /// Return instance
    pub fn new(field: Field2, kappa: f64, dt: f64) -> Self {
        let solver =
            SolverField::Hholtz(Hholtz::from_space(&field.space, [dt * kappa, dt * kappa]));
        let rhs = Array2::zeros(field.v.raw_dim());
        Diffusion2D {
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
    pub fn add_fieldbc(&mut self, fieldbc: Field2) {
        assert!(fieldbc.v.shape() == self.field.v.shape());
        self.fieldbc = Some(fieldbc);
    }
}

impl Integrate for Diffusion2D {
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
            self.rhs += &(self.dt * self.kappa * x.grad([2, 0], None));
            self.rhs += &(self.dt * self.kappa * x.grad([0, 2], None));
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

    fn write(&mut self) {
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
