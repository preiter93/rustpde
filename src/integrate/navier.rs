//! Solve 2-dimensional Navier-Stokes equations
//! coupled with temperature equations
use super::conv_term;
use super::Integrate;
use crate::bases::{cheb_dirichlet, cheb_dirichlet_bc, cheb_neumann, chebyshev};
use crate::hdf5::{read_from_hdf5, write_to_hdf5, Hdf5};
use crate::solver::{Hholtz, Poisson, Solve, SolverField};
use crate::Field2;
use crate::Space2;
use ndarray::{array, s, Array2, Zip};

/// Return viscosity from Ra, Pr, and height of the cell
pub fn get_nu(ra: &f64, pr: &f64, height: &f64) -> f64 {
    let f = pr / (ra / height.powf(3.0));
    f.sqrt()
}

/// Return diffusivity from Ra, Pr, and height of the cell
pub fn get_ka(ra: &f64, pr: &f64, height: &f64) -> f64 {
    let f = 1. / ((ra / height.powf(3.0)) * pr);
    f.sqrt()
}

/// Solve 2-dimensional Navier-Stokes equations
/// coupled with temperature equations
///
/// Bases: Chebyshev in x & y
///
/// Struct must be mutable, to perform the
/// update step, which advances the solution
/// by 1 timestep.
///
/// # Arguments
///
/// * `nx,ny` - The number of modes in x and y -direction
///
/// * `ra,pr` - Rayleigh and Prandtl number
///
/// * `dt` - Timestep size
///
/// * `adiabatic` - Boolean, sidewall temperature boundary condition
///
/// * `aspect` - Aspect ratio L/H
///
/// # Examples
///
/// ```
/// use rustpde::integrate::{integrate, Integrate, Navier2D};
/// let (nx, ny) = (33, 33);
/// let ra = 1e5;
/// let pr = 1.;
/// let adiabatic = true;
/// let aspect = 1.0;
/// let dt = 0.01;
/// let mut navier = Navier2D::new(nx, ny, ra, pr, dt, adiabatic, aspect);
/// // Read initial field from file
/// // navier.read("data/flow0.000.h5");
/// integrate(navier, 0.2,  None);
/// ```
pub struct Navier2D {
    field: Field2,
    temp: Field2,
    ux: Field2,
    uy: Field2,
    pres: [Field2; 2],
    solver: [SolverField<f64, 2>; 4],
    rhs: Array2<f64>,
    fieldbc: Option<Field2>,
    nu: f64,
    ka: f64,
    ra: f64,
    pr: f64,
    time: f64,
    dt: f64,
    scale: [f64; 2],
}

impl Navier2D {
    /// Returns Navier-Stokes Solver, an integrable type, used with pde::integrate
    pub fn new(
        nx: usize,
        ny: usize,
        ra: f64,
        pr: f64,
        dt: f64,
        adiabatic: bool,
        aspect: f64,
    ) -> Self {
        let scale = [aspect, 1.];
        let nu = get_nu(&ra, &pr, &(scale[1] * 2.0));
        let ka = get_ka(&ra, &pr, &(scale[1] * 2.0));
        let ux = Field2::new(Space2::new([cheb_dirichlet(nx), cheb_dirichlet(ny)]));
        let uy = Field2::new(Space2::new([cheb_dirichlet(nx), cheb_dirichlet(ny)]));
        let temp = if adiabatic {
            Field2::new(Space2::new([cheb_neumann(nx), cheb_dirichlet(ny)]))
        } else {
            Field2::new(Space2::new([cheb_dirichlet(nx), cheb_dirichlet(ny)]))
        };
        Self::from_fields(temp, ux, uy, nu, ka, ra, pr, dt, scale)
    }

    #[allow(clippy::too_many_arguments)]
    fn from_fields(
        temp: Field2,
        ux: Field2,
        uy: Field2,
        nu: f64,
        ka: f64,
        ra: f64,
        pr: f64,
        dt: f64,
        scale: [f64; 2],
    ) -> Self {
        // define additional fields
        let nx = temp.v.shape()[0];
        let ny = temp.v.shape()[1];
        let pres = Field2::new(Space2::new([chebyshev(nx), chebyshev(ny)]));
        let pseudo = Field2::new(Space2::new([cheb_neumann(nx), cheb_neumann(ny)]));
        let field = Field2::new(Space2::new([chebyshev(nx), chebyshev(ny)]));
        // define solver
        let solver_ux = SolverField::Hholtz(Hholtz::from_field(
            &ux,
            [dt * nu / scale[0].powf(2.), dt * nu / scale[1].powf(2.)],
        ));
        let solver_uy = SolverField::Hholtz(Hholtz::from_field(
            &uy,
            [dt * nu / scale[0].powf(2.), dt * nu / scale[1].powf(2.)],
        ));
        let solver_temp = SolverField::Hholtz(Hholtz::from_field(
            &temp,
            [dt * ka / scale[0].powf(2.), dt * ka / scale[1].powf(2.)],
        ));
        let solver_pres = SolverField::Poisson(Poisson::from_field(
            &pseudo,
            [1. / scale[0].powf(2.), 1. / scale[1].powf(2.)],
        ));
        let solver = [solver_ux, solver_uy, solver_temp, solver_pres];
        let rhs = Array2::zeros(temp.v.raw_dim());
        let mut navier = Navier2D {
            field,
            temp,
            ux,
            uy,
            pres: [pres, pseudo],
            solver,
            rhs,
            fieldbc: None,
            nu,
            ka,
            ra,
            pr,
            time: 0.0,
            dt,
            scale,
        };
        navier._scale();
        navier._rbc();
        //apply_sin_cos(&mut navier.temp, 0.2, 1., 1.);
        apply_sin_cos(&mut navier.ux, 0.2, 2., 1.);
        apply_cos_sin(&mut navier.uy, -0.2, 2., 1.);
        // Return
        navier
    }

    /// Rescale x & y coordinates of fields.
    /// Only affects output of files
    fn _scale(&mut self) {
        for field in [
            &mut self.temp,
            &mut self.ux,
            &mut self.uy,
            &mut self.pres[0],
        ]
        .iter_mut()
        {
            field.x[0] *= self.scale[0];
            field.x[1] *= self.scale[1];
        }
    }

    /// Add field_bc to self, which defines the
    /// inhomogeneous temperature boundary conditions.
    ///
    /// Specifically, add Rayleigh Benard Boundary
    /// Conditions with 0.5 at the bottom and 0.5
    /// at the top
    fn _rbc(&mut self) {
        //use crate::space::Spaced;
        //use crate::bases::Chebyshev;
        use crate::bases::Transform;
        // Apply boundary conditions
        let nx = self.temp.v.shape()[0];
        let ny = self.temp.v.shape()[1];
        let bases = [chebyshev(nx), cheb_dirichlet_bc(ny)];
        let mut fieldbc = Field2::new(Space2::new(bases));
        let mut bases = [chebyshev(nx), cheb_dirichlet_bc(ny)];

        let mut bc = fieldbc.vhat.to_owned();
        // bottom
        Zip::from(&mut bc.slice_mut(s![.., 0]))
            .and(&fieldbc.x[0])
            .for_each(|b, &_| {
                *b = 0.5; //(PI*x).cos();
            });
        // top
        Zip::from(&mut bc.slice_mut(s![.., 1]))
            .and(&fieldbc.x[0])
            .for_each(|b, &_| {
                *b = -0.5;
            });

        //let mut base = Chebyshev::new(nx);
        bases[0].forward_inplace(&mut bc, &mut fieldbc.vhat, 0);
        fieldbc.backward();
        fieldbc.forward();
        // Set fieldbc
        self.fieldbc = Some(fieldbc);
    }

    fn zero_rhs(&mut self) {
        for r in self.rhs.iter_mut() {
            *r = 0.;
        }
    }

    /// Convection term for temperature
    fn conv_temp(&mut self, ux: &Array2<f64>, uy: &Array2<f64>) -> Array2<f64> {
        // + ux * dTdx + uy * dTdy
        let mut conv = conv_term(&self.temp, &mut self.field, ux, [1, 0], Some(self.scale));
        conv += &conv_term(&self.temp, &mut self.field, uy, [0, 1], Some(self.scale));
        // + bc contribution
        if let Some(field) = &self.fieldbc {
            conv += &conv_term(field, &mut self.field, ux, [1, 0], Some(self.scale));
            conv += &conv_term(field, &mut self.field, uy, [0, 1], Some(self.scale));
        }
        // -> spectral space
        self.field.v.assign(&conv);
        self.field.forward();
        self.field.vhat.to_owned()
    }

    /// Convection term for ux
    fn conv_ux(&mut self, ux: &Array2<f64>, uy: &Array2<f64>) -> Array2<f64> {
        // + ux * dudx + uy * dudy
        let mut conv = conv_term(&self.ux, &mut self.field, ux, [1, 0], Some(self.scale));
        conv += &conv_term(&self.ux, &mut self.field, uy, [0, 1], Some(self.scale));
        // -> spectral space
        self.field.v.assign(&conv);
        self.field.forward();
        self.field.vhat.to_owned()
    }

    /// Convection term for uy
    fn conv_uy(&mut self, ux: &Array2<f64>, uy: &Array2<f64>) -> Array2<f64> {
        // + ux * dudx + uy * dudy
        let mut conv = conv_term(&self.uy, &mut self.field, ux, [1, 0], Some(self.scale));
        conv += &conv_term(&self.uy, &mut self.field, uy, [0, 1], Some(self.scale));
        // -> spectral space
        self.field.v.assign(&conv);
        self.field.forward();
        self.field.vhat.to_owned()
    }

    /// Solve horizontal momentum equation
    ///
    /// (1 - dt*D) u_new = -dt*C(u) - dt*grad(p) + dt*f + u
    fn solve_ux(&mut self, ux: &Array2<f64>, uy: &Array2<f64>) {
        self.zero_rhs();
        // + old field
        self.rhs += &self.ux.to_parent();
        // + pres
        self.rhs -= &(self.dt * self.pres[0].grad([1, 0], Some(self.scale)));
        // + convection
        let conv = self.conv_ux(ux, uy);
        self.rhs -= &(self.dt * conv);
        // solve lhs
        //self.ux.vhat.assign(&self.solver[0].solve(&self.rhs));
        self.solver[0].solve(&self.rhs, &mut self.ux.vhat, 0);
    }

    /// Solve vertical momentum equation
    fn solve_uy(&mut self, ux: &Array2<f64>, uy: &Array2<f64>, buoy: &Array2<f64>) {
        self.zero_rhs();
        // + old field
        self.rhs += &self.uy.to_parent();
        // + pres
        self.rhs -= &(self.dt * self.pres[0].grad([0, 1], Some(self.scale)));
        // + buoyancy
        self.rhs += &(self.dt * buoy);
        // + convection
        let conv = self.conv_uy(ux, uy);
        self.rhs -= &(self.dt * conv);
        // solve lhs
        //self.uy.vhat.assign(&self.solver[1].solve(&self.rhs));
        self.solver[1].solve(&self.rhs, &mut self.uy.vhat, 0);
    }

    /// Divergence: duxdx + duydy
    fn divergence(&mut self) -> Array2<f64> {
        self.zero_rhs();
        self.rhs += &self.ux.grad([1, 0], Some(self.scale));
        self.rhs += &self.uy.grad([0, 1], Some(self.scale));
        self.rhs.to_owned()
    }

    /// Correct velocity field.
    ///
    /// uxnew = ux - c*dpdx
    ///
    /// uynew = uy - c*dpdy
    fn project_velocity(&mut self, c: f64) {
        let dpdx = self.pres[1].grad([1, 0], Some(self.scale));
        let dpdy = self.pres[1].grad([0, 1], Some(self.scale));

        // self.ux.vhat -= &(c * self.ux.from_parent(&dpdx));
        // self.uy.vhat -= &(c * self.uy.from_parent(&dpdy));

        let ux_old = self.ux.vhat.clone();
        let uy_old = self.uy.vhat.clone();
        self.ux.from_parent(&dpdx);
        self.uy.from_parent(&dpdy);
        self.ux.vhat *= -c;
        self.uy.vhat *= -c;
        self.ux.vhat += &ux_old;
        self.uy.vhat += &uy_old;
    }

    /// Solve temperature equation:
    ///
    /// (1 - dt*D) temp_new = -dt*C(temp) + dt*f_bc + temp
    fn solve_temp(&mut self, ux: &Array2<f64>, uy: &Array2<f64>) {
        self.zero_rhs();
        // + old field
        self.rhs += &self.temp.to_parent();
        // + diffusion bc contribution
        if let Some(field) = &self.fieldbc {
            self.rhs += &(self.dt * self.ka * field.grad([2, 0], Some(self.scale)));
            self.rhs += &(self.dt * self.ka * field.grad([0, 2], Some(self.scale)));
        }
        // + convection
        let conv = self.conv_temp(ux, uy);
        self.rhs -= &(self.dt * conv);
        // solve lhs
        // self.temp.vhat.assign(&self.solver[2].solve(&self.rhs));
        self.solver[2].solve(&self.rhs, &mut self.temp.vhat, 0);
    }

    /// Solve pressure poisson equation
    ///
    /// D2 pres = f
    ///
    /// pseu: pseudo pressure ( in code it is pres\[1\] )
    fn solve_pres(&mut self, f: &Array2<f64>) {
        //self.pres[1].vhat.assign(&self.solver[3].solve(&f));
        self.solver[3].solve(&f, &mut self.pres[1].vhat, 0);
        // Singularity
        self.pres[1].vhat[[0, 0]] = 0.;
    }

    fn update_pres(&mut self, div: &Array2<f64>) {
        self.pres[0].vhat -= &(self.nu * div);
        self.pres[0].vhat += &(&self.pres[1].to_parent() / self.dt);
    }
}

impl Integrate for Navier2D {
    ///         Update Navier Stokes
    fn update(&mut self) {
        // Buoyancy
        let mut that = self.temp.to_parent();
        if let Some(field) = &self.fieldbc {
            that += &field.to_parent();
        }

        // Convection Veclocity
        self.ux.backward();
        self.uy.backward();
        let ux = self.ux.v.to_owned();
        let uy = self.uy.v.to_owned();

        // Solve Velocity
        self.solve_ux(&ux, &uy);
        self.solve_uy(&ux, &uy, &that);

        // Projection
        let div = self.divergence();
        self.solve_pres(&div);
        self.project_velocity(1.0);
        self.update_pres(&div);

        // Solve Temperature
        self.solve_temp(&ux, &uy);

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
        let fname = format!("data/flow{:.*}.h5", 3, self.time);
        self.temp.backward();
        self.ux.backward();
        self.uy.backward();
        self.pres[0].backward();
        // Add boundary contribution
        if let Some(x) = &self.fieldbc {
            self.temp.v = &self.temp.v + &x.v;
        }
        // Field
        self.temp.write(&fname, Some("temp"));
        self.ux.write(&fname, Some("ux"));
        self.uy.write(&fname, Some("uy"));
        self.pres[0].write(&fname, Some("pres"));
        // Additional info
        let mut time = array![self.time];
        let mut ra = array![self.ra];
        let mut pr = array![self.pr];
        write_to_hdf5(&fname, "time", None, Hdf5::Array1(&mut time)).ok();
        write_to_hdf5(&fname, "ra", None, Hdf5::Array1(&mut ra)).ok();
        write_to_hdf5(&fname, "pr", None, Hdf5::Array1(&mut pr)).ok();
        // Undo addition of bc
        if self.fieldbc.is_some() {
            self.temp.backward();
        }

        println!(" ==> {:?}", fname);

        // I/O
        let div = self.divergence();
        println!(
            "time = {:4.2}      |div| = {:4.2e}",
            self.time,
            norm_l2(&div)
        );
    }
}

fn norm_l2(array: &Array2<f64>) -> f64 {
    array.iter().map(|x| x.powf(2.0)).sum::<f64>().sqrt()
}

impl Navier2D {
    /// Read from existing file
    pub fn read(&mut self, fname: &str) {
        // Field
        self.temp.read(&fname, Some("temp"));
        self.ux.read(&fname, Some("ux"));
        self.uy.read(&fname, Some("uy"));
        self.pres[0].read(&fname, Some("pres"));
        // Additional info
        let mut time = array![0.];
        read_from_hdf5(&fname, "time", None, Hdf5::Array1(&mut time)).ok();
        self.time = time[0];
        println!(" <== {:?}", fname);
    }
}

fn apply_sin_cos(field: &mut Field2, amp: f64, m: f64, n: f64) {
    use std::f64::consts::PI;
    let nx = field.v.shape()[0];
    let ny = field.v.shape()[1];
    let x = &field.x[0];
    let y = &field.x[1];
    let x = &((x - x[0]) / (x[x.len() - 1] - x[0]));
    let y = &((y - y[0]) / (y[y.len() - 1] - y[0]));
    let arg_x = PI * m;
    let arg_y = PI * n;
    for i in 0..nx {
        for j in 0..ny {
            field.v[[i, j]] = amp * (arg_x * x[i]).sin() * (arg_y * y[j]).cos();
        }
    }
    field.forward()
}

fn apply_cos_sin(field: &mut Field2, amp: f64, m: f64, n: f64) {
    use std::f64::consts::PI;
    let nx = field.v.shape()[0];
    let ny = field.v.shape()[1];
    let x = &field.x[0];
    let y = &field.x[1];
    let x = &((x - x[0]) / (x[x.len() - 1] - x[0]));
    let y = &((y - y[0]) / (y[y.len() - 1] - y[0]));
    let arg_x = PI * m;
    let arg_y = PI * n;
    for i in 0..nx {
        for j in 0..ny {
            field.v[[i, j]] = amp * (arg_x * x[i]).cos() * (arg_y * y[j]).sin();
        }
    }
    field.forward()
}
