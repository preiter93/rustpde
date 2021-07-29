//! # Adjoint descent method for steady state solutions
//! Solve adjoint 2-dimensional Navier-Stokes equations
//! coupled with temperature equations to obtain steady
//! state solutions
//!
//! # Example
//! Find steady state solution of large scale circulation
//! ```ignore
//! use rustpde::integrate;
//! use rustpde::integrate::Navier2DAdjoint;
//! use rustpde::Integrate;
//!
//! fn main() {
//!     // Parameters
//!     let (nx, ny) = (64, 64);
//!     let ra = 1e5;
//!     let pr = 1.;
//!     let adiabatic = true;
//!     let aspect = 1.0;
//!     let dt = 0.02;
//!     let mut navier_adjoint = Navier2DAdjoint::new(nx, ny, ra, pr, dt, adiabatic, aspect);
//!     // Set initial conditions
//!     navier_adjoint.set_temperature(0.5, 1., 1.);
//!     navier_adjoint.set_velocity(0.5, 1., 1.);
//!     // // Want to restart?
//!     // navier_adjoint.read("data/flow100.000.h5");
//!     // Write first field
//!     navier_adjoint.write();
//!     integrate(&mut navier_adjoint, 100., Some(1.0));
//! }
//! ```
//!
//! ## References
//! <a id="1">\[1\]</a>
//! M. Farazmand (2016).
//! An adjoint-based approach for finding invariant solutions of Navier--Stokes equations
//! J. Fluid Mech., 795, 278-312.
use super::conv_term;
use super::navier::{apply_cos_sin, apply_sin_cos};
use super::navier::{get_ka, get_nu, Navier2D};
use super::Integrate;
use crate::bases::{cheb_dirichlet, cheb_neumann, chebyshev};
use crate::hdf5::{read_from_hdf5, write_to_hdf5, Hdf5};
use crate::solver::{Poisson, Solve, SolverField};
use crate::Field2;
use crate::Space2;
use ndarray::{array, Array2};
use std::collections::HashMap;

/// Tolerance criteria for residual
const RES_TOL: f64 = 1e-6;

/// Container for Adjoint Navier-Stokes solver
pub struct Navier2DAdjoint {
    /// Navier Stokes solver
    navier: Navier2D,
    /// Field for derivatives and transforms
    field: Field2,
    /// Temperature \[Adjoint Field, NS Residual\]
    pub temp: [Field2; 2],
    /// Horizontal Velocity \[Adjoint Field, NS Residual\]
    pub ux: [Field2; 2],
    /// Vertical Velocity \[Adjoint Field, NS Residual\]
    pub uy: [Field2; 2],
    /// Pressure [pres, pseudo pressure]
    pub pres: [Field2; 2],
    /// Collection of solvers \[pres\]
    solver: [SolverField<f64, 2>; 1],
    /// Smooth fields for better convergence \[ux, uy, temp\]
    smoother: [SolverField<f64, 2>; 3],
    /// Buffer
    rhs: Array2<f64>,
    /// Fields unsmoothed (for diffusion) \[ux, uy, temp\]
    fields_unsmoothed: [Array2<f64>; 3],
    /// Field for temperature boundary condition
    pub fieldbc: Option<Field2>,
    /// Viscosity
    nu: f64,
    /// Thermal diffusivity
    ka: f64,
    /// Rayleigh number
    pub ra: f64,
    /// Prandtl number
    pub pr: f64,
    /// Time
    pub time: f64,
    /// Time step size
    pub dt: f64,
    /// Scale of physical dimension [scale_x, scale_y]
    pub scale: [f64; 2],
    /// Scale adjoint fields
    scale_adjoint: f64,
    /// diagnostics like Nu, ...
    pub diagnostics: HashMap<String, Vec<f64>>,
    /// Time intervall for write fields
    /// If none, same intervall as diagnostics
    pub write_intervall: Option<f64>,
    /// residual tolerance (exit if below)
    res_tol: f64,
}

impl Navier2DAdjoint {
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
        let ux = [
            Field2::new(Space2::new([cheb_dirichlet(nx), cheb_dirichlet(ny)])),
            Field2::new(Space2::new([cheb_dirichlet(nx), cheb_dirichlet(ny)])),
        ];
        let uy = [
            Field2::new(Space2::new([cheb_dirichlet(nx), cheb_dirichlet(ny)])),
            Field2::new(Space2::new([cheb_dirichlet(nx), cheb_dirichlet(ny)])),
        ];
        let temp = if adiabatic {
            [
                Field2::new(Space2::new([cheb_neumann(nx), cheb_dirichlet(ny)])),
                Field2::new(Space2::new([cheb_neumann(nx), cheb_dirichlet(ny)])),
            ]
        } else {
            [
                Field2::new(Space2::new([cheb_dirichlet(nx), cheb_dirichlet(ny)])),
                Field2::new(Space2::new([cheb_dirichlet(nx), cheb_dirichlet(ny)])),
            ]
        };
        // define underlying naver-stokes solver
        let navier = Navier2D::new(nx, ny, ra, pr, dt, adiabatic, aspect);
        // Construct the rest from fields
        Self::from_fields(navier, temp, ux, uy, nu, ka, ra, pr, dt, scale)
    }

    #[allow(clippy::too_many_arguments)]
    fn from_fields(
        navier: Navier2D,
        temp: [Field2; 2],
        ux: [Field2; 2],
        uy: [Field2; 2],
        nu: f64,
        ka: f64,
        ra: f64,
        pr: f64,
        dt: f64,
        scale: [f64; 2],
    ) -> Self {
        // define additional fields
        let nx = temp[0].v.shape()[0];
        let ny = temp[0].v.shape()[1];
        let field = Field2::new(Space2::new([chebyshev(nx), chebyshev(ny)]));
        let pres = [
            Field2::new(Space2::new([chebyshev(nx), chebyshev(ny)])),
            Field2::new(Space2::new([cheb_neumann(nx), cheb_neumann(ny)])),
        ];
        // define solver
        let smooth_ux = SolverField::Poisson(Poisson::from_field(
            &ux[0],
            [1. / scale[0].powf(2.), 1. / scale[1].powf(2.)],
        ));
        let smooth_uy = SolverField::Poisson(Poisson::from_field(
            &uy[0],
            [1. / scale[0].powf(2.), 1. / scale[1].powf(2.)],
        ));
        let smooth_temp = SolverField::Poisson(Poisson::from_field(
            &temp[0],
            [1. / scale[0].powf(2.), 1. / scale[1].powf(2.)],
        ));
        let solver_pres = SolverField::Poisson(Poisson::from_field(
            &pres[1],
            [1. / scale[0].powf(2.), 1. / scale[1].powf(2.)],
        ));
        let smoother = [smooth_ux, smooth_uy, smooth_temp];
        let solver = [solver_pres];
        let rhs = Array2::zeros((nx, ny));
        let fields_unsmoothed = [
            Array2::zeros((nx, ny)),
            Array2::zeros((nx, ny)),
            Array2::zeros((nx, ny)),
        ];
        // Diagnostics
        let mut diagnostics = HashMap::new();
        diagnostics.insert("time".to_string(), Vec::<f64>::new());
        diagnostics.insert("Nu".to_string(), Vec::<f64>::new());
        diagnostics.insert("Nuvol".to_string(), Vec::<f64>::new());
        diagnostics.insert("Re".to_string(), Vec::<f64>::new());
        // Initialize
        let mut navier_adjoint = Navier2DAdjoint {
            navier,
            field,
            temp,
            ux,
            uy,
            pres,
            smoother,
            solver,
            rhs,
            fields_unsmoothed,
            fieldbc: None,
            nu,
            ka,
            ra,
            pr,
            time: 0.0,
            dt,
            scale,
            scale_adjoint: nu,
            diagnostics,
            write_intervall: None,
            res_tol: RES_TOL,
        };
        navier_adjoint._scale();
        // Boundary condition
        navier_adjoint.set_temp_bc(Navier2D::bc_rbc(nx, ny));
        // Return
        navier_adjoint
    }

    /// Rescale x & y coordinates of fields.
    /// Only affects output of files
    fn _scale(&mut self) {
        for field in [
            &mut self.temp[0],
            &mut self.ux[0],
            &mut self.uy[0],
            &mut self.pres[0],
        ]
        .iter_mut()
        {
            field.x[0] *= self.scale[0];
            field.x[1] *= self.scale[1];
        }
        for field in [&mut self.temp[1], &mut self.ux[1], &mut self.uy[1]].iter_mut() {
            field.x[0] *= self.scale[0];
            field.x[1] *= self.scale[1];
        }
    }

    /// Set boundary condition field for temperature
    /// Both fields should be the same, necessary beceause
    /// fields do not implement the copy trait yet
    pub fn set_temp_bc(&mut self, fieldbc: Field2) {
        self.fieldbc = Some(fieldbc.clone());
        self.navier.fieldbc = Some(fieldbc);
    }

    /// Reset rhs array
    fn zero_rhs(&mut self) {
        for r in self.rhs.iter_mut() {
            *r = 0.;
        }
    }

    /// Convection term for temperature
    fn conv_temp(&mut self, ux: &Array2<f64>, uy: &Array2<f64>) -> Array2<f64> {
        // + ux * dTdx + uy * dTdy
        let mut conv = conv_term(&self.temp[1], &mut self.field, ux, [1, 0], Some(self.scale));
        conv += &conv_term(&self.temp[1], &mut self.field, uy, [0, 1], Some(self.scale));
        // // + bc contribution
        // if let Some(field) = &self.fieldbc {
        //     conv += &conv_term(field, &mut self.field, ux, [1, 0], Some(self.scale));
        //     conv += &conv_term(field, &mut self.field, uy, [0, 1], Some(self.scale));
        // }
        // -> spectral space
        self.field.v.assign(&conv);
        self.field.forward();
        self.field.vhat.to_owned()
    }

    /// Convection term for ux
    fn conv_ux(&mut self, ux: &Array2<f64>, uy: &Array2<f64>, t: &Array2<f64>) -> Array2<f64> {
        // + ux * dudx + uy * dudy
        let mut conv = conv_term(&self.ux[1], &mut self.field, ux, [1, 0], Some(self.scale));
        conv += &conv_term(&self.ux[1], &mut self.field, uy, [0, 1], Some(self.scale));
        // adjoint contributions
        conv += &conv_term(&self.ux[1], &mut self.field, ux, [1, 0], Some(self.scale));
        conv += &conv_term(&self.uy[1], &mut self.field, uy, [1, 0], Some(self.scale));
        conv += &conv_term(&self.temp[1], &mut self.field, t, [1, 0], Some(self.scale));
        if let Some(x) = &self.fieldbc {
            conv += &conv_term(
                &self.temp[1],
                &mut self.field,
                &x.v,
                [1, 0],
                Some(self.scale),
            );
        }
        // -> spectral space
        self.field.v.assign(&conv);
        self.field.forward();
        self.field.vhat.to_owned()
    }

    /// Convection term for uy
    fn conv_uy(&mut self, ux: &Array2<f64>, uy: &Array2<f64>, t: &Array2<f64>) -> Array2<f64> {
        // + ux * dudx + uy * dudy
        let mut conv = conv_term(&self.uy[1], &mut self.field, ux, [1, 0], Some(self.scale));
        conv += &conv_term(&self.uy[1], &mut self.field, uy, [0, 1], Some(self.scale));
        // adjoint contributions
        conv += &conv_term(&self.ux[1], &mut self.field, ux, [0, 1], Some(self.scale));
        conv += &conv_term(&self.uy[1], &mut self.field, uy, [0, 1], Some(self.scale));
        conv += &conv_term(&self.temp[1], &mut self.field, t, [0, 1], Some(self.scale));
        if let Some(x) = &self.fieldbc {
            conv += &conv_term(
                &self.temp[1],
                &mut self.field,
                &x.v,
                [0, 1],
                Some(self.scale),
            );
        }
        // -> spectral space
        self.field.v.assign(&conv);
        self.field.forward();
        self.field.vhat.to_owned()
    }

    /// Solve adjoint horizontal momentum equation
    fn solve_ux(&mut self, ux: &Array2<f64>, uy: &Array2<f64>, temp: &Array2<f64>) {
        self.zero_rhs();
        // + old field
        self.rhs += &self.ux[0].to_parent();
        // + pres
        self.rhs -= &(self.dt * self.pres[0].grad([1, 0], Some(self.scale)));
        // + convection
        let conv = self.conv_ux(ux, uy, temp);
        self.rhs += &(self.dt * conv);
        // + diffusion
        let nu = self.nu / self.scale_adjoint;
        self.rhs += &(self.dt * nu * &self.fields_unsmoothed[0]);
        // update ux
        self.ux[0].from_parent(&self.rhs);
    }

    /// Solve adjoint vertical momentum equation
    fn solve_uy(&mut self, ux: &Array2<f64>, uy: &Array2<f64>, temp: &Array2<f64>) {
        self.zero_rhs();
        // + old field
        self.rhs += &self.uy[0].to_parent();
        // + pres
        self.rhs -= &(self.dt * self.pres[0].grad([0, 1], Some(self.scale)));
        // + convection
        let conv = self.conv_uy(ux, uy, temp);
        self.rhs += &(self.dt * conv);
        // + diffusion
        let nu = self.nu / self.scale_adjoint;
        self.rhs += &(self.dt * nu * &self.fields_unsmoothed[1]);
        // update uy
        self.uy[0].from_parent(&self.rhs);
    }

    /// Solve adjoint temperature equation
    fn solve_temp(&mut self, ux: &Array2<f64>, uy: &Array2<f64>) {
        self.zero_rhs();
        // + old field
        self.rhs += &self.temp[0].to_parent();
        // + convection
        let conv = self.conv_temp(ux, uy);
        self.rhs += &(self.dt * conv);
        // + buoyancy (adjoint)
        let buoy = self.uy[1].to_parent();
        self.rhs += &(self.dt * buoy);
        // + diffusion
        let ka = self.ka / self.scale_adjoint;
        self.rhs += &(self.dt * ka * &self.fields_unsmoothed[2]);
        // update temp
        self.temp[0].from_parent(&self.rhs);
    }

    /// Solve pressure poisson equation
    ///
    /// D2 pres = f
    ///
    /// pseu: pseudo pressure ( in code it is pres\[1\] )
    fn solve_pres(&mut self, f: &Array2<f64>) {
        self.solver[0].solve(&f, &mut self.pres[1].vhat, 0);
        // Singularity
        self.pres[1].vhat[[0, 0]] = 0.;
    }

    fn update_pres(&mut self, _div: &Array2<f64>) {
        //self.pres[0].vhat -= &(self.nu * div);
        self.pres[0].vhat += &(&self.pres[1].to_parent() / self.dt);
    }

    /// Correct velocity field.
    ///
    /// uxnew = ux - c*dpdx
    ///
    /// uynew = uy - c*dpdy
    fn project_velocity(&mut self, c: f64) {
        let dpdx = self.pres[1].grad([1, 0], Some(self.scale));
        let dpdy = self.pres[1].grad([0, 1], Some(self.scale));
        let ux_old = self.ux[0].vhat.clone();
        let uy_old = self.uy[0].vhat.clone();
        self.ux[0].from_parent(&dpdx);
        self.uy[0].from_parent(&dpdy);
        self.ux[0].vhat *= -c;
        self.uy[0].vhat *= -c;
        self.ux[0].vhat += &ux_old;
        self.uy[0].vhat += &uy_old;
    }

    /// Divergence: duxdx + duydy
    fn divergence(&mut self) -> Array2<f64> {
        self.zero_rhs();
        self.rhs += &self.ux[0].grad([1, 0], Some(self.scale));
        self.rhs += &self.uy[0].grad([0, 1], Some(self.scale));
        self.rhs.to_owned()
    }

    /// Update navier stokes residual
    fn update_residual(&mut self) {
        // Update residual
        self.navier.ux.vhat.assign(&self.ux[0].vhat);
        self.navier.uy.vhat.assign(&self.uy[0].vhat);
        self.navier.temp.vhat.assign(&self.temp[0].vhat);
        self.navier.update();
        let res = (&self.navier.ux.vhat - &self.ux[0].vhat) / self.dt;
        self.navier.ux.vhat.assign(&res);
        let res = (&self.navier.uy.vhat - &self.uy[0].vhat) / self.dt;
        self.navier.uy.vhat.assign(&res);
        let res = (&self.navier.temp.vhat - &self.temp[0].vhat) / self.dt;
        self.navier.temp.vhat.assign(&res);
        // Save unsmoothed fields for diffusion
        self.fields_unsmoothed[0].assign(&self.navier.ux.to_parent());
        self.fields_unsmoothed[1].assign(&self.navier.uy.to_parent());
        self.fields_unsmoothed[2].assign(&self.navier.temp.to_parent());
        // Smooth fields
        self.smoother[0].solve(&self.fields_unsmoothed[0], &mut self.ux[1].vhat, 0);
        self.smoother[1].solve(&self.fields_unsmoothed[1], &mut self.uy[1].vhat, 0);
        self.smoother[2].solve(&self.fields_unsmoothed[2], &mut self.temp[1].vhat, 0);
        self.ux[1].vhat /= self.scale_adjoint;
        self.uy[1].vhat /= self.scale_adjoint;
        self.temp[1].vhat /= self.scale_adjoint;
    }
}

impl Integrate for Navier2DAdjoint {
    /// Update of adjoint Navier Stokes
    fn update(&mut self) {
        // Convection fields
        self.ux[0].backward();
        self.uy[0].backward();
        self.temp[0].backward();
        let ux = self.ux[0].v.to_owned();
        let uy = self.uy[0].v.to_owned();
        let temp = self.temp[0].v.to_owned();

        // Update residual
        self.update_residual();

        // Solve Velocity
        self.solve_ux(&ux, &uy, &temp);
        self.solve_uy(&ux, &uy, &temp);

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
        use std::io::Write;
        std::fs::create_dir_all("data").unwrap();

        // Write flow field
        let fname = format!("data/adjoint{:.*}.h5", 3, self.time);
        if let Some(dt_save) = &self.write_intervall {
            if (self.time % dt_save) < self.dt / 2.
                || (self.time % dt_save) > dt_save - self.dt / 2.
            {
                self.write_to_file(&fname);
            }
        } else {
            self.write_to_file(&fname);
        }

        // I/O
        let div = self.divergence();
        let nu = self.eval_nu();
        let nuvol = self.eval_nuvol();
        let re = self.eval_re();
        //println!("Diagnostics:");
        println!(
            "time = {:4.2}      |div| = {:4.2e}     Nu = {:5.3e}     Nuv = {:5.3e}    Re = {:5.3e}",
            self.time,
            norm_l2(&div),
            nu,
            nuvol,
            re,
        );
        //println!("Residuals:");
        println!("|U| = {:10.2e}", norm_l2(&self.fields_unsmoothed[0]),);
        println!("|V| = {:10.2e}", norm_l2(&self.fields_unsmoothed[1]),);
        println!("|T| = {:10.2e}", norm_l2(&self.fields_unsmoothed[2]),);

        // diagnostics
        if let Some(d) = self.diagnostics.get_mut("time") {
            d.push(self.time);
        }
        if let Some(d) = self.diagnostics.get_mut("Nu") {
            d.push(nu);
        }
        if let Some(d) = self.diagnostics.get_mut("Nuvol") {
            d.push(nuvol);
        }
        if let Some(d) = self.diagnostics.get_mut("Re") {
            d.push(re);
        }
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .append(true)
            .create(true)
            .open("data/adjoint_info.txt")
            .unwrap();
        //write!(file, "{} {}", time, nu);
        if let Err(e) = writeln!(file, "{} {} {} {}", self.time, nu, nuvol, re) {
            eprintln!("Couldn't write to file: {}", e);
        }
    }

    fn exit(&mut self) -> bool {
        // Break if divergence is nan
        let div = self.divergence();
        if norm_l2(&div).is_nan() {
            return true;
        }
        // Break if residual is small enough
        let mut res = norm_l2(&self.fields_unsmoothed[0]) / 3.;
        res += norm_l2(&self.fields_unsmoothed[1]) / 3.;
        res += norm_l2(&self.fields_unsmoothed[2]) / 3.;
        if res < self.res_tol {
            return true;
        }
        false
    }
}

fn norm_l2(array: &Array2<f64>) -> f64 {
    array.iter().map(|x| x.powf(2.0)).sum::<f64>().sqrt()
}

impl Navier2DAdjoint {
    /// Returns Nusselt number (heat flux at the plates)
    /// $$
    /// Nu = \langle - dTdz \rangle_x (0/H))
    /// $$
    pub fn eval_nu(&mut self) -> f64 {
        use super::functions::eval_nu;
        eval_nu(
            &mut self.temp[0],
            &mut self.field,
            &self.fieldbc,
            &self.scale,
        )
    }

    /// Returns volumetric Nusselt number
    /// $$
    /// Nuvol = \langle uy*T/kappa - dTdz \rangle_V
    /// $$
    pub fn eval_nuvol(&mut self) -> f64 {
        use super::functions::eval_nuvol;
        eval_nuvol(
            &mut self.temp[0],
            &mut self.uy[0],
            &mut self.field,
            &self.fieldbc,
            self.ka,
            &self.scale,
        )
    }

    /// Returns Reynolds number based on kinetic energy
    pub fn eval_re(&mut self) -> f64 {
        use super::functions::eval_re;
        eval_re(
            &mut self.ux[0],
            &mut self.uy[0],
            &mut self.field,
            self.nu,
            &self.scale,
        )
    }

    /// Read from existing file
    pub fn read(&mut self, fname: &str) {
        // Field
        self.temp[0].read(&fname, Some("temp"));
        self.ux[0].read(&fname, Some("ux"));
        self.uy[0].read(&fname, Some("uy"));
        //self.pres[0].read(&fname, Some("pres"));
        // Additional info
        let time = read_from_hdf5::<ndarray::Ix1, 1>(&fname, "time", None).unwrap();
        self.time = time[0];
        println!(" <== {:?}", fname);
    }

    /// Initialize velocity with fourier modes
    ///
    /// ux = amp \* sin(mx)cos(nx)
    /// uy = -amp \* cos(mx)sin(nx)
    pub fn set_velocity(&mut self, amp: f64, m: f64, n: f64) {
        apply_sin_cos(&mut self.ux[0], amp, m, n);
        apply_cos_sin(&mut self.uy[0], -amp, m, n);
    }
    /// Initialize temperature with fourier modes
    ///
    /// temp = -amp \* cos(mx)sin(ny)
    pub fn set_temperature(&mut self, amp: f64, m: f64, n: f64) {
        apply_cos_sin(&mut self.temp[0], -amp, m, n);
    }

    /// Reset time
    pub fn reset_time(&mut self) {
        self.time = 0.;
        self.navier.time = 0.;
    }

    /// Write fields to hdf5 file
    pub fn write_to_file(&mut self, fname: &str) {
        self.temp[0].backward();
        self.ux[0].backward();
        self.uy[0].backward();
        self.pres[0].backward();
        // Add boundary contribution
        if let Some(x) = &self.fieldbc {
            self.temp[0].v = &self.temp[0].v + &x.v;
        }
        // Field
        self.temp[0].write(&fname, Some("temp"));
        self.ux[0].write(&fname, Some("ux"));
        self.uy[0].write(&fname, Some("uy"));
        self.pres[0].write(&fname, Some("pres"));
        // Additional info
        let mut time = array![self.time];
        let mut ra = array![self.ra];
        let mut pr = array![self.pr];
        let mut nu = array![self.nu];
        let mut ka = array![self.ka];
        write_to_hdf5(&fname, "time", None, Hdf5::Array1(&mut time)).ok();
        write_to_hdf5(&fname, "ra", None, Hdf5::Array1(&mut ra)).ok();
        write_to_hdf5(&fname, "pr", None, Hdf5::Array1(&mut pr)).ok();
        write_to_hdf5(&fname, "nu", None, Hdf5::Array1(&mut nu)).ok();
        write_to_hdf5(&fname, "kappa", None, Hdf5::Array1(&mut ka)).ok();
        // Undo addition of bc
        if self.fieldbc.is_some() {
            self.temp[0].backward();
        }

        println!(" ==> {:?}", fname);
    }
}
