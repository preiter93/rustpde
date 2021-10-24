//! # Adjoint descent method for steady state solutions
//! Solve adjoint 2-dimensional Navier-Stokes equations
//! coupled with temperature equations to obtain steady
//! state solutions
//!
//! # Example
//! Find steady state solution of large scale circulation
//! ```ignore
//! use rustpde::{Integrate, integrate};
//! use rustpde::navier::Navier2DAdjoint;
//!
//! fn main() {
//!     // Parameters
//!     let (nx, ny) = (64, 64);
//!     let ra = 1e5;
//!     let pr = 1.;
//!     let adiabatic = true;
//!     let aspect = 1.0;
//!     let dt = 0.02;
//!     let mut navier_adjoint = Navier2DAdjoint::new(nx, ny, ra, pr, dt, aspect, adiabatic);
//!     // Set initial conditions
//!     navier_adjoint.set_temperature(0.5, 1., 1.);
//!     navier_adjoint.set_velocity(0.5, 1., 1.);
//!     // // Want to restart?
//!     // navier_adjoint.read("data/flow100.000.h5", None);
//!     // Write first field
//!     navier_adjoint.callback();
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
use super::navier::{apply_cos_sin, apply_sin_cos, dealias};
use super::navier::{get_ka, get_nu, Navier2D};
use crate::bases::fourier_r2c;
use crate::bases::{cheb_dirichlet, cheb_neumann, chebyshev};
use crate::bases::{BaseR2c, BaseR2r};
use crate::field::{BaseSpace, Field2, ReadField, Space2, WriteField};
use crate::hdf5::{read_scalar_from_hdf5, write_scalar_to_hdf5, Result};
use crate::solver::{Hholtz, Poisson, Solve, SolverField};
use crate::Integrate;
use ndarray::Array2;
use num_complex::Complex;
use num_traits::Zero;
use std::collections::HashMap;
use std::ops::{Div, Mul};

/// Implement the ndividual terms of the Navier-Stokes equation
/// as a trait. This is necessary to support both real and complex
/// valued spectral spaces
pub trait NavierConvectionAdjoint {
    /// Type in physical space (ususally f64)
    type Physical;
    /// Type in spectral space (f64 or Complex<f64>)
    type Spectral;

    /// Convection term for velocity ux
    fn conv_ux(
        &mut self,
        ux: &Array2<Self::Physical>,
        uy: &Array2<Self::Physical>,
        t: &Array2<Self::Physical>,
    ) -> Array2<Self::Spectral>;

    /// Convection term for velocity uy
    fn conv_uy(
        &mut self,
        ux: &Array2<Self::Physical>,
        uy: &Array2<Self::Physical>,
        t: &Array2<Self::Physical>,
    ) -> Array2<Self::Spectral>;

    /// Convection term for temperature
    fn conv_temp(
        &mut self,
        ux: &Array2<Self::Physical>,
        uy: &Array2<Self::Physical>,
    ) -> Array2<Self::Spectral>;

    /// Solve horizontal momentum equation
    fn solve_ux(
        &mut self,
        ux: &Array2<Self::Physical>,
        uy: &Array2<Self::Physical>,
        temp: &Array2<Self::Physical>,
    );

    /// Solve vertical momentum equation
    fn solve_uy(
        &mut self,
        ux: &Array2<Self::Physical>,
        uy: &Array2<Self::Physical>,
        temp: &Array2<Self::Physical>,
    );

    /// Solve temperature equation:
    fn solve_temp(&mut self, ux: &Array2<Self::Physical>, uy: &Array2<Self::Physical>);

    /// Correct velocity field.
    fn project_velocity(&mut self, c: f64);

    /// Divergence: duxdx + duydy
    fn divergence(&mut self) -> Array2<Self::Spectral>;

    /// Solve pressure poisson equation
    /// pseu: pseudo pressure ( in code it is pres\[1\] )
    fn solve_pres(&mut self, f: &Array2<Self::Spectral>);

    /// Update pressure term by divergence
    fn update_pres(&mut self, div: &Array2<Self::Spectral>);

    /// Update navier stokes residual
    fn update_residual(&mut self);
}

/// Tolerance criteria for residual
const RES_TOL: f64 = 1e-8;

type Space2R2r = Space2<BaseR2r<f64>, BaseR2r<f64>>;
type Space2R2c = Space2<BaseR2c<f64>, BaseR2r<f64>>;

/// Container for Adjoint Navier-Stokes solver
pub struct Navier2DAdjoint<T, S> {
    /// Navier Stokes solver
    navier: Navier2D<T, S>,
    /// Field for derivatives and transforms
    field: Field2<T, S>,
    /// Temperature \[Adjoint Field, NS Residual\]
    pub temp: [Field2<T, S>; 2],
    /// Horizontal Velocity \[Adjoint Field, NS Residual\]
    pub ux: [Field2<T, S>; 2],
    /// Vertical Velocity \[Adjoint Field, NS Residual\]
    pub uy: [Field2<T, S>; 2],
    /// Pressure [pres, pseudo pressure]
    pub pres: [Field2<T, S>; 2],
    /// Collection of solvers \[pres\]
    solver: [SolverField<f64, 2>; 1],
    /// Smooth fields for better convergence \[ux, uy, temp\]
    smoother: [SolverField<f64, 2>; 3],
    /// Buffer
    rhs: Array2<T>,
    /// Fields unsmoothed (for diffusion) \[ux, uy, temp\]
    fields_unsmoothed: [Array2<T>; 3],
    /// Field for temperature boundary condition
    pub fieldbc: Option<Field2<T, S>>,
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
    /// Time step size for residual calculation
    pub dt_navier: f64,
    /// Scale of physical dimension [scale_x, scale_y]
    pub scale: [f64; 2],
    /// diagnostics like Nu, ...
    pub diagnostics: HashMap<String, Vec<f64>>,
    /// Time intervall for write fields
    /// If none, same intervall as diagnostics
    pub write_intervall: Option<f64>,
    /// residual tolerance (exit if below)
    res_tol: f64,
    /// Set true and the fields will be dealiased
    pub dealias: bool,
}

impl Navier2DAdjoint<f64, Space2R2r> {
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
    /// * `aspect` - Aspect ratio L/H
    ///
    /// * `adiabatic` - Boolean, sidewall temperature boundary condition
    #[allow(clippy::similar_names)]
    pub fn new(
        nx: usize,
        ny: usize,
        ra: f64,
        pr: f64,
        dt: f64,
        aspect: f64,
        adiabatic: bool,
    ) -> Navier2DAdjoint<f64, Space2R2r> {
        let scale = [aspect, 1.];
        let nu = get_nu(ra, pr, scale[1] * 2.0);
        let ka = get_ka(ra, pr, scale[1] * 2.0);
        let ux = [
            Field2::new(&Space2::new(&cheb_dirichlet(nx), &cheb_dirichlet(ny))),
            Field2::new(&Space2::new(&cheb_dirichlet(nx), &cheb_dirichlet(ny))),
        ];
        let uy = [
            Field2::new(&Space2::new(&cheb_dirichlet(nx), &cheb_dirichlet(ny))),
            Field2::new(&Space2::new(&cheb_dirichlet(nx), &cheb_dirichlet(ny))),
        ];
        let temp = if adiabatic {
            [
                Field2::new(&Space2::new(&cheb_neumann(nx), &cheb_dirichlet(ny))),
                Field2::new(&Space2::new(&cheb_neumann(nx), &cheb_dirichlet(ny))),
            ]
        } else {
            [
                Field2::new(&Space2::new(&cheb_dirichlet(nx), &cheb_dirichlet(ny))),
                Field2::new(&Space2::new(&cheb_dirichlet(nx), &cheb_dirichlet(ny))),
            ]
        };
        // define underlying naver-stokes solver
        let dt_navier = 1e-2;
        let navier = Navier2D::new(nx, ny, ra, pr, dt_navier, aspect, adiabatic);
        // pressure
        let pres = [
            Field2::new(&Space2::new(&chebyshev(nx), &chebyshev(ny))),
            Field2::new(&Space2::new(&cheb_neumann(nx), &cheb_neumann(ny))),
        ];
        // fields for derivatives
        let field = Field2::new(&Space2::new(&chebyshev(nx), &chebyshev(ny)));

        // pressure solver
        let solver_pres = SolverField::Poisson(Poisson::new(
            &pres[1],
            [1.0 / scale[0].powf(2.), 1.0 / scale[1].powf(2.)],
        ));
        let solver = [solver_pres];

        // define smoother (hholtz type) (1-weight*D2)
        let weight_laplacian = 1e0;
        let smooth_ux = SolverField::Hholtz(Hholtz::new(
            &ux[1],
            [
                weight_laplacian / scale[0].powf(2.),
                weight_laplacian / scale[1].powf(2.),
            ],
        ));
        let smooth_uy = SolverField::Hholtz(Hholtz::new(
            &uy[1],
            [
                weight_laplacian / scale[0].powf(2.),
                weight_laplacian / scale[1].powf(2.),
            ],
        ));
        let smooth_temp = SolverField::Hholtz(Hholtz::new(
            &temp[1],
            [
                weight_laplacian / scale[0].powf(2.),
                weight_laplacian / scale[1].powf(2.),
            ],
        ));

        // // define smoother (poisson type)
        // let smooth_ux = SolverField::Poisson(Poisson::new(
        //     &ux[0],
        //     [
        //         -1.0 / (1. * scale[0].powf(2.)),
        //         -1.0 / (1. * scale[1].powf(2.)),
        //     ],
        // ));
        // let smooth_uy = SolverField::Poisson(Poisson::new(
        //     &uy[0],
        //     [
        //         -1.0 / (1. * scale[0].powf(2.)),
        //         -1.0 / (1. * scale[1].powf(2.)),
        //     ],
        // ));
        // let smooth_temp = SolverField::Poisson(Poisson::new(
        //     &temp[0],
        //     [
        //         -1.0 / (1. * scale[0].powf(2.)),
        //         -1.0 / (1. * scale[1].powf(2.)),
        //     ],
        // ));

        let smoother = [smooth_ux, smooth_uy, smooth_temp];
        let fields_unsmoothed = [
            Array2::zeros(field.vhat.raw_dim()),
            Array2::zeros(field.vhat.raw_dim()),
            Array2::zeros(field.vhat.raw_dim()),
        ];

        // Buffer for rhs
        let rhs = Array2::zeros(field.vhat.raw_dim());

        // Diagnostics
        let mut diagnostics = HashMap::new();
        diagnostics.insert("time".to_string(), Vec::<f64>::new());
        diagnostics.insert("Nu".to_string(), Vec::<f64>::new());
        diagnostics.insert("Nuvol".to_string(), Vec::<f64>::new());
        diagnostics.insert("Re".to_string(), Vec::<f64>::new());

        // Initialize
        let mut navier_adjoint = Navier2DAdjoint::<f64, Space2R2r> {
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
            dt_navier,
            scale,
            diagnostics,
            write_intervall: None,
            res_tol: RES_TOL,
            dealias: true,
        };
        navier_adjoint._scale();
        // Boundary condition
        navier_adjoint.set_temp_bc(Navier2D::bc_rbc(nx, ny));
        // Return
        navier_adjoint
    }
}

impl Navier2DAdjoint<Complex<f64>, Space2R2c> {
    /// Bases: Fourier in x and chebyshev in y
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
    /// * `aspect` - Aspect ratio L/H (unity is assumed to be to 2pi)
    #[allow(clippy::similar_names)]
    pub fn new_periodic(
        nx: usize,
        ny: usize,
        ra: f64,
        pr: f64,
        dt: f64,
        aspect: f64,
    ) -> Navier2DAdjoint<Complex<f64>, Space2R2c> {
        let scale = [aspect, 1.];
        let nu = get_nu(ra, pr, scale[1] * 2.0);
        let ka = get_ka(ra, pr, scale[1] * 2.0);
        let ux = [
            Field2::new(&Space2::new(&fourier_r2c(nx), &cheb_dirichlet(ny))),
            Field2::new(&Space2::new(&fourier_r2c(nx), &cheb_dirichlet(ny))),
        ];
        let uy = [
            Field2::new(&Space2::new(&fourier_r2c(nx), &cheb_dirichlet(ny))),
            Field2::new(&Space2::new(&fourier_r2c(nx), &cheb_dirichlet(ny))),
        ];
        let temp = [
            Field2::new(&Space2::new(&fourier_r2c(nx), &cheb_dirichlet(ny))),
            Field2::new(&Space2::new(&fourier_r2c(nx), &cheb_dirichlet(ny))),
        ];
        // define underlying naver-stokes solver
        let dt_navier = 1e-2;
        let navier = Navier2D::new_periodic(nx, ny, ra, pr, dt_navier, aspect);
        // pressure
        let pres = [
            Field2::new(&Space2::new(&fourier_r2c(nx), &chebyshev(ny))),
            Field2::new(&Space2::new(&fourier_r2c(nx), &cheb_neumann(ny))),
        ];
        // fields for derivatives
        let field = Field2::new(&Space2::new(&fourier_r2c(nx), &chebyshev(ny)));

        // pressure solver
        let solver_pres = SolverField::Poisson(Poisson::new(
            &pres[1],
            [1.0 / scale[0].powf(2.), 1.0 / scale[1].powf(2.)],
        ));
        let solver = [solver_pres];

        // define smoother (hholtz type) (1-weight*D2)
        let weight_laplacian = 1e0;
        let smooth_ux = SolverField::Hholtz(Hholtz::new(
            &ux[0],
            [
                weight_laplacian / scale[0].powf(2.),
                weight_laplacian / scale[1].powf(2.),
            ],
        ));
        let smooth_uy = SolverField::Hholtz(Hholtz::new(
            &uy[0],
            [
                weight_laplacian / scale[0].powf(2.),
                weight_laplacian / scale[1].powf(2.),
            ],
        ));
        let smooth_temp = SolverField::Hholtz(Hholtz::new(
            &temp[0],
            [
                weight_laplacian / scale[0].powf(2.),
                weight_laplacian / scale[1].powf(2.),
            ],
        ));

        // // define smoother (poisson type)
        // let smooth_ux = SolverField::Poisson(Poisson::new(
        //     &ux[0],
        //     [
        //         -1.0 / (1. * scale[0].powf(2.)),
        //         -1.0 / (1. * scale[1].powf(2.)),
        //     ],
        // ));
        // let smooth_uy = SolverField::Poisson(Poisson::new(
        //     &uy[0],
        //     [
        //         -1.0 / (1. * scale[0].powf(2.)),
        //         -1.0 / (1. * scale[1].powf(2.)),
        //     ],
        // ));
        // let smooth_temp = SolverField::Poisson(Poisson::new(
        //     &temp[0],
        //     [
        //         -1.0 / (1. * scale[0].powf(2.)),
        //         -1.0 / (1. * scale[1].powf(2.)),
        //     ],
        // ));

        let smoother = [smooth_ux, smooth_uy, smooth_temp];
        let fields_unsmoothed = [
            Array2::zeros(field.vhat.raw_dim()),
            Array2::zeros(field.vhat.raw_dim()),
            Array2::zeros(field.vhat.raw_dim()),
        ];
        // buffer for rhs
        let rhs = Array2::zeros(field.vhat.raw_dim());

        // Diagnostics
        let mut diagnostics = HashMap::new();
        diagnostics.insert("time".to_string(), Vec::<f64>::new());
        diagnostics.insert("Nu".to_string(), Vec::<f64>::new());
        diagnostics.insert("Nuvol".to_string(), Vec::<f64>::new());
        diagnostics.insert("Re".to_string(), Vec::<f64>::new());
        // Initialize
        let mut navier_adjoint = Navier2DAdjoint::<Complex<f64>, Space2R2c> {
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
            dt_navier,
            scale,
            diagnostics,
            write_intervall: None,
            res_tol: RES_TOL,
            dealias: true,
        };
        navier_adjoint._scale();
        // Boundary condition
        navier_adjoint.set_temp_bc(Navier2D::bc_rbc_periodic(nx, ny));
        // Return
        navier_adjoint
    }
}

impl<T, S> Navier2DAdjoint<T, S>
where
    T: num_traits::Zero,
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T>,
{
    /// Rescale x & y coordinates of fields.
    /// Only affects output of files
    fn _scale(&mut self) {
        for field in &mut [
            &mut self.temp[0],
            &mut self.ux[0],
            &mut self.uy[0],
            &mut self.pres[0],
        ] {
            field.x[0] *= self.scale[0];
            field.x[1] *= self.scale[1];
            field.dx[0] *= self.scale[0];
            field.dx[1] *= self.scale[1];
        }
        for field in &mut [&mut self.temp[1], &mut self.ux[1], &mut self.uy[1]] {
            field.x[0] *= self.scale[0];
            field.x[1] *= self.scale[1];
            field.dx[0] *= self.scale[0];
            field.dx[1] *= self.scale[1];
        }
    }

    /// Set boundary condition field for temperature
    pub fn set_temp_bc(&mut self, fieldbc: Field2<T, S>) {
        self.fieldbc = Some(fieldbc);
    }

    fn zero_rhs(&mut self) {
        for r in self.rhs.iter_mut() {
            *r = T::zero();
        }
    }
}

macro_rules! impl_navier_convection {
    ($s: ty) => {
        impl<S> NavierConvectionAdjoint for Navier2DAdjoint<$s, S>
        where
            S: BaseSpace<f64, 2, Physical = f64, Spectral = $s>,
        {
            type Physical = f64;
            type Spectral = $s;

            /// Convection term for ux
            fn conv_ux(
                &mut self,
                ux: &Array2<Self::Physical>,
                uy: &Array2<Self::Physical>,
                t: &Array2<Self::Physical>,
            ) -> Array2<Self::Spectral> {
                // + ux * dudx + uy * dudy
                let mut conv =
                    conv_term(&self.ux[1], &mut self.field, ux, [1, 0], Some(self.scale));
                conv += &conv_term(&self.ux[1], &mut self.field, uy, [0, 1], Some(self.scale));
                // + adjoint contributions
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
                if self.dealias {
                    dealias(&mut self.field);
                }
                self.field.vhat.to_owned()
            }

            /// Convection term for uy
            fn conv_uy(
                &mut self,
                ux: &Array2<Self::Physical>,
                uy: &Array2<Self::Physical>,
                t: &Array2<Self::Physical>,
            ) -> Array2<Self::Spectral> {
                // + ux * dudx + uy * dudy
                let mut conv =
                    conv_term(&self.uy[1], &mut self.field, ux, [1, 0], Some(self.scale));
                conv += &conv_term(&self.uy[1], &mut self.field, uy, [0, 1], Some(self.scale));
                // + adjoint contributions
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
                if self.dealias {
                    dealias(&mut self.field);
                }
                self.field.vhat.to_owned()
            }

            /// Convection term for temperature
            fn conv_temp(
                &mut self,
                ux: &Array2<Self::Physical>,
                uy: &Array2<Self::Physical>,
            ) -> Array2<Self::Spectral> {
                // + ux * dTdx + uy * dTdy
                let mut conv =
                    conv_term(&self.temp[1], &mut self.field, ux, [1, 0], Some(self.scale));
                conv += &conv_term(&self.temp[1], &mut self.field, uy, [0, 1], Some(self.scale));
                // -> spectral space
                self.field.v.assign(&conv);
                self.field.forward();
                if self.dealias {
                    dealias(&mut self.field);
                }
                self.field.vhat.to_owned()
            }

            /// Solve adjoint horizontal momentum equation
            fn solve_ux(
                &mut self,
                ux: &Array2<Self::Physical>,
                uy: &Array2<Self::Physical>,
                temp: &Array2<Self::Physical>,
            ) {
                self.zero_rhs();
                // + old field
                self.rhs += &self.ux[0].to_ortho();
                // + pres
                self.rhs -= &(self.pres[0].gradient([1, 0], Some(self.scale)) * self.dt);
                // + convection
                let conv = self.conv_ux(ux, uy, temp);
                self.rhs += &(conv * self.dt);
                // + diffusion
                self.rhs += &(self.ux[1].gradient([2, 0], Some(self.scale)) * self.dt * self.nu);
                self.rhs += &(self.ux[1].gradient([0, 2], Some(self.scale)) * self.dt * self.nu);
                // update ux
                self.ux[0].from_ortho(&self.rhs);
            }

            /// Solve adjoint vertical momentum equation
            fn solve_uy(
                &mut self,
                ux: &Array2<Self::Physical>,
                uy: &Array2<Self::Physical>,
                temp: &Array2<Self::Physical>,
            ) {
                self.zero_rhs();
                // + old field
                self.rhs += &self.uy[0].to_ortho();
                // + pres
                self.rhs -= &(self.pres[0].gradient([0, 1], Some(self.scale)) * self.dt);
                // + convection
                let conv = self.conv_uy(ux, uy, temp);
                self.rhs += &(conv * self.dt);
                // + diffusion
                self.rhs += &(self.uy[1].gradient([2, 0], Some(self.scale)) * self.dt * self.nu);
                self.rhs += &(self.uy[1].gradient([0, 2], Some(self.scale)) * self.dt * self.nu);
                // update uy
                self.uy[0].from_ortho(&self.rhs);
            }

            /// Solve adjoint temperature equation
            fn solve_temp(&mut self, ux: &Array2<Self::Physical>, uy: &Array2<Self::Physical>) {
                self.zero_rhs();
                // + old field
                self.rhs += &self.temp[0].to_ortho();
                // + convection
                let conv = self.conv_temp(ux, uy);
                self.rhs += &(conv * self.dt);
                // + buoyancy (adjoint)
                let buoy = self.uy[1].to_ortho();
                self.rhs += &(buoy * self.dt);
                // + diffusion
                self.rhs += &(self.temp[1].gradient([2, 0], Some(self.scale)) * self.dt * self.ka);
                self.rhs += &(self.temp[1].gradient([0, 2], Some(self.scale)) * self.dt * self.ka);
                // update temp
                self.temp[0].from_ortho(&self.rhs);
            }

            /// Correct velocity field.
            ///
            /// uxnew = ux - c*dpdx
            ///
            /// uynew = uy - c*dpdy
            #[allow(clippy::similar_names)]
            fn project_velocity(&mut self, c: f64) {
                let dpdx = self.pres[1].gradient([1, 0], Some(self.scale));
                let dpdy = self.pres[1].gradient([0, 1], Some(self.scale));
                let old_ux = self.ux[0].vhat.clone();
                let old_uy = self.uy[0].vhat.clone();
                self.ux[0].from_ortho(&dpdx);
                self.uy[0].from_ortho(&dpdy);
                let cinto: Self::Spectral = (-c).into();
                self.ux[0].vhat *= cinto;
                self.uy[0].vhat *= cinto;
                self.ux[0].vhat += &old_ux;
                self.uy[0].vhat += &old_uy;
            }

            /// Divergence: duxdx + duydy
            fn divergence(&mut self) -> Array2<Self::Spectral> {
                self.zero_rhs();
                self.rhs += &self.ux[0].gradient([1, 0], Some(self.scale));
                self.rhs += &self.uy[0].gradient([0, 1], Some(self.scale));
                self.rhs.to_owned()
            }

            /// Solve pressure poisson equation
            ///
            /// D2 pres = f
            ///
            /// pseu: pseudo pressure ( in code it is pres\[1\] )
            fn solve_pres(&mut self, f: &Array2<Self::Spectral>) {
                self.solver[0].solve(&f, &mut self.pres[1].vhat, 0);
                // Singularity
                self.pres[1].vhat[[0, 0]] = Self::Spectral::zero();
            }

            fn update_pres(&mut self, _div: &Array2<Self::Spectral>) {
                // self.pres[0].vhat = &self.pres[0].vhat - &(div * self.nu);
                let inv_dt: Self::Spectral = (1. / self.dt).into();
                self.pres[0].vhat += &(&self.pres[1].to_ortho() * inv_dt);
            }

            /// Update navier stokes residual
            fn update_residual(&mut self) {
                // Update residual
                self.navier.ux.vhat.assign(&self.ux[0].vhat);
                self.navier.uy.vhat.assign(&self.uy[0].vhat);
                self.navier.temp.vhat.assign(&self.temp[0].vhat);
                self.navier.update();
                let res = (&self.navier.ux.vhat - &self.ux[0].vhat) / self.navier.dt;
                self.navier.ux.vhat.assign(&res);
                let res = (&self.navier.uy.vhat - &self.uy[0].vhat) / self.navier.dt;
                self.navier.uy.vhat.assign(&res);
                let res = (&self.navier.temp.vhat - &self.temp[0].vhat) / self.navier.dt;
                self.navier.temp.vhat.assign(&res);
                // Save "unsmoothed" residual fields
                self.fields_unsmoothed[0].assign(&self.navier.ux.to_ortho());
                self.fields_unsmoothed[1].assign(&self.navier.uy.to_ortho());
                self.fields_unsmoothed[2].assign(&self.navier.temp.to_ortho());
                // Smooth fields
                self.smoother[0].solve(&self.fields_unsmoothed[0], &mut self.ux[1].vhat, 0);
                self.smoother[1].solve(&self.fields_unsmoothed[1], &mut self.uy[1].vhat, 0);
                self.smoother[2].solve(&self.fields_unsmoothed[2], &mut self.temp[1].vhat, 0);
                let rescale: Self::Spectral = (-1.0).into();
                self.ux[1].vhat *= rescale;
                self.uy[1].vhat *= rescale;
                self.temp[1].vhat *= rescale;
            }
        }
    };
}

impl_navier_convection!(f64);
impl_navier_convection!(Complex<f64>);

macro_rules! impl_integrate {
    ($s: ty, $norm: ident) => {
        impl<S> Integrate for Navier2DAdjoint<$s, S>
        where
            S: BaseSpace<f64, 2, Physical = f64, Spectral = $s>,
        {
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

            fn callback(&mut self) {
                use std::io::Write;
                std::fs::create_dir_all("data").unwrap();

                // Write flow field
                //let fname = format!("data/adjoint{:.*}.h5", 3, self.time);
                let fname = format!("data/adjoint{:0>8.2}.h5", self.time);
                if let Some(dt_save) = &self.write_intervall {
                    if (self.time % dt_save) < self.dt / 2.
                        || (self.time % dt_save) > dt_save - self.dt / 2.
                    {
                        self.write(&fname);
                    }
                } else {
                    self.write(&fname);
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
                    $norm(&div),
                    nu,
                    nuvol,
                    re,
                );

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
                if let Err(e) = writeln!(file, "{} {} {} {}", self.time, nu, nuvol, re) {
                    eprintln!("Couldn't write to file: {}", e);
                }
                // Write residual
                let res_u = $norm(&self.fields_unsmoothed[0]);
                let res_v = $norm(&self.fields_unsmoothed[1]);
                let res_t = $norm(&self.fields_unsmoothed[2]);
                let res_total = res_u + res_v + res_t;
                let res_u2 = $norm(&self.ux[1].vhat);
                let res_v2 = $norm(&self.uy[1].vhat);
                let res_t2 = $norm(&self.temp[1].vhat);
                let res_total2 = (res_u2 + res_v2 + res_t2);
                println!("|U| = {:10.2e}", res_u2,);
                println!("|V| = {:10.2e}", res_v2,);
                println!("|T| = {:10.2e}", res_t2,);
                let mut residual = std::fs::OpenOptions::new()
                    .write(true)
                    .append(true)
                    .create(true)
                    .open("data/residual.txt")
                    .unwrap();
                if let Err(e) = writeln!(residual, "{} {} {}", self.time, res_total, res_total2) {
                    eprintln!("Couldn't write to file: {}", e);
                }
            }

            fn exit(&mut self) -> bool {
                // Break if divergence is nan
                let div = self.divergence();
                if $norm(&div).is_nan() {
                    println!("Divergence is nan!");
                    return true;
                }
                // Break if residual is small enough
                let res_u = $norm(&self.ux[1].vhat);
                let res_v = $norm(&self.uy[1].vhat);
                let res_t = $norm(&self.temp[1].vhat);
                if res_u + res_v + res_t < self.res_tol {
                    println!("Residual reached!");
                    return true;
                }
                false
            }
        }

    };
}
impl_integrate!(f64, norm_l2_f64);
impl_integrate!(Complex<f64>, norm_l2_c64);

fn norm_l2_f64(array: &Array2<f64>) -> f64 {
    array.iter().map(|x| x.powi(2)).sum::<f64>().sqrt()
}

fn norm_l2_c64(array: &Array2<Complex<f64>>) -> f64 {
    array
        .iter()
        .map(|x| x.re.powi(2) + x.im.powi(2))
        .sum::<f64>()
        .sqrt()
}

// fn norm_l2_f64(q1: &Array2<f64>, q2: &Array2<f64>) -> f64 {
//     q1.iter().zip(q2).map(|(x, y)| x * y).sum::<f64>().sqrt()
// }
//
// fn norm_l2_c64(q1: &Array2<Complex<f64>>, q2: &Array2<Complex<f64>>) -> f64 {
//     q1.iter()
//         .zip(q2)
//         .map(|(x, y)| {
//             let z = x * y;
//             z.re.powi(2) + z.im.powi(2)
//         })
//         .sum::<f64>()
//         .sqrt()
// }

impl<T, S> Navier2DAdjoint<T, S>
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T>,
    T: crate::types::Scalar + Mul<f64, Output = T> + Div<f64, Output = T>,
{
    /// Returns Nusselt number (heat flux at the plates)
    /// $$
    /// Nu = \langle - dTdz \rangle\\_x (0/H))
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
    /// Nuvol = \langle uy*T/kappa - dTdz \rangle\\_V
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
}

macro_rules! impl_read_write_navier {
    ($s: ty) => {
        impl<S> Navier2DAdjoint<$s, S>
        where
            S: BaseSpace<f64, 2, Physical = f64, Spectral = $s>,
        {
            /// Restart from file
            pub fn read(&mut self, filename: &str) {
                self.temp[0].read(&filename, Some("temp"));
                self.ux[0].read(&filename, Some("ux"));
                self.uy[0].read(&filename, Some("uy"));
                // Read scalars
                self.time = read_scalar_from_hdf5::<f64>(&filename, "time", None).unwrap();
                println!(" <== {:?}", filename);
            }

            /// Write Field data to hdf5 file
            pub fn write(&mut self, filename: &str) {
                let result = self.write_return_result(filename);
                match result {
                    Ok(_) => println!(" ==> {:?}", filename),
                    Err(_) => println!("Error while writing file {:?}.", filename),
                }
            }

            fn write_return_result(&mut self, filename: &str) -> Result<()> {
                self.temp[0].backward();
                self.ux[0].backward();
                self.uy[0].backward();
                self.pres[0].backward();
                // Add boundary contribution
                if let Some(x) = &self.fieldbc {
                    self.temp[0].v = &self.temp[0].v + &x.v;
                }
                // Field
                self.temp[0].write(&filename, Some("temp"));
                self.ux[0].write(&filename, Some("ux"));
                self.uy[0].write(&filename, Some("uy"));
                self.pres[0].write(&filename, Some("pres"));
                // Write scalars
                write_scalar_to_hdf5(&filename, "time", None, self.time).ok();
                write_scalar_to_hdf5(&filename, "ra", None, self.ra).ok();
                write_scalar_to_hdf5(&filename, "pr", None, self.pr).ok();
                write_scalar_to_hdf5(&filename, "nu", None, self.nu).ok();
                write_scalar_to_hdf5(&filename, "kappa", None, self.ka).ok();
                // Undo addition of bc
                if self.fieldbc.is_some() {
                    self.temp[0].backward();
                }
                Ok(())
            }
        }
    };
}

impl_read_write_navier!(f64);
impl_read_write_navier!(Complex<f64>);
