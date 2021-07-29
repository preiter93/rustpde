//! # Multidimensional field of basis functions
#![allow(dead_code)]
pub mod average;
pub mod read;
pub mod write;
use crate::bases::Differentiate;
use crate::bases::FromOrtho;
use crate::bases::TransformPar;
use crate::space::{Space1, Space2, Spaced};
use crate::{Real, SolverField};
use ndarray::prelude::*;
use ndarray::Ix;
use std::collections::HashMap;

/// One dimensional Field
pub type Field1 = Field<Space1, f64, 1>;
/// Two dimensional Field
pub type Field2 = Field<Space2, f64, 2>;

/// Field struct is rustpdes backbone
///
/// v: ndarray
///
///   Holds data in physical space
///
/// vhat: ndarray
///
///   Holds data in spectral space
///
/// x: list of ndarrays
///
///   Grid points (physical space)
///
/// solvers: HashMap<String, SolverField>
///
///  Add plans for various equations
///
/// Field is derived from space.
/// space mplements trait IsSpace, which defines
/// forward / backward transform from physical
/// to spectral space + derivative in spectral space
///
///
/// let (nx, ny) = (5, 5);
/// let bases = [cheb_dirichlet(nx), cheb_dirichlet(ny)];
/// let mut field = Field::new(Space2D::new(bases));
///
#[derive(Clone)]
pub struct Field<S, T, const N: usize> {
    /// Number of dimensions
    pub ndim: usize,
    /// Space
    pub space: S,
    /// Field in physical space
    pub v: Array<Real, Dim<[Ix; N]>>,
    /// Field in spectral space
    pub vhat: Array<T, Dim<[Ix; N]>>,
    /// Grid coordinates
    pub x: [Array1<f64>; N],
    /// Grid deltas
    pub dx: [Array1<f64>; N],
    /// Collection of mathematical solvers
    pub solvers: HashMap<String, SolverField<T, N>>,
}

impl<S, T, const N: usize> Field<S, T, N>
where
    S: Spaced<T, N>,
{
    /// Returns field
    ///
    /// # Example
    /// 2-D field in chebyshev space
    ///```
    /// use rustpde::cheb_dirichlet;
    /// use rustpde::Space2;
    /// use rustpde::Field2;
    /// let cdx = cheb_dirichlet(8);
    /// let cdy = cheb_dirichlet(6);
    /// let field = Field2::new(Space2::new([cdx,cdy]));
    ///```
    pub fn new(space: S) -> Self {
        let ndim = N;
        let v = space.ndarr_phys();
        let vhat = space.ndarr_spec();
        let x = space.get_x();
        let dx = Self::get_dx(&x);
        Field {
            ndim,
            space,
            v,
            vhat,
            x,
            dx,
            solvers: HashMap::new(),
        }
    }

    /// Generate grid deltas from coordinates
    fn get_dx(x_arr: &[Array1<f64>; N]) -> [Array1<f64>; N] {
        use std::convert::TryInto;
        let mut dx_vec = Vec::new();
        for x in x_arr.iter() {
            let mut dx = Array1::<f64>::zeros(x.len());
            for (i, dxi) in dx.iter_mut().enumerate() {
                let xs_left = if i == 0 { x[0] } else { (x[i] + x[i - 1]) / 2. };
                let xs_right = if i == x.len() - 1 {
                    x[x.len() - 1]
                } else {
                    (x[i + 1] + x[i]) / 2.
                };
                *dxi = xs_right - xs_left;
            }
            dx_vec.push(dx);
        }
        dx_vec.try_into().unwrap_or_else(|v: Vec<Array1<f64>>| {
            panic!("Expected Vec of length {} but got {}", N, v.len())
        })
    }
}

impl<S: Spaced<f64, 1>> Field<S, Real, 1> {
    /// Forward transform 1d
    pub fn forward(&mut self) {
        self.space.get_bases_mut()[0].forward_inplace_par(&mut self.v, &mut self.vhat, 0);
    }
    /// Backward transform 1d
    pub fn backward(&mut self) {
        self.space.get_bases_mut()[0].backward_inplace_par(&mut self.vhat, &mut self.v, 0);
    }

    /// Transform to parent space
    pub fn to_parent(&self) -> Array1<Real> {
        self.space.get_bases()[0].to_ortho(&self.vhat, 0)
    }

    /// Transform to child space
    pub fn from_parent(&mut self, input: &Array1<f64>) {
        self.vhat
            .assign(&self.space.get_bases()[0].from_ortho(input, 0))
    }

    /// Gradient
    fn grad(&self, deriv: [usize; 1], scale: Option<[f64; 1]>) -> Array1<Real> {
        let mut output = self.space.get_bases()[0].differentiate(&self.vhat, deriv[0], 0);
        if let Some(s) = scale {
            output /= s[0].powi(deriv[0] as i32);
        }
        output
    }
}

impl<S: Spaced<f64, 2>> Field<S, Real, 2> {
    /// Forward transform 2d
    pub fn forward(&mut self) {
        let mut buffer = self.space.get_bases_mut()[1].forward_par(&mut self.v, 1);
        self.space.get_bases_mut()[0].forward_inplace_par(&mut buffer, &mut self.vhat, 0);
    }
    /// Backward transform 2d
    pub fn backward(&mut self) {
        let mut buffer = self.space.get_bases_mut()[0].backward_par(&mut self.vhat, 0);
        self.space.get_bases_mut()[1].backward_inplace_par(&mut buffer, &mut self.v, 1);
    }

    /// Transform to parent space
    pub fn to_parent(&self) -> Array2<Real> {
        let axis0 = self.space.get_bases()[0].to_ortho(&self.vhat, 0);
        self.space.get_bases()[1].to_ortho(&axis0, 1)
    }

    /// Transform to child space
    pub fn from_parent(&mut self, input: &Array2<f64>) {
        let axis0 = self.space.get_bases()[0].from_ortho(input, 0);
        self.vhat
            .assign(&self.space.get_bases()[1].from_ortho(&axis0, 1));
    }

    /// Gradient
    pub fn grad(&self, deriv: [usize; 2], scale: Option<[f64; 2]>) -> Array2<Real> {
        let buffer = self.space.get_bases()[0].differentiate(&self.vhat, deriv[0], 0);
        let mut output = self.space.get_bases()[1].differentiate(&buffer, deriv[1], 1);
        if let Some(s) = scale {
            output /= s[0].powi(deriv[0] as i32);
            output /= s[1].powi(deriv[1] as i32);
        }
        output
    }
}
