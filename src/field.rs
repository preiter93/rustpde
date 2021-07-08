//! # Multidimensional field of basis functions
#![allow(dead_code)]
pub mod read;
pub mod write;
use crate::bases::Differentiate;
use crate::bases::Parental;
use crate::bases::Size;
use crate::bases::Transform;
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
    /// use ndspectral::cheb_dirichlet;
    /// use ndspectral::Space2;
    /// use ndspectral::Field2;
    /// let cdx = cheb_dirichlet(8);
    /// let cdy = cheb_dirichlet(6);
    /// let field = Field2::new(Space2::new([cdx,cdy]));
    ///```
    pub fn new(space: S) -> Self {
        let ndim = N;
        let v = space.ndarr_phys();
        let vhat = space.ndarr_spec();
        let x = space.get_x();
        Field {
            ndim,
            space,
            v,
            vhat,
            x,
            solvers: HashMap::new(),
        }
    }
}

impl<S: Spaced<f64, 1>> Field<S, Real, 1> {
    /// Forward transform 1d
    pub fn forward(&mut self) {
        self.space.get_bases_mut()[0].forward(&mut self.v, &mut self.vhat, 0);
    }
    /// Backward transform 1d
    pub fn backward(&mut self) {
        self.space.get_bases_mut()[0].backward(&mut self.vhat, &mut self.v, 0);
    }

    /// Transform to parent space
    pub fn to_parent(&self) -> Array1<Real> {
        self.space.get_bases()[0].to_parent(&self.vhat, 0)
    }

    /// Transform to child space
    pub fn from_parent(&mut self, input: &Array1<f64>) {
        self.vhat
            .assign(&self.space.get_bases()[0].from_parent(input, 0))
    }

    /// Gradient
    fn grad(&self, deriv: [usize; 1], scale: Option<[f64; 1]>) -> Array1<Real> {
        let mut output = Array1::<f64>::zeros(self.v.raw_dim());
        self.space.get_bases()[0].differentiate(&self.vhat, &mut output, deriv[0], 0);
        if let Some(s) = scale {
            output /= s[0].powi(deriv[0] as i32);
        }
        output
    }
}

impl<S: Spaced<f64, 2>> Field<S, Real, 2> {
    /// Forward transform 2d
    pub fn forward(&mut self) {
        let shape = [
            self.space.get_bases()[0].len_phys(),
            self.space.get_bases()[1].len_spec(),
        ];
        let mut buffer = Array2::<f64>::zeros(shape);
        self.space.get_bases_mut()[1].forward(&mut self.v, &mut buffer, 1);
        self.space.get_bases_mut()[0].forward(&mut buffer, &mut self.vhat, 0);
    }
    /// Backward transform 2d
    pub fn backward(&mut self) {
        let shape = [
            self.space.get_bases()[0].len_phys(),
            self.space.get_bases()[1].len_spec(),
        ];
        let mut buffer = Array2::<f64>::zeros(shape);
        self.space.get_bases_mut()[0].backward(&mut self.vhat, &mut buffer, 0);
        self.space.get_bases_mut()[1].backward(&mut buffer, &mut self.v, 1);
    }

    /// Transform to parent space
    pub fn to_parent(&self) -> Array2<Real> {
        let axis0 = self.space.get_bases()[0].to_parent(&self.vhat, 0);
        self.space.get_bases()[1].to_parent(&axis0, 1)
    }

    /// Transform to child space
    pub fn from_parent(&mut self, input: &Array2<f64>) {
        let axis0 = self.space.get_bases()[0].from_parent(input, 0);
        self.vhat
            .assign(&self.space.get_bases()[1].from_parent(&axis0, 1));
    }

    /// Gradient
    pub fn grad(&self, deriv: [usize; 2], scale: Option<[f64; 2]>) -> Array2<Real> {
        let mut output = Array2::<f64>::zeros(self.v.raw_dim());
        let shape = [
            self.space.get_bases()[0].len_phys(),
            self.space.get_bases()[1].len_spec(),
        ];
        let mut buffer = Array2::<f64>::zeros(shape);
        self.space.get_bases()[0].differentiate(&self.vhat, &mut buffer, deriv[0], 0);
        self.space.get_bases()[1].differentiate(&buffer, &mut output, deriv[1], 1);
        if let Some(s) = scale {
            output /= s[0].powi(deriv[0] as i32);
            output /= s[1].powi(deriv[1] as i32);
        }
        output
    }
}
