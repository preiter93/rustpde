//! # Multidimensional field of basis functions

use crate::bases::Size;
use crate::bases::Transform;
use crate::{Base, Real, Solver};
use ndarray::prelude::*;
use ndarray::Ix;
use std::collections::HashMap;
use std::convert::TryInto;

/// Two dimensional Space
pub type Space2 = Space<2>;
/// Two dimensional Field
pub type Field2 = Field<Space2, f64, 2>;

/// Implement on all supported dimensions.
pub trait Spaced<T, const N: usize> {
    /// Return ndarray with shape of physical space
    fn ndarr_phys(&self) -> Array<Real, Dim<[Ix; N]>>;
    /// Return ndarray with shape of spectral space
    fn ndarr_spec(&self) -> Array<T, Dim<[Ix; N]>>;
    /// Return array of coordinates [x,y,..]
    fn get_x(&self) -> [Array1<f64>; N];
    /// Return array of enum Base
    fn get_bases(&mut self) -> &mut [Base; N];
}

/// Create multidimensional field
///
/// First create a space, then
/// initialize field with it.
pub struct Space<const N: usize> {
    bases: [Base; N],
}

impl<const N: usize> Space<N> {
    /// Return new space
    pub fn new(bases: [Base; N]) -> Self {
        Space { bases }
    }

    /// Shape in physical space
    fn shape_phys(&self) -> [usize; N] {
        self.bases
            .iter()
            .map(|x| x.len_phys())
            .collect::<Vec<usize>>()
            .try_into()
            .unwrap()
    }
    /// Shape in spectral space
    fn shape_spec(&self) -> [usize; N] {
        self.bases
            .iter()
            .map(|x| x.len_spec())
            .collect::<Vec<usize>>()
            .try_into()
            .unwrap()
    }
}

impl<T> Spaced<T, 1> for Space<1>
where
    T: num_traits::Zero + Clone,
{
    fn get_bases(&mut self) -> &mut [Base; 1] {
        &mut self.bases
    }

    fn ndarr_phys(&self) -> Array<Real, Ix1> {
        Array::<Real, Ix1>::zeros(self.shape_phys())
    }

    fn ndarr_spec(&self) -> Array<T, Ix1> {
        Array::<T, Ix1>::zeros(self.shape_spec())
    }

    fn get_x(&self) -> [Array1<f64>; 1] {
        [self.bases[0].coords().to_owned()]
    }
}

impl<T> Spaced<T, 2> for Space<2>
where
    T: num_traits::Zero + Clone,
{
    fn get_bases(&mut self) -> &mut [Base; 2] {
        &mut self.bases
    }

    fn ndarr_phys(&self) -> Array<Real, Dim<[Ix; 2]>> {
        Array::<Real, Dim<[Ix; 2]>>::zeros(self.shape_phys())
    }

    fn ndarr_spec(&self) -> Array<T, Dim<[Ix; 2]>> {
        Array::<T, Dim<[Ix; 2]>>::zeros(self.shape_spec())
    }

    fn get_x(&self) -> [Array1<f64>; 2] {
        [
            self.bases[0].coords().to_owned(),
            self.bases[1].coords().to_owned(),
        ]
    }
}

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
/// solvers: HashMap<String, Solver>
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
    pub solvers: HashMap<String, Solver<T, N>>,
}

impl<S, T, const N: usize> Field<S, T, N>
where
    S: Spaced<T, N>,
{
    /// Returns field
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
        self.space.get_bases()[0].forward(&mut self.v, &mut self.vhat, 0);
    }
    /// Backward transform 1d
    pub fn backward(&mut self) {
        self.space.get_bases()[0].backward(&mut self.vhat, &mut self.v, 0);
    }
}

impl<S: Spaced<f64, 2>> Field<S, Real, 2> {
    /// Forward transform 2d
    pub fn forward(&mut self) {
        let shape = [
            self.space.get_bases()[0].len_phys(),
            self.space.get_bases()[1].len_spec(),
        ];
        println!("{:?}", shape);
        let mut buffer = Array2::<f64>::zeros(shape);
        self.space.get_bases()[1].forward(&mut self.v, &mut buffer, 1);
        self.space.get_bases()[0].forward(&mut buffer, &mut self.vhat, 0);
    }
    /// Backward transform 2d
    pub fn backward(&mut self) {
        let shape = [
            self.space.get_bases()[0].len_phys(),
            self.space.get_bases()[1].len_spec(),
        ];
        let mut buffer = Array2::<f64>::zeros(shape);
        self.space.get_bases()[0].backward(&mut self.vhat, &mut buffer, 0);
        self.space.get_bases()[1].backward(&mut buffer, &mut self.v, 1);
    }
}
