//! Space initializes Field
use crate::bases::BaseBasics;
use crate::{Base, Real};
use ndarray::prelude::*;
use ndarray::Ix;
use std::convert::TryInto;

/// One dimensional Space
pub type Space1 = Space<1>;
/// Two dimensional Space
pub type Space2 = Space<2>;

/// Implement on all supported dimensions.
pub trait Spaced<T, const N: usize> {
    /// Return ndarray with shape of physical space
    fn ndarr_phys(&self) -> Array<Real, Dim<[Ix; N]>>;
    /// Return ndarray with shape of spectral space
    fn ndarr_spec(&self) -> Array<T, Dim<[Ix; N]>>;
    /// Return array of coordinates [x,y,..]
    fn get_x(&self) -> [Array1<f64>; N];
    /// Return array of enum Base
    fn get_bases(&self) -> &[Base<f64>; N];
    /// Return array of enum Base
    fn get_bases_mut(&mut self) -> &mut [Base<f64>; N];
}

/// Create multidimensional space
///
/// First create a space, then
/// initialize field with it.
#[derive(Clone)]
pub struct Space<const N: usize> {
    bases: [Base<f64>; N],
}

impl<const N: usize> Space<N> {
    /// Return new space
    #[must_use]
    pub fn new(bases: [Base<f64>; N]) -> Self {
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
    fn get_bases(&self) -> &[Base<f64>; 1] {
        &self.bases
    }

    fn get_bases_mut(&mut self) -> &mut [Base<f64>; 1] {
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
    fn get_bases(&self) -> &[Base<f64>; 2] {
        &self.bases
    }

    fn get_bases_mut(&mut self) -> &mut [Base<f64>; 2] {
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
