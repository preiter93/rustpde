//! # Bases
//! Collection of various basis functions which implement forward/backward transforms,
//! differentiation and other methods to conveniently work in different spaces.
//!
//! Implemented:
//! - Chebyshev
//! - ChebDirichlet
//! - ChebNeumann
pub mod chebyshev;
pub mod composite;
use crate::Real;
pub use chebyshev::Chebyshev;
pub use chebyshev::{ChebDirichlet, ChebNeumann};
use ndarray::prelude::*;
use ndarray::{Data, DataMut, LinalgScalar, RawDataClone, RemoveAxis};

/// Function space for Chebyshev Polynomials
///             T{k}
/// ```
/// let ch = chebyshev(n);
/// ```
pub fn chebyshev(n: usize) -> Base {
    Base::Chebyshev(Chebyshev::new(n))
}

/// Function space with Dirichlet boundary conditions
///         phi{k} = T{k} - T{k+2}
/// ```
/// let cd = cheb_dirichlet(n);
/// ```
pub fn cheb_dirichlet(n: usize) -> Base {
    Base::ChebDirichlet(ChebDirichlet::new(n))
}

// Function space with Neumann boundary conditions
///         phi{k} = T{k} - k^2/(k+2)^2 T{k+2}
/// ```
/// let cn = cheb_neumann(n);
/// ```
pub fn cheb_neumann(n: usize) -> Base {
    Base::ChebNeumann(ChebNeumann::new(n))
}

/// Enum of all implemented basis functions.
///
/// All bases must implement the transform and differentiation trait,
/// which from there derived for this enum.
//#[enum_dispatch(Differentiate, Transform)]
#[enum_dispatch(Differentiate, Mass, LaplacianInverse, Size)]
pub enum Base {
    /// Orthonormal Chebyshev base
    Chebyshev(Chebyshev),
    /// Composite Chebyshev base with Dirichlet boundary conditions
    ChebDirichlet(ChebDirichlet),
    /// Composite Chebyshev base with Neumann boundary conditions
    ChebNeumann(ChebNeumann),
    //Fourier(Fourier),
}

impl Transform<Real> for Base {
    fn forward<S1, S2, D>(
        &mut self,
        input: &mut ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: Data<Elem = Real> + DataMut + RawDataClone,
        S2: Data<Elem = Real> + DataMut + RawDataClone,
        D: Dimension + RemoveAxis,
    {
        match self {
            Base::Chebyshev(ref mut b) => b.forward(input, output, axis),
            Base::ChebDirichlet(ref mut b) => b.forward(input, output, axis),
            Base::ChebNeumann(ref mut b) => b.forward(input, output, axis),
        }
    }

    fn backward<S1, S2, D>(
        &mut self,
        input: &mut ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: Data<Elem = Real> + DataMut + RawDataClone,
        S2: Data<Elem = Real> + DataMut + RawDataClone,
        D: Dimension + RemoveAxis,
    {
        match self {
            Base::Chebyshev(ref mut b) => b.backward(input, output, axis),
            Base::ChebDirichlet(ref mut b) => b.backward(input, output, axis),
            Base::ChebNeumann(ref mut b) => b.backward(input, output, axis),
        }
    }
}

/// Defines transform from physical to spectral space and vice versa,
/// together with other methods that all Bases should implement.
//#[enum_dispatch]
pub trait Transform<T> {
    /// Transform array from physical to spectral space along axis
    fn forward<S1, S2, D>(
        &mut self,
        input: &mut ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: Data<Elem = Real> + DataMut + RawDataClone,
        S2: Data<Elem = T> + DataMut + RawDataClone,
        D: Dimension + RemoveAxis;

    /// Transform array from physical to spectral space along axis
    fn backward<S1, S2, D>(
        &mut self,
        input: &mut ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: Data<Elem = T> + DataMut + RawDataClone,
        S2: Data<Elem = Real> + DataMut + RawDataClone,
        D: Dimension + RemoveAxis;
}

/// Defines differentiation in spectral space
#[enum_dispatch]
pub trait Differentiate {
    /// Differentiate n_times along axis (performed in spectral space)
    fn differentiate<T, S1, S2, D>(
        &self,
        input: &ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        n_times: usize,
        axis: usize,
    ) where
        T: LinalgScalar + Send + From<f64>,
        f64: Into<T>,
        S1: Data<Elem = T>,
        S2: Data<Elem = T> + RawDataClone + DataMut,
        D: Dimension;
}

/// Return mass matrix and size of basis
#[enum_dispatch]
pub trait Mass {
    /// Return mass matrix
    fn mass<T>(&self) -> Array2<T>
    where
        T: LinalgScalar + From<f64>;
    /// Return size of basis
    fn size(&self) -> usize;
}

/// Define (Pseudo-) Inverse of Laplacian
#[enum_dispatch]
pub trait LaplacianInverse {
    /// Pseudoinverse mtrix of Laplacian
    fn pinv<T>(&self) -> Array2<T>
    where
        T: LinalgScalar + From<f64>;
    /// Pseudoidentity matrix of laplacian
    fn pinv_eye<T>(&self) -> Array2<T>
    where
        T: LinalgScalar + From<f64>;
}

/// Defines size of basis
#[enum_dispatch]
pub trait Size {
    /// Size in physical space
    fn len_phys(&self) -> usize;
    /// Size in spectral space
    fn len_spec(&self) -> usize;
    /// Coordinates in physical space
    fn coords(&self) -> &Array1<f64>;
}
