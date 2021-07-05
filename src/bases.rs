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

/// Enum of all implemented basis functions.
///
/// All bases must implement the transform trait,
/// which is then derived for the enum with enum_dispatch.
//#[enum_dispatch(Differentiate, Transform)]
#[enum_dispatch(Differentiate)]
pub enum Base {
    /// Orthonormal Chebyshev base
    Chebyshev(Chebyshev),
    /// Composite Chebyshev base with Dirichlet boundary conditions
    ChebDirichlet(ChebDirichlet),
    /// Composite Chebyshev base with Neumann boundary conditions
    ChebNeumann(ChebNeumann),
    //Fourier(Fourier),
}

impl Transform<Real, Real> for Base {
    fn forward<R, S, D>(
        &mut self,
        input: &mut ArrayBase<R, D>,
        output: &mut ArrayBase<S, D>,
        axis: usize,
    ) where
        R: Data<Elem = Real> + DataMut + RawDataClone,
        S: Data<Elem = Real> + DataMut + RawDataClone,
        D: Dimension + RemoveAxis,
    {
        match self {
            Base::Chebyshev(ref mut b) => b.forward(input, output, axis),
            Base::ChebDirichlet(ref mut b) => b.forward(input, output, axis),
            Base::ChebNeumann(ref mut b) => b.forward(input, output, axis),
        }
    }

    fn backward<R, S, D>(
        &mut self,
        input: &mut ArrayBase<R, D>,
        output: &mut ArrayBase<S, D>,
        axis: usize,
    ) where
        R: Data<Elem = Real> + DataMut + RawDataClone,
        S: Data<Elem = Real> + DataMut + RawDataClone,
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
#[enum_dispatch]
pub trait Transform<A, B> {
    /// Transform array from physical to spectral space along axis
    fn forward<R, S, D>(
        &mut self,
        input: &mut ArrayBase<R, D>,
        output: &mut ArrayBase<S, D>,
        axis: usize,
    ) where
        R: Data<Elem = A> + DataMut + RawDataClone,
        S: Data<Elem = B> + DataMut + RawDataClone,
        D: Dimension + RemoveAxis;

    /// Transform array from physical to spectral space along axis
    fn backward<R, S, D>(
        &mut self,
        input: &mut ArrayBase<R, D>,
        output: &mut ArrayBase<S, D>,
        axis: usize,
    ) where
        R: Data<Elem = A> + DataMut + RawDataClone,
        S: Data<Elem = B> + DataMut + RawDataClone,
        D: Dimension + RemoveAxis;
}

/// Defines differentiation in spectral space
#[enum_dispatch]
pub trait Differentiate {
    /// Differentiate n_times along axis (performed in spectral space)
    fn differentiate<T, R, S, D>(
        &self,
        input: &ArrayBase<R, D>,
        output: &mut ArrayBase<S, D>,
        n_times: usize,
        axis: usize,
    ) where
        T: LinalgScalar + Send + From<f64>,
        f64: Into<T>,
        R: Data<Elem = T>,
        S: Data<Elem = T> + RawDataClone + DataMut,
        D: Dimension;
}
