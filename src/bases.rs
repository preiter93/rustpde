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
pub use chebyshev::Chebyshev;
pub use chebyshev::{ChebDirichlet, ChebNeumann};
use ndarray::prelude::*;
use ndarray::{Data, DataMut, LinalgScalar, RawDataClone, RemoveAxis};
//use crate::Real;

// pub struct StructA{}
// pub struct StructB{}
// impl test for StructA{
//     type A=f64;
// }
// impl test for StructB{
//     type A=f64;
// }
// pub trait test {
//     type A;
// }
// pub enum Test {
//     StructA(StructA),
//     StructB(StructB),
// }
//
// impl test for Test {
//     type A = f64;
//     // match self {
//     //         Test::StructA => f64,
//     //         Test::StructB => f64,
//     // };
// }

/// Enum of all implemented basis functions.
///
/// All bases must implement the transform trait,
/// which is then derived for the enum with enum_dispatch.
//#[enum_dispatch(Transform)]
pub enum Base {
    /// Orthonormal Chebyshev base
    Chebyshev(Chebyshev),
    /// Composite Chebyshev base with Dirichlet boundary conditions
    ChebDirichlet(ChebDirichlet),
    /// Composite Chebyshev base with Neumann boundary conditions
    ChebNeumann(ChebNeumann),
    //Fourier(Fourier),
}

/// Defines transform from physical to spectral space and vice versa,
/// together with other methods that all Bases should implement.
pub trait Transform {
    /// Data Type in physical space
    type PhType;
    /// DataType in spectral space
    type SpType;
    /// Transform array from physical to spectral space along axis
    fn forward<R, S, D>(
        &mut self,
        input: &mut ArrayBase<R, D>,
        output: &mut ArrayBase<S, D>,
        axis: usize,
    ) where
        R: Data<Elem = Self::SpType> + DataMut + RawDataClone,
        S: Data<Elem = Self::PhType> + DataMut + RawDataClone,
        D: Dimension + RemoveAxis;

    /// Transform array from physical to spectral space along axis
    fn backward<R, S, D>(
        &mut self,
        input: &mut ArrayBase<R, D>,
        output: &mut ArrayBase<S, D>,
        axis: usize,
    ) where
        R: Data<Elem = Self::SpType> + DataMut + RawDataClone,
        S: Data<Elem = Self::PhType> + DataMut + RawDataClone,
        D: Dimension + RemoveAxis;

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
    // fn len_phys(&self) -> usize;
    // fn len_spec(&self) -> usize;
    // fn coords(&self) -> &Array1<f64>;
}
