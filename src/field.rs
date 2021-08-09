//! # Multidimensional field of basis functions
#![allow(dead_code)]
pub mod average;
pub mod read;
pub mod write;
use crate::bases::{Space, SpaceBase};
use crate::types::{FloatNum, Scalar};
use crate::Base;
use ndarray::prelude::*;
use ndarray::IntoDimension;
use ndarray::Ix;
use ndarray::ScalarOperand;
use num_complex::Complex;
pub use read::ReadField;
use std::convert::TryInto;
pub use write::WriteField;
// use crate::SolverField;
// use std::collections::HashMap;

/// One dimensional Field (Spectral Space is Real)
pub type Field1 = FieldBase<f64, f64, 1>;
/// One dimensional Field (Spectral Space is Complex)
pub type Field1Complex = FieldBase<f64, Complex<f64>, 1>;
/// Two dimensional Field (Spectral Space is Real)
pub type Field2 = FieldBase<f64, f64, 2>;
/// Two dimensional Field (Spectral Space is Complex)
pub type Field2Complex = FieldBase<f64, Complex<f64>, 2>;

/// Transform and gradient operations
pub trait Field<T1, T2, const N: usize> {
    /// Type in Spectral space
    type Output;
    /// Forward transform of full field
    fn forward(&mut self);
    /// Backward transform of full field
    fn backward(&mut self);
    /// Transform full field to orthogonal space
    fn to_ortho(&self) -> Array<T2, Dim<[usize; N]>>;
    /// Transform full field from orthogonal space
    fn from_ortho(&mut self, input: &Array<T2, Dim<[usize; N]>>);
    /// Take gradient along axis. Optional: Rescale result by a constant.
    fn grad(&self, deriv: [usize; N], scale: Option<[T1; N]>) -> Array<T2, Dim<[usize; N]>>;
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
/// dx: list of ndarrays
///
///   Grid points deltas (physical space)
///
/// solvers: HashMap<String, `SolverField`>
///
///  Add plans for various equations
///
/// `FieldBase` is derived from `SpaceBase` struct,
/// defined in the `funspace` crate.
/// It implements forward / backward transform from physical
/// to spectral space, differentation and casting
/// from an orthonormal space to its galerkin space (`from_ortho`
/// and `to_ortho`).
///
/// # Example
/// 2-D field in chebyshev space
///```
/// use rustpde::cheb_dirichlet;
/// use rustpde::Field2;
/// let cdx = cheb_dirichlet(8);
/// let cdy = cheb_dirichlet(6);
/// let field = Field2::new(&[cdx,cdy]);
///```
#[derive(Clone)]
pub struct FieldBase<T: FloatNum, T2, const N: usize> {
    /// Number of dimensions
    pub ndim: usize,
    /// Space
    pub space: SpaceBase<T, N>,
    /// Field in physical space
    pub v: Array<T, Dim<[Ix; N]>>,
    /// Field in spectral space
    pub vhat: Array<T2, Dim<[Ix; N]>>,
    /// Grid coordinates
    pub x: [Array1<T>; N],
    /// Grid deltas
    pub dx: [Array1<T>; N],
    // /// Collection of numerical solvers (Poisson, Hholtz, ...)
    // pub solvers: HashMap<String, SolverField<T, N>>,
}

impl<T, T2, const N: usize> FieldBase<T, T2, N>
where
    T: FloatNum,
    T2: Scalar,
    [usize; N]: IntoDimension<Dim = Dim<[usize; N]>>,
    Dim<[usize; N]>: Dimension,
{
    /// Return a new field from an array of Bases
    pub fn new(bases: &[Base<T>; N]) -> Self {
        let space = SpaceBase::new(bases);
        Self {
            ndim: N,
            space: space.clone(),
            v: space.ndarray_physical(),
            vhat: space.ndarray_spectral(),
            x: space.coords(),
            dx: Self::get_dx(&space.coords()),
        }
    }

    /// Return a new field from `SpaceBase` struct
    pub fn from_space(space: &SpaceBase<T, N>) -> Self {
        Self::new(&space.bases)
    }

    /// Generate grid deltas from coordinates
    ///
    /// ## Panics
    /// When vec to array convection fails
    fn get_dx(x_arr: &[Array1<T>; N]) -> [Array1<T>; N] {
        let mut dx_vec = Vec::new();
        let two = T::one() + T::one();
        for x in x_arr.iter() {
            let mut dx = Array1::<T>::zeros(x.len());
            for (i, dxi) in dx.iter_mut().enumerate() {
                let xs_left = if i == 0 {
                    x[0]
                } else {
                    (x[i] + x[i - 1]) / two
                };
                let xs_right = if i == x.len() - 1 {
                    x[x.len() - 1]
                } else {
                    (x[i + 1] + x[i]) / two
                };
                *dxi = xs_right - xs_left;
            }
            dx_vec.push(dx);
        }
        dx_vec.try_into().unwrap_or_else(|v: Vec<Array1<T>>| {
            panic!("Expected Vec of length {} but got {}", N, v.len())
        })
    }
}

macro_rules! impl_space_functions {
    ($a: ty, $n: expr) => {
        impl<T> Field<T, $a, $n> for FieldBase<T, $a, $n>
        where
            T: FloatNum,
            $a: std::ops::DivAssign,
            Complex<T>: ScalarOperand,
        {
            type Output = $a;
            /// Forward transform 1d
            fn forward(&mut self) {
                self.space
                    .forward_space_inplace_par(&mut self.v, &mut self.vhat);
            }

            /// Backward transform 1d
            fn backward(&mut self) {
                self.space
                    .backward_space_inplace_par(&mut self.vhat, &mut self.v);
            }

            /// Transform to parent space
            fn to_ortho(&self) -> Array<$a, Dim<[usize; $n]>> {
                self.space.to_ortho_space(&self.vhat)
            }

            /// Transform to child space
            fn from_ortho(&mut self, input: &Array<$a, Dim<[usize; $n]>>) {
                self.space.from_ortho_space_inplace(input, &mut self.vhat);
            }

            /// Gradient
            // #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
            fn grad(
                &self,
                deriv: [usize; $n],
                scale: Option<[T; $n]>,
            ) -> Array<$a, Dim<[usize; $n]>> {
                self.space.gradient(&self.vhat, deriv, scale)
            }
        }
    };
}
// Float
impl_space_functions!(T, 1);
impl_space_functions!(T, 2);
// Complex
impl_space_functions!(Complex<T>, 1);
impl_space_functions!(Complex<T>, 2);

// macro_rules! impl_field1 {
//     ($a: ty) => {
//         impl<T> Field<T, $a, 1> for FieldBase<T, $a, 1>
//         where
//             T: FloatNum,
//             $a: std::ops::DivAssign,
//             Complex<T>: ScalarOperand,
//         {
//             type Output = $a;
//             /// Forward transform 1d
//             fn forward(&mut self) {
//                 self.space
//                     .forward_inplace_par(&mut self.v, &mut self.vhat, 0);
//             }

//             /// Backward transform 1d
//             fn backward(&mut self) {
//                 self.space
//                     .backward_inplace_par(&mut self.vhat, &mut self.v, 0);
//             }

//             /// Transform to parent space
//             fn to_ortho(&self) -> Array1<Self::Output> {
//                 self.space.to_ortho(&self.vhat, 0)
//             }

//             /// Transform to child space
//             fn from_ortho(&mut self, input: &Array1<Self::Output>) {
//                 self.vhat.assign(&self.space.from_ortho(input, 0));
//             }

//             /// Gradient
//             #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
//             fn grad(&self, deriv: [usize; 1], scale: Option<[T; 1]>) -> Array1<Self::Output> {
//                 let mut output = self.space.differentiate(&self.vhat, deriv[0], 0);
//                 if let Some(s) = scale {
//                     let scale_1: $a = (s[0].powi(deriv[0] as i32)).into();
//                     output /= scale_1;
//                 }
//                 output
//             }
//         }
//     };
// }
// // Float
// impl_field1!(T);
// // Complex
// impl_field1!(Complex<T>);

// macro_rules! impl_field2 {
//     ($a: ty) => {
//         impl<T> Field<T, $a, 2> for FieldBase<T, $a, 2>
//         where
//             T: FloatNum,
//             $a: std::ops::DivAssign,
//             Complex<T>: ScalarOperand,
//         {
//             type Output = $a;
//             /// Forward transform 1d
//             fn forward(&mut self) {
//                 let mut buffer = self.space.forward_par(&mut self.v, 1);
//                 self.space
//                     .forward_inplace_par(&mut buffer, &mut self.vhat, 0);
//             }

//             /// Backward transform 1d
//             fn backward(&mut self) {
//                 let mut buffer = self.space.backward_par(&mut self.vhat, 0);
//                 self.space.backward_inplace_par(&mut buffer, &mut self.v, 1);
//             }

//             /// Transform to parent space
//             fn to_ortho(&self) -> Array2<Self::Output> {
//                 let buffer = self.space.to_ortho(&self.vhat, 1);
//                 self.space.to_ortho(&buffer, 0)
//             }

//             /// Transform to child space
//             fn from_ortho(&mut self, input: &Array2<Self::Output>) {
//                 let buffer = self.space.from_ortho(input, 1);
//                 self.vhat.assign(&self.space.from_ortho(&buffer, 0));
//             }

//             /// Gradient
//             #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
//             fn grad(&self, deriv: [usize; 2], scale: Option<[T; 2]>) -> Array2<Self::Output> {
//                 let buffer = self.space.differentiate(&self.vhat, deriv[0], 0);
//                 let mut output = self.space.differentiate(&buffer, deriv[1], 1);
//                 if let Some(s) = scale {
//                     let scale_1: $a = (s[0].powi(deriv[0] as i32)).into();
//                     let scale_2: $a = (s[1].powi(deriv[0] as i32)).into();
//                     output /= scale_1;
//                     output /= scale_2;
//                 }
//                 output
//             }
//         }
//     };
// }

// // Float
// impl_field2!(T);
// // Complex
// impl_field2!(Complex<T>);
