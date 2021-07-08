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
pub use chebyshev::{DirichletBc, NeumannBc};
use ndarray::prelude::*;
use ndarray::{Data, DataMut, LinalgScalar, RawDataClone, RemoveAxis};

/// Function space for Chebyshev Polynomials
///
/// .. math:
///             T{k}
/// ```
/// use ndspectral::chebyshev;
/// let ch = chebyshev(10);
/// ```
pub fn chebyshev(n: usize) -> Base {
    Base::Chebyshev(Chebyshev::new(n))
}

/// Function space with Dirichlet boundary conditions
///
/// .. math:
///         phi{k} = T{k} - T{k+2}
/// ```
/// use ndspectral::cheb_dirichlet;
/// let cd = cheb_dirichlet(10);
/// ```
pub fn cheb_dirichlet(n: usize) -> Base {
    Base::ChebDirichlet(ChebDirichlet::new(n))
}

// Function space with Neumann boundary conditions
///
/// .. math:
///     phi{k} = T{k} - k^2/(k+2)^2 T{k+2}
/// ```
/// use ndspectral::cheb_neumann;
/// let cn = cheb_neumann(10);
/// ```
pub fn cheb_neumann(n: usize) -> Base {
    Base::ChebNeumann(ChebNeumann::new(n))
}

/// Functions space for inhomogenoeus Dirichlet
/// boundary conditiosn
pub fn cheb_dirichlet_bc(n: usize) -> Base {
    Base::DirichletBc(DirichletBc::new(n))
}

/// Functions space for inhomogenoeus Neumann
/// boundary conditiosn
pub fn cheb_neumann_bc(n: usize) -> Base {
    Base::NeumannBc(NeumannBc::new(n))
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
    /// Composite Chebyshev base for enforcing Dirichlet boundary conditions
    DirichletBc(DirichletBc),
    /// Composite Chebyshev base for enforcing Neumann boundary conditions
    NeumannBc(NeumannBc),
    //Fourier(Fourier),
}

/// Transform from physical to spectral space and vice versa
///
/// The generic parameter <T> in this trait describes the
/// Scalar in spectral space. While Chebyshev spaces transform
/// to the real space, other transforms like the Fourier transform
/// end up in space.
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

/// Differentiation in spectral space
///
/// The generic type T describes the Scalar
/// type in spectral space, i.e. the type of the
/// in- and output arrays.
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
            Base::DirichletBc(ref mut b) => b.forward(input, output, axis),
            Base::NeumannBc(ref mut b) => b.forward(input, output, axis),
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
            Base::DirichletBc(ref mut b) => b.backward(input, output, axis),
            Base::NeumannBc(ref mut b) => b.backward(input, output, axis),
        }
    }
}

/// Transformation from and to parent
///
/// Returns itself for non-composite spaces like chebyshev
/// For composite spaces, transforms from e.g ChebDirichlet
/// to Chebyshev coefficients along axis
pub trait Parental<D> {
    /// From child to parent
    fn to_parent<T, S1>(&self, input: &ArrayBase<S1, D>, axis: usize) -> Array<T, D>
    where
        T: LinalgScalar + From<f64>,
        S1: Data<Elem = T>;

    /// From parent to child
    fn from_parent<T, S1>(&self, input: &ArrayBase<S1, D>, axis: usize) -> Array<T, D>
    where
        T: LinalgScalar + From<f64> + std::ops::MulAssign,
        S1: Data<Elem = T>;
}

impl Parental<Ix1> for Base {
    /// From child to parent
    #[allow(unused_variables)]
    fn to_parent<T, S1>(&self, input: &ArrayBase<S1, Ix1>, axis: usize) -> Array<T, Ix1>
    where
        T: LinalgScalar + From<f64>,
        S1: Data<Elem = T>,
    {
        let n = self.len_phys();
        let mut output = Array1::<T>::zeros(n);
        match self {
            Base::Chebyshev(_) => {
                output.assign(input);
            }
            Base::ChebDirichlet(ref b) => {
                b.stencil.to_parent(input, &mut output, 0);
            }
            Base::ChebNeumann(ref b) => {
                b.stencil.to_parent(input, &mut output, 0);
            }
            Base::DirichletBc(ref b) => {
                b.stencil.to_parent(input, &mut output, 0);
            }
            Base::NeumannBc(ref b) => {
                b.stencil.to_parent(input, &mut output, 0);
            }
        }
        output
    }

    /// From parent to child
    #[allow(unused_variables)]
    fn from_parent<T, S1>(&self, input: &ArrayBase<S1, Ix1>, axis: usize) -> Array<T, Ix1>
    where
        T: LinalgScalar + From<f64> + std::ops::MulAssign,
        S1: Data<Elem = T>,
    {
        let n = self.len_phys();
        let mut output = Array1::<T>::zeros(n);
        match self {
            Base::Chebyshev(_) => {
                output.assign(input);
            }
            Base::ChebDirichlet(ref b) => {
                b.stencil.from_parent(input, &mut output, 0);
            }
            Base::ChebNeumann(ref b) => {
                b.stencil.from_parent(input, &mut output, 0);
            }
            Base::DirichletBc(ref b) => {
                b.stencil.from_parent(input, &mut output, 0);
            }
            Base::NeumannBc(ref b) => {
                b.stencil.from_parent(input, &mut output, 0);
            }
        }
        output
    }
}

impl Parental<Ix2> for Base {
    /// From child to parent
    #[allow(unused_variables)]
    fn to_parent<T, S1>(&self, input: &ArrayBase<S1, Ix2>, axis: usize) -> Array<T, Ix2>
    where
        T: LinalgScalar + From<f64>,
        S1: Data<Elem = T>,
    {
        let shape = input.shape();
        let n = self.len_phys();
        let mut output = if axis == 0 {
            Array2::<T>::zeros((n, shape[1]))
        } else {
            Array2::<T>::zeros((shape[0], n))
        };
        match self {
            Base::Chebyshev(_) => {
                output.assign(input);
            }
            Base::ChebDirichlet(ref b) => {
                b.stencil.to_parent(input, &mut output, axis);
            }
            Base::ChebNeumann(ref b) => {
                b.stencil.to_parent(input, &mut output, axis);
            }
            Base::DirichletBc(ref b) => {
                b.stencil.to_parent(input, &mut output, axis);
            }
            Base::NeumannBc(ref b) => {
                b.stencil.to_parent(input, &mut output, axis);
            }
        }
        output
    }

    /// From parent to child
    #[allow(unused_variables)]
    fn from_parent<T, S1>(&self, input: &ArrayBase<S1, Ix2>, axis: usize) -> Array<T, Ix2>
    where
        T: LinalgScalar + From<f64> + std::ops::MulAssign,
        S1: Data<Elem = T>,
    {
        let shape = input.shape();
        let n = self.len_spec();
        let mut output = if axis == 0 {
            Array2::<T>::zeros((n, shape[1]))
        } else {
            Array2::<T>::zeros((shape[0], n))
        };
        match self {
            Base::Chebyshev(_) => {
                output.assign(input);
            }
            Base::ChebDirichlet(ref b) => {
                b.stencil.from_parent(input, &mut output, axis);
            }
            Base::ChebNeumann(ref b) => {
                b.stencil.from_parent(input, &mut output, axis);
            }
            Base::DirichletBc(ref b) => {
                b.stencil.from_parent(input, &mut output, axis);
            }
            Base::NeumannBc(ref b) => {
                b.stencil.from_parent(input, &mut output, axis);
            }
        }
        output
    }
}
