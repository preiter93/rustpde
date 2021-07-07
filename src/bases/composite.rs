//! Composite bases are produced by a combination of basis functions from a orthonormal set.
//use ndarray::{Data, DataMut, Zip};

/// Transform from parent space to composite space and vice versa.
///
/// Parent (p) and composite space (c) are simply connected by a stencil matrix S, i.e.:
/// p = S c. The transform to_parent is done via matrix multiplication, while for the inverse
/// from_parent function, a system of linear equations is solved.
pub struct Composite {
    /// Number of coefficients in parent space
    pub n: usize,
    /// Number of coefficients in composite space
    pub m: usize,
}

/// Procedural macro which derives a composite Base
/// from its parent base (p) and a transform
/// stencil (s). Additionally, the identifier
/// which generate the stencil must be supplied (a);
/// it can deviate from the standard new() method.
/// Lastly, the type and after (t1) a forward
/// transform (type in spectral space) must be supplied.
#[macro_export]
macro_rules! derive_composite {
    (
        $(#[$meta:meta])* $i: ident, $p: ty, $s: ty, $a: ident, $t1: ty
    ) => {
        $(#[$meta])*
        pub struct $i {
            /// Number of coefficients in parent space
            pub n: usize,
            /// Number of coefficients in composite space
            pub m: usize,
            parent: $p,
            stencil: $s,
        }

        impl $i {
            /// Create new Basis.
            pub fn new(n: usize) -> Self {
                let m = <$s>::get_m(n);
                let stencil = <$s>::$a(n);
                let parent = <$p>::new(n);
                $i {
                    n,
                    m,
                    stencil,
                    parent,
                }
            }

            /// Return size of physical space
            pub fn len_phys(&self) -> usize {
                self.n
            }

            /// Return size of spectral space
            pub fn len_spec(&self) -> usize {
                self.m
            }

            /// Return grid coordinates
            pub fn coords(&self) -> &Array1<f64> {
                &self.parent.x
            }
        }

        impl Transform<$t1> for $i {
            /// Transform: Physical space --> Spectral space
            fn forward<R, S, D>(
                &mut self,
                input: &mut ArrayBase<R, D>,
                output: &mut ArrayBase<S, D>,
                axis: usize,
            ) where
                R: Data<Elem = Real> + DataMut + RawDataClone,
                S: Data<Elem = $t1> + DataMut,
                D: Dimension + RemoveAxis,
            {
                let mut buffer = input.clone();
                self.parent.forward(input, &mut buffer, axis);
                self.stencil.from_parent(&buffer, output, axis);
            }

            /// Transform: Spectral space --> Physical space
            fn backward<R, S, D>(
                &mut self,
                input: &mut ArrayBase<R, D>,
                output: &mut ArrayBase<S, D>,
                axis: usize,
            ) where
                R: Data<Elem = $t1> + DataMut + RawDataClone,
                S: Data<Elem = Real> + DataMut + RawDataClone,
                D: Dimension + RemoveAxis,
            {
                let mut buffer = output.clone();
                self.stencil.to_parent(input,&mut buffer,axis);
                self.parent.backward(&mut buffer, output, axis);
            }
        }

        impl Differentiate for $i {
            /// Differentiate array n_times in spectral space along
            /// axis.
            ///
            /// Returns derivative coefficients in parent space
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
                D: Dimension,
            {
                let mut buffer = output.clone();
                self.stencil.to_parent(input,&mut buffer,axis);
                self.parent.differentiate(&buffer, output, n_times, axis);
            }
        }

        impl Mass for $i {
            /// Return mass matrix
            fn mass<T>(&self) -> Array2<T>
            where
                T: LinalgScalar + From<f64>
            {
                self.stencil.to_array()
            }
            /// Return size of basis
            fn size(&self) -> usize {
                self.n
            }
        }

        impl Size for $i {
            fn len_phys(&self) -> usize {
                self.n
            }

            fn len_spec(&self) -> usize {
                self.m
            }

            fn coords(&self) -> &Array1<f64> {
                &self.parent.x
            }
        }

        impl LaplacianInverse for $i {
            /// Pseudoinverse mtrix of Laplacian
            fn pinv<T>(&self) -> Array2<T>
            where
                T: LinalgScalar + From<f64>
            {
                self.parent.pinv()
            }
            /// Pseudoidentity matrix of laplacian
            fn pinv_eye<T>(&self) -> Array2<T>
            where
                T: LinalgScalar + From<f64>
            {
                self.parent.pinv_eye()
            }
        }

    };
}
