//! # Chebyshev
//! Real-to-real transform of ndarrays from physical space to chebyshev spectral
//! space and vice versa.
//!
//! In the context of fluid simulations, chebyshev polynomials are especially
//! usefull for wall bounded flows.
use super::{Differentiate, LaplacianInverse, Mass, Size, Transform};
use crate::derive_composite;
use crate::Real;
use ndarray::prelude::*;
use ndarray::LinalgScalar;
use ndarray::{Data, DataMut, RawDataClone, RemoveAxis, Zip};
use ndrustfft::DctHandler;
// use ndarray::iter::Lanes;

/// # Orthonormal set of basis functions: Chebyshev polynomials
pub struct Chebyshev {
    /// Number of coefficients in parent space
    pub n: usize,
    /// Number of coefficients in composite space (equal to n for chebyshev)
    pub m: usize,
    /// Grid points of chebyshev polynomials in physical space.
    pub x: Array1<Real>,
    dct_handler: DctHandler<Real>,
    correct_dct: Array1<Real>,
}

impl Chebyshev {
    /// Creates a new Basis.
    ///
    /// # Arguments
    /// * `n` - Length of array's dimension which shall live in chebyshev space.
    ///
    /// # Examples
    /// ```
    /// use ndspectral::bases::Chebyshev;
    /// let cheby = Chebyshev::new(10);
    /// ```
    pub fn new(n: usize) -> Self {
        Chebyshev {
            n,
            m: n,
            x: Chebyshev::nodes_2nd_kind(n),
            dct_handler: DctHandler::new(n),
            correct_dct: Chebyshev::alternating_ones(n),
        }
    }

    /// Chebyshev nodes of the second kind, includes -1 and 1
    fn nodes_2nd_kind(n: usize) -> Array1<Real> {
        use std::f64::consts::PI;
        let m: Real = (n - 1) as Real;
        let mut grid = Array1::zeros(n);
        for (k, x) in grid.iter_mut().enumerate() {
            let arg = PI as Real * (m - 2.0 as Real * k as Real) / (2. as Real * m);
            *x = -arg.sin();
        }
        grid
    }

    /// Array of ones with alternating signs. This array is used
    /// as a correction to the DCT in the forward and
    /// backward transform, which deviates slightly from the
    /// chebyshev transform.
    /// For performance reason, it is also stored in the chebyshev
    /// struct.
    fn alternating_ones(n: usize) -> Array1<Real> {
        let mut sign = Array1::<Real>::zeros(n);
        for (i, a) in sign.iter_mut().enumerate() {
            *a = (-1. as Real).powf(i as Real);
        }
        sign
    }

    fn check_array<T, S, D>(&self, data: &ArrayBase<S, D>, axis: usize)
    where
        T: LinalgScalar,
        S: Data<Elem = T>,
        D: Dimension,
    {
        assert!(
            self.n == data.shape()[axis],
            "Size mismatch in fft, got {} expected {}",
            data.shape()[axis],
            self.n
        );
    }
}

impl Chebyshev {
    /// Pseudoinverse matrix of chebyshev spectral
    /// differentiation matrices
    ///
    /// When preconditioned with the pseudoinverse Matrix,
    /// systems become banded and thus efficient to solve.
    ///
    /// Literature:
    /// Sahuck Oh - An Efficient Spectral Method to Solve Multi-Dimensional
    /// Linear Partial Different Equations Using Chebyshev Polynomials
    ///
    /// Output:
    /// ndarray (n x n) matrix, acts in spectral space
    fn _pinv<T>(n: usize, deriv: usize) -> Array2<T>
    where
        T: LinalgScalar,
        f64: Into<T>,
    {
        if deriv > 2 {
            panic!("pinv does only support deriv's 1 & 2, got {}", deriv)
        }
        let mut pinv = Array2::<f64>::zeros([n, n]);
        if deriv == 1 {
            pinv[[1, 0]] = 1.;
            for i in 2..n {
                pinv[[i, i - 1]] = 1. / (2. * i as f64); // diag - 1
            }
            for i in 1..n - 2 {
                pinv[[i, i + 1]] = -1. / (2. * i as f64); // diag + 1
            }
        } else if deriv == 2 {
            pinv[[2, 0]] = 0.25;
            for i in 3..n {
                pinv[[i, i - 2]] = 1. / (4 * i * (i - 1)) as f64; // diag - 2
            }
            for i in 2..n - 2 {
                pinv[[i, i]] = -1. / (2 * (i * i - 1)) as f64; // diag 0
            }
            for i in 2..n - 4 {
                pinv[[i, i + 2]] = 1. / (4 * i * (i + 1)) as f64; // diag + 2
            }
        }
        //pinv
        pinv.mapv(|elem| elem.into())
    }

    /// Returns eye matrix, where the n ( = deriv) upper rows
    /// are removed
    fn _pinv_eye<T>(n: usize, deriv: usize) -> Array2<T>
    where
        T: LinalgScalar,
        f64: Into<T>,
    {
        let pinv_eye = Array2::<f64>::eye(n).slice(s![deriv.., ..]).to_owned();
        pinv_eye.mapv(|elem| elem.into())
    }
}

impl Transform<Real> for Chebyshev {
    /// Transform: Physical space --> Chebyshev space
    ///
    /// The transform is conducted along a single axis.
    /// Size of axis must match chebyshev's parameter *n*.
    /// The major workload is done by ndrustfft's discrete
    /// cosine transform. Ndrustffts features parallelization.
    ///
    /// The inverse backward transform is done by: [`Chebyshev::backward`]
    ///
    /// # Example
    /// Forward transform along first axis
    /// ```
    /// use ndspectral::bases::{Chebyshev, Transform};
    /// use ndarray::{Array, Dim, Ix};
    /// let (nx, ny) = (6, 4);
    /// let mut cheby = Chebyshev::new(nx);
    /// let mut data = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, ny));
    /// let mut vhat = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, ny));
    /// for (i, v) in data.iter_mut().enumerate() {
    ///     *v = i as f64;
    /// }
    /// cheby.forward(&mut data, &mut vhat, 0);
    /// ```
    fn forward<S1, S2, D>(
        &mut self,
        input: &mut ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: Data<Elem = Real> + DataMut + RawDataClone,
        S2: Data<Elem = Real> + DataMut,
        D: Dimension + RemoveAxis,
    {
        use ndrustfft::nddct1_par as nddct1;
        self.check_array(input, axis);
        self.check_array(output, axis);
        // discrete cosine tranform (type 1)
        nddct1(
            &mut input.view_mut(),
            &mut output.view_mut(),
            &mut self.dct_handler,
            axis,
        );
        // Correct DCT-I to find chebyshev coefficients
        let corrector = 1. / ((self.n - 1) as Real * &self.correct_dct);
        for mut v in output.lanes_mut(Axis(axis)) {
            v *= &corrector;
            v[0] /= 2 as Real;
            v[self.n - 1] /= 2 as Real;
        }
    }

    /// Transform: Chebyshev space --> Physical space
    ///
    /// The inverse forward transform is done by: [`Chebyshev::forward`]
    ///
    /// # Example
    /// Backward transform along first axis
    /// ```
    /// use ndspectral::bases::{Chebyshev, Transform};
    /// use ndarray::{Array, Dim, Ix};
    /// let (nx, ny) = (6, 4);
    /// let mut cheby = Chebyshev::new(nx);
    /// let mut data = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, ny));
    /// let mut vhat = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, ny));
    /// for (i, v) in data.iter_mut().enumerate() {
    ///     *v = i as f64;
    /// }
    /// cheby.backward(&mut data, &mut vhat, 0);
    /// ```
    fn backward<S1, S2, D>(
        &mut self,
        input: &mut ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: Data<Elem = Real> + DataMut + RawDataClone,
        S2: Data<Elem = Real> + DataMut,
        D: Dimension + RemoveAxis,
    {
        use ndrustfft::nddct1_par as nddct1;
        self.check_array(input, axis);
        self.check_array(output, axis);
        let mut buffer = input.clone();
        // correction step
        let corrector = &self.correct_dct / 2 as Real;
        for mut v in buffer.lanes_mut(Axis(axis)) {
            v *= &corrector;
            v[0] *= 2 as Real;
            v[self.n - 1] *= 2 as Real;
        }
        // dct
        nddct1(
            &mut buffer.view_mut(),
            &mut output.view_mut(),
            &mut self.dct_handler,
            axis,
        );
    }
}

impl Differentiate for Chebyshev {
    /// Differentiate array n_times in spectral space along axis.
    ///
    /// Size of axis must match chebyshev's parameter *n*.
    ///
    /// # Example
    /// Differentiate once along first axis
    /// ```
    /// use ndspectral::bases::{Chebyshev, Differentiate};
    /// use ndarray::{Array, Dim, Ix};
    /// let (nx, ny) = (6, 4);
    /// let cheby = Chebyshev::new(nx);
    /// let mut data = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, ny));
    /// let mut diff = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, ny));
    /// for (i, v) in data.iter_mut().enumerate() {
    ///     *v = i as f64;
    /// }
    /// cheby.differentiate(&data,&mut diff, 1, 0);
    /// ```
    fn differentiate<T, S1, S2, D>(
        &self,
        input: &ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        n_times: usize,
        axis: usize,
    ) where
        T: LinalgScalar + Send,
        f64: Into<T>,
        S1: Data<Elem = T>,
        S2: Data<Elem = T> + DataMut,
        D: Dimension,
    {
        self.check_array(input, axis);
        output.assign(&input);
        if n_times > 0 {
            Zip::from(output.lanes_mut(Axis(axis))).par_for_each(|mut out| {
                for _ in 0..n_times {
                    out[0] = out[1];
                    for i in 1..out.len() - 1 {
                        out[i] = (2. * (i as f64 + 1.)).into() * out[i + 1];
                    }
                    out[self.n - 1] = T::zero();
                    // Add d_x(T_(n-2))
                    for i in (1..self.n - 2).rev() {
                        out[i] = out[i] + out[i + 2];
                    }
                    out[0] = out[0] + out[2] / 2.0.into();
                }
            });
        }
    }
}

impl Mass for Chebyshev {
    /// Return mass matrix
    fn mass<T: LinalgScalar + From<f64>>(&self) -> Array2<T> {
        Array2::<T>::eye(self.n)
    }
    /// Return size of basis
    fn size(&self) -> usize {
        self.n
    }
}

impl Size for Chebyshev {
    fn len_phys(&self) -> usize {
        self.n
    }

    fn len_spec(&self) -> usize {
        self.n
    }

    fn coords(&self) -> &Array1<f64> {
        &self.x
    }
}

impl LaplacianInverse for Chebyshev {
    /// Pseudoinverse matrix of chebyshev spectral
    /// differentiation matrices
    ///
    /// When preconditioned with the pseudoinverse Matrix,
    /// systems become banded and thus efficient to solve.
    ///
    /// For example:
    /// ```
    /// use ndspectral::bases::{Chebyshev};
    /// use ndarray::Array2;
    /// use crate::ndspectral::bases::LaplacianInverse;
    /// let nx = 5;
    /// let cheby = Chebyshev::new(nx);
    /// let pinv: Array2<f64> = cheby.pinv();
    /// ```
    fn pinv<T>(&self) -> Array2<T>
    where
        T: LinalgScalar + From<f64>,
    {
        Self::_pinv(self.n, 2)
    }
    /// Pseudoidentity matrix of laplacian
    fn pinv_eye<T>(&self) -> Array2<T>
    where
        T: LinalgScalar + From<f64>,
    {
        Self::_pinv_eye(self.n, 2)
    }
}

derive_composite!(
    /// # ChebDirichlet
    /// Composite set of basis functions based on Chebyshev polynomials
    /// with Dirichlet type boundary conditions.
    ///
    /// # Examples
    /// ```
    /// use ndspectral::bases::ChebDirichlet;
    /// let cd = ChebDirichlet::new(10);
    /// ```
    ChebDirichlet,
    Chebyshev,
    StencilChebyshev,
    dirichlet,
    Real
);

derive_composite!(
    /// # ChebNeumann
    /// Composite set of basis functions based on Chebyshev polynomials
    /// with Neumann type boundary conditions.
    ///
    /// # Examples
    /// ```
    /// use ndspectral::bases::ChebNeumann;
    /// let cn = ChebNeumann::new(10);
    /// ```
    ChebNeumann,
    Chebyshev,
    StencilChebyshev,
    neumann,
    Real
);

derive_composite!(
    /// # DirichletBc
    /// Composite set of basis functions based on Chebyshev polynomials
    /// for Dirichlet boundary conditions.
    ///
    /// This basis should only be used for inhomogeneous BC's together
    /// with [`ChebDirichlet`]
    ///
    /// # Examples
    /// ```
    /// use ndspectral::bases::DirichletBc;
    /// let cd = DirichletBc::new(10);
    /// ```
    DirichletBc,
    Chebyshev,
    StencilChebyshevBoundary,
    dirichlet,
    Real
);

derive_composite!(
    /// # NeumannBc
    /// Composite set of basis functions based on Chebyshev polynomials
    /// for Neummann boundary conditions.
    ///
    /// This basis should only be used for inhomogeneous BC's together
    /// with [`ChebNeumann`]
    ///
    /// # Examples
    /// ```
    /// use ndspectral::bases::NeumannBc;
    /// let cd = NeumannBc::new(10);
    /// ```
    NeumannBc,
    Chebyshev,
    StencilChebyshevBoundary,
    dirichlet,
    Real
);

/// Stencil for composite chebyshev bases.
///
/// chebdirichlet:
/// .. math::
///  phi_k = T_k - T_k+2
///
/// chebneumann:
/// .. math::
///     phi_k = T_k - k^2/(k+2)^2 * T_k+2
pub struct StencilChebyshev {
    /// Number of coefficients in parent space
    n: usize,
    /// Number of coefficients in parent space
    m: usize,
    /// Main diagonal
    diag: Array1<Real>,
    /// Subdiagonal offset -2
    low2: Array1<Real>,
}

impl StencilChebyshev {
    /// Return stencil of chebyshev dirichlet space
    /// .. math::
    /// \phi_k = T_k - T_{k+2}
    ///
    /// Reference:
    /// J. Shen: Effcient Spectral-Galerkin Method II.
    ///
    /// # Example
    ///```
    /// use ndspectral::bases::chebyshev::StencilChebyshev;
    /// let stencil = StencilChebyshev::dirichlet(5);
    ///```
    pub fn dirichlet(n: usize) -> Self {
        let m = Self::get_m(n);
        let diag = Array::from_vec(vec![1.0_f64; m]);
        let low2 = Array::from_vec(vec![-1.0_f64; m]);
        StencilChebyshev { n, m, diag, low2 }
    }

    /// Return stencil of chebyshev neumann space
    /// .. math::
    /// \phi_k = T_k - k^2/(k+2)^2 * T_k+2
    ///
    /// Reference:
    /// J. Shen: Effcient Spectral-Galerkin Method II.
    ///
    /// # Example
    ///```
    /// use ndspectral::bases::chebyshev::StencilChebyshev;
    /// let stencil = StencilChebyshev::neumann(5);
    ///```
    pub fn neumann(n: usize) -> Self {
        let m = Self::get_m(n);
        let diag = Array::from_vec(vec![1.0_f64; m]);
        let mut low2 = Array::from_vec(vec![0.0_f64; m]);
        for (k, v) in low2.iter_mut().enumerate() {
            *v = -1. * k.pow(2) as Real / (k + 2).pow(2) as Real;
        }
        StencilChebyshev { n, m, diag, low2 }
    }

    /// Multiply stencil with a vector. (see test)
    pub fn to_parent<T, R, S, D>(
        &self,
        composite: &ArrayBase<R, D>,
        parent: &mut ArrayBase<S, D>,
        axis: usize,
    ) where
        T: LinalgScalar,
        f64: Into<T>,
        R: Data<Elem = T>,
        S: Data<Elem = T> + DataMut,
        D: Dimension,
    {
        self.check_array(composite, axis, self.m);
        self.check_array(parent, axis, self.n);

        for p in parent.iter_mut() {
            *p = T::zero();
        }

        Zip::from(parent.lanes_mut(Axis(axis)))
            .and(composite.lanes(Axis(axis)))
            .for_each(|mut p, c| {
                p[0] = self.diag[0].into() * c[0];
                p[1] = self.diag[1].into() * c[1];
                for i in 2..self.n - 2 {
                    p[i] = self.diag[i].into() * c[i] + self.low2[i - 2].into() * c[i - 2];
                }
                p[self.n - 2] = self.low2[self.n - 4].into() * c[self.n - 4];
                p[self.n - 1] = self.low2[self.n - 3].into() * c[self.n - 3];
            });
    }

    /// Mutliply inverse of stencil with a vector. (see test)
    ///
    /// This is done by solving a linear sytem, not
    /// by actually calculating the inverse of S.
    pub fn from_parent<T, R, S, D>(
        &self,
        parent: &ArrayBase<R, D>,
        composite: &mut ArrayBase<S, D>,
        axis: usize,
    ) where
        T: LinalgScalar + std::fmt::Debug,
        f64: Into<T>,
        R: Data<Elem = T>,
        S: Data<Elem = T> + DataMut,
        D: Dimension,
        //Lanes<'_, T, <D as ndarray::Dimension>::Smaller>: Send,
    {
        self.check_array(parent, axis, self.n);
        self.check_array(composite, axis, self.m);

        // Construct diagonal of S^T@S which is symmetric
        let mut d = Array1::zeros(self.m);
        let mut u = Array1::zeros(self.m - 2);
        for (i, x) in d.iter_mut().enumerate() {
            *x = (self.diag[i] * self.diag[i] + self.low2[i] * self.low2[i]).into();
        }
        for (i, x) in u.iter_mut().enumerate() {
            *x = (self.diag[i + 2] * self.low2[i]).into();
        }

        for c in composite.iter_mut() {
            *c = T::zero();
        }

        Zip::from(parent.lanes(Axis(axis)))
            .and(composite.lanes_mut(Axis(axis)))
            .for_each(|p, mut c| {
                // Multiply rhs
                for i in 0..self.m {
                    c[i] = self.diag[i].into() * p[i] + self.low2[i].into() * p[i + 2];
                }
                // Solve 3-diag system
                Self::tdma(&u.view(), &d.view(), &u.view(), &mut c.view_mut())
            });
    }

    /// Returns transform stencil as 2d ndarray
    pub fn to_array<T>(&self) -> Array2<T>
    where
        T: LinalgScalar,
        f64: Into<T>,
    {
        let mut mat = Array2::<f64>::zeros((self.n, self.m).f());
        for (i, (d, l)) in self.diag.iter().zip(self.low2.iter()).enumerate() {
            mat[[i, i]] = *d;
            mat[[i + 2, i]] = *l;
        }
        mat.mapv(|elem| elem.into())
    }

    /// Tridiagonal matrix solver
    ///     Ax = d
    /// where A is banded with diagonals in offsets -2, 0, 2
    ///
    /// a: sub-diagonal (-2)
    /// b: main-diagonal
    /// c: sub-diagonal (+2)
    #[allow(clippy::many_single_char_names)]
    fn tdma<T: LinalgScalar>(
        a: &ArrayView1<T>,
        b: &ArrayView1<T>,
        c: &ArrayView1<T>,
        d: &mut ArrayViewMut1<T>,
    ) {
        let n = d.len();
        let mut x = Array1::zeros(n);
        let mut w = Array1::zeros(n - 2);
        let mut g = Array1::zeros(n);

        // Forward sweep
        w[0] = c[0] / b[0];
        g[0] = d[0] / b[0];
        if c.len() > 1 {
            w[1] = c[1] / b[1];
        }
        g[1] = d[1] / b[1];

        for i in 2..n - 2 {
            w[i] = c[i] / (b[i] - a[i - 2] * w[i - 2]);
        }
        for i in 2..n {
            g[i] = (d[i] - a[i - 2] * g[i - 2]) / (b[i] - a[i - 2] * w[i - 2]);
        }

        // Back substitution
        x[n - 1] = g[n - 1];
        x[n - 2] = g[n - 2];
        for i in (1..n - 1).rev() {
            x[i - 1] = g[i - 1] - w[i - 1] * x[i + 1]
        }

        d.assign(&x);
    }

    fn check_array<T, S, D: Dimension>(&self, data: &ArrayBase<S, D>, axis: usize, n: usize)
    where
        T: LinalgScalar,
        S: Data<Elem = T>,
        D: Dimension,
    {
        assert!(
            n == data.shape()[axis],
            "Size mismatch in StencilChebyshev, got {} expected {}",
            data.shape()[axis],
            n
        );
    }

    /// Return size of spectral space (number of coefficients) from size in physical space
    pub fn get_m(n: usize) -> usize {
        n - 2
    }
}

/// Stencil for boundary conditions of chebyshev bases.
///
/// dirichlet_bc:
/// .. math::
///     phi_0 = 0.5*T_0 - 0.5*T_1
///     phi_1 = 0.5*T_0 + 0.5*T_1
///
/// neumann_bc:
/// .. math::
///     phi_0 = 0.5*T_0 - 1/8*T_1
///     phi_1 = 0.5*T_0 + 1/8*T_1
///
/// This stencil is fully defined by the
/// the 2 coefficients that act on 'T_0' and the
/// 2 coefficients that act on 'T_1'
pub struct StencilChebyshevBoundary {
    /// Number of coefficients in parent space
    n: usize,
    /// Number of coefficients in parent space
    m: usize,
    /// T0
    t0: Array1<Real>,
    /// T1
    t1: Array1<Real>,
}

impl StencilChebyshevBoundary {
    /// dirichlet_bc basis
    /// .. math::
    ///     phi_0 = 0.5*T_0 - 0.5*T_1
    ///     phi_1 = 0.5*T_0 + 0.5*T_1
    ///
    /// # Example
    ///```
    /// use ndspectral::bases::chebyshev::StencilChebyshevBoundary;
    /// let stencil = StencilChebyshevBoundary::dirichlet(5);
    ///```
    pub fn dirichlet(n: usize) -> Self {
        let m = Self::get_m(n);
        let t0 = Array::from_vec(vec![0.5, 0.5]);
        let t1 = Array::from_vec(vec![-0.5, 0.5]);
        StencilChebyshevBoundary { n, m, t0, t1 }
    }

    /// neumann_bc basis
    /// .. math::
    ///     phi_0 = 0.5*T_0 - 0.5*T_1
    ///     phi_1 = 0.5*T_0 + 0.5*T_1
    ///
    /// # Example
    ///```
    /// use ndspectral::bases::chebyshev::StencilChebyshevBoundary;
    /// let stencil = StencilChebyshevBoundary::neumann(5);
    ///```
    pub fn neumann(n: usize) -> Self {
        let m = Self::get_m(n);
        let t0 = Array::from_vec(vec![0.5, 0.5]);
        let t1 = Array::from_vec(vec![-1. / 8., 1. / 8.]);
        StencilChebyshevBoundary { n, m, t0, t1 }
    }

    /// Multiply stencil with a vector. (see test)
    pub fn to_parent<T, R, S, D>(
        &self,
        composite: &ArrayBase<R, D>,
        parent: &mut ArrayBase<S, D>,
        axis: usize,
    ) where
        T: LinalgScalar,
        f64: Into<T>,
        R: Data<Elem = T>,
        S: Data<Elem = T> + DataMut,
        D: Dimension,
    {
        self.check_array(composite, axis, self.m);
        self.check_array(parent, axis, self.n);

        for p in parent.iter_mut() {
            *p = T::zero();
        }

        Zip::from(parent.lanes_mut(Axis(axis)))
            .and(composite.lanes(Axis(axis)))
            .for_each(|mut p, c| {
                p[0] = self.t0[0].into() * c[0] + self.t0[1].into() * c[1];
                p[1] = self.t1[0].into() * c[0] + self.t1[1].into() * c[1];
            });
    }

    /// Mutliply inverse of stencil with a vector. (see test)
    ///
    /// This is done by solving a linear sytem, not
    /// by actually calculating the inverse of S.
    #[allow(clippy::many_single_char_names)]
    pub fn from_parent<T, R, S, D>(
        &self,
        parent: &ArrayBase<R, D>,
        composite: &mut ArrayBase<S, D>,
        axis: usize,
    ) where
        T: LinalgScalar + std::ops::MulAssign,
        f64: Into<T>,
        R: Data<Elem = T>,
        S: Data<Elem = T> + DataMut,
        D: Dimension,
        //Lanes<'_, T, <D as ndarray::Dimension>::Smaller>: Send,
    {
        self.check_array(composite, axis, self.m);
        self.check_array(parent, axis, self.n);

        for c in composite.iter_mut() {
            *c = T::zero();
        }

        Zip::from(parent.lanes(Axis(axis)))
            .and(composite.lanes_mut(Axis(axis)))
            .for_each(|p, mut comp| {
                //comp *= Array1T::zero();
                // S^T @ c
                let c0: T = self.t0[0].into() * p[0] + self.t1[0].into() * p[1];
                let c1: T = self.t0[1].into() * p[0] + self.t1[1].into() * p[1];
                // S^T@S = 2 x 2 matrix
                let a: T =
                    self.t0[0].into() * self.t0[0].into() + self.t1[0].into() * self.t1[0].into();
                let b: T =
                    self.t0[0].into() * self.t0[1].into() + self.t1[0].into() * self.t1[1].into();
                let c: T =
                    self.t0[1].into() * self.t0[0].into() + self.t1[1].into() * self.t1[0].into();
                let d: T =
                    self.t0[1].into() * self.t0[1].into() + self.t1[1].into() * self.t1[1].into();
                // Inverse of 2x2 matrix
                let det: T = 1.0.into() / (a * d - b * c);
                comp[0] = det * (d * c0 - b * c1);
                comp[1] = det * (a * c1 - c * c0);
            });
    }

    fn check_array<T, S, D: Dimension>(&self, data: &ArrayBase<S, D>, axis: usize, n: usize)
    where
        T: LinalgScalar,
        S: Data<Elem = T>,
        D: Dimension,
    {
        assert!(
            n == data.shape()[axis],
            "Size mismatch in StencilChebyshevBoundary, got {} expected {}",
            data.shape()[axis],
            n
        );
    }

    /// Returns transform stencil as 2d ndarray
    pub fn to_array<T>(&self) -> Array2<T>
    where
        T: LinalgScalar,
        f64: Into<T>,
    {
        let mut mat = Array2::<f64>::zeros((self.n, self.m).f());
        mat[[0, 0]] = self.t0[0];
        mat[[0, 1]] = self.t0[1];
        mat[[1, 0]] = self.t1[0];
        mat[[1, 1]] = self.t1[1];
        mat.mapv(|elem| elem.into())
    }

    /// Return size of spectral space (number of coefficients) from size in physical space
    pub fn get_m(_n: usize) -> usize {
        2
    }
}

/// Tests
#[cfg(test)]
mod test {
    use super::*;
    use ndarray::{Array, Dim, Ix};

    fn approx_eq<S, D>(result: &ArrayBase<S, D>, expected: &ArrayBase<S, D>)
    where
        S: Data<Elem = Real>,
        D: Dimension,
    {
        let dif = 1e-3;
        for (a, b) in expected.iter().zip(result.iter()) {
            if (a - b).abs() > dif {
                panic!("Large difference of values, got {} expected {}.", b, a)
            }
        }
    }

    #[test]
    /// Differantiate 2d array along first and second axis
    fn test_cheby_differentiate() {
        let (nx, ny) = (6, 4);
        let mut data = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, ny));
        let mut diff = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, ny));

        // Axis 0
        let cheby = Chebyshev::new(nx);
        for (i, v) in data.iter_mut().enumerate() {
            *v = i as f64;
        }
        let expected = array![
            [140.0, 149.0, 158.0, 167.0],
            [160.0, 172.0, 184.0, 196.0],
            [272.0, 288.0, 304.0, 320.0],
            [128.0, 136.0, 144.0, 152.0],
            [200.0, 210.0, 220.0, 230.0],
            [0.0, 0.0, 0.0, 0.0],
        ];
        cheby.differentiate(&data, &mut diff, 1, 0);
        approx_eq(&diff, &expected);

        // Axis 1
        let cheby = Chebyshev::new(ny);
        for (i, v) in data.iter_mut().enumerate() {
            *v = i as f64;
        }
        let expected = array![
            [10.0, 8.0, 18.0, 0.0],
            [26.0, 24.0, 42.0, 0.0],
            [42.0, 40.0, 66.0, 0.0],
            [58.0, 56.0, 90.0, 0.0],
            [74.0, 72.0, 114.0, 0.0],
            [90.0, 88.0, 138.0, 0.0],
        ];
        cheby.differentiate(&data, &mut diff, 1, 1);
        approx_eq(&diff, &expected);
    }

    #[test]
    /// Forward transform of 2d array along first and second axis
    fn test_cheby_forward() {
        let (nx, ny) = (6, 4);
        let mut data = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, ny));
        let mut vhat = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, ny));

        // Axis 0
        let mut cheby = Chebyshev::new(nx);
        for (i, v) in data.iter_mut().enumerate() {
            *v = i as f64;
        }
        let expected = array![
            [10.0, 11.0, 12.0, 13.0],
            [8.3777, 8.3777, 8.3777, 8.3777],
            [0.0, 0.0, 0.0, 0.0],
            [1.2222, 1.2222, 1.2222, 1.2222],
            [0.0, 0.0, 0.0, 0.0],
            [0.4, 0.4, 0.4, 0.4],
        ];
        cheby.forward(&mut data, &mut vhat, 0);
        approx_eq(&vhat, &expected);

        // Axis 1
        let mut cheby = Chebyshev::new(ny);
        for (i, v) in data.iter_mut().enumerate() {
            *v = i as f64;
        }
        let expected = array![
            [1.5, 1.3333, 0.0, 0.1667],
            [5.5, 1.3333, 0.0, 0.1667],
            [9.5, 1.3333, 0.0, 0.1667],
            [13.5, 1.3333, 0.0, 0.1667],
            [17.5, 1.3333, 0.0, 0.1667],
            [21.5, 1.3333, 0.0, 0.1667],
        ];
        cheby.forward(&mut data, &mut vhat, 1);
        approx_eq(&vhat, &expected);
    }

    #[test]
    /// Successive forward and inverse transform
    fn test_cheby_fwd_bwd() {
        let (nx, ny) = (6, 4);
        let mut data = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, ny));
        let mut vhat = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, ny));
        for (i, v) in data.iter_mut().enumerate() {
            *v = i as f64;
        }
        let expected = data.clone();

        // Axis 0
        let mut cheby = Chebyshev::new(nx);
        cheby.forward(&mut data, &mut vhat, 0);
        cheby.backward(&mut vhat, &mut data, 0);
        approx_eq(&data, &expected);

        // Axis 1
        let mut cheby = Chebyshev::new(ny);
        cheby.forward(&mut data, &mut vhat, 1);
        cheby.backward(&mut vhat, &mut data, 1);
        approx_eq(&data, &expected);
    }

    #[test]
    /// Differantiate ChebDirichlet (2d array) twice along first and second axis
    fn test_chebdirichlet_differentiate() {
        let (nx, ny) = (6, 4);
        let mut data = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, ny));

        // Axis 0
        let cheby = ChebDirichlet::new(nx + 2);
        let mut diff = Array::<f64, Dim<[Ix; 2]>>::zeros((nx + 2, ny));
        for (i, v) in data.iter_mut().enumerate() {
            *v = i as f64;
        }
        let expected = array![
            [-1440.0, -1548.0, -1656.0, -1764.0],
            [-5568.0, -5904.0, -6240.0, -6576.0],
            [-2688.0, -2880.0, -3072.0, -3264.0],
            [-4960.0, -5240.0, -5520.0, -5800.0],
            [-1920.0, -2040.0, -2160.0, -2280.0],
            [-3360.0, -3528.0, -3696.0, -3864.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ];
        cheby.differentiate(&data, &mut diff, 2, 0);
        approx_eq(&diff, &expected);

        // Axis 1
        let cheby = ChebDirichlet::new(ny + 2);
        let mut diff = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, ny + 2));
        for (i, v) in data.iter_mut().enumerate() {
            *v = i as f64;
        }
        let expected = array![
            [-56.0, -312.0, -96.0, -240.0, 0.0, 0.0],
            [-184.0, -792.0, -288.0, -560.0, 0.0, 0.0],
            [-312.0, -1272.0, -480.0, -880.0, 0.0, 0.0],
            [-440.0, -1752.0, -672.0, -1200.0, 0.0, 0.0],
            [-568.0, -2232.0, -864.0, -1520.0, 0.0, 0.0],
            [-696.0, -2712.0, -1056.0, -1840.0, 0.0, 0.0],
        ];
        cheby.differentiate(&data, &mut diff, 2, 1);
        approx_eq(&diff, &expected);
    }

    #[test]
    /// Differantiate ChebNeumann (2d array) twice along first and second axis
    fn test_chebneumann_differentiate() {
        let (nx, ny) = (6, 4);
        let mut data = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, ny));

        // Axis 0
        let cheby = ChebNeumann::new(nx + 2);
        let mut diff = Array::<f64, Dim<[Ix; 2]>>::zeros((nx + 2, ny));
        for (i, v) in data.iter_mut().enumerate() {
            *v = i as f64;
        }
        let expected = array![
            [-288.0, -308.0, -328.0, -348.0],
            [-1269.6381, -1342.9333, -1416.2286, -1489.5238],
            [-693.3333, -742.6667, -792.0, -841.3333],
            [-1602.74286, -1694.4, -1786.0571, -1877.71428],
            [-853.3333, -906.6667, -960.0, -1013.333],
            [-1714.2857, -1800.0, -1885.7143, -1971.4286],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ];
        cheby.differentiate(&data, &mut diff, 2, 0);
        approx_eq(&diff, &expected);

        // Axis 1
        let cheby = ChebNeumann::new(ny + 2);
        let mut diff = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, ny + 2));
        for (i, v) in data.iter_mut().enumerate() {
            *v = i as f64;
        }
        let expected = array![
            [-8.0, -60.2667, -24.0, -86.4, 0.0, 0.0],
            [-24.0, -147.7333, -72.0, -201.6, 0.0, 0.0],
            [-40.0, -235.2, -120.0, -316.8, 0.0, 0.0],
            [-56.0, -322.6667, -168.0, -432.0, 0.0, 0.0],
            [-72.0, -410.1333, -216.0, -547.2, 0.0, 0.0],
            [-88.0, -497.6, -264.0, -662.4, 0.0, 0.0],
        ];
        cheby.differentiate(&data, &mut diff, 2, 1);
        approx_eq(&diff, &expected);
    }

    #[test]
    /// Test transform of ChebDirichlet (2d array)
    fn test_chebdirichlet_transform() {
        let (nx, ny) = (6, 4);
        let mut data = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, ny));

        // Axis 0
        let mut cheby = ChebDirichlet::new(nx + 2);
        let mut res = Array::<f64, Dim<[Ix; 2]>>::zeros((nx + 2, ny));
        for (i, v) in data.iter_mut().enumerate() {
            *v = i as f64;
        }
        let expected = array![
            [0., 0., 0., 0.],
            [-2.773, -2.773, -2.773, -2.773],
            [7.53, 8.283, 9.036, 9.789],
            [-21.769, -21.769, -21.769, -21.769],
            [24.45, 26.895, 29.341, 31.786],
            [-45.458, -45.458, -45.458, -45.458],
            [38.019, 41.821, 45.623, 49.425],
            [0., 0., 0., 0.],
        ];
        cheby.backward(&mut data, &mut res, 0);
        approx_eq(&res, &expected);
        let expected = data.clone();
        cheby.forward(&mut res, &mut data, 0);
        approx_eq(&data, &expected);

        // Axis 1
        let mut cheby = ChebDirichlet::new(ny + 2);
        let mut res = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, ny + 2));
        for (i, v) in data.iter_mut().enumerate() {
            *v = i as f64;
        }
        let expected = array![
            [0., -0.955, 2.073, -6.545, 5.427, 0.],
            [0., -0.955, 7.601, -6.545, 19.899, 0.],
            [0., -0.955, 13.129, -6.545, 34.371, 0.],
            [0., -0.955, 18.657, -6.545, 48.843, 0.],
            [0., -0.955, 24.184, -6.545, 63.316, 0.],
            [0., -0.955, 29.712, -6.545, 77.788, 0.],
        ];
        cheby.backward(&mut data, &mut res, 1);
        approx_eq(&res, &expected);
        let expected = data.clone();
        cheby.forward(&mut res, &mut data, 1);
        approx_eq(&data, &expected);
    }

    #[test]
    /// Test transform of ChebNeumann (2d array)
    fn test_chebneumann_transform() {
        let (nx, ny) = (6, 4);
        let mut data = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, ny));

        // Axis 0
        let mut cheby = ChebNeumann::new(nx + 2);
        let mut res = Array::<f64, Dim<[Ix; 2]>>::zeros((nx + 2, ny));
        for (i, v) in data.iter_mut().enumerate() {
            *v = i as f64;
        }
        let expected = array![
            [-6.143, -5.856, -5.569, -5.282],
            [1.677, 2.324, 2.971, 3.618],
            [2.783, 3.438, 4.092, 4.747],
            [-14.913, -15.003, -15.092, -15.182],
            [21.12, 22.541, 23.961, 25.382],
            [-40.438, -41.443, -42.449, -43.454],
            [14.883, 17.949, 21.016, 24.083],
            [35.92, 40.245, 44.569, 48.893],
        ];
        cheby.backward(&mut data, &mut res, 0);
        approx_eq(&res, &expected);
        let expected = data.clone();
        cheby.forward(&mut res, &mut data, 0);
        approx_eq(&data, &expected);

        // Axis 1
        let mut cheby = ChebNeumann::new(ny + 2);
        let mut res = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, ny + 2));
        for (i, v) in data.iter_mut().enumerate() {
            *v = i as f64;
        }
        let expected = array![
            [-1.309, 0.026, 1.336, -4.881, 2.019, 4.309],
            [-0.424, 2.494, 4.871, -7.506, 11.641, 17.424],
            [0.46, 4.962, 8.406, -10.132, 21.264, 30.54],
            [1.344, 7.429, 11.942, -12.757, 30.886, 43.656],
            [2.229, 9.897, 15.477, -15.383, 40.509, 56.771],
            [3.113, 12.365, 19.012, -18.008, 50.131, 69.887],
        ];
        cheby.backward(&mut data, &mut res, 1);
        approx_eq(&res, &expected);
        let expected = data.clone();
        cheby.forward(&mut res, &mut data, 1);
        approx_eq(&data, &expected);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    fn approx_eq<S, D>(result: &ArrayBase<S, D>, expected: &ArrayBase<S, D>)
    where
        S: Data<Elem = Real>,
        D: Dimension,
    {
        let dif = 1e-3;
        for (a, b) in expected.iter().zip(result.iter()) {
            if (a - b).abs() > dif {
                panic!("Large difference of values, got {} expected {}.", b, a)
            }
        }
    }

    #[test]
    fn test_stencil_chebdirichlet() {
        let stencil = StencilChebyshev::dirichlet(5);
        // to_parent
        let mut composite = Array::from_vec(vec![2., 0.70710678, 1.]);
        let mut parent = Array1::zeros(5);
        stencil.to_parent(&composite, &mut parent, 0);
        let expected: Array1<f64> = Array::from_vec(vec![2., 0.7071, -1., -0.7071, -1.]);
        approx_eq(&parent, &expected);
        // frin_parent
        let expected = composite.clone();
        stencil.from_parent(&parent, &mut composite, 0);
        approx_eq(&composite, &expected);
        assert!(1==2);
    }

    #[test]
    fn test_stencil_dirichlet_bc() {
        let stencil = StencilChebyshevBoundary::dirichlet(4);
        // to_parent
        let mut composite = Array::from_vec(vec![1., 2.]);
        let mut parent = Array1::zeros(4);
        stencil.to_parent(&composite, &mut parent, 0);
        let expected: Array1<f64> = Array::from_vec(vec![1.5, 0.5, 0., 0.]);
        approx_eq(&parent, &expected);
        // from_parent
        let expected = composite.clone();
        stencil.from_parent(&parent, &mut composite, 0);
        approx_eq(&composite, &expected);
    }
}
