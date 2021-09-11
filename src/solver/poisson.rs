//! Poisson Solver
//! Solve equations of the form:
//! ..math:
//!  c * D2 vhat = f
//!
//! where D2 is the second derivative.
//! Alternatively, if defined, multiply rhs
//! before the solve step, i.e.
//! ..math:
//!  c * D2 vhat = A f
//!
//! For multidimensional equations, apply
//! eigendecomposition on the non - outermost
//! dimensions of the form
//! ..math:
//!   ``` (A + lam_i*C) x_i^* = b_i^* ```
//!
//! Chebyshev bases: The equation becomes
//! banded after multiplication with the pseudoinverse
//! of D2 (B2). In this case, the second equation is
//! solved, with A = B2.
use super::{MatVec, MatVecFdma, SolverScalar};
use crate::bases::BaseSpace;
use crate::field::FieldBase;
use crate::solver::utils::vec_to_array;
use crate::solver::{FdmaTensor, Solve, SolveReturn};
use ndarray::prelude::*;
use std::ops::{Add, Div, Mul};

/// Container for Poisson Solver
#[derive(Clone)]
pub struct Poisson<T, const N: usize> {
    solver: Box<FdmaTensor<T, N>>,
    matvec: Vec<Option<MatVec<T>>>,
}

impl<const N: usize> Poisson<f64, N> {
    /// Construct Poisson solver from field:
    ///
    ///  [(D2x x Iy) + (Ix x D2y)] vhat = [(Ax x Iy) + (Ix + Ay)] f
    ///
    /// Multiplication with right side is only necessary for bases
    /// who need a preconditioner to make the laplacian banded, like
    /// chebyshev bases.
    ///
    /// Bases are diagonal, when there laplacian is a diagonal matrix.
    /// This is the case for fourier bases. Other bases will be made
    /// diagonal by an eigendecomposition. This is entirely done in
    /// the `FdmaTensor` solver.
    pub fn new<T2, S>(field: &FieldBase<f64, f64, T2, S, N>, c: [f64; N]) -> Self
    where
        S: BaseSpace<f64, N, Physical = f64, Spectral = T2>,
    {
        // Gather matrices and preconditioner
        let mut laplacians: Vec<Array2<f64>> = Vec::new();
        let mut masses: Vec<Array2<f64>> = Vec::new();
        let mut is_diags: Vec<bool> = Vec::new();
        let mut matvec: Vec<Option<MatVec<f64>>> = Vec::new();
        for (axis, ci) in c.iter().enumerate() {
            // Matrices and preconditioner
            let (mat_a, mat_b, precond, is_diag) = field.ingredients_for_poisson(axis);
            let mass = mat_a;
            let laplacian = mat_b * *ci;
            let matvec_axis = precond.map(|x| MatVec::MatVecFdma(MatVecFdma::new(&x)));

            laplacians.push(laplacian);
            masses.push(mass);
            matvec.push(matvec_axis);
            is_diags.push(is_diag);
        }

        // Vectors -> Arrays
        let laplacians = vec_to_array::<&Array2<f64>, N>(laplacians.iter().collect());
        let masses = vec_to_array::<&Array2<f64>, N>(masses.iter().collect());
        let is_diag = vec_to_array::<&bool, N>(is_diags.iter().collect());

        // Solver
        let mut solver = FdmaTensor::from_matrix(laplacians, masses, is_diag, 0.);
        // Handle singularity (2D)
        if N == 2 && solver.lam[0][0].abs() < 1e-10 {
            solver.lam[0] -= 1e-10;
            println!("Poisson seems singular! Eigenvalue 0 is manipulated to help out.");
        }

        // let solver = Box::new(solver);
        Self {
            solver: Box::new(solver),
            matvec,
        }
    }
}

#[allow(unused_variables)]
impl<A> Solve<A, ndarray::Ix1> for Poisson<f64, 1>
where
    A: SolverScalar
        + Div<f64, Output = A>
        + Mul<f64, Output = A>
        + Add<f64, Output = A>
        + From<f64>,
{
    /// # Example
    fn solve<S1, S2>(
        &self,
        input: &ArrayBase<S1, Ix1>,
        output: &mut ArrayBase<S2, Ix1>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = A>,
        S2: ndarray::Data<Elem = A> + ndarray::DataMut,
    {
        if let Some(matvec) = &self.matvec[0] {
            let buffer = matvec.solve(input, 0);
            self.solver.solve(&buffer, output, 0);
        } else {
            self.solver.solve(input, output, 0);
        }
    }
}

#[allow(unused_variables)]
impl<A> Solve<A, ndarray::Ix2> for Poisson<f64, 2>
where
    A: SolverScalar
        + Div<f64, Output = A>
        + Mul<f64, Output = A>
        + Add<f64, Output = A>
        + From<f64>,
{
    /// # Example
    fn solve<S1, S2>(
        &self,
        input: &ArrayBase<S1, Ix2>,
        output: &mut ArrayBase<S2, Ix2>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = A>,
        S2: ndarray::Data<Elem = A> + ndarray::DataMut,
    {
        // Matvec
        let mut rhs = self.matvec[0]
            .as_ref()
            .map_or_else(|| input.to_owned(), |x| x.solve(input, 0));
        if let Some(x) = &self.matvec[1] {
            rhs = x.solve(&rhs, 1);
        };
        // Solve fdma-tensor
        self.solver.solve(&rhs, output, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{Field1, Field2, Space1, Space2};
    use crate::{cheb_dirichlet, fourier_r2c};
    use ndarray::array;
    use num_complex::Complex;
    //use std::f64::consts::PI;

    fn approx_eq<S, D>(result: &ndarray::ArrayBase<S, D>, expected: &ndarray::ArrayBase<S, D>)
    where
        S: ndarray::Data<Elem = f64>,
        D: ndarray::Dimension,
    {
        let dif = 1e-3;
        for (a, b) in expected.iter().zip(result.iter()) {
            if (a - b).abs() > dif {
                panic!("Large difference of values, got {} expected {}.", b, a)
            }
        }
    }

    fn approx_eq_complex<S, D>(result: &ArrayBase<S, D>, expected: &ArrayBase<S, D>)
    where
        S: ndarray::Data<Elem = Complex<f64>>,
        D: ndarray::Dimension,
    {
        let dif = 1e-3;
        for (a, b) in expected.iter().zip(result.iter()) {
            if (a.re - b.re).abs() > dif || (a.im - b.im).abs() > dif {
                panic!("Large difference of values, got {} expected {}.", b, a)
            }
        }
    }

    #[test]
    fn test_poisson1d() {
        let nx = 8;
        let space = Space1::new(&cheb_dirichlet(nx));
        let field = Field1::new(&space);
        let poisson = Poisson::new(&field, [1.0]);
        let mut b: Array1<f64> = Array1::zeros(nx);
        let mut x = Array1::<f64>::zeros(nx - 2);
        for (i, bi) in b.iter_mut().enumerate() {
            *bi = (i + 1) as f64;
        }

        // Solve Poisson
        poisson.solve(&b, &mut x, 0);
        // Python (pypde's) solution
        let y = array![0.1042, 0.0809, 0.0625, 0.0393, -0.0417, -0.0357];

        approx_eq(&x, &y);
    }

    #[test]
    fn test_poisson2d() {
        let (nx, ny) = (8, 7);
        let space = Space2::new(&cheb_dirichlet(nx), &cheb_dirichlet(ny));
        let field = Field2::new(&space);
        let poisson = Poisson::new(&field, [1.0, 1.0]);
        let mut x = Array2::<f64>::zeros((nx - 2, ny - 2));
        let b: Array2<f64> = array![
            [1., 2., 3., 4., 5., 6., 7.],
            [1., 2., 3., 4., 5., 6., 7.],
            [1., 2., 3., 4., 5., 6., 7.],
            [1., 2., 3., 4., 5., 6., 7.],
            [1., 2., 3., 4., 5., 6., 7.],
            [1., 2., 3., 4., 5., 6., 7.],
            [1., 2., 3., 4., 5., 6., 7.],
            [1., 2., 3., 4., 5., 6., 7.],
        ];

        // Solve Poisson
        poisson.solve(&b, &mut x, 0);
        // Python (pypde's) solution
        let y = array![
            [0.01869736, 0.0244178, 0.01403203, -0.0202917, -0.0196697],
            [-0.0027890, -0.004035, -0.0059870, -0.0023490, -0.0046850],
            [-0.0023900, -0.007947, -0.0085570, -0.0189310, -0.0223680],
            [-0.0038940, -0.006622, -0.0096270, -0.0079020, -0.0120490],
            [0.00025400, -0.006752, -0.0082940, -0.0316230, -0.0361640],
            [-0.0001120, -0.004374, -0.0066430, -0.0216410, -0.0262570],
        ];

        approx_eq(&x, &y);
    }

    #[test]
    fn test_poisson2d_complex() {
        let (nx, ny) = (8, 7);
        let space = Space2::new(&cheb_dirichlet(nx), &cheb_dirichlet(ny));
        let field = Field2::new(&space);
        let poisson = Poisson::new(&field, [1.0, 1.0]);
        let mut x_cmpl = Array2::<Complex<f64>>::zeros((nx - 2, ny - 2));
        let b: Array2<f64> = array![
            [1., 2., 3., 4., 5., 6., 7.],
            [1., 2., 3., 4., 5., 6., 7.],
            [1., 2., 3., 4., 5., 6., 7.],
            [1., 2., 3., 4., 5., 6., 7.],
            [1., 2., 3., 4., 5., 6., 7.],
            [1., 2., 3., 4., 5., 6., 7.],
            [1., 2., 3., 4., 5., 6., 7.],
            [1., 2., 3., 4., 5., 6., 7.],
        ];

        let b_cmpl: Array2<Complex<f64>> = b.mapv(|x| Complex::new(x, x));

        // Solve Poisson
        poisson.solve(&b_cmpl, &mut x_cmpl, 0);
        // Python (pypde's) solution
        let y = array![
            [0.01869736, 0.0244178, 0.01403203, -0.0202917, -0.0196697],
            [-0.0027890, -0.004035, -0.0059870, -0.0023490, -0.0046850],
            [-0.0023900, -0.007947, -0.0085570, -0.0189310, -0.0223680],
            [-0.0038940, -0.006622, -0.0096270, -0.0079020, -0.0120490],
            [0.00025400, -0.006752, -0.0082940, -0.0316230, -0.0361640],
            [-0.0001120, -0.004374, -0.0066430, -0.0216410, -0.0262570],
        ];
        let y_cmpl: Array2<Complex<f64>> = y.mapv(|x| Complex::new(x, x));

        approx_eq_complex(&x_cmpl, &y_cmpl);
    }

    #[test]
    fn test_poisson2d_cd_cd() {
        // Init
        let (nx, ny) = (8, 7);
        let space = Space2::new(&cheb_dirichlet(nx), &cheb_dirichlet(ny));
        let mut field = Field2::new(&space);
        let poisson = Poisson::new(&field, [1.0, 1.0]);
        let x = &field.x[0];
        let y = &field.x[1];

        // Analytical field and solution
        let n = std::f64::consts::PI / 2.;
        let mut expected = field.v.clone();
        for (i, xi) in x.iter().enumerate() {
            for (j, yi) in y.iter().enumerate() {
                field.v[[i, j]] = (n * xi).cos() * (n * yi).cos();
                expected[[i, j]] = -1. / (n * n * 2.) * field.v[[i, j]];
            }
        }

        // Solve
        field.forward();
        let input = field.to_ortho();
        let mut result = Array2::<f64>::zeros(field.vhat.raw_dim());
        poisson.solve(&input, &mut result, 0);
        field.vhat.assign(&result);
        field.backward();

        // Compare
        approx_eq(&field.v, &expected);
    }

    #[test]
    fn test_poisson2d_fo_cd() {
        // Init
        let (nx, ny) = (16, 7);
        let space = Space2::new(&fourier_r2c(nx), &cheb_dirichlet(ny));
        let mut field = Field2::new(&space);
        let poisson = Poisson::new(&field, [1.0, 1.0]);
        let x = &field.x[0];
        let y = &field.x[1];

        // Analytical field and solution
        let ny = std::f64::consts::PI / 2.;
        let nx = 2.;
        let mut expected = field.v.clone();
        for (i, xi) in x.iter().enumerate() {
            for (j, yi) in y.iter().enumerate() {
                field.v[[i, j]] = (nx * xi).cos() * (ny * yi).cos();
                expected[[i, j]] = -1. / (nx * nx + ny * ny) * field.v[[i, j]];
            }
        }

        // Solve
        field.forward();
        let input = field.to_ortho();
        let mut result = Array2::<Complex<f64>>::zeros(field.vhat.raw_dim());
        poisson.solve(&input, &mut result, 0);
        field.vhat.assign(&result);
        field.backward();

        // Compare
        approx_eq(&field.v, &expected);
    }
}
