//! # Helmoltz Solver
//!  Solve equations of the form:
//!
//!  (I-c*D2) vhat = f
//!
//! where D2 is the second derivative.
//! Alternatively, if defined, multiply rhs
//! before the solve step, i.e.
//!
//!  (I-c*D2) vhat = A f
//!
//! For multidimensional equations, apply
//! alternating-direction implicit method (ADI)
//! to solve each dimension individually. But take
//! in mind that this method introduces a numerical
//! error, which is large if *c* is large.
//!
//! Chebyshev bases: The equation becomes
//! banded after multiplication with the pseudoinverse
//! of D2 (B2). In this case, the second equation is
//! solved, with A = B2.
use super::{MatVec, Solver, SolverScalar};
use crate::bases::BaseBasics;
use crate::bases::LaplacianInverse;
use crate::bases::SpaceBase;
use crate::solver::{Solve, SolveReturn};
use crate::Base;
use ndarray::prelude::*;
use ndarray::{Data, DataMut};

/// Container for Hholtz
#[derive(Clone)]
pub struct Hholtz<T, const N: usize>
// where
//     T: SolverScalar,
{
    solver: Vec<Solver<T>>,
    matvec: Vec<Option<MatVec<T>>>,
}

impl<const N: usize> Hholtz<f64, N>
// where
//     T: SolverScalar + ndarray::ScalarOperand,
//     f64: Into<T>,
{
    // /// Construct Helmholtz solver from field
    // pub fn from_field<S>(field: &Field<S, T, N>, c: [f64; N]) -> Self
    // where
    //     S: Spaced<T, N>,
    // {
    //     Self::from_space(&field.space, c)
    // }

    /// Construct Helmholtz solver from space
    pub fn from_space(space: &SpaceBase<f64, N>, c: [f64; N]) -> Self {
        let solver: Vec<Solver<f64>> = space
            .bases
            .iter()
            .enumerate()
            .map(|(i, base)| Self::solver_from_base(base, c[i]))
            .collect();

        let matvec: Vec<Option<MatVec<f64>>> = space
            .bases
            .iter()
            .map(|base| Self::matvec_from_base(base))
            .collect();

        Hholtz { solver, matvec }
    }

    /// Returns the solver for the lhs, depending on the base
    fn solver_from_base(base: &Base<f64>, c: f64) -> Solver<f64> {
        let mass = base.mass();
        let lap = base.laplace();
        let peye = base.laplace_inv_eye();
        let pinv = peye.dot(&base.laplace_inv());

        let mat = match base {
            Base::Chebyshev(_) => {
                let mass_sliced = mass.slice(s![.., 2..]);
                pinv.dot(&mass_sliced) - peye.dot(&mass_sliced) * c
            }
            Base::CompositeChebyshev(_) => pinv.dot(&mass) - peye.dot(&mass) * c,
            Base::FourierC2c(_) | Base::FourierR2c(_) => mass - lap * c,
        };
        Solver::Fdma(crate::solver::Fdma::from_matrix(&mat))
    }

    /// Returns the solver for the rhs, depending on the base
    #[allow(clippy::unnecessary_wraps)]
    fn matvec_from_base(base: &Base<f64>) -> Option<MatVec<f64>> {
        use crate::solver::MatVecDot;
        match base {
            Base::Chebyshev(_) | Base::CompositeChebyshev(_) => {
                let peye = base.laplace_inv_eye();
                let pinv = base.laplace_inv();
                //let mat = pinv.slice(ndarray::s![2.., ..]).to_owned();
                let mat = peye.dot(&pinv);
                let matvec = MatVec::MatVecDot(MatVecDot::new(&mat));
                Some(matvec)
            }
            Base::FourierC2c(_) | Base::FourierR2c(_) => None,
        }
    }
}

#[allow(unused_variables)]
impl<T, A> Solve<A, ndarray::Ix1> for Hholtz<T, 1>
where
    T: SolverScalar,
    A: SolverScalar
        + std::ops::Div<T, Output = A>
        + std::ops::Mul<T, Output = A>
        + std::ops::Add<T, Output = A>
        + From<T>,
{
    /// # Example
    fn solve<S1, S2>(
        &self,
        input: &ArrayBase<S1, Ix1>,
        output: &mut ArrayBase<S2, Ix1>,
        axis: usize,
    ) where
        S1: Data<Elem = A>,
        S2: Data<Elem = A> + DataMut,
    {
        if let Some(matvec) = &self.matvec[0] {
            let buffer = matvec.solve(input, 0);
            self.solver[0].solve(&buffer, output, 0);
        } else {
            self.solver[0].solve(input, output, 0);
        }
    }
}

#[allow(unused_variables)]
impl<T, A> Solve<A, ndarray::Ix2> for Hholtz<T, 2>
where
    T: SolverScalar,
    A: SolverScalar
        + std::ops::Div<T, Output = A>
        + std::ops::Mul<T, Output = A>
        + std::ops::Add<T, Output = A>
        + From<T>,
{
    /// # Example
    fn solve<S1, S2>(
        &self,
        input: &ArrayBase<S1, Ix2>,
        output: &mut ArrayBase<S2, Ix2>,
        axis: usize,
    ) where
        S1: Data<Elem = A>,
        S2: Data<Elem = A> + DataMut,
    {
        // Matvec
        let mut rhs = self.matvec[0]
            .as_ref()
            .map_or_else(|| input.to_owned(), |x| x.solve(input, 0));
        if let Some(x) = &self.matvec[1] {
            rhs = x.solve(&rhs, 1);
        }

        // Solve
        self.solver[0].solve(&rhs, output, 0);
        self.solver[1].solve(&output.to_owned(), output, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bases::BaseBasics;
    use crate::field::{Field, Field1, Field2, Field2Complex};
    use crate::{cheb_dirichlet, fourier_r2c};
    use ndarray::array;

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

    #[test]
    fn test_hholtz() {
        let nx = 7;
        let bases = [cheb_dirichlet(nx)];
        let field = Field1::new(&bases);
        let hholtz = Hholtz::from_space(&field.space, [1.0]);
        let b: Array1<f64> = array![1., 2., 3., 4., 5., 6., 7.];
        let mut x = Array1::<f64>::zeros(nx - 2);
        // Solve Hholtz
        hholtz.solve(&b, &mut x, 0);
        // Python's (pypde) solution
        let y = array![
            -0.08214845,
            -0.10466761,
            -0.06042153,
            0.04809052,
            0.04082296
        ];

        approx_eq(&x, &y);
    }

    #[test]
    fn test_hholtz2d() {
        let nx = 7;
        let bases = [cheb_dirichlet(nx), cheb_dirichlet(nx)];
        let field = Field2::new(&bases);
        let hholtz = Hholtz::from_space(&field.space, [1.0, 1.0]);
        let b: Array2<f64> = array![
            [1., 2., 3., 4., 5., 6., 7.],
            [1., 2., 3., 4., 5., 6., 7.],
            [1., 2., 3., 4., 5., 6., 7.],
            [1., 2., 3., 4., 5., 6., 7.],
            [1., 2., 3., 4., 5., 6., 7.],
            [1., 2., 3., 4., 5., 6., 7.],
            [1., 2., 3., 4., 5., 6., 7.],
        ];
        let mut x = Array2::<f64>::zeros((nx - 2, nx - 2));
        // Solve Hholtz
        hholtz.solve(&b, &mut x, 0);

        // Python's (pypde) solution
        let y = array![
            [-7.083e-03, -9.025e-03, -5.210e-03, 4.146e-03, 3.520e-03],
            [5.809e-04, 7.402e-04, 4.273e-04, -3.401e-04, -2.887e-04],
            [1.699e-04, 2.165e-04, 1.250e-04, -9.951e-05, -8.447e-05],
            [-1.007e-03, -1.283e-03, -7.406e-04, 5.895e-04, 5.004e-04],
            [-6.775e-04, -8.632e-04, -4.983e-04, 3.966e-04, 3.366e-04],
        ];

        // Assert
        approx_eq(&x, &y);
    }

    #[test]
    fn test_hholtz2d_cd_cd() {
        // Init
        let (nx, ny) = (16, 7);
        let bases = [cheb_dirichlet::<f64>(nx), cheb_dirichlet::<f64>(ny)];
        let mut field = Field2::new(&bases);
        let alpha = 1e-5;
        let hholtz = Hholtz::from_space(&field.space, [alpha, alpha]);
        let x = bases[0].coords();
        let y = bases[1].coords();

        // Analytical field and solution
        let n = std::f64::consts::PI / 2.;
        let mut expected = field.v.clone();
        for (i, xi) in x.iter().enumerate() {
            for (j, yi) in y.iter().enumerate() {
                field.v[[i, j]] = (n * xi).cos() * (n * yi).cos();
                expected[[i, j]] = 1. / (1. + alpha * n * n * 2.) * field.v[[i, j]];
            }
        }

        // Solve
        field.forward();
        hholtz.solve(&field.to_ortho(), &mut field.vhat, 0);
        //field.vhat.assign(&result);
        field.backward();

        // Compare
        approx_eq(&field.v, &expected);
    }

    #[test]
    fn test_hholtz2d_fo_cd() {
        // Init
        let (nx, ny) = (16, 7);
        let bases = [fourier_r2c::<f64>(nx), cheb_dirichlet::<f64>(ny)];
        let mut field = Field2Complex::new(&bases);
        let alpha = 1e-5;
        let hholtz = Hholtz::from_space(&field.space, [alpha, alpha]);
        let x = bases[0].coords();
        let y = bases[1].coords();

        // Analytical field and solution
        let n = std::f64::consts::PI / 2.;
        let mut expected = field.v.clone();
        for (i, xi) in x.iter().enumerate() {
            for (j, yi) in y.iter().enumerate() {
                field.v[[i, j]] = xi.cos() * (n * yi).cos();
                expected[[i, j]] = 1. / (1. + alpha * n * n + alpha) * field.v[[i, j]];
            }
        }

        // Solve
        field.forward();
        hholtz.solve(&field.to_ortho(), &mut field.vhat, 0);
        //field.vhat.assign(&result);
        field.backward();

        // Compare
        approx_eq(&field.v, &expected);
    }
}
