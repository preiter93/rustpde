//! Four-diagonal matrix solver for multidimensional problems
//!
//! Only one dimension needs to be four-diagonal. All other
//! dimensions are diagonalized by an eigendecomposition. This
//! adds two matrix multiplications per dimension to the solve
//! step, one before the fdma solver, and one after.
#![allow(clippy::doc_markdown)]
use super::utils::{diag, eig, inv};
use super::Fdma;
use super::Solve;
use super::SolverScalar;
use ndarray::{Array1, Array2, ArrayBase, Ix1, Ix2, Zip};
use ndarray::{Data, DataMut};
use std::ops::{Add, Div, Mul};

/// Tensor solver handles non-seperable multidimensional
/// systems, by diagonalizing all, but one, dimension
/// via a eigendecomposition. This makes the problem,
/// banded along the not-diagonalized direction.
///
/// In 2-D, the equations for each row look like:
/// .. math::
/// (A + (lam_i + alpha)*C)x_i = b_i
///
///  x,b: Matrix ( M x N )
///
///  A: Matrix ( N x N )
///    banded with diagonals in offsets  0, 2
///
///  C: Matrix ( N x N )
///    banded with diagonals in offsets -2, 0, 2, 4
///
///  lam: Eigenvector ( M )
///
/// Derivation:
///
/// Starting from the equation
/// .. math::
///
///  [(Ax x Cy) + (Cx x Ay) + alpha (Cx x Cy)] g = f
///
/// where 'x' is the Kronecker product operator.
///
/// Multiplying it by the inverse of Cx, CxI
/// .. math::
///
/// [(CxI @ Ax x Cy) + (Ix x Ay) + alpha (Ix x Cy)] g = (CxI x Iy) f
///
/// Applying a eigen-decomposition on CxI @ Ax = Qx lam QxI,
/// and multiplying the above equation with QxI from the left
/// .. math::
///
/// [(lam*QxI x Cy) + (QxI x Ay) + + alpha (QxI x Cy)] g = (QxI@CxI x Iy) f
///
/// This equation is solved in 3 steps:
///
/// 1. Transform f (x):
/// .. math::
///
///    fhat = (QxI @ CxI x Iy) f = self.p.dot( f )
///
///  2. Solve the system, that is now seperable and banded (y)
/// .. math::
///
///    (Ay + (lam_i + alpha)*Cy)ghat_i = fhat_i
///
///  3. Transfrom ghat back to g (x)
/// .. math::
///
///    g = Qx ghat = self.q.dot(ghat)
#[derive(Debug, Clone)]
#[allow(clippy::similar_names)]
pub struct FdmaTensor<T, const N: usize> {
    /// Problem size
    pub n: usize,
    /// One dimensional fdma solver (four diagonal sparse)
    pub fdma: [Fdma<T>; 2],
    /// Multiply before, of size (N-1)
    pub fwd: Vec<Option<Array2<T>>>,
    /// Multiply after, of size (N-1)
    pub bwd: Vec<Option<Array2<T>>>,
    /// Eigenvalues, of size (N-1)
    pub lam: Vec<Array1<T>>,
    /// Define wether problem is singular (pure neumann for example)
    pub singular: bool,
    /// Additional constant for hholtz problems
    pub alpha: T,
}

impl<const N: usize> FdmaTensor<f64, N> {
    /// Supply array of matrices a and c, as defined in the definition of `FdmaTensor`.
    ///
    /// Eigendecompoiton:
    ///
    /// The first N-1 dimensions are diagonalized by an eigendecomposition,
    /// If the matrices of a particular dimension are already diagonal,
    /// the respective place in variable `a_is_diag` should be set to true.
    /// In this case, the eigenvalues must be supplied in 'a' as a diagonal matrix,
    /// and c is not used any further.
    ///
    /// 1-Dimensional problems:
    ///
    /// In this case, only a, which must be a banded matrix, is used in solve.
    #[allow(clippy::many_single_char_names, clippy::similar_names)]
    pub fn from_matrix(
        a: [&Array2<f64>; N],
        c: [&Array2<f64>; N],
        a_is_diag: [&bool; N],
        alpha: f64,
    ) -> Self {
        //todo!()
        let mut fwd: Vec<Option<Array2<f64>>> = Vec::new();
        let mut bwd: Vec<Option<Array2<f64>>> = Vec::new();
        let mut lam: Vec<Array1<f64>> = Vec::new();
        // Inner dimensions
        for i in 0..N - 1 {
            if *a_is_diag[i] {
                lam.push(diag(a[i], 0));
                fwd.push(None);
                bwd.push(None);
            } else {
                let xmat = inv(c[i]).dot(a[i]);
                let (l, q, p) = eig(&xmat);
                lam.push(l);
                fwd.push(Some(p.dot(&inv(c[i]))));
                bwd.push(Some(q));
            }
        }
        // Outermost
        let n = a[N - 1].shape()[0];
        let fdma = [
            Fdma::from_matrix_raw(a[N - 1]),
            Fdma::from_matrix_raw(c[N - 1]),
        ];
        // Initialize
        let mut tensor = FdmaTensor {
            n,
            fdma,
            fwd,
            bwd,
            lam,
            singular: false,
            alpha,
        };

        // For 1-D problems, the forward sweep
        // can already perfomered beforehand
        if N == 1 {
            tensor.fdma[0].sweep();
        }
        // Return
        tensor
    }
}

impl<S> Solve<S, Ix1> for FdmaTensor<f64, 1>
where
    S: SolverScalar
        + std::ops::Div<f64>
        + std::ops::Mul<f64>
        + std::ops::Add<f64>
        + Div<f64, Output = S>
        + Mul<f64, Output = S>
        + Add<f64, Output = S>,
{
    /// Solve 1-D
    fn solve<S1: Data<Elem = S>, S2: Data<Elem = S> + DataMut>(
        &self,
        input: &ArrayBase<S1, Ix1>,
        output: &mut ArrayBase<S2, Ix1>,
        axis: usize,
    ) {
        if input.shape()[0] != self.n {
            panic!(
                "Dimension mismatch in Tensor! Got {} vs. {}.",
                input.len(),
                self.n
            );
        }
        self.fdma[0].solve(input, output, axis);
    }
}

impl<S> Solve<S, Ix2> for FdmaTensor<f64, 2>
where
    S: SolverScalar
        + std::ops::Div<f64>
        + std::ops::Mul<f64>
        + std::ops::Add<f64>
        + Div<f64, Output = S>
        + Mul<f64, Output = S>
        + Add<f64, Output = S>,
{
    /// Solve 2-D Problem with real in and output
    fn solve<S1: Data<Elem = S>, S2: Data<Elem = S> + DataMut>(
        &self,
        input: &ArrayBase<S1, Ix2>,
        output: &mut ArrayBase<S2, Ix2>,
        _axis: usize,
    ) {
        if input.shape()[0] != self.lam[0].len() || input.shape()[1] != self.n {
            panic!(
                "Dimension mismatch in Tensor! Got {} vs. {} (0) and {} vs. {} (1).",
                input.shape()[0],
                self.lam[0].len(),
                input.shape()[1],
                self.n
            );
        }

        // Step 1: Forward Transform rhs along x
        if let Some(p) = &self.fwd[0] {
            let p_cast: Array2<S> = p.mapv(|x| x.into());
            output.assign(&p_cast.dot(input));
        } else {
            output.assign(input);
        }

        // Step 2: Solve along y (but iterate over all lanes in x)
        Zip::from(output.outer_iter_mut())
            .and(self.lam[0].outer_iter())
            .par_for_each(|mut out, lam| {
                let l = lam.as_slice().unwrap()[0] + self.alpha;
                let mut fdma = &self.fdma[0] + &(&self.fdma[1] * l);
                fdma.sweep();
                fdma.solve(&out.to_owned(), &mut out, 0);
            });

        // Step 3: Backward Transform solution along x
        if let Some(q) = &self.bwd[0] {
            let q_cast: Array2<S> = q.mapv(|x| x.into());
            output.assign(&q_cast.dot(output));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Dimension;
    use ndarray::{Array1, Array2};
    use num_complex::Complex;

    fn approx_eq<S, D>(result: &ArrayBase<S, D>, expected: &ArrayBase<S, D>)
    where
        S: Data<Elem = f64>,
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
        S: Data<Elem = Complex<f64>>,
        D: Dimension,
    {
        let dif = 1e-3;
        for (a, b) in expected.iter().zip(result.iter()) {
            if (a.re - b.re).abs() > dif || (a.im - b.im).abs() > dif {
                panic!("Large difference of values, got {} expected {}.", b, a)
            }
        }
    }

    fn test_matrix(nx: usize) -> Array2<f64> {
        let mut matrix = Array2::<f64>::zeros((nx, nx));
        for i in 0..nx {
            let j = (i + 1) as f64;
            matrix[[i, i]] = 0.5 * j;
            if i > 1 {
                matrix[[i, i - 2]] = 10. * j;
            }
            if i < nx - 2 {
                matrix[[i, i + 2]] = 1.5 * j;
            }
            if i < nx - 4 {
                matrix[[i, i + 4]] = 2.5 * j;
            }
        }
        matrix
    }

    #[test]
    fn test_tensor1d() {
        type Ty = f64;
        let nx = 6;
        let mut data = Array1::<Ty>::zeros(nx);
        let mut result = Array1::<Ty>::zeros(nx);
        for (i, v) in data.iter_mut().enumerate() {
            *v = i as f64;
        }
        let matrix = test_matrix(nx);
        let solver = FdmaTensor::from_matrix([&matrix], [&matrix], [&false], 0.);
        solver.solve(&data, &mut result, 0);
        let recover: Array1<f64> = matrix.dot(&result);
        approx_eq(&recover, &data);
    }

    #[test]
    fn test_tensor1d_complex() {
        type Ty = Complex<f64>;
        let nx = 6;
        let mut data = Array1::<Ty>::zeros(nx);
        let mut result = Array1::<Ty>::zeros(nx);
        for (i, v) in data.iter_mut().enumerate() {
            v.re = (i + 0) as f64;
            v.im = (i + 1) as f64;
        }
        let matrix = test_matrix(nx);
        let solver = FdmaTensor::from_matrix([&matrix], [&matrix], [&false], 0.);
        solver.solve(&data, &mut result, 0);
        let recover: Array1<Ty> = matrix.mapv(|x| Complex::new(x, 0.)).dot(&result);
        approx_eq_complex(&recover, &data);
    }

    #[test]
    fn test_tensor2d() {
        type Ty = f64;
        let nx = 6;

        let mut data: Array2<Ty> = Array2::zeros((6, 6));
        let mut result = Array2::<Ty>::zeros((nx, nx));
        for (i, v) in data.iter_mut().enumerate() {
            *v = i as f64;
        }
        // Test arrays
        let a = ndarray::array![
            [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, -1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
        ];
        let c = ndarray::array![
            [0.41666, 0.0, -0.2083, 0.0, 0.041666, 0.0],
            [0.0, 0.104166, 0.0, -0.0833, 0.0, 0.0208],
            [-0.0208, 0.0, 0.0542, 0.0, -0.0333, 0.0],
            [0.0, -0.0125, 0.0, 0.033333, 0.0, -0.020833],
            [0.0, 0.0, -0.00833, 0.0, 0.00833, 0.0],
            [0.0, 0.0, 0.0, -0.00595, 0.0, 0.00595]
        ];

        let solver = FdmaTensor::from_matrix([&a, &a], [&c, &c], [&false, &false], 0.);
        solver.solve(&data, &mut result, 0);

        // Recover b
        let x = result.clone();
        let recover = a.dot(&x).dot(&(c.t())) + c.dot(&x).dot(&(a.t()));
        approx_eq(&recover, &data);
    }

    #[test]
    fn test_tensor2d_complex() {
        type Ty = Complex<f64>;
        let nx = 6;

        let mut data: Array2<Ty> = Array2::zeros((6, 6));
        let mut result = Array2::<Ty>::zeros((nx, nx));
        for (i, v) in data.iter_mut().enumerate() {
            v.re = (i + 0) as f64;
            v.im = (i + 1) as f64;
        }
        // Test arrays
        let a = ndarray::array![
            [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, -1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
        ];
        let c = ndarray::array![
            [0.41666, 0.0, -0.2083, 0.0, 0.041666, 0.0],
            [0.0, 0.104166, 0.0, -0.0833, 0.0, 0.0208],
            [-0.0208, 0.0, 0.0542, 0.0, -0.0333, 0.0],
            [0.0, -0.0125, 0.0, 0.033333, 0.0, -0.020833],
            [0.0, 0.0, -0.00833, 0.0, 0.00833, 0.0],
            [0.0, 0.0, 0.0, -0.00595, 0.0, 0.00595]
        ];
        let ac = a.mapv(|x| Complex::new(x, 0.));
        let cc = c.mapv(|x| Complex::new(x, 0.));

        let solver = FdmaTensor::from_matrix([&a, &a], [&c, &c], [&false, &false], 0.);
        solver.solve(&data, &mut result, 0);

        // Recover b
        let x = result.clone();
        let recover = ac.dot(&x).dot(&(cc.t())) + cc.dot(&x).dot(&(ac.t()));
        approx_eq_complex(&recover, &data);
    }
}
