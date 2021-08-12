//! Four-diagonal matrix solver
use super::Solve;
use super::{diag, SolverScalar, Tdma};
use ndarray::prelude::*;
use ndarray::ScalarOperand;
use ndarray::{Data, DataMut, RemoveAxis, Zip};
use std::ops::{Add, Div, Mul};

/// Solve banded system with diagonals-offsets: -2, 0, 2, 4
#[derive(Debug, Clone)]
pub struct Fdma<T> {
    /// Size of matrix (= size of main diagonal)
    pub n: usize,
    /// Lower diagonal (-2)
    pub low: Array1<T>,
    /// Main diagonal
    pub dia: Array1<T>,
    /// Upper diagonal (+2)
    pub up1: Array1<T>,
    /// Upper diagonal (+4)
    pub up2: Array1<T>,
    /// ensure forward sweep is performed before solve
    sweeped: bool,
}

impl<T> Fdma<T>
where
    T: SolverScalar,
{
    /// Initialize Fdma from matrix.
    /// Extracts the diagonals.
    /// Precomputes the forward sweep.
    pub fn from_matrix(a: &Array2<T>) -> Self {
        let mut fdma = Fdma::from_matrix_raw(a);
        fdma.sweep();
        fdma
    }

    /// Initialize Fdma from matrix.
    /// Extracts only diagonals; no forward sweep is performed.
    /// Note that self.solve, for performance reasons, does not
    /// do the `forward_sweep` itself. So, if `from_matrix_raw`
    /// is used, this step must be executed manually before solve
    pub fn from_matrix_raw(a: &Array2<T>) -> Self {
        Fdma {
            n: a.shape()[0],
            low: diag(a, -2),
            dia: diag(a, 0),
            up1: diag(a, 2),
            up2: diag(a, 4),
            sweeped: false,
        }
    }

    /// Initialize `Fdma` from diagonals
    /// Precomputes the forward sweep.
    pub fn from_diags(low: &Array1<T>, dia: &Array1<T>, up1: &Array1<T>, up2: &Array1<T>) -> Self {
        let mut fdma = Fdma {
            n: dia.len(),
            low: low.to_owned(),
            dia: dia.to_owned(),
            up1: up1.to_owned(),
            up2: up2.to_owned(),
            sweeped: false,
        };
        fdma.sweep();
        fdma
    }

    /// Precompute forward sweep.
    /// The Arrays l,m,u1,u2 will deviate from the
    /// diagonals of the original matrix.
    pub fn sweep(&mut self) {
        for i in 2..self.n {
            self.low[i - 2] /= self.dia[i - 2];
            self.dia[i] -= self.low[i - 2] * self.up1[i - 2];
            if i < self.n - 2 {
                self.up1[i] -= self.low[i - 2] * self.up2[i - 2];
            }
        }
        self.sweeped = true;
    }

    fn solve_lane<A>(&self, input: &mut ArrayViewMut1<A>)
    where
        A: SolverScalar + Div<T, Output = A> + Mul<T, Output = A> + Add<T, Output = A>,
    {
        self.fdma(input);
    }

    /// Banded matrix solver
    ///     Ax = b
    /// where A is banded with diagonals in offsets -2, 0, 2, 4
    ///
    /// l:  sub-diagonal (-2)
    /// m:  main-diagonal (0)
    /// u1: sub-diagonal (+2)
    /// u2: sub-diagonal (+2)
    #[allow(clippy::many_single_char_names)]
    #[allow(clippy::assign_op_pattern)]
    pub fn fdma<A>(&self, x: &mut ArrayViewMut1<A>)
    where
        A: SolverScalar + Div<T, Output = A> + Mul<T, Output = A> + Add<T, Output = A>,
    {
        let n = self.n;

        for i in 2..n {
            x[i] = x[i] - x[i - 2] * self.low[i - 2];
        }

        x[n - 1] = x[n - 1] / self.dia[n - 1];
        x[n - 2] = x[n - 2] / self.dia[n - 2];
        x[n - 3] = (x[n - 3] - x[n - 1] * self.up1[n - 3]) / self.dia[n - 3];
        x[n - 4] = (x[n - 4] - x[n - 2] * self.up1[n - 4]) / self.dia[n - 4];
        for i in (0..n - 4).rev() {
            x[i] = (x[i] - x[i + 2] * self.up1[i] - x[i + 4] * self.up2[i]) / self.dia[i];
        }
    }
}

impl<T, A, D> Solve<A, D> for Fdma<T>
where
    T: SolverScalar,
    A: SolverScalar + Div<T, Output = A> + Mul<T, Output = A> + Add<T, Output = A> + From<T>,
    D: Dimension + RemoveAxis,
{
    /// # Example
    ///```
    /// use rustpde::solver::Fdma;
    /// use rustpde::solver::Solve;
    /// use ndarray::prelude::*;
    /// let nx =  6;
    /// let mut data = Array1::<f64>::zeros(nx);
    /// let mut result = Array1::<f64>::zeros(nx);
    /// let mut matrix = Array2::<f64>::zeros((nx,nx));
    /// for (i, v) in data.iter_mut().enumerate() {
    ///     *v = i as f64;
    /// }
    /// for i in 0..nx {
    ///     let j = (i+1) as f64;
    ///     matrix[[i,i]] = 0.5*j;
    ///     if i>1 {
    ///         matrix[[i,i-2]] = 10.*j;
    ///     }
    ///     if i<nx-2 {
    ///         matrix[[i,i+2]] = 1.5*j;
    ///     }
    ///  if i<nx-4 {
    ///         matrix[[i,i+4]] = 2.5*j;
    ///     }
    /// }
    /// let solver = Fdma::from_matrix(&matrix);
    /// solver.solve(&data, &mut result,0);
    /// let recover = matrix.dot(&result);
    /// for (a, b) in recover.iter().zip(data.iter()) {
    ///     if (a - b).abs() > 1e-4 {
    ///         panic!("Large difference of values, got {} expected {}.", b, a)
    ///     }
    /// }
    ///```
    fn solve<S1: Data<Elem = A>, S2: Data<Elem = A> + DataMut>(
        &self,
        input: &ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) {
        assert!(
            self.sweeped,
            "Fdma: Forward sweep must be performed for solve! Abort."
        );
        output.assign(input);
        Zip::from(output.lanes_mut(Axis(axis))).par_for_each(|mut out| {
            self.solve_lane(&mut out);
        });
    }
}

/// Addition : Fdma + Fdma
impl<'a, 'b, T: SolverScalar> Add<&'b Fdma<T>> for &'a Fdma<T> {
    type Output = Fdma<T>;

    fn add(self, other: &'b Fdma<T>) -> Fdma<T> {
        assert!(!self.sweeped, "Add only unsweeped Fdma!");
        Fdma {
            n: self.n,
            low: &self.low + &other.low,
            dia: &self.dia + &other.dia,
            up1: &self.up1 + &other.up1,
            up2: &self.up2 + &other.up2,
            sweeped: false,
        }
    }
}

/// Addition : Fdma + Tdma
impl<'a, 'b, T: SolverScalar> Add<&'b Tdma<T>> for &'a Fdma<T> {
    type Output = Fdma<T>;

    fn add(self, other: &'b Tdma<T>) -> Fdma<T> {
        assert!(!self.sweeped, "Add only unsweeped Fdma!");
        Fdma {
            n: self.n,
            low: &self.low + &other.low,
            dia: &self.dia + &other.dia,
            up1: &self.up1 + &other.upp,
            up2: self.up2.to_owned(),
            sweeped: false,
        }
    }
}

/// Elementwise multiplication with scalar
impl<'a, T: SolverScalar + ScalarOperand> Mul<T> for &'a Fdma<T> {
    type Output = Fdma<T>;

    fn mul(self, other: T) -> Fdma<T> {
        assert!(!self.sweeped, "Mul only unsweeped Fdma!");
        Fdma {
            n: self.n,
            low: &self.low * other,
            dia: &self.dia * other,
            up1: &self.up1 * other,
            up2: &self.up2 * other,
            sweeped: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};
    use num_complex::Complex;

    fn approx_eq<S, D>(result: &ArrayBase<S, D>, expected: &ArrayBase<S, D>)
    where
        S: Data<Elem = f64>,
        D: Dimension,
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

    #[test]
    fn test_fdma_dim1() {
        let nx = 6;
        type Ty = f64;
        let mut data = Array1::<Ty>::zeros(nx);
        let mut result = Array1::<Ty>::zeros(nx);
        let mut matrix = Array2::<Ty>::zeros((nx, nx));
        for (i, v) in data.iter_mut().enumerate() {
            *v = i as f64;
        }
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
        let solver = Fdma::from_matrix(&matrix);
        solver.solve(&data, &mut result, 0);
        let recover: Array1<Ty> = matrix.dot(&result);
        approx_eq(&recover, &data);
    }

    #[test]
    fn test_fdma_dim1_complex() {
        let nx = 6;
        type Ty = Complex<f64>;
        let mut data = Array1::<Ty>::zeros(nx);
        let mut result = Array1::<Ty>::zeros(nx);
        let mut matrix = Array2::<Ty>::zeros((nx, nx));
        for (i, v) in data.iter_mut().enumerate() {
            v.re = (i + 0) as f64;
            v.im = (i + 1) as f64;
        }
        for i in 0..nx {
            let j = (i + 1) as f64;
            matrix[[i, i]].re = 0.5 * j;
            matrix[[i, i]].im = 1.5 * j;
            if i > 1 {
                matrix[[i, i - 2]].re = 10. * j;
                matrix[[i, i - 2]].im = 12. * j;
            }
            if i < nx - 2 {
                matrix[[i, i + 2]].re = 1.5 * j;
                matrix[[i, i + 2]].im = 4.5 * j;
            }
            if i < nx - 4 {
                matrix[[i, i + 4]].re = 2.5 * j;
            }
        }
        let solver = Fdma::from_matrix(&matrix);
        solver.solve(&data, &mut result, 0);
        let recover: Array1<Ty> = matrix.dot(&result);
        approx_eq_complex(&recover, &data);
    }
}
