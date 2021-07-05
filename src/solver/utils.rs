//! Collection of usefull algebra methods
use ndarray::LinalgScalar;
use ndarray::{Array1, Array2};
//use ndarray_linalg::*;
use std::cmp::Ordering;
use std::convert::TryInto;

/// Return the diagonal of a one-dimensional array.
/// Parameter offset defines which diagonal is returned
pub fn diag<T: LinalgScalar>(a: &Array2<T>, offset: i8) -> Array1<T> {
    assert!(
        a.is_square(),
        "Array for method diag() must be square, but has shape {:?}",
        a.shape()
    );
    let n: usize = a.shape()[0];
    let m: usize = offset.abs().try_into().unwrap();
    if m > n {
        panic!(
            "Size of Array must be larger than offset, got {} and {}.",
            n, offset
        );
    }
    let mut diag: Array1<T> = Array1::zeros(n - m);
    if offset >= 0 {
        for (i, d) in &mut diag.iter_mut().enumerate() {
            *d = a[[i, i + m]];
        }
    } else {
        for (i, d) in &mut diag.iter_mut().enumerate() {
            *d = a[[i + m, i]];
        }
    }

    diag
}
//
// /// Return inverse of square matrix
// pub fn inv(a: &Array2<f64>) -> Array2<f64> {
//     a.inv().unwrap()
// }
//
// /// Returns real-valued eigendecomposition A = Q lam Qi,
// /// where A is a square matrix.
// /// The output is already sorted with respect to the
// /// eigenvalues, i.e. largest -> smallest.
// /// ```
// /// let test = array![
// ///         [1., 2., 3., 4., 5.],
// ///         [1., 2., 3., 4., 5.],
// ///         [1., 2., 3., 4., 5.],
// ///         [1., 2., 3., 4., 5.],
// ///         [1., 2., 3., 4., 5.]
// ///     ];
// /// let (e, evec, evec_inv) = eig(&test);
// /// ```
// pub fn eig(a: &Array2<f64>) -> (Array1<f64>, Array2<f64>, Array2<f64>) {
//     // Call eig
//     let (eval_c, evec_c) = a.clone().eig().unwrap();
//
//     // Convert complex -> f64
//     let mut eval = Array1::zeros(eval_c.raw_dim());
//     let mut evec = Array2::zeros(evec_c.raw_dim());
//     for (e, ec) in eval.iter_mut().zip(eval_c.iter()) {
//         *e = ec.re;
//     }
//     for (e, ec) in evec.iter_mut().zip(evec_c.iter()) {
//         *e = ec.re;
//     }
//
//     // Order Eigenvalues, largest first
//     let permut: Vec<usize> = argsort(eval.as_slice().unwrap())
//         .into_iter()
//         .rev()
//         .collect();
//     let eval = eval.select(Axis(0), &permut).to_owned();
//     let evec = evec.select(Axis(1), &permut).to_owned();
//
//     // Inverse of evec
//     let evec_inv = inv(&evec);
//     (eval, evec, evec_inv)
// }

/// Argsort Vector ( smallest -> largest ).
/// Returns permutation vector.
///
/// ```
/// use ndspectral::solver::utils::argsort;
/// use ndarray::{array,Axis};
/// let vec = array![3., 1., 2., 9., 7.];
/// let permut: Vec<usize> = argsort(vec.as_slice().unwrap());
/// let vec = vec.select(Axis(0), &permut).to_owned();
/// assert_eq!(vec,array![1.0, 2.0, 3.0, 7.0, 9.0]);
/// ```
pub fn argsort(vec: &[f64]) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..vec.len()).collect();

    perm.sort_by(|i, j| {
        if vec[*i] < vec[*j] {
            Ordering::Less
        } else if vec[*i] > vec[*j] {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    });
    perm
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use assert_approx_eq::assert_approx_eq;
//     use ndarray::array;
//
//     #[test]
//     fn test_eig() {
//         let test = array![
//             [1., 2., 3., 4., 5.],
//             [1., 2., 3., 4., 5.],
//             [1., 2., 3., 4., 5.],
//             [1., 2., 3., 4., 5.],
//             [1., 2., 3., 4., 5.]
//         ];
//         let (e, evec, evec_inv) = eig(&test);
//
//         // Eigenvalue vector -> Diagonal matrix
//         let mut lam = Array2::<f64>::eye(e.shape()[0]);
//         for (i, v) in e.iter().enumerate() {
//             lam[[i, i]] = v.clone();
//         }
//
//         // Check if eval and evec reproduce origonal array
//         let t = lam.dot(&evec_inv);
//         let t = evec.dot(&t);
//         for (a, b) in t.iter().zip(test.iter()) {
//             assert_approx_eq!(a, b, 1e-4f64);
//         }
//     }
// }

// pub fn vec_to_array<T, const N: usize>(v: Vec<T>) -> [T; N] {
//     v.try_into()
//         .unwrap_or_else(|v: Vec<T>| panic!("Expected a Vec of length {} but it was {}", N, v.len()))
// }
