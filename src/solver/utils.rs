//! Collection of usefull algebra methods
use ndarray::LinalgScalar;
use ndarray::{Array1, Array2};

// use ndarray_02::Array1 as Array1_old;
// use ndarray_02::Array2 as Array2_old;
use std::cmp::Ordering;
use std::convert::TryInto;

// use nalgebra::{Dynamic, OMatrix, Scalar};
// use nalgebra_lapack::Eigen;
/// Return the diagonal of a one-dimensional array.
/// Parameter offset defines which diagonal is returned
/// ## Panics
/// Panics when input is not square or requested diag
/// is larger than matrix size.
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

/// Returns real-valued eigendecomposition A = Q lam Qi,
/// where A is a square matrix.
/// The output is already sorted with respect to the
/// eigenvalues, i.e. largest -> smallest.
///
/// # Example
/// ```
/// use ndarray::array;
/// use rustpde::solver::utils::eig;
/// let test = array![
///         [1., 2., 3., 4., 5.],
///         [1., 2., 3., 4., 5.],
///         [1., 2., 3., 4., 5.],
///         [1., 2., 3., 4., 5.],
///         [1., 2., 3., 4., 5.]
///     ];
/// let (e, evec, evec_inv) = eig(&test);
/// ```
///
/// ## Panics
/// Panics if eigendecomposition or inverse fails.
pub fn eig(a: &Array2<f64>) -> (Array1<f64>, Array2<f64>, Array2<f64>) {
    use ndarray::Axis;
    use ndarray_linalg::Eig;

    // use old ndarray version, which supports linalg
    let (n, m) = (a.shape()[0], a.shape()[1]);
    let mut m = Array2::<f64>::zeros((n, m));
    for (oldv, newv) in m.iter_mut().zip(a.iter()) {
        *oldv = *newv;
    }
    let (eval_c, evec_c) = m.eig().unwrap();
    // let eval_c = ndarray_vec_to_new(&eval_c);
    // let evec_c = ndarray_to_new(&evec_c);
    // Convert complex -> f64
    let mut eval = Array1::zeros(eval_c.raw_dim());
    let mut evec = Array2::zeros(evec_c.raw_dim());
    for (e, ec) in eval.iter_mut().zip(eval_c.iter()) {
        *e = ec.re;
    }
    for (e, ec) in evec.iter_mut().zip(evec_c.iter()) {
        *e = ec.re;
    }
    // Order Eigenvalues, largest first
    let permut: Vec<usize> = argsort(eval.as_slice().unwrap())
        .into_iter()
        .rev()
        .collect();
    let eval = eval.select(Axis(0), &permut).to_owned();
    let evec = evec.select(Axis(1), &permut).to_owned();
    // Inverse of evec
    let evec_inv = inv(&evec);
    (eval, evec, evec_inv)
}

/// Return inverse of square matrix
/// ## Panics
/// Panics when computation of inverse fails.
pub fn inv(a: &Array2<f64>) -> Array2<f64> {
    use ndarray_linalg::Inverse;
    a.inv().unwrap()
}

// // Convert 2d to old ndarray
// fn ndarray_to_old<T: LinalgScalar>(new: &Array2<T>) -> Array2_old<T> {
//     let (n, m) = (new.shape()[0], new.shape()[1]);
//     let mut old = Array2_old::<T>::zeros((n, m));
//     for (oldv, newv) in old.iter_mut().zip(new.iter()) {
//         *oldv = *newv;
//     }
//     old
// }

// // Convert 2d from old ndarray
// fn ndarray_to_new<T: LinalgScalar>(old: &Array2_old<T>) -> Array2<T> {
//     let (n, m) = (old.shape()[0], old.shape()[1]);
//     let mut new = Array2::<T>::zeros((n, m));
//     for (newv, oldv) in new.iter_mut().zip(old.iter()) {
//         *newv = *oldv;
//     }
//     new
// }

// // Convert 1d from old ndarray
// fn ndarray_vec_to_new<T: LinalgScalar>(old: &Array1_old<T>) -> Array1<T> {
//     let mut new = Array1::<T>::zeros(old.len());
//     for (newv, oldv) in new.iter_mut().zip(old.iter()) {
//         *newv = *oldv;
//     }
//     new
// }

/// Argsort Vector ( smallest -> largest ).
/// Returns permutation vector.
///
/// ```
/// use rustpde::solver::utils::argsort;
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, ArrayBase, Data, Dimension};

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

    #[test]
    fn test_eig() {
        let test = array![
            [1., 2., 3., 4., 5.],
            [1., 2., 3., 4., 5.],
            [1., 2., 3., 4., 5.],
            [1., 2., 3., 4., 5.],
            [1., 2., 3., 4., 5.]
        ];
        let (e, evec, evec_inv) = eig(&test);

        // Eigenvalue vector -> Diagonal matrix
        let mut lam = Array2::<f64>::eye(e.shape()[0]);
        for (i, v) in e.iter().enumerate() {
            lam[[i, i]] = v.clone();
        }

        // Check if eval and evec reproduce original array
        let t = lam.dot(&evec_inv);
        let t = evec.dot(&t);
        approx_eq(&t, &test);
    }
}

// /// Convert 2d array from ndarray to nalgebra
// fn ndarray_to_nalgebra<T: Scalar + num_traits::Zero>(
//     ndarr: &Array2<T>,
// ) -> OMatrix<T, Dynamic, Dynamic> {
//     let (n, m) = (ndarr.shape()[0], ndarr.shape()[1]);
//     let mut nalg = OMatrix::<T, Dynamic, Dynamic>::from_element(n, m, T::zero());
//     for (nd, na) in ndarr.into_iter().zip(nalg.iter_mut()) {
//         *na = nd.clone();
//     }
//     nalg
// }
//
// /// Convert 2d array from nalgebra to ndarray
// fn nalgebra_to_ndarray<T: Scalar + num_traits::Zero>(
//     nalg: &OMatrix<T, Dynamic, Dynamic>,
// ) -> Array2<T> {
//     let (n, m) = nalg.shape();
//     let mut ndarr = Array2::<T>::zeros((n, m));
//     for (na, nd) in nalg.iter().zip(ndarr.iter_mut()) {
//         *nd = na.clone();
//     }
//     ndarr
// }

/// Convert dynamically sized vector to static array
///
/// ## Panics
/// Mismatching size of vector and array
pub fn vec_to_array<T, const N: usize>(v: Vec<T>) -> [T; N] {
    v.try_into()
        .unwrap_or_else(|v: Vec<T>| panic!("Expected a Vec of length {} but it was {}", N, v.len()))
}
