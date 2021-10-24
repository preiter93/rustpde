//! # Calculate convective terms u*dvdx
use crate::field::{BaseSpace, FieldBase};
use crate::types::Scalar;
use ndarray::Array2;
/// Calculate u*dvdx
///
/// # Input
///
///    *field*: Field<Space2D, 2>
///        Contains field variable vhat in spectral space
///
///   *u*:  ndarray (2D)
///        Velocity field in physical space
///
///   *deriv*: [usize; 2]
///        \[1,0\] for partial x, \[0,1\] for partial y
///
/// # Return
/// Array of u*dvdx term in physical space.
///
/// Collect all convective terms, thatn transform to spectral space.
pub fn conv_term<T2, S>(
    field: &FieldBase<f64, f64, T2, S, 2>,
    deriv_field: &mut FieldBase<f64, f64, T2, S, 2>,
    u: &Array2<f64>,
    deriv: [usize; 2],
    scale: Option<[f64; 2]>,
) -> Array2<f64>
where
    //FieldBase<f64, T2, 2>: Field<f64, T2, 2>,
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T2>,
    T2: Scalar,
{
    //dvdx
    for v in deriv_field.vhat.iter_mut() {
        *v = T2::zero();
    }
    deriv_field.vhat.assign(&field.gradient(deriv, scale));
    deriv_field.backward();
    //u*dvdx
    u * &deriv_field.v
}

#[cfg(test)]
mod navier {
    use super::*;
    use crate::{cheb_dirichlet, cheb_neumann, chebyshev};
    use crate::{Field2, Space2};
    use std::f64::consts::PI;

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
    fn test_conv_term() {
        let (nx, ny) = (12, 12);
        // Define fields
        let mut temp = Field2::new(&Space2::new(&cheb_dirichlet(nx), &cheb_neumann(nx)));
        let mut ux = Field2::new(&Space2::new(&cheb_dirichlet(nx), &cheb_dirichlet(nx)));
        let mut field = Field2::new(&Space2::new(&chebyshev(nx), &chebyshev(nx)));

        //
        let x = field.x[0].to_owned();
        let y = field.x[1].to_owned();

        let m = 1.;
        let arg_x = PI * m;
        let arg_y = PI;
        for i in 0..nx {
            for j in 0..ny {
                temp.v[[i, j]] = (arg_x * x[i]).sin() * (arg_y * y[j]).cos();
                ux.v[[i, j]] = (arg_x * x[i]).sin() * (arg_y * y[j]).sin();
            }
        }
        temp.forward();

        // dudx
        let conv = conv_term(&temp, &mut field, &ux.v, [1, 0], None);

        // Exact
        for i in 0..nx {
            for j in 0..ny {
                let dtemp = arg_x * (arg_x * x[i]).cos() * (arg_y * y[j]).cos();
                field.v[[i, j]] = dtemp;
                field.v[[i, j]] *= ux.v[[i, j]];
            }
        }

        // Assert
        approx_eq(&conv, &field.v);

        // dudy
        let conv = conv_term(&temp, &mut field, &ux.v, [0, 1], None);
        println!("{:?}", conv);

        // Exact
        for i in 0..nx {
            for j in 0..ny {
                let dtemp = (arg_x * x[i]).sin() * -1.0 * arg_y * (arg_y * y[j]).sin();
                field.v[[i, j]] = dtemp;
                field.v[[i, j]] *= ux.v[[i, j]];
            }
        }

        println!("{:?}", field.v);

        // Assert
        approx_eq(&conv, &field.v);
    }
}
