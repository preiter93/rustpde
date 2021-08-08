//! Some useful post-processing functions
use crate::types::Scalar;
use crate::{Field, FieldBase};
use std::ops::{Div, Mul};
/// Returns Nusselt number (heat flux at the plates)
/// $$
/// Nu = \langle - dTdz \rangle\\_x (0/H))
/// $$
pub fn eval_nu<T2>(
    temp: &mut FieldBase<f64, T2, 2>,
    field: &mut FieldBase<f64, T2, 2>,
    tempbc: &Option<FieldBase<f64, T2, 2>>,
    scale: &[f64; 2],
) -> f64
where
    FieldBase<f64, T2, 2>: Field<f64, T2, 2>,
    T2: Scalar + Mul<f64, Output = T2>,
{
    //self.temp.backward();
    field.vhat.assign(&temp.to_ortho());
    if let Some(x) = &tempbc {
        field.vhat = &field.vhat + &x.to_ortho();
    }
    let mut dtdz = field.grad([0, 1], None) * -1.;
    dtdz = dtdz * (1. / (scale[1] * 0.5));
    field.vhat.assign(&dtdz);
    field.backward();
    let x_avg = field.average_axis(0);
    (x_avg[x_avg.len() - 1] + x_avg[0]) / 2.
}

/// Returns volumetric Nusselt number
/// $$
/// Nuvol = \langle uy*T/kappa - dTdz \rangle\\_V
/// $$
pub fn eval_nuvol<T2>(
    temp: &mut FieldBase<f64, T2, 2>,
    uy: &mut FieldBase<f64, T2, 2>,
    field: &mut FieldBase<f64, T2, 2>,
    tempbc: &Option<FieldBase<f64, T2, 2>>,
    kappa: f64,
    scale: &[f64; 2],
) -> f64
where
    FieldBase<f64, T2, 2>: Field<f64, T2, 2>,
    T2: Scalar + Div<f64, Output = T2>,
{
    // temp
    field.vhat.assign(&temp.to_ortho());
    if let Some(x) = &tempbc {
        field.vhat = &field.vhat + &x.to_ortho();
    }
    field.backward();
    // uy
    uy.backward();
    let uy_temp = &field.v * &uy.v;
    // dtdz
    let dtdz = field.grad([0, 1], None) / (scale[1] * -1.);
    field.vhat.assign(&dtdz);
    field.backward();
    let dtdz = &field.v;
    // Nuvol
    field.v = (dtdz + uy_temp / kappa) * 2. * scale[1];
    //average
    field.average()
}

/// Returns Reynolds number base on kinetic energy
/// $$
/// Re = U*L / nu
/// U = \sqrt{(ux^2 + uy^2)}
/// $$
pub fn eval_re<T2>(
    ux: &mut FieldBase<f64, T2, 2>,
    uy: &mut FieldBase<f64, T2, 2>,
    field: &mut FieldBase<f64, T2, 2>,
    nu: f64,
    scale: &[f64; 2],
) -> f64
where
    FieldBase<f64, T2, 2>: Field<f64, T2, 2>,
{
    ux.backward();
    uy.backward();
    let ekin = &ux.v.mapv(|x| x.powi(2)) + &uy.v.mapv(|x| x.powi(2));
    field.v.assign(&ekin.mapv(f64::sqrt));
    field.v *= 2. * scale[1] / nu;
    field.average()
}
