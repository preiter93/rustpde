use crate::Field2;
/// Returns Nusselt number (heat flux at the plates)
/// $$
/// Nu = \langle - dTdz \rangle_x (0/H))
/// $$
pub fn eval_nu(
    temp: &mut Field2,
    field: &mut Field2,
    tempbc: &Option<Field2>,
    scale: &[f64; 2],
) -> f64 {
    //self.temp.backward();
    field.vhat.assign(&temp.to_parent());
    if let Some(x) = &tempbc {
        field.vhat += &x.to_parent();
    }
    //self.field.forward();
    let mut dtdz = -1. * field.grad([0, 1], None);
    dtdz /= scale[1] * 0.5;
    field.vhat.assign(&dtdz);
    field.backward();
    let x_avg = field.average_axis(0);
    (x_avg[x_avg.len() - 1] + x_avg[0]) / 2.
}

/// Returns volumetric Nusselt number
/// $$
/// Nuvol = \langle uy*T/kappa - dTdz \rangle_V
/// $$
pub fn eval_nuvol(
    temp: &mut Field2,
    uy: &mut Field2,
    field: &mut Field2,
    tempbc: &Option<Field2>,
    kappa: f64,
    scale: &[f64; 2],
) -> f64 {
    // temp
    field.vhat.assign(&temp.to_parent());
    if let Some(x) = &tempbc {
        field.vhat += &x.to_parent();
    }
    field.backward();
    // uy
    uy.backward();
    let uy_temp = &field.v * &uy.v;
    // dtdz
    let dtdz = -1. * field.grad([0, 1], None) / scale[1];
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
pub fn eval_re(
    ux: &mut Field2,
    uy: &mut Field2,
    field: &mut Field2,
    nu: f64,
    scale: &[f64; 2],
) -> f64 {
    ux.backward();
    uy.backward();
    let ux2 = ux.v.mapv(|x| x.powi(2));
    let uy2 = uy.v.mapv(|x| x.powi(2));
    let ekin = &ux2 + &uy2;
    field.v.assign(&ekin.mapv(f64::sqrt));
    field.v *= 2. * scale[1] / nu;
    field.average()
}
