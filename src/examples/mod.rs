//! Collection of partial diff equations for *rustpde*
pub mod conv_term;
pub mod diffusion;
pub mod functions;
pub mod navier;
pub mod navier_adjoint;
// pub mod navier_periodic;
pub mod solid_masks;
pub mod vorticity;
pub use conv_term::conv_term;
pub use navier::Navier2D;
pub use navier_adjoint::Navier2DAdjoint;
pub use solid_masks::solid_cylinder_inner;
pub use vorticity::vorticity_from_file;
