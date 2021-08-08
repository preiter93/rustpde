//! Collection of some implemented partial diff equations
mod conv_term;
pub mod diffusion;
pub mod functions;
pub mod navier;
pub mod navier_adjoint;
pub mod navier_periodic;
pub mod solid_masks;
pub use conv_term::conv_term;
pub use navier::Navier2D;
pub use navier_periodic::Navier2DPeriodic;
pub use solid_masks::solid_cylinder_inner;
