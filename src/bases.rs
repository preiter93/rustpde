//! # Bases
//! `Funspace` now as independent package
//!
//! Implemented:
//! - `Chebyshev` (Orthonormal), see [`chebyshev()`]
//! - `ChebDirichlet` (Composite), see [`cheb_dirichlet()`]
//! - `ChebNeumann` (Composite), see [`cheb_neumann()`]
//! - `FourierC2c` (Orthonormal), see [`fourier_c2c()`]
//! - `FourierR2c` (Orthonormal), see [`fourier_r2c()`]
pub use funspace::cheb_dirichlet;
pub use funspace::cheb_dirichlet_bc;
pub use funspace::cheb_neumann;
pub use funspace::cheb_neumann_bc;
pub use funspace::chebyshev;
pub use funspace::fourier_c2c;
pub use funspace::fourier_r2c;
pub use funspace::Basics;
pub use funspace::Differentiate;
pub use funspace::FromOrtho;
pub use funspace::FromOrthoPar;
pub use funspace::LaplacianInverse;
pub use funspace::Transform;
pub use funspace::TransformPar;
pub use funspace::{BaseAll, BaseC2c, BaseR2c, BaseR2r};
pub use funspace::{BaseSpace, Space1, Space2};
