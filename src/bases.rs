//! # Bases
//! `Funspace` now as independent package
//!
//! Implemented:
//! - `Chebyshev` (Orthonormal), see [`chebyshev()`]
//! - `ChebDirichlet` (Composite), see [`cheb_dirichlet()`]
//! - `ChebNeumann` (Composite), see [`cheb_neumann()`]
//! - `Fourier` (Orthonormal), see [`fourier()`]
//! - `FourierR2c` (Orthonormal), see [`fourier_r2c()`]
pub use funspace::cheb_dirichlet;
pub use funspace::cheb_dirichlet_bc;
pub use funspace::cheb_neumann;
pub use funspace::cheb_neumann_bc;
pub use funspace::chebyshev;
pub use funspace::fourier_c2c;
pub use funspace::fourier_r2c;
pub use funspace::BaseBasics;
pub use funspace::BaseKind as Base;
pub use funspace::Differentiate;
pub use funspace::FromOrtho;
pub use funspace::LaplacianInverse;
pub use funspace::Transform;
pub use funspace::TransformPar;
