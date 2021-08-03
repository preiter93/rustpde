//! # Bases
//! Use external package funspace
//!
//! Implemented:
//! - `Chebyshev` (Orthonormal), see [`chebyshev()`]
//! - `ChebDirichlet` (Composite), see [`cheb_dirichlet()`]
//! - `ChebNeumann` (Composite), see [`cheb_neumann()`]
pub use funspace::cheb_dirichlet;
pub use funspace::cheb_dirichlet_bc;
pub use funspace::cheb_neumann;
pub use funspace::cheb_neumann_bc;
pub use funspace::chebyshev;
pub use funspace::Base;
pub use funspace::Differentiate;
pub use funspace::FromOrtho;
pub use funspace::LaplacianInverse;
pub use funspace::Mass;
pub use funspace::Size;
pub use funspace::Transform;
pub use funspace::TransformPar;
