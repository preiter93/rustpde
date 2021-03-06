# funspace

## Funspace
<img align="right" src="https://rustacean.net/assets/cuddlyferris.png" width="80">

Collection of function spaces.

A function space is made up of elements of basis functions.
Every function in the function space can be represented as a
linear combination of basis functions, represented by real/complex
coefficients (spectral space).

### Implemented function spaces:
- `Chebyshev` (Orthogonal), see [`chebyshev()`]
- `ChebDirichlet` (Composite), see [`cheb_dirichlet()`]
- `ChebNeumann` (Composite), see [`cheb_neumann()`]
- `FourierC2c` (Orthogonal), see [`fourier_c2c()`]
- `FourierR2c` (Orthogonal), see [`fourier_r2c()`]

### Transform
A transform describes a change from the physical space to the function
space. For example, a fourier transform describes a transform from
values of a function on a regular grid to coefficents of sine/cosine
polynomials. This is analogous to other function spaces. The transforms
are implemented by the [`Transform`] trait.

#### Example
Apply forward transform of 1d array in `cheb_dirichlet` space
```rust
use funspace::{Transform, cheb_dirichlet};
use ndarray::prelude::*;
use ndarray::Array1;
let mut cd = cheb_dirichlet::<f64>(5);
let input = array![1., 2., 3., 4., 5.];
let output: Array1<f64> = cd.forward(&input, 0);
```

### Differentiation
One key advantage representing a function with coefficents in
the function space is its ease of differentiation. Differentiation in
fourier space becomes multiplication with the wavenumbe vector.
Differentiation in Chebyshev space is done by a recurrence
relation and almost as fast as in Fourier space.
Each base implements a differentiation method, which must be applied on
an array of coefficents. This is defined by the [`Differentiate`] trait.

#### Example
Apply differentiation
```rust
use funspace::{Transform, Differentiate, Basics, fourier_r2c};
use ndarray::prelude::*;
use ndarray::Array1;
use num_complex::Complex;
// Define base
let mut fo = fourier_r2c(8);
// Get coordinates in physical space
let x = fo.coords().clone();
let v = x.mapv(|xi: f64| (2. * xi).sin());
// Transform to physical space
let vhat: Array1<Complex<f64>> = fo.forward(&v, 0);

// Apply differentiation twice along first axis
let dvhat = fo.differentiate(&vhat, 2, 0);
// Transform back to spectral space
let dv: Array1<f64> = fo.backward(&dvhat, 0);
// Compare with correct derivative
for (exp, ist) in x
    .mapv(|xi: f64| -4. * (2. * xi).sin())
    .iter()
    .zip(dv.iter())
{
    assert!((exp - ist).abs() < 1e-5);
}
```

### Composite Bases
Bases like those of fourier polynomials or chebyshev polynomials are
considered orthonormal bases, i.e. the dot product of each individual
polynomial with any other of its set vanishes and the dot product with
itself is unity. In these cases, the mass matrix is equal to the
identity matrix.
However, other function spaces can be constructed by a linearly combiningn
the orthonormal basis functions. By doing so, one can construct
bases which satisfy particular boundary conditions
like dirichlet (zero at the ends) or neumann (zero derivative at the ends).
This is usefull for solving partial differential equations. When expressed
in those composite function space, the boundary condition is automatically
satisfied. This is known as the *Galerkin* Method.

To switch from its composite form to the orthonormal form, each base implements
a [`FromOrtho`] trait, which defines the transform `to_ortho` and `from_ortho`.
If the base is already orthogonal, the input will be returned, otherwise it
is transformed from the composite space to the orthonormal space.
Note that the size of the composite space is usually
less than its orthogonal counterpart.  Therefore the output array must
not maintain the same shape (but its dimensionality is conserved).

#### Example
Transform composite space `cheb_dirichlet` to its orthogonal counterpart
`chebyshev`. Note that `cheb_dirichlet` has 6 spectral coefficients,
while the `chebyshev` bases has 8.
```rust
use funspace::{Transform, FromOrtho, Basics};
use funspace::{cheb_dirichlet, chebyshev};
use std::f64::consts::PI;
use ndarray::prelude::*;
use ndarray::Array1;
use num_complex::Complex;
// Define base
let mut ch = chebyshev(8);
let mut cd = cheb_dirichlet(8);
// Get coordinates in physical space
let x = ch.coords().clone();
let v = x.mapv(|xi: f64| (PI / 2. * xi).cos());
// Transform to physical space
let ch_vhat: Array1<f64> = ch.forward(&v, 0);
let cd_vhat: Array1<f64> = cd.forward(&v, 0);
// Send array to orthogonal space (cheb_dirichlet
// to chebyshev in this case)
let cd_vhat_ortho = cd.to_ortho(&cd_vhat, 0);
// Both arrays are equal, because field was
// initialized with correct boundary conditions,
// i.e. dirichlet ones
for (exp, ist) in ch_vhat.iter().zip(cd_vhat_ortho.iter()) {
    assert!((exp - ist).abs() < 1e-5);
}

// However, if the physical field values do not
// satisfy dirichlet boundary conditions, they
// will be enforced by the transform to cheb_dirichle
// and ultimately the transformed values will deviate
// from a pure chebyshev transform (which does not)
// enfore the boundary conditions.
let mut v = x.mapv(|xi: f64| (PI / 2. * xi).sin());
let ch_vhat: Array1<f64> = ch.forward(&v, 0);
let cd_vhat: Array1<f64> = cd.forward(&v, 0);
let cd_vhat_ortho = cd.to_ortho(&cd_vhat, 0);
// They will deviate
println!("chebyshev     : {:?}", ch_vhat);
println!("cheb_dirichlet: {:?}", cd_vhat_ortho);
```

### Multidimensional Spaces
A collection of bases makes up a Space, on which one can again defines operations
along a specfic dimension (= axis). But special care must be taken in order to transform
a field from the physical space to the spectral space on how the transforms
are chained in a multidimensional space. Not all combinations are possible.
For example, `cheb_dirichlet` is a real-to-real transform,
while `fourier_r2c` defines a real-to-complex transform.
So, for a given real valued physical field, the chebyshev transform must be applied
before the fourier transform in the forward transform, and in opposite order in
the backward transform.

**Note**: Currently `funspace` supports 1- 2- and 3 - dimensional spaces.

#### Example
Apply transform from physical to spectral in a two-dimensional space
```rust
use funspace::{fourier_r2c, cheb_dirichlet, Space2, BaseSpace};
use ndarray::prelude::*;
use std::f64::consts::PI;
use num_complex::Complex;
// Define the space and allocate arrays
let mut space = Space2::new(&fourier_r2c(5), &cheb_dirichlet(5));
let mut v: Array2<f64> = space.ndarray_physical();
// Set some field values
let x = space.coords_axis(0);
let y = space.coords_axis(1);
for (i,xi) in x.iter().enumerate() {
    for (j,yi) in y.iter().enumerate() {
        v[[i,j]] = xi.sin() * (PI/2.*yi).cos();
    }
}
// Transform forward (vhat is complex)
let mut vhat = space.forward(&v);
// Transform backward (v is real)
let v = space.backward(&vhat);
```

License: MIT
