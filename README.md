# rustpde

## `rustpde`: Spectral method solver for Navier-Stokes equations
<img align="right" src="https://rustacean.net/assets/cuddlyferris.png" width="80">

## Dependencies
- cargo >= v1.49
- `hdf5` (sudo apt-get install -y libhdf5-dev)

## Details

This library is intended for simulation softwares which solve the
partial differential equations using spectral methods.

Currently `rustpde` implements transforms from physical to spectral space
for the following basis functions:
- `Chebyshev` (Orthonormal), see [`chebyshev()`]
- `ChebDirichlet` (Composite), see [`cheb_dirichlet()`]
- `ChebNeumann` (Composite), see [`cheb_neumann()`]
- `FourierR2c` (Orthonormal), see [`fourier_r2c()`]

Composite basis combine several basis functions of its parent space to
satisfy the needed boundary conditions, this is often called a Galerkin method.

### Implemented solver

- `2-D Rayleigh Benard Convection: Direct numerical simulation`,
see [`navier::navier`]
- `2-D Rayleigh Benard Convection: Steady state solver`,
see [`navier::navier_adjoint`]

## Example
Solve 2-D Rayleigh Benard Convection ( Run with `cargo run --release` )
```rust
use rustpde::integrate;
use rustpde::navier::Navier2D;
use rustpde::Integrate;

fn main() {
    // Parameters
    let (nx, ny) = (64, 64);
    let ra = 1e5;
    let pr = 1.;
    let adiabatic = true;
    let aspect = 1.0;
    let dt = 0.02;
    let mut navier = Navier2D::new(nx, ny, ra, pr, dt, aspect, adiabatic);
    // Set initial conditions
    navier.set_velocity(0.2, 1., 1.);
    // // Want to restart?
    // navier.read("data/flow100.000.h5");
    // Write first field
    navier.callback();
    integrate(&mut navier, 100., Some(1.0));
}
```
Solve 2-D Rayleigh Benard Convection with periodic sidewall
```rust
use rustpde::integrate;
use rustpde::navier::Navier2D;
use rustpde::Integrate;

fn main() {
    // Parameters
    let (nx, ny) = (64, 64);
    let ra = 1e5;
    let pr = 1.;
    let aspect = 1.0;
    let dt = 0.02;
    let mut navier = Navier2D::new_periodic(nx, ny, ra, pr, dt, aspect);
    integrate(&mut navier, 100., Some(1.0));
}
```

### Postprocess the output

`rustpde` contains a `python` folder with some scripts.
If you have run the above example, and you specified
to save snapshots ( replace *None* with Some(1.) or any
other value), you will see `hdf5` in the `data` folder.

You can create an animation with python's matplotlib by typing

`python3 python/anim2d.py`

Or just plot a single snapshot

`python3 python/plot2d.py`

Provided python has all librarys installed, you should now
see an animation.

#### Paraview

The xmf files, corresponding to the h5 files can be created
by the script

`./bin/create_xmf`.

This script works only for fields from the `Navier2D`
solver with the attributes temp, ux, uy and pres.
The bin folder contains also the full crate `create_xmf`, which
can be adapted for specific usecases.

### Documentation

Download and run:

`cargo doc --open`
