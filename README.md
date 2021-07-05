# ndspectral

## ndspectral: *n*-dimensional transforms of various basis functions

This library is intended for simulation softwares which solve the
partial differential equations using spectral methods.

Currently ndspectral implements transforms from physical to spectral space
for the following basis functions:
- Chebyshev (Orthonormal)
- ChebDirichlet (Composite)
- ChebNeumann (Composite)

Composite basis combine several basis functions of its parent space to
satisfy the needed boundary conditions, this is often called a Galerkin method.
