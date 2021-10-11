import numpy as np
import matplotlib.pyplot as plt
from utils.colors import *

SETTINGS = {
    "cmap": "gfcmap",
}

def _calculate_streamfunction(x,y,u,v):
    from scipy import integrate
    import numpy

    # grid
    xx,yy=numpy.meshgrid(x,y,indexing='xy')

    # integrate
    intx=integrate.cumtrapz(v,xx,axis=1,initial=0)[0]
    inty=integrate.cumtrapz(u,yy,axis=0,initial=0)

    psi1=intx-inty

    intx2=integrate.cumtrapz(v,xx,axis=1,initial=0)
    inty2=integrate.cumtrapz(u,yy,axis=0,initial=0)[:,0][:,None]

    psi2=intx2-inty2
    psi=(psi1+psi2)/2.

    return psi

def plot_contour(x, y, t, u, v, skip=None, return_fig=False):
    if SETTINGS["cmap"] == "gfcmap":
        set_gfcmap()
    xx, yy = np.meshgrid(x,y,indexing="ij")

    fig, ax = plt.subplots()
    ax.contourf(
        xx, yy, t, levels=np.linspace(t.min(), t.max(), 101), cmap=SETTINGS["cmap"]
    )
    ax.set_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())

    if return_fig:
        return fig, ax
    plt.show()

def plot_quiver(x, y, t, u, v, skip=None, return_fig=False):
    if SETTINGS["cmap"] == "gfcmap":
        set_gfcmap()
    xx, yy = np.meshgrid(x,y,indexing="ij")

    fig, ax = plt.subplots()
    ax.contourf(
        xx, yy, t, levels=np.linspace(t.min(), t.max(), 101), cmap=SETTINGS["cmap"]
    )
    ax.set_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())

    speed = 2.*np.max(np.sqrt(u ** 2 + v ** 2))
    if skip is None:
        skip = t.shape[0] // 16
    ax.quiver(
        xx[::skip, ::skip],
        yy[::skip, ::skip],
        u[::skip, ::skip] / speed,
        v[::skip, ::skip] / speed,
        scale=7.9,
        width=0.005,
        alpha=0.5,
        headwidth=4,
    )
    if return_fig:
        return fig, ax
    plt.show()

def plot_streamplot(x, y, t, u, v, return_fig=False):
    from scipy import interpolate
    if SETTINGS["cmap"] == "gfcmap":
        set_gfcmap()
    x = x-x.min()
    y = y-y.min()
    xx, yy = np.meshgrid(x,y,indexing="ij")

    fig, ax = plt.subplots()
    tmax = np.amax(np.abs(t))
    tmin = -tmax
    ax.contourf(
        xx, yy, t, levels=np.linspace(tmin, tmax, 401), cmap=SETTINGS["cmap"]
    )
    ax.set_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())

    nx, ny = int(41*x.size/y.size), 41
    xi = np.linspace(x.min(), x.max(), nx)
    yi = np.linspace(y.min(), y.max(), ny)

    f = interpolate.interp2d(y, x, u, kind="cubic")
    ui = f(yi, xi)
    f = interpolate.interp2d(y, x, v, kind="cubic")
    vi = f(yi, xi)

    speed = np.sqrt(ui*ui + vi*vi)
    lw = 0.8 * speed.T / np.abs(speed).max()
    ax.streamplot(xi, yi, ui.T, vi.T, density=0.75, color="k", linewidth=lw)
    if return_fig:
        return fig, ax
    plt.show()

def plot_streamfunction(x, y, t, u, v, return_fig=False):
    from scipy import interpolate
    if SETTINGS["cmap"] == "gfcmap":
        set_gfcmap()
    x = x-x.min()
    y = y-y.min()
    xx, yy = np.meshgrid(x,y,indexing="ij")

    fig, ax = plt.subplots()
    ax.contourf(
        xx, yy, t, levels=np.linspace(t.min(), t.max(), 101), cmap=SETTINGS["cmap"]
    )
    ax.set_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())

    psi = _calculate_streamfunction(x, y, v, u)
    ax.contour(xx,yy,psi,12,colors="k",linewidths=0.6)

    if return_fig:
        return fig, ax
    plt.show()
