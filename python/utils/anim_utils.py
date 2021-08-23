import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits import mplot3d
from utils.colors import *

global_settings = {
    "fps": None,
    "duration": 20,  # time in seconds; determines fps
    "filename": "data/anim.mp4",
    "gifname": "data/anim.gif",
}

FFMPEG = ["-vcodec", "libx264", "-r", "5"]


def create_output_folder(full_path):
    import pathlib
    import os

    dirname = os.path.dirname(full_path)
    pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)


def set_fps_from_duration(dict, nframes):
    dict["fps"] = nframes / dict["duration"]


def animate_line(x, Y, **kwargs):
    """
    Animate collection of lines (Y)
    Input:
    x (1d ndarray): Grid points
    Y (2d ndarray): Collection of lines [time, Nx]
    """
    settings = {
        "color": gfblue3,
        "lw": 2,
    }
    settings.update(**global_settings)
    settings.update(**kwargs)
    set_fps_from_duration(settings, Y.shape[0])
    assert len(x.shape) == 1
    assert len(Y.shape) == 2

    # First set up the figure, the axis, and the plot element we want to animate
    fig, ax = plt.subplots()
    ax.set_xlim((np.amin(x), np.amax(x)))
    ax.set_ylim((np.amin(Y), np.amax(Y)))
    (line,) = ax.plot([], [], lw=settings["lw"], color=settings["color"])

    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        return (line,)

    # animation function.  This is called sequentially
    def animate(i):
        y = Y[i, :]
        line.set_data(x, y)
        return (line,)

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=Y.shape[0],
        interval=1 / settings["fps"] * 1000,
        blit=True,
    )

    # write animation
    create_output_folder(settings["filename"])
    anim.save(settings["filename"], fps=settings["fps"], extra_args=FFMPEG)
    print("Save animation to {:s}".format(settings["filename"]))
    return anim


def animate_contour(x, y, Z, **kwargs):

    """
    Animate collection of contours (Z)
    Input:
    x (1d ndarray): Grid points in x
    y (1d ndarray): Grid points in y
    Z (3d ndarray): Collection of lines [time, Nx, Ny]
    """
    settings = {
        "cmap": "gfcmap",
    }
    settings.update(**global_settings)
    settings.update(**kwargs)
    set_fps_from_duration(settings, Z.shape[0])
    if settings["cmap"] == "gfcmap":
        set_gfcmap()  # Register goldfish colormap

    global cont
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert len(Z.shape) == 3

    fig, ax = plt.subplots()
    ax.set_xlim((np.amin(x), np.amax(x)))
    ax.set_ylim((np.amin(y), np.amax(y)))
    ax.set_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])
    cvals = np.linspace(np.amin(Z), np.amax(Z), 101)  # set contour value
    x, y = np.meshgrid(x, y, indexing="ij")
    cont = ax.contourf(
        x, y, Z[0, :, :], cvals, cmap=settings["cmap"]
    )  # first image on screen

    # animation function
    def animate(i):
        global cont
        z = Z[i, :, :]
        for c in cont.collections:
            c.remove()  # removes only the contours
        cont = plt.contourf(x, y, z, cvals, cmap=settings["cmap"])
        # plt.title('t = %i:  %.2f' % (i,z[5,5]))
        return cont

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(
        fig, animate, frames=Z.shape[0], interval=1 / settings["fps"] * 1000, blit=False
    )

    # write animation
    create_output_folder(settings["filename"])
    anim.save(settings["filename"], fps=settings["fps"], extra_args=FFMPEG)
    anim.save(settings["gifname"], writer="imagemagick", fps=settings["fps"])
    print("Save animation to {:s}".format(settings["filename"]))
    return anim


def animate_wireframe(x, y, Z, **kwargs):
    """
    Animate collection of contours (Z)
    Input:
    x (1d ndarray): Grid points in x
    y (1d ndarray): Grid points in y
    Z (3d ndarray): Collection of lines [time, Nx, Ny]
    """

    settings = {"cmap": "gfcmap", "edgecolor": "none"}
    settings.update(**global_settings)
    settings.update(**kwargs)
    set_fps_from_duration(settings, Z.shape[0])
    if settings["cmap"] == "gfcmap":
        set_gfcmap()  # Register goldfish colormap

    global wframe
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert len(Z.shape) == 3
    xx, yy = np.meshgrid(x, y, indexing="ij")
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlim((np.amin(x), np.amax(x)))
    ax.set_ylim((np.amin(y), np.amax(y)))
    ax.set_zlim((np.amin(Z), np.amax(Z)))
    ax.locator_params(tight=True, nbins=4)

    # begin plotting
    wframe = None

    # animation function
    def animate(i):
        global wframe
        z = Z[i, :, :]
        if wframe:
            ax.collections.remove(wframe)
        wframe = ax.plot_surface(
            xx,
            yy,
            z,
            rstride=1,
            cstride=1,
            cmap=settings["cmap"],
            edgecolor=settings["edgecolor"],
        )

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(
        fig, animate, frames=Z.shape[0], interval=1 / settings["fps"] * 1000, blit=False
    )

    # write animation
    create_output_folder(settings["filename"])
    anim.save(settings["filename"], fps=settings["fps"], extra_args=FFMPEG)
    print("Save animation to {:s}".format(settings["filename"]))
    return anim
