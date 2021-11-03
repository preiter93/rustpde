import numpy as np
import h5py
import glob
import re
import matplotlib.pyplot as plt
from utils.plot_utils import plot_streamplot

# -- Read hd5 file
filename = "data/statistics.nc"
with h5py.File(filename, "r") as f:
    t = np.array(f["temp/v"])
    u = np.array(f["ux/v"])
    v = np.array(f["uy/v"])
    n = np.array(f["nusselt/v"])
    x = np.array(f["x"])
    y = np.array(f["y"])
    try:
        vorticity = np.array(f["vorticity/v"])
    except:
        vorticity = None

print("Plot {:}".format(filename))
fig, ax = plot_streamplot(x, y, t, u, v, return_fig=True, cbar=True)
# fig.savefig("fig.png", bbox_inches="tight", dpi=200)
plt.show()

fig, ax = plot_streamplot(x, y, n, u, v, return_fig=True, cbar=True)
# fig.savefig("fig.png", bbox_inches="tight", dpi=200)
plt.show()
