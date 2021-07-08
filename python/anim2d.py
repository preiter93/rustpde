import numpy as np
import h5py
import glob
import re
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from utils.anim_utils import animate_wireframe, animate_contour

# -- Get list of files
fname, time = [], []
for file in glob.glob("data/*.h5"):
    try:
        time.append(float(re.findall("\d+\.\d+", file)[0]))
        fname.append(file)
    except:
        print("No number found in {:}".format(file))
idx = np.argsort(time)
fname = np.array(fname)[idx]
time = np.array(time)[idx]


# -- Read hd5 file
filename = fname[0]
V = []
for filename in fname:
    with h5py.File(filename, "r") as f:
        # List all groups
        # print("Keys: %s" % f.keys())
        print(filename)
        a_group_key = list(f.keys())[0]

        # Get the data
        # data = list(f[a_group_key])
        v = np.array(f["temp/v"])
        x = np.array(f["x"])
        y = np.array(f["y"])
        V.append(v)

kwargs = {}
# List to ndarray
VS = np.rollaxis(np.dstack(V).squeeze(), -1)
# Animate
# anim = animate_wireframe(x, y, VS, **kwargs)
anim = animate_contour(x, y, VS, **kwargs)
plt.show()
# xx,yy = np.meshgrid(x,y,indexing="ij")
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot_wireframe(xx,yy,v)
# # Plot
# plt.show()
