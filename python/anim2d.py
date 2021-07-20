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

for i, f in enumerate(fname):
    print("# {:3d}: {:}".format(i, f))
print("From number:")
i0 = int(input())
print("To number:")
i9 = int(input())
print("Step:")
step = int(input())

# -- Read hd5 file
filename = fname[0]
V = []
for filename in fname[i0:i9:step]:
    with h5py.File(filename, "r") as f:
        # List all groups
        print(filename)
        a_group_key = list(f.keys())[0]

        # Get the data
        v = np.array(f["temp/v"])
        x = np.array(f["x"])
        y = np.array(f["y"])
        V.append(v)

kwargs = {}
# List to ndarray
VS = np.rollaxis(np.dstack(V).squeeze(), -1)
# Animate
anim = animate_contour(x, y, VS, **kwargs)
plt.show()
