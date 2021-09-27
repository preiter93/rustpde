import numpy as np
import h5py
import glob
import re
import matplotlib.pyplot as plt
from utils.plot_utils import plot_quiver
import os.path
import ffmpeg

settings = {
    "duration": None,  # time in seconds; determines fps
    "filename": "data/out.mp4",
}

# -- Get list of files
fname, time = [], []
for file in [*glob.glob("*.h5"), *glob.glob("data/*.h5")]:
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

for i, f in enumerate(fname[i0:i9:step]):
    # -- Read hd5 file
    filename = f
    figname = f.replace(".h5", ".png")

    if os.path.isfile(figname):
        print("{} already exists".format(figname))
        continue

    with h5py.File(filename, "r") as f:
        t = np.array(f["temp/v"])
        u = np.array(f["ux/v"])
        v = np.array(f["uy/v"])
        x = np.array(f["x"])
        y = np.array(f["y"])

    print("Plot {:}".format(filename))
    fig, ax = plot_quiver(x, y, t, u, v, return_fig=True)
    fig.savefig(figname)
    plt.close("all")

# -- Get list of pngs
fname, time = [], []
for file in [*glob.glob("data/*.png")]:
    try:
        time.append(float(re.findall("\d+\.\d+", file)[0]))
        fname.append(file)
    except:
        print("No number found in {:}".format(file))
idx = np.argsort(time)
files = list(np.array(fname)[idx])
time = np.array(time)[idx]

for i, f in enumerate(files):
    print("# {:3d}: {:}".format(i, f))

if settings["duration"] is None:
    print("How long should the movie be? (seconds)")
    settings["duration"] = float(input())

n_frames = len(files)
fps = n_frames / settings["duration"]

# Execute FFmpeg sub-process, with stdin pipe as input, and jpeg_pipe input format
process = (
    ffmpeg.input("pipe:", r=fps, f="png_pipe")
    .output(settings["filename"], vcodec="libx264", pix_fmt="yuv420p")
    .overwrite_output()
    .run_async(pipe_stdin=True)
)

# Iterate jpeg_files, read the content of each file and write it to stdin
for in_file in files:
    with open(in_file, "rb") as f:
        # Read the JPEG file content to jpeg_data (bytes array)
        data = f.read()

        # Write JPEG data to stdin pipe of FFmpeg process
        process.stdin.write(data)

# Close stdin pipe - FFmpeg fininsh encoding the output file.
process.stdin.close()
process.wait()
