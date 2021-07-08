import matplotlib.pyplot as plt

# -- Colors ---------------------------------------
gfblue3 = (0 / 255, 137 / 255, 204 / 255)
gfred3 = (196 / 255, 0, 96 / 255)

# -- Colormap -------------------------------------
def gfcmap(filename="gfcmap.json"):
    """ "
    Read goldfish colormap

    Args:
        filename (str): Filename (.json) of goldfish colormap
    """
    import json
    import os
    from matplotlib.colors import LinearSegmentedColormap

    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(dir_path + "/" + filename, "r") as fp:
        gfcextdict = json.load(fp)
    return LinearSegmentedColormap("goldfishext", gfcextdict)


def set_gfcmap():
    """Register goldfish colormap"""
    plt.cm.register_cmap(name="gfcmap", cmap=gfcmap())
