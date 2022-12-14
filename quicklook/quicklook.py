"""
quicklook

A simple way to visualize numpy arrays
"""

# standard library
import os
import sys
from pathlib import Path
from numbers import Number

# 3rd party distribution packages
import numpy as np
from numpy import min, max
import matplotlib.pyplot as plt
from matplotlib import image
import typer

# local modules
here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(here)
from example_arrays import static

#
#
#


def replace_nan(arr):
    """replace array elements that are "not a number" with min valid number"""
    return np.nan_to_num(arr, nan=np.nanmin(arr))


def percentile_clip(arr, clip):
    """
    (optional) numpy.clip extreme pixel values using percentiles

    An array has a distribution of pixel values.
    If there is one or more extreme outlier from this distribution the image can look bad.
    This is because we can get a significantly suboptimal linear scaling of the image for visualization.
    If we replace these extreme values using percentiles the result can look a lot better
    """

    if not clip:
        return arr

    if not isinstance(clip, Number):
        raise TypeError(
            f"Percentile clip needs to be a number but was given a {type(clip)}"
        )

    if clip > 0 and clip < 50:
        arr = np.clip(arr, np.percentile(arr, clip), np.percentile(arr, 100 - clip))
    else:
        print("Image not clipped. Clip percentile must be between 0 and 50")

    return arr


def bytescale(arr):
    """
    scale pixels values to 0 - 255 and 8bit depth

    Standard screens (e.g. computer, phone, tablet, TV, etc.) expect pixel values
    to be between 0 and 255 and have 8bits (i.e .a byte) of depth per channel.
    """

    # all zeros
    if not np.any(arr):
        return arr

    # float dtype (temporarily) during scaling
    arr = arr.astype(float)

    # linear scale between 0 and 255
    arr = 255 * ((arr - np.min(arr)) / (np.max(arr) - np.min(arr)))

    # byte data type
    arr = np.uint8(arr)

    return arr


def reshape_array(arr):
    """reshape arrays (if needed) for display"""

    if arr.ndim == 1:
        raise ValueError("Array needs to have 2 or 3 dimension")

    if arr.ndim == 2:
        return arr

    if arr.ndim > 3:
        raise ValueError("Array needs to have 2 or 3 dimension")

    # channel last ordering (matplotlib requirement)
    if arr.shape[-1] != min(arr.shape):

        # assume smallest dimension is the channel index
        channel_index = int(np.where(arr.shape == np.min(arr.shape))[0])

        # channel last ordering
        arr = np.moveaxis(arr, channel_index, -1)

    # if there are more than 3 channels then pick first 3
    if min(arr.shape) > 3:
        arr = arr[:, :, 0:3]

    # if there are 2 channels take their mean average
    if min(arr.shape) == 2:
        arr = np.mean(arr, axis=2)

    # remove flat dimensions (matplotlib requirement)
    arr = np.squeeze(arr)

    return arr


def matplotlib_display(arr, figsize, title, cmap):
    """display array in a window using matplotlib"""

    plt.figure(figsize=figsize)
    plt.imshow(arr, vmin=0, vmax=255, cmap=cmap)
    plt.title(title)
    plt.show()


def show(arr: np.ndarray, clip=0, title="", cmap="viridis", figsize=(5, 5)):
    """show numpy array"""

    if not isinstance(arr, (np.ndarray)):
        raise TypeError(f"Input must be a numpy array was given a {type(arr)}")

    arr = replace_nan(arr)

    arr = percentile_clip(arr, clip)

    arr = bytescale(arr)

    arr = reshape_array(arr)

    matplotlib_display(arr, figsize, title, cmap)


def load(fpath: str) -> np.ndarray:
    """loads an image from file into a numpy array"""

    return image.imread(fpath)


def cli(filepath: str =typer.Argument(''), title: str = "", clip: int = 0, cmap: str = "viridis"):
    """
    Quicklook command line interface

    Args:
        filepath (str): The file path of the image to look at

    Kwargs:
        title (str): The title for the display window when looking at the image
        clip (int): The percentile clip to remove from edges of pixel value distribution
        cmap (str): Name of colormap to use for 2D arrays (ignored for arrays with more than 2 dimensions)

    """

    if not filepath:
        print(
            """
      No filepath given, to load file from path:

      quicklook path/to/file.png"""
        )
        arr = static()
    else:
        arr = load(filepath)
        if not title:
            title = Path(filepath).stem

    show(arr, clip=clip, title=title, cmap=cmap)


def main():
    """pretty command line interface using typer"""
    typer.run(cli)


if __name__ == "__main__":
    main()
