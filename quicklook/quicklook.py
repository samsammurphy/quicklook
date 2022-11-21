import numpy as np
from numpy import min, max
import matplotlib.pyplot as plt
from numbers import Number


def replace_nan(arr):
    """replace array elements that are "not a number" with the minimum valid number"""
    return np.nan_to_num(arr, nan=np.nanmin(arr))


def percentile_clip(arr, clip):
    """clip extreme pixel values using percentiles"""

    if not clip:
        return arr

    if not isinstance(clip, Number):
        raise TypeError(
            f"Percentile clip needs to be a number but was given a {type(clip)}"
        )

    if clip > 0 and clip < 50:
        arr = np.clip(arr, np.percentile(arr, clip), np.percentile(arr, 100 - clip))

    return arr


def scale(arr, a=0, b=255, dtype=int):
    """scale pixel values and datatype. Default to integer between 0 and 255"""

    # float dtype (temporarily) during scaling
    arr = arr.astype(float)

    if min(arr) == max(arr):
        # simple scale if all pixels same value
        scaled = arr / b
    else:
        # generic linear scale between integer values a and b
        scaled = (b - a) * ((arr - np.min(arr)) / (np.max(arr) - np.min(arr))) + a

    # output data type
    arr = scaled.astype(dtype)

    return arr


def reshape_array(arr):
    """ensures arrays have the correct shape for display using matplotlib"""

    if len(arr.shape) == 1:
        raise ValueError("Array needs to have more than 1 dimension")

    if len(arr.shape) == 2:
        return arr

    # channel last ordering (matplotlib requirement) TODO min might be second dimension
    if arr.shape[0] == min(arr.shape):
        arr = np.moveaxis(arr, 0, -1)

    # first three wavebands TODO support for two channel array
    if min(arr.shape) > 3:
        arr = arr[:, :, 0:3]

    # remove flat dimensions (matplotlib requirement)
    arr = np.squeeze(arr)

    return arr


def static(width=100, height=100):
    """a random 2D array"""

    if not (isinstance(width, int) and isinstance(height, int)):
        raise TypeError("width and heigh must be integers")

    return np.random.rand(height, width)


def matplotlib_display(arr, figsize, title, cmap):
    """display array in a window using matplotlib"""

    plt.figure(figsize=figsize)
    plt.imshow(arr, vmin=0, vmax=255, cmap=cmap)
    plt.title(title)
    plt.show()


def show(arr, clip=None, figsize=(5, 5), title="", cmap="viridis"):
    """quick and easy numpy array visualization"""

    # must be a numpy array
    if not isinstance(arr, (np.ndarray)):
        raise TypeError("Input must be a numpy array")

    arr = replace_nan(arr)

    arr = percentile_clip(arr, clip)

    arr = scale(arr)

    arr = reshape_array(arr)

    matplotlib_display(arr, figsize, title, cmap)


if __name__ == "__main__":
    # TODO command line tool
    main()
