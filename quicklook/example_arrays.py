"""
Example Arrays

Some functions to create example numpy arrays
"""

import numpy as np


def static(width: int = 100, height: int = 100) -> np.ndarray:
    """
    a random 2D array

    This function is called "static" because the output looks like
    the white noise signal that televisions show(ed) when
    there is no incoming broadcast signal
    """

    if not (isinstance(width, int) and isinstance(height, int)):
        raise TypeError("width and heigh must be integers")

    return np.random.rand(height, width)


def distances_from_centre(size: int = 201) -> np.ndarray:
    """an array which measures x and y distance from centre"""

    coords = range(size)

    # distance from origin
    X, Y = np.meshgrid(coords, coords)

    # distance from centre
    X = X - np.floor(size / 2)
    Y = Y - np.floor(size / 2)

    return (X, Y)


def pretty_pattern(size: int = 201, XY: tuple = None) -> np.ndarray:
    """
    an array with a pretty pattern
    """

    if not (isinstance(size, int)):
        raise TypeError("size must be integers")

    if not XY:
        XY = distances_from_centre(size)

    # x and y distances
    X, Y = XY

    # creates a cool pattern
    hsq = X**2 + Y**2  # hypotenuse squared
    R = np.sin(hsq)
    G = np.cos(hsq)
    B = np.sin(np.sqrt(hsq) / size)  # radial gradient

    # 3D array (3 channels)
    arr = np.stack([R, G, B], axis=2)

    # min = 0
    arr = arr - np.min(arr)

    return arr


def egg_array(size: int = 201, XY: tuple = None) -> np.ndarray:
    """
    an array that has an egg shape inside of it

    thanks to John Cook for the formula to make an egg ^.^
    https://www.johndcook.com/blog/2018/04/18/equation-to-fit-an-egg/
    """

    if not (isinstance(size, int)):
        raise TypeError("size must be integers")

    if not XY:
        XY = distances_from_centre(size)

    # x and y distances
    X, Y = XY

    # egg parameters
    h = float(size) * 0.67  # height
    w = float(size) * 0.5  # width
    k = 1 / size  # asymmetry factor (0 is an ellipse)

    a = X**2 / (h / 2) ** 2
    b = Y**2 / (w / 2) ** 2
    c = 1 + k * X

    egg = a + b * c

    egg = egg < 1

    egg = np.rot90(egg)

    return egg


def easter_egg(size: int = 201) -> np.ndarray:
    """super secret easter egg"""

    XY = distances_from_centre(size)

    pattern = pretty_pattern(size, XY=XY)

    egg = egg_array(size, XY=XY)
    egg = np.repeat(egg[:, :, np.newaxis], 3, axis=2)

    arr = egg * pattern

    return arr
