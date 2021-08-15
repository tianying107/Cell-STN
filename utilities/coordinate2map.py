import numpy as np
from scipy import signal


def xy2gaussian(x_c, y_c, col, row, outer_range=24):
    """
    This function project the x, y coordinates to a 2-D map with three values. The points within the window follows
    2-D gaussian distribution, otherwise 0. The kernel windows set the point (x,y) as center. For example, if the
    outer_range is 24, the size of kernel window is 49x49;
    :param x_c: (int) coordinate x of the center point
    :param y_c: (int) coordinate y of the center point
    :param col: (int) total columns of the output
    :param row: (int) total rows of the output
    :param outer_range: (int) radius of the kernel window
    :return:
    """

    confidence_map = np.zeros((row+2*outer_range, col+2*outer_range))
    confidence_map[y_c:(y_c + 2*outer_range + 1), x_c:(x_c + 2*outer_range + 1)] = gkern(2*outer_range+1)
    confidence_map = confidence_map[outer_range:-outer_range, outer_range:-outer_range]

    return confidence_map


def gkern(kernlen=25, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d


def xy2square(x_c, y_c, col, row, inner_range=3, outer_range=10):
    """
    This function project the x, y coordinates to a 2-D map with three values. The points within the inner window have
    a value of 1, points within the outer area have a value of 0.7, otherwise 0. Both windows set the point (x,y) as
    center with radius of inner_range and outer_range, respectively. For example, if the outer_range is 10, the size of
    outer window is 21x21; if the inner_range is 3, the size of inner window is 7x7.
    :param x_c: (int) coordinate x of the center point
    :param y_c: (int) coordinate y of the center point
    :param col: (int) total columns of the output
    :param row: (int) total rows of the output
    :param inner_range: (int) radius of the inner window
    :param outer_range: (int) radius of the outer window
    :return: confidence_map: 2-D np array
    """
    confidence_map = np.zeros((row, col))
    confidence_map[np.maximum(0, y_c - outer_range):np.minimum(row, np.maximum(0, y_c + outer_range + 1)),
                   np.maximum(0, x_c - outer_range):np.minimum(col, np.maximum(0, x_c + outer_range + 1))] = 0.7
    confidence_map[np.maximum(0, y_c - inner_range):np.minimum(row, np.maximum(0, y_c + inner_range + 1)),
                   np.maximum(0, x_c - inner_range):np.minimum(col, np.maximum(0, x_c + inner_range + 1))] = 1
    return confidence_map
