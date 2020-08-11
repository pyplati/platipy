"""
Various useful utility functions for atlas based segmentation algorithms.
"""
import numpy as np

from scipy.stats import norm as scipy_norm


def vectorised_transform_index_to_physical_point(image, point_array, rotate=True):
    """
    Transforms a set of points from array indices to real-space
    """
    if rotate:
        spacing = image.GetSpacing()[::-1]
        origin = image.GetOrigin()[::-1]
    else:
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
    return point_array * spacing + origin


def vectorised_transform_physical_point_to_index(image, point_array, rotate=True):
    """
    Transforms a set of points from real-space to array indices
    """
    if rotate:
        spacing = image.GetSpacing()[::-1]
        origin = image.GetOrigin()[::-1]
    else:
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
    return (point_array - origin) / spacing


def median_absolute_deviation(data, axis=None):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    return np.median(np.abs(data - np.median(data, axis=axis)), axis=axis)


def gaussian_curve(x, a, m, s):
    return a * scipy_norm.pdf(x, loc=m, scale=s)
