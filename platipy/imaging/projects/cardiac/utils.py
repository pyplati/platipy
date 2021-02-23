# Copyright 2020 University of New South Wales, University of Sydney, Ingham Institute

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import re
import sys
import datetime
import warnings

from functools import reduce

import numpy as np
import SimpleITK as sitk
import vtk

from scipy.stats import norm as scipy_norm
from scipy.optimize import curve_fit
from scipy.ndimage import filters
from scipy.ndimage import measurements
from scipy.interpolate import griddata
from scipy.interpolate import RectSphereBivariateSpline

from platipy.imaging.atlas.label_fusion import combine_labels

from platipy.imaging.utils.tools import (
    vectorised_transform_index_to_physical_point,
    vectorised_transform_physical_point_to_index,
)

from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

def ThresholdAndMeasureLungVolume(image, l=0, u=1):
    """
    Thresholds an image using upper and lower bounds in the intensity.
    Each non-connected component has the volume and perimeter/border ratio measured

    Args
        image (sitk.Image)  : the input image
        l (float)           : the lower threshold
        u (float)           : the upper threshold

    Returns
        NP (np.ndarray)     : a one-dimensional array of the number of pixels in each component
        PBR (np.ndarray)    : a one-dimensional array of the perimeter/border ratio for each component
        mask (sitk.Image)   : the connected component label map
        maxVals (np.ndarray): a one-dimensional array of the label map values

    """
    # Perform the threshold
    imThresh = sitk.Threshold(image, lower=l, upper=u)

    # Create a connected component image
    mask = sitk.ConnectedComponent(sitk.Cast(imThresh * 1024, sitk.sitkInt32), True)

    # Get the number of pixels that fall into each label map value
    cts = np.bincount(sitk.GetArrayFromImage(mask).flatten())

    # Keep only the largest 6 components for analysis
    maxVals = cts.argsort()[-6:][::-1]

    # Calculate metrics that describe segmentations
    PBR = np.zeros_like(maxVals, dtype=np.float32)
    NP = np.zeros_like(maxVals, dtype=np.float32)
    label_shape_analysis = sitk.LabelShapeStatisticsImageFilter()
    for i, val in enumerate(maxVals):
        binaryVol = sitk.Equal(mask, val.astype(np.float))
        label_shape_analysis.Execute(binaryVol)
        PBR[i] = label_shape_analysis.GetPerimeterOnBorderRatio(True)
        NP[i] = label_shape_analysis.GetNumberOfPixels(True)

    return NP, PBR, mask, maxVals


def AutoLungSegment(image, l=0.05, u=0.4, NPthresh=1e5):
    """
    Segments the lungs, generating a bounding box

    Args
        image (sitk.Image)  : the input image
        l (float)           : the lower (normalised) threshold
        u (float)           : the upper (normalised) threshold
        NPthresh (int)      : lower limit of voxel counts for a structure to be tested

    Returns
        maskBox (np.ndarray)    : bounding box of the automatically segmented lungs
        maskBinary (sitk.Image) : the segmented lungs (+/- airways)

    """

    # Normalise image intensity
    imNorm = sitk.Normalize(sitk.Threshold(image, -1000, 500, outsideValue=-1000))

    # Calculate the label maps and metrics on non-connected regions
    NP, PBR, mask, labels = ThresholdAndMeasureLungVolume(imNorm, l, u)
    indices = np.array(np.where(np.logical_and(PBR <= 5e-4, NP > NPthresh)))

    if indices.size == 0:
        print("     Warning - non-zero perimeter/border ratio")
        indices = np.argmin(PBR)

    if indices.size == 1:
        validLabels = labels[indices]
        maskBinary = sitk.Equal(mask, int(validLabels))

    else:
        validLabels = labels[indices[0]]
        maskBinary = sitk.Equal(mask, int(validLabels[0]))
        for i in range(len(validLabels) - 1):
            maskBinary = sitk.Add(maskBinary, sitk.Equal(mask, int(validLabels[i + 1])))
    maskBinary = maskBinary > 0
    label_shape_analysis = sitk.LabelShapeStatisticsImageFilter()
    label_shape_analysis.Execute(maskBinary)
    maskBox = label_shape_analysis.GetBoundingBox(True)

    return maskBox, maskBinary


def CropImage(image, cropBox):
    """
    Crops an image using a bounding box

    Args
        image (sitk.Image)          : the input image
        cropBox (list, np.ndarray)  : the bounding box
                                      (sag0, cor0, ax0, sagD, corD, axD)

    Returns
        imCrop (sitk.Image)         : the cropped image

    """
    imCrop = sitk.RegionOfInterest(image, size=cropBox[3:], index=cropBox[:3])
    return imCrop


def norm(x, mean, sd):
    result = []
    for i in range(x.size):
        result += [
            1.0
            / (sd * np.sqrt(2 * np.pi))
            * np.exp(-((x[i] - mean) ** 2) / (2 * sd ** 2))
        ]
    return np.array(result)


def res(p, y, x):
    m, dm, sd1, sd2 = p
    m1 = m
    m2 = m1 + dm
    y_fit = norm(x, m1, sd1) + norm(x, m2, sd2)
    err = y - y_fit
    return err



