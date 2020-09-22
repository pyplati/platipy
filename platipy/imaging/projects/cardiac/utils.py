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


def COMFromImageList(
    sitkImageList,
    conditionType="count",
    conditionValue=0,
    scanDirection="z",
    debug=False,
):
    """
    Input: list of SimpleITK images
           minimum total slice area required for the tube to be inserted at that slice
           scan direction: x = sagittal, y=coronal, z=axial
    Output: mean centre of mass positions, with shape (NumSlices, 2)
    Note: positions are converted into image space by default
    """
    if scanDirection.lower() == "x":
        if debug:
            print("Scanning in sagittal direction")
        COMZ = []
        COMY = []
        W = []
        C = []

        referenceImage = sitkImageList[0]
        referenceArray = sitk.GetArrayFromImage(referenceImage)
        z, y = np.mgrid[
            0 : referenceArray.shape[0] : 1, 0 : referenceArray.shape[1] : 1
        ]

        with np.errstate(divide="ignore", invalid="ignore"):
            for sitkImage in sitkImageList:
                volumeArray = sitk.GetArrayFromImage(sitkImage)
                comZ = 1.0 * (z[:, :, np.newaxis] * volumeArray).sum(axis=(1, 0))
                comY = 1.0 * (y[:, :, np.newaxis] * volumeArray).sum(axis=(1, 0))
                weights = np.sum(volumeArray, axis=(1, 0))
                W.append(weights)
                C.append(np.any(volumeArray, axis=(1, 0)))
                comZ /= 1.0 * weights
                comY /= 1.0 * weights
                COMZ.append(comZ)
                COMY.append(comY)

        with warnings.catch_warnings():
            """
            It's fairly likely some slices have just np.NaN values - it raises a warning but we can suppress it here
            """
            warnings.simplefilter("ignore", category=RuntimeWarning)
            meanCOMZ = np.nanmean(COMZ, axis=0)
            meanCOMY = np.nanmean(COMY, axis=0)
            if conditionType.lower() == "area":
                meanCOM = (
                    np.dstack((meanCOMZ, meanCOMY))[0]
                    * np.array((np.sum(W, axis=0) > (conditionValue),) * 2).T
                )
            elif conditionType.lower() == "count":
                meanCOM = (
                    np.dstack((meanCOMZ, meanCOMY))[0]
                    * np.array((np.sum(C, axis=0) > (conditionValue),) * 2).T
                )
            else:
                print("Invalid condition type, please select from 'area' or 'count'.")
                sys.exit()

        pointArray = []
        for index, COM in enumerate(meanCOM):
            if np.all(np.isfinite(COM)):
                if np.all(COM > 0):
                    pointArray.append(
                        referenceImage.TransformIndexToPhysicalPoint(
                            (index, int(COM[1]), int(COM[0]))
                        )
                    )

        return pointArray

    elif scanDirection.lower() == "z":
        if debug:
            print("Scanning in axial direction")
        COMX = []
        COMY = []
        W = []
        C = []

        referenceImage = sitkImageList[0]
        referenceArray = sitk.GetArrayFromImage(referenceImage)
        x, y = np.mgrid[
            0 : referenceArray.shape[1] : 1, 0 : referenceArray.shape[2] : 1
        ]

        with np.errstate(divide="ignore", invalid="ignore"):
            for sitkImage in sitkImageList:
                volumeArray = sitk.GetArrayFromImage(sitkImage)
                comX = 1.0 * (x * volumeArray).sum(axis=(1, 2))
                comY = 1.0 * (y * volumeArray).sum(axis=(1, 2))
                weights = np.sum(volumeArray, axis=(1, 2))
                W.append(weights)
                C.append(np.any(volumeArray, axis=(1, 2)))
                comX /= 1.0 * weights
                comY /= 1.0 * weights
                COMX.append(comX)
                COMY.append(comY)

        with warnings.catch_warnings():
            """
            It's fairly likely some slices have just np.NaN values - it raises a warning but we
            can suppress it here
            """
            warnings.simplefilter("ignore", category=RuntimeWarning)
            meanCOMX = np.nanmean(COMX, axis=0)
            meanCOMY = np.nanmean(COMY, axis=0)
            if conditionType.lower() == "area":
                meanCOM = (
                    np.dstack((meanCOMX, meanCOMY))[0]
                    * np.array((np.sum(W, axis=0) > (conditionValue),) * 2).T
                )
            elif conditionType.lower() == "count":
                meanCOM = (
                    np.dstack((meanCOMX, meanCOMY))[0]
                    * np.array((np.sum(C, axis=0) > (conditionValue),) * 2).T
                )
            else:
                print("Invalid condition type, please select from 'area' or 'count'.")
                quit()
        pointArray = []
        for index, COM in enumerate(meanCOM):
            if np.all(np.isfinite(COM)):
                if np.all(COM > 0):
                    pointArray.append(
                        referenceImage.TransformIndexToPhysicalPoint(
                            (int(COM[1]), int(COM[0]), index)
                        )
                    )

        return pointArray


def tubeFromCOMList(COMList, radius, debug=False):
    """
    Input: image-space positions along the tube centreline.
    Output: VTK tube
    Note: positions do not have to be continuous - the tube is interpolated in real space
    """
    points = vtk.vtkPoints()
    for i, pt in enumerate(COMList):
        points.InsertPoint(i, pt[0], pt[1], pt[2])

    # Fit a spline to the points
    if debug:
        print("Fitting spline")
    spline = vtk.vtkParametricSpline()
    spline.SetPoints(points)

    functionSource = vtk.vtkParametricFunctionSource()
    functionSource.SetParametricFunction(spline)
    functionSource.SetUResolution(10 * points.GetNumberOfPoints())
    functionSource.Update()

    # Generate the radius scalars
    tubeRadius = vtk.vtkDoubleArray()
    n = functionSource.GetOutput().GetNumberOfPoints()
    tubeRadius.SetNumberOfTuples(n)
    tubeRadius.SetName("TubeRadius")
    for i in range(n):
        # We can set the radius based on the given propagated segmentations in that slice?
        # Typically segmentations are elliptical, this could be an issue so for now a constant
        # radius is used
        tubeRadius.SetTuple1(i, radius)

    # Add the scalars to the polydata
    tubePolyData = vtk.vtkPolyData()
    tubePolyData = functionSource.GetOutput()
    tubePolyData.GetPointData().AddArray(tubeRadius)
    tubePolyData.GetPointData().SetActiveScalars("TubeRadius")

    # Create the tubes
    tuber = vtk.vtkTubeFilter()
    tuber.SetInputData(tubePolyData)
    tuber.SetNumberOfSides(50)
    tuber.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
    tuber.Update()

    return tuber


def writeVTKTubeToFile(tube, filename):
    """
    Input: VTK tube
    Output: exit success
    Note: format is XML VTP
    """
    print("Writing tube to polydata file (VTP)")
    polyDataWriter = vtk.vtkXMLPolyDataWriter()
    polyDataWriter.SetInputData(tube.GetOutput())

    polyDataWriter.SetFileName(filename)
    polyDataWriter.SetCompressorTypeToNone()
    polyDataWriter.SetDataModeToAscii()
    s = polyDataWriter.Write()

    return s


def SimpleITKImageFromVTKTube(tube, SITKReferenceImage, debug=False):
    """
    Input: VTK tube, referenceImage (used for spacing, etc.)
    Output: SimpleITK image
    Note: Uses binary output (background 0, foreground 1)
    """
    size = list(SITKReferenceImage.GetSize())
    origin = list(SITKReferenceImage.GetOrigin())
    spacing = list(SITKReferenceImage.GetSpacing())
    ncomp = SITKReferenceImage.GetNumberOfComponentsPerPixel()

    # convert the SimpleITK image to a numpy array
    arr = sitk.GetArrayFromImage(SITKReferenceImage).transpose(2, 1, 0).flatten()

    # send the numpy array to VTK with a vtkImageImport object
    dataImporter = vtk.vtkImageImport()

    dataImporter.CopyImportVoidPointer(arr, len(arr))
    dataImporter.SetDataScalarTypeToUnsignedChar()
    dataImporter.SetNumberOfScalarComponents(ncomp)

    # Set the new VTK image's parameters
    dataImporter.SetDataExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)
    dataImporter.SetWholeExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)
    dataImporter.SetDataOrigin(origin)
    dataImporter.SetDataSpacing(spacing)

    dataImporter.Update()

    VTKReferenceImage = dataImporter.GetOutput()

    # fill the image with foreground voxels:
    inval = 1
    outval = 0
    VTKReferenceImage.GetPointData().GetScalars().Fill(inval)

    if debug:
        print("Using polydaya to generate stencil.")
    pol2stenc = vtk.vtkPolyDataToImageStencil()
    pol2stenc.SetTolerance(0.5)  # points within 0.5 voxels are included
    pol2stenc.SetInputConnection(tube.GetOutputPort())
    pol2stenc.SetOutputOrigin(VTKReferenceImage.GetOrigin())
    pol2stenc.SetOutputSpacing(VTKReferenceImage.GetSpacing())
    pol2stenc.SetOutputWholeExtent(VTKReferenceImage.GetExtent())
    pol2stenc.Update()

    if debug:
        print("using stencil to generate image.")
    imgstenc = vtk.vtkImageStencil()
    imgstenc.SetInputData(VTKReferenceImage)
    imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
    imgstenc.ReverseStencilOff()
    imgstenc.SetBackgroundValue(outval)
    imgstenc.Update()

    if debug:
        print("Generating SimpleITK image.")
    finalImage = imgstenc.GetOutput()
    finalArray = finalImage.GetPointData().GetScalars()
    finalArray = vtk_to_numpy(finalArray).reshape(SITKReferenceImage.GetSize()[::-1])
    if debug:
        print(f"Volume = {finalArray.sum()*sum(spacing):.3f} mm^3")
    finalImageSITK = sitk.GetImageFromArray(finalArray)
    finalImageSITK.CopyInformation(SITKReferenceImage)

    return finalImageSITK


def ConvertSimpleITKtoVTK(img):
    """

    """
    size = list(img.GetSize())
    origin = list(img.GetOrigin())
    spacing = list(img.GetSpacing())
    ncomp = img.GetNumberOfComponentsPerPixel()

    # convert the SimpleITK image to a numpy array
    arr = sitk.GetArrayFromImage(img).transpose(2, 1, 0).flatten()
    arr_string = arr.tostring()

    # send the numpy array to VTK with a vtkImageImport object
    dataImporter = vtk.vtkImageImport()

    dataImporter.CopyImportVoidPointer(arr_string, len(arr_string))
    dataImporter.SetDataScalarTypeToUnsignedChar()
    dataImporter.SetNumberOfScalarComponents(ncomp)

    # Set the new VTK image's parameters
    dataImporter.SetDataExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)
    dataImporter.SetWholeExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)
    dataImporter.SetDataOrigin(origin)
    dataImporter.SetDataSpacing(spacing)

    dataImporter.Update()

    vtk_image = dataImporter.GetOutput()
    return vtk_image


def vesselSplineGeneration(
    referenceImage,
    atlasSet,
    vesselNameList,
    vesselRadiusDict,
    stopConditionTypeDict,
    stopConditionValueDict,
    scanDirectionDict,
    debug=False,
):
    """

    """
    splinedVessels = {}
    for vesselName in vesselNameList:

        # We must set the image direction to identity
        # This is because it is not possible to modify VTK Image directions
        # This may get fixed in a future VTK version

        initial_image_direction = referenceImage.GetDirection()

        imageList = [atlasSet[i]["DIR"][vesselName] for i in atlasSet.keys()]
        for im in imageList:
            im.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))

        vesselRadius = vesselRadiusDict[vesselName]
        stopConditionType = stopConditionTypeDict[vesselName]
        stopConditionValue = stopConditionValueDict[vesselName]
        scanDirection = scanDirectionDict[vesselName]

        pointArray = COMFromImageList(
            imageList,
            conditionType=stopConditionType,
            conditionValue=stopConditionValue,
            scanDirection=scanDirection,
            debug=debug,
        )
        tube = tubeFromCOMList(pointArray, radius=vesselRadius, debug=debug)

        SITKReferenceImage = imageList[0]

        vessel_delineation = SimpleITKImageFromVTKTube(
            tube, SITKReferenceImage, debug=debug
        )

        vessel_delineation.SetDirection(initial_image_direction)

        splinedVessels[vesselName] = vessel_delineation

        # We also have to reset the direction to whatever it was
        # This is because SimpleITK doesn't use deep copying
        # And it isn't necessary here as we can save some sweet, sweet memory
        for im in imageList:
            im.SetDirection(initial_image_direction)

    return splinedVessels
