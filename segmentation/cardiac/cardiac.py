#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues Aug 13

"""
import os, re, sys, datetime, warnings

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

from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

debug=True

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
    mask = sitk.ConnectedComponent(sitk.Cast(imThresh*1024, sitk.sitkInt32),True)

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


def AutoLungSegment(image, l = 0.05, u = 0.4, NPthresh=1e5):
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
    imNorm = sitk.Normalize(sitk.Threshold(image, -1000,500, outsideValue=-1000))

    # Calculate the label maps and metrics on non-connected regions
    NP, PBR, mask, labels = ThresholdAndMeasureLungVolume(imNorm,l,u)
    indices = np.array(np.where(np.logical_and(PBR<=5e-4, NP>NPthresh)))

    if indices.size==0:
        print("     Warning - non-zero perimeter/border ratio")
        indices = np.argmin(PBR)

    if indices.size==1:
        validLabels = labels[indices]
        maskBinary = sitk.Equal(mask, int(validLabels))

    else:
        validLabels = labels[indices[0]]
        maskBinary = sitk.Equal(mask, int(validLabels[0]))
        for i in range(len(validLabels)-1):
            maskBinary = sitk.Add(maskBinary, sitk.Equal(mask, int(validLabels[i+1])))
    maskBinary = maskBinary>0
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


def InitialRegistration(fixedImage, movingImage, movingStructure=False, fixedStructure=False, options=None, trace=False, regMethod = 'Rigid'):
    """
    Rigid image registration using ITK

    Args
        fixedImage (sitk.Image) : the fixed image
        movingImage (sitk.Image): the moving image, transformed to match fixedImage
        options (dict)          : registration options
        structure (bool)        : True if the image is a structure image

    Returns
        registeredImage (sitk.Image): the rigidly registered moving image
        transform (transform        : the transform, can be used directly with sitk.ResampleImageFilter

    """

    # Re-cast
    fixedImage = sitk.Cast(fixedImage, sitk.sitkFloat32)
    movingImage = sitk.Cast(movingImage, sitk.sitkFloat32)

    if not options:
        options = { 'shrinkFactors': [8,2,1],
                    'smoothSigmas' : [4,2,0],
                    'samplingRate' : 0.1,
                    'finalInterp'  : sitk.sitkBSpline
        }

    # Get the options
    shrinkFactors = options['shrinkFactors']
    smoothSigmas  = options['smoothSigmas']
    samplingRate  = options['samplingRate']
    finalInterp  = options['finalInterp']

    if regMethod == 'Rigid':
        # Select the rigid transform
        transform = sitk.VersorRigid3DTransform()
    elif regMethod == 'Affine':
        # Select the affine transform
        transform = sitk.AffineTransform(3)
    elif regMethod == 'Translation':
        # Select the translation transform
        transform = sitk.TranslationTransform(3)
    else:
        print('[ERROR] Registration method must be Rigid, Affine or Translation.')
        sys.exit()

    # Set up image registration method
    registration = sitk.ImageRegistrationMethod()

    registration.SetShrinkFactorsPerLevel(shrinkFactors)
    registration.SetSmoothingSigmasPerLevel(smoothSigmas)
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5,
                                      numberOfIterations=512,
                                      maximumNumberOfCorrections=50,
                                      maximumNumberOfFunctionEvaluations=1024,
                                      costFunctionConvergenceFactor=1e+7,
                                      trace=trace)

    registration.SetMetricAsMeanSquares()
    registration.SetInterpolator(sitk.sitkLinear) # Perhaps a small gain in improvement
    registration.SetMetricSamplingPercentage(samplingRate)
    registration.SetMetricSamplingStrategy(sitk.ImageRegistrationMethod.REGULAR)

    if movingStructure:
        registration.SetMetricMovingMask(movingStructure)

    if fixedStructure:
        registration.SetMetricFixedMask(fixedStructure)

    initializer = sitk.CenteredTransformInitializerFilter()
    initializer.GeometryOn()
    initialTransform = initializer.Execute(fixedImage, movingImage, sitk.VersorRigid3DTransform())

    registration.SetInitialTransform(initialTransform)
    outputTransform = registration.Execute(fixed=fixedImage, moving=movingImage)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixedImage)
    resampler.SetTransform(outputTransform)
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetDefaultPixelValue(-1024)

    registeredImage = resampler.Execute(movingImage)

    return registeredImage, outputTransform

def TransformPropagation(fixedImage, movingImage, transform, structure=False, interp=sitk.sitkNearestNeighbor):
    """
    Transform propagation using ITK

    Args
        fixedImage (sitk.Image)     : the fixed image
        movingImage (sitk.Image)    : the moving image, to be propagated
        transform (sitk.transform)  : the transformation; e.g. VersorRigid3DTransform, AffineTransform
        structure (bool)            : True if the image is a structure image
        interp (int)                : the interpolation
                                        sitk.sitkNearestNeighbor
                                        sitk.sitkLinear
                                        sitk.sitkBSpline

    Returns
        registeredImage (sitk.Image)        : the rigidly registered moving image

    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixedImage)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(interp)
    if structure:
        resampler.SetDefaultPixelValue(0)
    else:
        resampler.SetDefaultPixelValue(-1024)

    outputImage = resampler.Execute(movingImage)

    if structure and interp>1:
        print("Note:  Higher order interpolation on binary mask - using 32-bit floating point output.")
        outputIm = sitk.Cast(outputImage, sitk.sitkFloat32)
        # Safe way to remove dodgy values that can cause issues later
        outputImage = sitk.Threshold(outputIm, lower=1e-5, upper=100.0)
    else:
        outputImage = sitk.Cast(outputImage, movingImage.GetPixelID())

    return outputImage


def smooth_and_resample(image, shrink_factor, smoothing_sigma):
    """
    Args:
        image: The image we want to resample.
        shrink_factor: A number greater than one, such that the new image's size is original_size/shrink_factor.
        smoothing_sigma: Sigma for Gaussian smoothing, this is in physical (image spacing) units, not pixels.
    Return:
        Image which is a result of smoothing the input and then resampling it using the given sigma and shrink factor.
    """
    if smoothing_sigma>0:
        smoothed_image = sitk.SmoothingRecursiveGaussian(image, smoothing_sigma)
    else:
        smoothed_image = image

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [int(sz/float(shrink_factor) + 0.5) for sz in original_size]
    new_spacing = [((original_sz-1)*original_spc)/(new_sz-1)
                   for original_sz, original_spc, new_sz in zip(original_size, original_spacing, new_size)]
    return sitk.Resample(smoothed_image, new_size, sitk.Transform(),
                         sitk.sitkLinear, image.GetOrigin(),
                         new_spacing, image.GetDirection(), 0.0,
                         image.GetPixelID())



def multiscale_demons(registration_algorithm,
                      fixed_image, moving_image, initial_transform = None,
                      shrink_factors=None, smoothing_sigmas=None, iterationStaging=None, sgFlag=0):
    """
    Run the given registration algorithm in a multiscale fashion. The original scale should not be given as input as the
    original images are implicitly incorporated as the base of the pyramid.
    Args:
        registration_algorithm: Any registration algorithm that has an Execute(fixed_image, moving_image, displacement_field_image)
                                method.
        fixed_image: Resulting transformation maps points from this image's spatial domain to the moving image spatial domain.
        moving_image: Resulting transformation maps points from the fixed_image's spatial domain to this image's spatial domain.
        initial_transform: Any SimpleITK transform, used to initialize the displacement field.
        shrink_factors: Shrink factors relative to the original image's size.
        smoothing_sigmas: Amount of smoothing which is done prior to resmapling the image using the given shrink factor. These
                          are in physical (image spacing) units.
    Returns:
        SimpleITK.DisplacementFieldTransform
    """
    # Create image pyramid.
    fixed_images = []
    moving_images = []
    if shrink_factors:
        for shrink_factor, smoothing_sigma in reversed(list(zip(shrink_factors, smoothing_sigmas))):
            fixed_images.append(smooth_and_resample(fixed_image, shrink_factor, smoothing_sigma))
            moving_images.append(smooth_and_resample(moving_image, shrink_factor, smoothing_sigma))

    # Create initial displacement field at lowest resolution.
    # Currently, the pixel type is required to be sitkVectorFloat64 because of a constraint imposed by the Demons filters.
    if initial_transform:
        initial_displacement_field = sitk.TransformToDisplacementField(initial_transform,
                                                                       sitk.sitkVectorFloat64,
                                                                       fixed_images[-1].GetSize(),
                                                                       fixed_images[-1].GetOrigin(),
                                                                       fixed_images[-1].GetSpacing(),
                                                                       fixed_images[-1].GetDirection())
    else:
        initial_displacement_field = sitk.Image(fixed_images[-1].GetWidth(),
                                                fixed_images[-1].GetHeight(),
                                                fixed_images[-1].GetDepth(),
                                                sitk.sitkVectorFloat64)
        initial_displacement_field.CopyInformation(fixed_images[-1])

    # Run the registration.
    iters = iterationStaging[0]
    registration_algorithm.SetNumberOfIterations(iters)
    initial_displacement_field = registration_algorithm.Execute(fixed_images[-1],
                                                                moving_images[-1],
                                                                initial_displacement_field)
    # Start at the top of the pyramid and work our way down.
    for i, (f_image, m_image) in enumerate(reversed(list(zip(fixed_images[0:-1], moving_images[0:-1])))):
        initial_displacement_field = sitk.Resample (initial_displacement_field, f_image)
        iters = iterationStaging[i+1]
        registration_algorithm.SetNumberOfIterations(iters)
        initial_displacement_field = registration_algorithm.Execute(f_image, m_image, initial_displacement_field)
    return sitk.DisplacementFieldTransform(initial_displacement_field)

def command_iteration(filter) :
    """
    Utility function to print information during demons registration
    """
    print("{0:3} = {1:10.5f}".format(filter.GetElapsedIterations(),filter.GetMetric()))

def FastSymmetricForcesDemonsRegistration(fixed_image, moving_image, resolutionStaging=[8,4,1], iterationStaging=[10,10,10], smoothingSigmaFactor=1, ncores=1, structure=False, interpOrder=2, trace=False):
    """
    Deformable image propagation using Fast Symmetric-Forces Demons

    Args
        fixed_image (sitk.Image)        : the fixed image
        moving_image (sitk.Image)       : the moving image, to be deformable registered (must be in the same image space)
        resolutionStaging (list[int])   : down-sampling factor for each resolution level
        iterationStaging (list[int])    : number of iterations for each resolution level
        ncores (int)                    : number of processing cores to use
        structure (bool)                : True if the image is a structure image
        smoothingSigmaFactor (float)    : the relative width of the Gaussian smoothing kernel
        interpOrder (int)               : the interpolation order
                                            1 = Nearest neighbour
                                            2 = Bi-linear splines
                                            3 = B-Spline (cubic)

    Returns
        registeredImage (sitk.Image)    : the registered moving image
        outputTransform                 : the displacement field transform
    """

    # Cast to floating point representation, if necessary
    if fixed_image.GetPixelID()!=6:
        fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    if moving_image.GetPixelID()!=6:
        moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)

    # Set up the appropriate image filter
    registration_method = sitk.FastSymmetricForcesDemonsRegistrationFilter()

    # Multi-resolution framework
    registration_method.SetNumberOfThreads(ncores)
    registration_method.SetSmoothUpdateField(True)
    registration_method.SetSmoothDisplacementField(True)
    registration_method.SetStandardDeviations(1.5)

    # This allows monitoring of the progress
    if trace:
        registration_method.AddCommand( sitk.sitkIterationEvent, lambda: command_iteration(registration_method) )

    outputTransform = multiscale_demons(registration_algorithm=registration_method,
                                        fixed_image = fixed_image,
                                        moving_image = moving_image,
                                        shrink_factors = resolutionStaging,
                                        smoothing_sigmas = [i*smoothingSigmaFactor for i in resolutionStaging],
                                        iterationStaging = iterationStaging)
    deformationField = outputTransform.GetDisplacementField()

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(interpOrder)

    if structure:
        resampler.SetDefaultPixelValue(0)
    else:
        resampler.SetDefaultPixelValue(-1024)

    resampler.SetTransform(outputTransform)
    registeredImage = resampler.Execute(moving_image)

    if structure:
        registeredImage = sitk.Cast(registeredImage, sitk.sitkFloat32)
        registeredImage = sitk.Threshold(registeredImage, lower=1e-5, upper=100)

    registeredImage.CopyInformation(fixed_image)

    return registeredImage, outputTransform

def ApplyField(inputImage, outputTransform, structure=False, interp=sitk.sitkNearestNeighbor):

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(inputImage)

    if structure:
        resampler.SetDefaultPixelValue(0)
    else:
        resampler.SetDefaultPixelValue(-1024)

    #transform = sitk.DisplacementFieldTransform(sitk.Cast(deformationField, sitk.sitkVectorFloat64))
    resampler.SetTransform(outputTransform)
    resampler.SetInterpolator(interp)

    registeredImage = resampler.Execute(sitk.Cast(inputImage, sitk.sitkFloat32))

    return registeredImage


def vectorisedTransformIndexToPhysicalPoint(image, pointArr, correct=True):
    """
    Transforms a set of points from array indices to real-space
    """
    if correct:
        spacing = image.GetSpacing()[::-1]
        origin = image.GetOrigin()[::-1]
    else:
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
    return pointArr*spacing + origin

def vectorisedTransformPhysicalPointToIndex(image, pointArr, correct=True):
    """
    Transforms a set of points from real-space to array indices
    """
    if correct:
        spacing = image.GetSpacing()[::-1]
        origin = image.GetOrigin()[::-1]
    else:
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
    return (pointArr-origin)/spacing


def evaluateDistanceOnSurface(referenceVolume, testVolume, absDistance=True, referenceAsDistanceMap=False):
    """
    Evaluates a distance map on a surface
    Input: referenceVolume: binary volume SimpleITK image, or alternatively a distance map
           testVolume: binary volume SimpleITK image
    Output: theta, phi, values
    """
    if referenceAsDistanceMap:
        referenceDistanceMap = referenceVolume
    else:
        if absDistance:
            referenceDistanceMap = sitk.Abs(sitk.SignedMaurerDistanceMap(referenceVolume, squaredDistance=False, useImageSpacing=True))

        else:
            referenceDistanceMap = sitk.SignedMaurerDistanceMap(referenceVolume, squaredDistance=False, useImageSpacing=True)

    testSurface = sitk.LabelContour(testVolume)

    distanceImage = sitk.Multiply(referenceDistanceMap, sitk.Cast(testSurface, sitk.sitkFloat32))
    distanceArray = sitk.GetArrayFromImage(distanceImage)

    # Calculate centre of mass in real coordinates
    testSurfaceArray = sitk.GetArrayFromImage(testSurface)
    testSurfaceLocations = np.where(testSurfaceArray==1)
    testSurfaceLocationsArray = np.array(testSurfaceLocations)
    COMIndex = testSurfaceLocationsArray.mean(axis=1)
    COMReal = vectorisedTransformIndexToPhysicalPoint(testSurface, COMIndex)

    # Calculate each point on the surface in real coordinates
    pts = testSurfaceLocationsArray.T
    ptsReal = vectorisedTransformIndexToPhysicalPoint(testSurface, pts)
    ptsDiff = ptsReal - COMReal

    # Convert to spherical polar coordinates - base at north pole
    rho = np.sqrt((ptsDiff*ptsDiff).sum(axis=1))
    theta = np.pi/2.-np.arccos(ptsDiff.T[0]/rho)
    phi =  -1*np.arctan2(ptsDiff.T[2],-1.0*ptsDiff.T[1])

    # Extract values
    values = distanceArray[testSurfaceLocations]

    return theta, phi, values


def regridSphericalData(theta, phi, values, resolution):
    """
    Re-grids spherical data
    Input: theta, phi, values
    Options: plot a figure (plotFig), save a figure (saveFig), case identifier (figName)
    Output: pLat, pLong, gridValues (, fig)
    """
    # Re-grid:
    #  Set up grid
    Dradian = resolution*np.pi/180
    pLong, pLat = np.mgrid[-np.pi:np.pi:Dradian, -np.pi/2.:np.pi/2.0:Dradian]

    # First pass - linear interpolation, works well but not for edges
    gridValues = griddata(list(zip(theta, phi)), values, (pLat, pLong), method='linear', rescale=False)

    # Second pass - nearest neighbour interpolation
    gridValuesNN = griddata(list(zip(theta, phi)), values, (pLat, pLong), method='nearest', rescale=False)

    # Third pass - wherever the linear interpolation isn't defined use nearest neighbour interpolation
    gridValues[~np.isfinite(gridValues)] = gridValuesNN[~np.isfinite(gridValues)]

    return pLat, pLong, gridValues


def medianAbsoluteDeviation(data, axis=None):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    return np.median(np.abs(data - np.median(data, axis=axis)), axis=axis)

def norm(x, mean, sd):
    norm = []
    for i in range(x.size):
        norm += [1.0/(sd*np.sqrt(2*np.pi))*np.exp(-(x[i] - mean)**2/(2*sd**2))]
    return np.array(norm)

def res(p, y, x):
    m, dm, sd1, sd2 = p
    m1 = m
    m2 = m1 + dm
    y_fit = norm(x, m1, sd1) + norm(x, m2, sd2)
    err = y - y_fit
    return err

def gaussianCurve(x, a, m, s):
    return a*scipy_norm.pdf(x, loc=m, scale=s)

def IAR(atlasSet, structureName, smoothMaps=False, smoothSigma=1, zScore='MAD', outlierMethod='IQR', minBestAtlases=10, N_factor=1.5, logFile='IAR_{0}.log'.format(datetime.datetime.now()), debug=False, iteration=0, singleStep=False):

    if iteration == 0:
        # Run some checks in the data?

        # Begin the process
        print('Iterative atlas removal: ')
        print('  Beginning process')
        logFile = open(logFile, 'w')
        logFile.write('Iteration,Atlases,Qvalue,Threshold\n')

    # Get remaining case identifiers to loop through
    remainingIdList = list(atlasSet.keys())

    #Modify resolution for better statistics
    if len(remainingIdList)<12:
        print('  Less than 12 atlases, resolution set: 3x3 sqr deg')
        resolution = 3
    elif len(remainingIdList)<7:
        print('  Less than 7 atlases, resolution set: 6x6 sqr deg')
        resolution = 6
    else:
        resolution = 1

    # Generate the surface projections
    #   1. Set the consensus surface using the reference volume
    probabilityLabel = combineLabels(atlasSet, structureName)[structureName]
    referenceVolume = processProbabilityImage(probabilityLabel, threshold=1)
    referenceDistanceMap = sitk.Abs(sitk.SignedMaurerDistanceMap(referenceVolume, squaredDistance=False, useImageSpacing=True))

    gValList = []
    print('  Calculating surface distance maps: ')
    #print('    ', end=' ')
    for testId in remainingIdList:
        print('    {0}'.format(testId), end=" ")
        sys.stdout.flush()
        #   2. Calculate the distance from the surface to the consensus surface

        testVolume = atlasSet[testId]['DIR'][structureName]

        # This next step ensures non-binary labels are treated properly
        # We use 0.1 to capture the outer edge of the test delineation, if it is probabilistic
        testVolume = processProbabilityImage(testVolume, 0.1)

        # Now compute the distance across the surface
        theta, phi, values = evaluateDistanceOnSurface(referenceDistanceMap, testVolume, referenceAsDistanceMap=True)
        pLat, pLong, gVals = regridSphericalData(theta, phi, values, resolution=resolution)

        gValList.append(gVals)
    print()
    QResults = {}

    for i, (testId, gVals) in enumerate(zip(remainingIdList, gValList)):

        gValListTest = gValList[:]
        gValListTest.pop(i)

        if smoothMaps:
            gVals = filters.gaussian_filter(gVals, sigma=smoothSigma, mode='wrap')

        #       b) i] Compute the Z-scores over the projected surface
        if zScore.lower()=='std':
            gValMean = np.mean(gValListTest, axis=0)
            gValStd = np.std(gValListTest, axis=0)

            if np.any(gValStd==0):
                print('    Std Dev zero count: {0}'.format(np.sum(gValStd==0)))
                gValStd[gValStd==0] = gValStd.mean()

            zScoreValsArr =  ( gVals - gValMean ) / gValStd

        elif zScore.lower()=='mad':
            gValMedian = np.median(gValListTest, axis=0)
            gValMAD    = 1.4826 * medianAbsoluteDeviation(gValListTest, axis=0)

            if np.any(~np.isfinite(gValMAD)):
                print('Error in MAD')
                print(gValMAD)

            if np.any(gValMAD==0):
                print('    MAD zero count: {0}'.format(np.sum(gValMAD==0)))
                gValMAD[gValMAD==0] = np.median(gValMAD)

            zScoreValsArr =  ( gVals - gValMedian ) / gValMAD

        else:
            print(' Error!')
            print(' zScore must be one of: MAD, STD')
            sys.exit()

        zScoreVals = np.ravel( zScoreValsArr )

        if debug:
            print('      [{0}] Statistics of mZ-scores'.format(testId))
            print('        Min(Z)    = {0:.2f}'.format(zScoreVals.min()))
            print('        Q1(Z)     = {0:.2f}'.format(np.percentile(zScoreVals, 25)))
            print('        Mean(Z)   = {0:.2f}'.format(zScoreVals.mean()))
            print('        Median(Z) = {0:.2f}'.format(np.percentile(zScoreVals, 50)))
            print('        Q3(Z)     = {0:.2f}'.format(np.percentile(zScoreVals, 75)))
            print('        Max(Z)    = {0:.2f}\n'.format(zScoreVals.max()))

        # Calculate excess area from Gaussian: the Q-metric
        bins = np.linspace(-15,15,501)
        zDensity, bin_edges = np.histogram(zScoreVals, bins=bins, density=True)
        bin_centers = (bin_edges[1:]+bin_edges[:-1])/2.0

        popt, pcov = curve_fit(f=gaussianCurve, xdata=bin_centers, ydata=zDensity)
        zIdeal = gaussianCurve(bin_centers, *popt)
        zDiff = np.abs(zDensity - zIdeal)

        # Integrate to get the Q_value
        Q_value = np.trapz(zDiff*np.abs(bin_centers)**2, bin_centers)
        QResults[testId] = np.float64(Q_value)

    # Exclude (at most) the worst 3 atlases for outlier detection
    # With a minimum number, this helps provide more robust estimates at low numbers
    RL = list(QResults.values())
    bestResults = np.sort(RL)[:max([minBestAtlases, len(RL)-3])]

    if outlierMethod.lower()=='iqr':
        outlierLimit = np.percentile(bestResults, 75, axis=0) + N_factor*np.subtract(*np.percentile(bestResults, [75, 25], axis=0))
    elif outlierMethod.lower()=='std':
        outlierLimit = np.mean(bestResults, axis=0) + N_factor*np.std(bestResults, axis=0)
    else:
        print(' Error!')
        print(' outlierMethod must be one of: IQR, STD')
        sys.exit()

    print('  Analysing results')
    print('   Outlier limit: {0:06.3f}'.format(outlierLimit))
    keepIdList = []

    logFile.write('{0},{1},{2},{3:.4g}\n'.format(iteration,
                                                 ' '.join(remainingIdList),
                                                 ' '.join(['{0:.4g}'.format(i) for i in list(QResults.values())]),
                                                 outlierLimit))
    logFile.flush()

    for ii, result in QResults.items():

        accept = (result <= outlierLimit)

        print('      {0}: Q = {1:06.3f} [{2}]'.format(ii, result, {True:'KEEP',False:'REMOVE'}[accept]))

        if accept:
            keepIdList.append(ii)

    if len(keepIdList)<len(remainingIdList):
        print('\n  Step {0} Complete'.format(iteration))
        print('   Num. Removed = {0} --\n'.format(len(remainingIdList)-len(keepIdList)))

        iteration += 1
        atlasSetNew = {i:atlasSet[i] for i in keepIdList}

        if singleStep:
            return atlasSetNew
        else:
            return IAR(atlasSet=atlasSetNew, structureName=structureName, smoothMaps=smoothMaps, smoothSigma=smoothSigma, zScore=zScore, outlierMethod=outlierMethod, minBestAtlases=minBestAtlases, N_factor=N_factor, logFile=logFile, debug=debug, iteration=iteration)

    else:
        print('  End point reached. Keeping:\n   {0}'.format(keepIdList))
        logFile.close()

        return atlasSet

def computeWeightMap(targetImage, movingImage, voteType='local', voteParams={'sigma':2.0, 'epsilon':1E-5, 'factor':1e12, 'gain':6, 'blockSize':5}):
    """
    Computes the weight map
    """
    # Cast to floating point representation, if necessary
    if targetImage.GetPixelID()!=6:
        targetImage = sitk.Cast(targetImage, sitk.sitkFloat32)
    if movingImage.GetPixelID()!=6:
        movingImage = sitk.Cast(movingImage, sitk.sitkFloat32)

    squareDifferenceImage = sitk.SquaredDifference(targetImage, movingImage)
    squareDifferenceImage = sitk.Cast(squareDifferenceImage, sitk.sitkFloat32)

    if voteType.lower()=='majority':
        weightMap = targetImage * 0.0 + 1.0

    elif voteType.lower()=='global':
        factor = voteParams['factor']
        sumSquaredDifference  = sitk.GetArrayFromImage(squareDifferenceImage).sum(dtype=np.float)
        globalWeight = factor / sumSquaredDifference

        weightMap = targetImage * 0.0 + globalWeight

    elif voteType.lower()=='local':
        sigma = voteParams['sigma']
        epsilon = voteParams['epsilon']

        rawMap = sitk.DiscreteGaussian(squareDifferenceImage, sigma*sigma)
        weightMap = sitk.Pow(rawMap + epsilon , -1.0)

    elif voteType.lower()=='block':
        factor = voteParams['factor']
        gain = voteParams['gain']
        blockSize = voteParams['blockSize']
        if type(blockSize)==int:
            blockSize = (blockSize,)*targetImage.GetDimension()

        #rawMap = sitk.Mean(squareDifferenceImage, blockSize)
        rawMap  = sitk.BoxMean(squareDifferenceImage, blockSize)
        weightMap = factor * sitk.Pow(rawMap, -1.0) ** abs(gain/2.0)
        # Note: we divide gain by 2 to account for using the squared difference image
        #       which raises the power by 2 already.

    else:
        raise ValueError('Weighting scheme not valid.')

    return sitk.Cast(weightMap, sitk.sitkFloat32)

def combineLabelsSTAPLE(labelListDict, threshold=1e-4):
    """
    Combine labels using STAPLE
    """

    combinedLabelDict = {}

    caseIdList = list(labelListDict.keys())
    structureNameList = [list(i.keys()) for i in labelListDict.values()]
    structureNameList = np.unique([item for sublist in structureNameList for item in sublist] )

    for structureName in structureNameList:
        # Ensure all labels are binarised
        binaryLabels = [sitk.BinaryThreshold(labelListDict[i][structureName], lowerThreshold=0.5) for i in labelListDict]

        # Perform STAPLE
        combinedLabel = sitk.STAPLE(binaryLabels)

        # Normalise
        combinedLabel = sitk.RescaleIntensity(combinedLabel, 0, 1)

        # Threshold - grants vastly improved compression performance
        if threshold:
            combinedLabel = sitk.Threshold(combinedLabel, lower=threshold, upper=1, outsideValue=0.0)

        combinedLabelDict[structureName] = combinedLabel

    return combinedLabelDict

def combineLabels(atlasSet, structureName, threshold=1e-4, smoothSigma=1.0):
    """
    Combine labels using weight maps
    """

    caseIdList = list(atlasSet.keys())

    if isinstance(structureName, str):
        structureNameList = [structureName]
    elif isinstance(structureName, list):
        structureNameList = structureName

    combinedLabelDict = {}

    for structureName in structureNameList:
        # Find the cases which have the strucure (in case some cases do not)
        validCaseIdList = [i for i in caseIdList if structureName in atlasSet[i]['DIR'].keys()]

        # Get valid weight images
        weightImageList = [atlasSet[caseId]['DIR']['Weight Map'] for caseId in validCaseIdList]

        # Sum the weight images
        weightSumImage = reduce(lambda x,y:x+y, weightImageList)
        weightSumImage = sitk.Mask(weightSumImage, weightSumImage==0, maskingValue=1, outsideValue=1)

        # Combine weight map with each label
        weightedLabels = [atlasSet[caseId]['DIR']['Weight Map']*sitk.Cast(atlasSet[caseId]['DIR'][structureName], sitk.sitkFloat32) for caseId in validCaseIdList]

        # Combine all the weighted labels
        combinedLabel = reduce(lambda x,y:x+y, weightedLabels) / weightSumImage

        # Smooth combined label
        combinedLabel = sitk.DiscreteGaussian(combinedLabel, smoothSigma*smoothSigma)

        # Normalise
        combinedLabel = sitk.RescaleIntensity(combinedLabel, 0, 1)

        # Threshold - grants vastly improved compression performance
        if threshold:
            combinedLabel = sitk.Threshold(combinedLabel, lower=threshold, upper=1, outsideValue=0.0)

        combinedLabelDict[structureName] = combinedLabel

    return combinedLabelDict

def processProbabilityImage(probabilityImage, threshold=0.5):

    # Check type
    if type(probabilityImage)!=sitk.Image:
        probabilityImage = sitk.GetImageFromArray(probabilityImage)

    # Normalise probability map
    probabilityImage = (probabilityImage / sitk.GetArrayFromImage(probabilityImage).max())

    # Get the starting binary image
    binaryImage = sitk.BinaryThreshold(probabilityImage, lowerThreshold=threshold)

    # Fill holes
    binaryImage = sitk.BinaryFillhole(binaryImage)

    # Apply the connected component filter
    labelledImage = sitk.ConnectedComponent(binaryImage)

    # Measure the size of each connected component
    labelShapeFilter = sitk.LabelShapeStatisticsImageFilter()
    labelShapeFilter.Execute(labelledImage)
    labelIndices = labelShapeFilter.GetLabels()
    voxelCounts  = [labelShapeFilter.GetNumberOfPixels(i) for i in labelIndices]
    if voxelCounts==[]:
        return binaryImage

    # Select the largest region
    largestComponentLabel = labelIndices[np.argmax(voxelCounts)]
    largestComponentImage = (labelledImage==largestComponentLabel)

    return sitk.Cast(largestComponentImage, sitk.sitkUInt8)


def COMFromImageList(sitkImageList, conditionType="count", conditionValue=0, scanDirection = 'z'):
    """
    Input: list of SimpleITK images
           minimum total slice area required for the tube to be inserted at that slice
           scan direction: x = sagittal, y=coronal, z=axial
    Output: mean centre of mass positions, with shape (NumSlices, 2)
    Note: positions are converted into image space by default
    """
    if scanDirection.lower()=='x':
        print("Scanning in sagittal direction")
        COMZ = []
        COMY = []
        W    = []
        C    = []

        referenceImage = sitkImageList[0]
        referenceArray = sitk.GetArrayFromImage(referenceImage)
        z,y = np.mgrid[0:referenceArray.shape[0]:1, 0:referenceArray.shape[1]:1]

        with np.errstate(divide='ignore', invalid='ignore'):
            for sitkImage in sitkImageList:
                volumeArray = sitk.GetArrayFromImage(sitkImage)
                comZ = 1.0*(z[:,:,np.newaxis]*volumeArray).sum(axis=(1,0))
                comY = 1.0*(y[:,:,np.newaxis]*volumeArray).sum(axis=(1,0))
                weights = np.sum(volumeArray, axis=(1,0))
                W.append(weights)
                C.append(np.any(volumeArray, axis=(1,0)))
                comZ/=(1.0*weights)
                comY/=(1.0*weights)
                COMZ.append(comZ)
                COMY.append(comY)

        with warnings.catch_warnings():
            """
            It's fairly likely some slices have just np.NaN values - it raises a warning but we can suppress it here
            """
            warnings.simplefilter("ignore", category=RuntimeWarning)
            meanCOMZ = np.nanmean(COMZ, axis=0)
            meanCOMY = np.nanmean(COMY, axis=0)
            if conditionType.lower()=="area":
                meanCOM = np.dstack((meanCOMZ, meanCOMY))[0]*np.array((np.sum(W, axis=0)>(conditionValue),)*2).T
            elif conditionType.lower()=="count":
                meanCOM = np.dstack((meanCOMZ, meanCOMY))[0]*np.array((np.sum(C, axis=0)>(conditionValue),)*2).T
            else:
                print("Invalid condition type, please select from 'area' or 'count'.")
                sys.exit()

        pointArray = []
        for index, COM in enumerate(meanCOM):
            if np.all(np.isfinite(COM)):
                if np.all(COM>0):
                    pointArray.append(referenceImage.TransformIndexToPhysicalPoint(( index, int(COM[1]), int(COM[0]))))

        return pointArray

    elif scanDirection.lower()=='z':
        print("Scanning in axial direction")
        COMX = []
        COMY = []
        W    = []
        C    = []

        referenceImage = sitkImageList[0]
        referenceArray = sitk.GetArrayFromImage(referenceImage)
        x,y = np.mgrid[0:referenceArray.shape[1]:1, 0:referenceArray.shape[2]:1]

        with np.errstate(divide='ignore', invalid='ignore'):
            for sitkImage in sitkImageList:
                volumeArray = sitk.GetArrayFromImage(sitkImage)
                comX = 1.0*(x*volumeArray).sum(axis=(1,2))
                comY = 1.0*(y*volumeArray).sum(axis=(1,2))
                weights = np.sum(volumeArray, axis=(1,2))
                W.append(weights)
                C.append(np.any(volumeArray, axis=(1,2)))
                comX/=(1.0*weights)
                comY/=(1.0*weights)
                COMX.append(comX)
                COMY.append(comY)

        with warnings.catch_warnings():
            """
            It's fairly likely some slices have just np.NaN values - it raises a warning but we can suppress it here
            """
            warnings.simplefilter("ignore", category=RuntimeWarning)
            meanCOMX = np.nanmean(COMX, axis=0)
            meanCOMY = np.nanmean(COMY, axis=0)
            if conditionType.lower()=="area":
                meanCOM = np.dstack((meanCOMX, meanCOMY))[0]*np.array((np.sum(W, axis=0)>(conditionValue),)*2).T
            elif conditionType.lower()=="count":
                meanCOM = np.dstack((meanCOMX, meanCOMY))[0]*np.array((np.sum(C, axis=0)>(conditionValue),)*2).T
            else:
                print("Invalid condition type, please select from 'area' or 'count'.")
                quit()
        pointArray = []
        for index, COM in enumerate(meanCOM):
            if np.all(np.isfinite(COM)):
                if np.all(COM>0):
                    pointArray.append(referenceImage.TransformIndexToPhysicalPoint((int(COM[1]), int(COM[0]), index)))

        return pointArray

def tubeFromCOMList(COMList, radius):
    """
    Input: image-space positions along the tube centreline.
    Output: VTK tube
    Note: positions do not have to be continuous - the tube is interpolated in real space
    """
    points = vtk.vtkPoints()
    for i,pt in enumerate(COMList):
        points.InsertPoint(i, pt[0], pt[1], pt[2])

    # Fit a spline to the points
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
        # Typically segmentations are elliptical, this could be an issue so for now a constant radius is used
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

def SimpleITKImageFromVTKTube(tube, SITKReferenceImage, verbose = False):
    """
    Input: VTK tube, referenceImage (used for spacing, etc.)
    Output: SimpleITK image
    Note: Uses binary output (background 0, foreground 1)
    """
    size     = list(SITKReferenceImage.GetSize())
    origin   = list(SITKReferenceImage.GetOrigin())
    spacing  = list(SITKReferenceImage.GetSpacing())
    ncomp    = SITKReferenceImage.GetNumberOfComponentsPerPixel()

    # convert the SimpleITK image to a numpy array
    arr = sitk.GetArrayFromImage(SITKReferenceImage).transpose(2,1,0).flatten()

    # send the numpy array to VTK with a vtkImageImport object
    dataImporter = vtk.vtkImageImport()

    dataImporter.CopyImportVoidPointer( arr, len(arr) )
    dataImporter.SetDataScalarTypeToUnsignedChar()
    dataImporter.SetNumberOfScalarComponents(ncomp)

    # Set the new VTK image's parameters
    dataImporter.SetDataExtent (0, size[0]-1, 0, size[1]-1, 0, size[2]-1)
    dataImporter.SetWholeExtent(0, size[0]-1, 0, size[1]-1, 0, size[2]-1)
    dataImporter.SetDataOrigin(origin)
    dataImporter.SetDataSpacing(spacing)

    dataImporter.Update()

    VTKReferenceImage = dataImporter.GetOutput()

    # fill the image with foreground voxels:
    inval = 1
    outval = 0
    count = VTKReferenceImage.GetNumberOfPoints()
    VTKReferenceImage.GetPointData().GetScalars().Fill(inval)

    if verbose:
        print("Generating volume using extrusion.")
    extruder = vtk.vtkLinearExtrusionFilter()
    extruder.SetInputData(tube.GetOutput())

    extruder.SetScaleFactor(1.)
    extruder.SetExtrusionTypeToNormalExtrusion()
    extruder.SetVector(0, 0, 1)
    extruder.Update()

    if verbose:
        print("Using polydaya to generate stencil.")
    pol2stenc = vtk.vtkPolyDataToImageStencil()
    pol2stenc.SetTolerance(0) # important if extruder.SetVector(0, 0, 1) !!!
    pol2stenc.SetInputConnection(tube.GetOutputPort())
    pol2stenc.SetOutputOrigin(VTKReferenceImage.GetOrigin())
    pol2stenc.SetOutputSpacing(VTKReferenceImage.GetSpacing())
    pol2stenc.SetOutputWholeExtent(VTKReferenceImage.GetExtent())
    pol2stenc.Update()

    if verbose:
        print("using stencil to generate image.")
    imgstenc = vtk.vtkImageStencil()
    imgstenc.SetInputData(VTKReferenceImage)
    imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
    imgstenc.ReverseStencilOff()
    imgstenc.SetBackgroundValue(outval)
    imgstenc.Update()

    if verbose:
        print("Generating SimpleITK image.")
    finalImage = imgstenc.GetOutput()
    finalArray = finalImage.GetPointData().GetScalars()
    finalArray = vtk_to_numpy(finalArray).reshape(SITKReferenceImage.GetSize()[::-1])
    finalImageSITK = sitk.GetImageFromArray(finalArray)
    finalImageSITK.CopyInformation(SITKReferenceImage)

    return finalImageSITK

def ConvertSimpleITKtoVTK(img):
    """

    """
    size     = list(img.GetSize())
    origin   = list(img.GetOrigin())
    spacing  = list(img.GetSpacing())
    ncomp    = img.GetNumberOfComponentsPerPixel()

    # convert the SimpleITK image to a numpy array
    arr = sitk.GetArrayFromImage(img).transpose(2,1,0).flatten()
    arr_string = arr.tostring()

    # send the numpy array to VTK with a vtkImageImport object
    dataImporter = vtk.vtkImageImport()

    dataImporter.CopyImportVoidPointer( arr_string, len(arr_string) )
    dataImporter.SetDataScalarTypeToUnsignedChar()
    dataImporter.SetNumberOfScalarComponents(ncomp)

    # Set the new VTK image's parameters
    dataImporter.SetDataExtent (0, size[0]-1, 0, size[1]-1, 0, size[2]-1)
    dataImporter.SetWholeExtent(0, size[0]-1, 0, size[1]-1, 0, size[2]-1)
    dataImporter.SetDataOrigin(origin)
    dataImporter.SetDataSpacing(spacing)

    dataImporter.Update()

    vtk_image = dataImporter.GetOutput()
    return vtk_image

def vesselSplineGeneration(atlasSet, vesselNameList, vesselRadiusDict, stopConditionTypeDict, stopConditionValueDict, scanDirectionDict):
    """

    """
    splinedVessels = {}
    for vesselName in vesselNameList:

        imageList    = [atlasSet[i]['DIR'][vesselName] for i in atlasSet.keys()]

        vesselRadius        = vesselRadiusDict[vesselName]
        stopConditionType   = stopConditionTypeDict[vesselName]
        stopConditionValue  = stopConditionValueDict[vesselName]
        scanDirection       = scanDirectionDict[vesselName]

        pointArray = COMFromImageList(imageList, conditionType=stopConditionType, conditionValue=stopConditionValue, scanDirection=scanDirection)
        tube       = tubeFromCOMList(pointArray, radius=vesselRadius)

        SITKReferenceImage  = imageList[0]

        splinedVessels[vesselName] = SimpleITKImageFromVTKTube(tube, SITKReferenceImage, verbose = False)
    return splinedVessels
