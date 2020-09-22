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

"""
Module name: LabelComparison
Author:      Robert Finnegan
Date:        December 2018
Description:
---------------------------------
- Overlap metrics
- Volume metrics
- Surface metrics
---------------------------------

"""

from __future__ import print_function
import os, sys

import numpy as np
import SimpleITK as sitk


def compute_surface_metrics(imFixed, imMoving, verbose=False):
    """
    HD, meanSurfDist, medianSurfDist, maxSurfDist, stdSurfDist
    """
    hausdorffDistance = sitk.HausdorffDistanceImageFilter()
    hausdorffDistance.Execute(imFixed, imMoving)
    HD = hausdorffDistance.GetHausdorffDistance()

    meanSDList = []
    maxSDList = []
    stdSDList = []
    medianSDList = []
    numPoints = []
    for (imA, imB) in ((imFixed, imMoving), (imMoving, imFixed)):

        labelIntensityStat = sitk.LabelIntensityStatisticsImageFilter()
        referenceDistanceMap = sitk.Abs(sitk.SignedMaurerDistanceMap(imA, squaredDistance=False, useImageSpacing=True))
        movingLabelContour = sitk.LabelContour(imB)
        labelIntensityStat.Execute(movingLabelContour,referenceDistanceMap)

        meanSDList.append(labelIntensityStat.GetMean(1))
        maxSDList.append(labelIntensityStat.GetMaximum(1))
        stdSDList.append(labelIntensityStat.GetStandardDeviation(1))
        medianSDList.append(labelIntensityStat.GetMedian(1))

        numPoints.append(labelIntensityStat.GetNumberOfPixels(1))

    if verbose:
        print("        Boundary points:  {0}  {1}".format(numPoints[0], numPoints[1]))

    meanSurfDist = np.dot(meanSDList, numPoints)/np.sum(numPoints)
    maxSurfDist = np.max(maxSDList)
    stdSurfDist = np.sqrt( np.dot(numPoints,np.add(np.square(stdSDList), np.square(np.subtract(meanSDList,meanSurfDist)))) )
    medianSurfDist = np.mean(medianSDList)

    resultDict = {}
    resultDict['hausdorffDistance'] = HD
    resultDict['meanSurfaceDistance'] = meanSurfDist
    resultDict['medianSurfaceDistance'] = medianSurfDist
    resultDict['maximumSurfaceDistance'] = maxSurfDist
    resultDict['sigmaSurfaceDistance'] = stdSurfDist

    return resultDict

def compute_volume_metrics(imFixed, imMoving):
    """
    DSC, VolOverlap, FracOverlap, TruePosFrac, TrueNegFrac, FalsePosFrac, FalseNegFrac
    """
    arrFixed = sitk.GetArrayFromImage(imFixed).astype(bool)
    arrMoving = sitk.GetArrayFromImage(imMoving).astype(bool)

    arrInter = arrFixed & arrMoving
    arrUnion = arrFixed | arrMoving

    voxVol = np.product(imFixed.GetSpacing())/1000. # Conversion to cm^3

    # 2|A & B|/(|A|+|B|)
    DSC =  (2.0*arrInter.sum())/(arrFixed.sum()+arrMoving.sum())

    #  |A & B|/|A | B|
    FracOverlap = arrInter.sum()/arrUnion.sum().astype(float)
    VolOverlap = arrInter.sum() * voxVol

    TruePos = arrInter.sum()
    TrueNeg = (np.invert(arrFixed) & np.invert(arrMoving)).sum()
    FalsePos = arrMoving.sum()-TruePos
    FalseNeg = arrFixed.sum()-TruePos

    #
    TruePosFrac = (1.0*TruePos)/(TruePos+FalseNeg)
    TrueNegFrac = (1.0*TrueNeg)/(TrueNeg+FalsePos)
    FalsePosFrac = (1.0*FalsePos)/(TrueNeg+FalsePos)
    FalseNegFrac = (1.0*FalseNeg)/(TruePos+FalseNeg)

    resultDict = {}
    resultDict['DSC'] = DSC
    resultDict['volumeOverlap'] = VolOverlap
    resultDict['fractionOverlap'] = FracOverlap
    resultDict['truePositiveFraction'] = TruePosFrac
    resultDict['trueNegativeFraction'] = TrueNegFrac
    resultDict['falsePositiveFraction'] = FalsePosFrac
    resultDict['falseNegativeFraction'] = FalseNegFrac

    return resultDict

def compute_metric_dsc(imA, imB):
    arrA = sitk.GetArrayFromImage(imA).astype(bool)
    arrB = sitk.GetArrayFromImage(imB).astype(bool)
    return  2*((arrA & arrB).sum())/(arrA.sum()+arrB.sum())

def compute_metric_specificity(imFixed, imMoving):
    arrFixed = sitk.GetArrayFromImage(imFixed).astype(bool)
    arrMoving = sitk.GetArrayFromImage(imMoving).astype(bool)

    arrInter = arrFixed & arrMoving

    TruePos = arrInter.sum()
    TrueNeg = (np.invert(arrFixed) & np.invert(arrMoving)).sum()
    FalsePos = arrMoving.sum()-TruePos

    return (1.0*TrueNeg)/(TrueNeg+FalsePos)

def compute_metric_sensitivity(imFixed, imMoving):
    arrFixed = sitk.GetArrayFromImage(imFixed).astype(bool)
    arrMoving = sitk.GetArrayFromImage(imMoving).astype(bool)

    arrInter = arrFixed & arrMoving

    TruePos = arrInter.sum()
    FalseNeg = arrFixed.sum()-TruePos

    return (1.0*TruePos)/(TruePos+FalseNeg)

def compute_metric_masd(imFixed, imMoving):
    meanSDList = []
    numPoints = []
    for (imA, imB) in ((imFixed, imMoving), (imMoving, imFixed)):

        labelIntensityStat = sitk.LabelIntensityStatisticsImageFilter()
        referenceDistanceMap = sitk.Abs(sitk.SignedMaurerDistanceMap(imA, squaredDistance=False, useImageSpacing=True))
        movingLabelContour = sitk.LabelContour(imB)
        labelIntensityStat.Execute(movingLabelContour,referenceDistanceMap)

        meanSDList.append(labelIntensityStat.GetMean(1))
        numPoints.append(labelIntensityStat.GetNumberOfPixels(1))

    meanSurfDist = np.dot(meanSDList, numPoints)/np.sum(numPoints)
    return meanSurfDist

def compute_metric_hd(imFixed, imMoving, verbose=False):
    hausdorffDistance = sitk.HausdorffDistanceImageFilter()
    hausdorffDistance.Execute(imFixed, imMoving)
    HD = hausdorffDistance.GetHausdorffDistance()

    return HD