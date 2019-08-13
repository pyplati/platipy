#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 16:39:27 2018

Bronchus segmentation.  The superior extent of bronchus is around 4cm SUP from Carina (From first slice where two airways become visible).
In this code I'm using 2cm as it's easier to detect where the airways are getting wider

Areas to improve: parameters could be improved (eg. the median filter, carina detection, etc).  The GenLungMask is based on old ITK code from a masters student.  I think we can replace this function
by checking the top (sup) slice for an airhole and then connected thresholding.

Code fails on two Liverpool cases:  13 (lungs appear in the sup slice) and 36 (the mask failed to generate - need to look at this)

@author: Jason Dowling (CSIRO)
"""

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
import os, re
from multiprocessing import Process, Pool
import signal, time

debug=True
fast_mode=True
iExtendFromCarina=20 #ie. 2cm


def FastMask(img, startZ, endZ):
    """Fast masking for area of a 3D volume .

    SimpleITK lacks iterators so voxels need to be set with SetPixel which is horribly slow.
    This code uses numpy arrays to reduce time for one volume from around one minute to about 0.5s

    Args:
        image: Input 3D binary image, start slice for masking (value=0), end slice for masking

    Returns:
        Masked image.  This may be in float, so it might need casting back to original pixel type.
    """
    npImg = sitk.GetArrayFromImage(img).astype(float)
    npImg[startZ:endZ, :, :]=0
    NewImg = sitk.GetImageFromArray(npImg)
    NewImg.SetSpacing(img.GetSpacing())
    NewImg.SetOrigin(img.GetOrigin())
    NewImg.SetDirection(img.GetDirection())
    return NewImg

def GetDistance(a_mask,b_mask,dest):
    """Get the nearest distance between two masks.

    Args:
        a_mask: The first mask
        b_mask: The second mask
        dest: Working directory to output intermediate files

    Returns:
        None
    """

    #load lung mask from previous step
    try:
        aMask=sitk.ReadImage(a_mask)
    except:
        print('File read failed ' + a_mask)
        raise

    try:
        bMask=sitk.ReadImage(b_mask)
    except:
        print('File read failed ' + b_mask)
        raise

    # 1. Generate distance map from surface of Contour1
    smdm=sitk.SignedMaurerDistanceMapImageFilter()
    smdm.UseImageSpacingOn();
    smdm.InsideIsPositiveOff()
    smdm.SquaredDistanceOff()
    result=smdm.Execute(aMask)

    # Subtract 1 from the mask, making 0's -> -1 and 1's -> 0.
    bMask=bMask-1

    # Multiply this result by -10000, making -1's -> 10000.
    # It is assumed that 10000mm > than the max distance
    bMask=bMask*-10000

    result=result+sitk.Cast ( bMask, result.GetPixelIDValue() )

    sif=sitk.StatisticsImageFilter()
    sif.Execute(result)
    dist = sif.GetMinimum()

    return dist

def GenLungMask(img,dest):
    """Generate initial airway mask (includes lungs).

    Args:
        image: path for CT image and mask destination.  The awful output filenames match
        the output from Antoine's C++ Masters code.

    Returns:
        None
    """

    print('Generating Lung Mask...')

    MidImg = sitk.GetArrayFromImage(img).astype(float)
    centre = ndimage.measurements.center_of_mass(MidImg)

    #get air mask external to body
    connT=sitk.ConnectedThresholdImageFilter()
    connT.SetSeed([0,0,0])
    connT.SetLower(-10000)
    connT.SetUpper(-300)
    result=connT.Execute(img)
    writer=sitk.ImageFileWriter()
    writer.SetFileName(dest+'/imageB.nii.gz')
    writer.Execute(result)

    #get body mask (remove couch) Seed this from centre of image.
    CCIF=sitk.ConfidenceConnectedImageFilter()
    CCIF.SetSeed([int(centre[1]),int(centre[2]),int(centre[0])])
    resultF=CCIF.Execute(result)
    writer=sitk.ImageFileWriter()
    writer.SetFileName(dest+'/imageF.nii.gz')
    writer.Execute(resultF)

    #get non-air voxels fom within body
    connT=sitk.ConnectedThresholdImageFilter()
    connT.SetSeed([251,240,65])
    connT.SetLower(-300)
    connT.SetUpper(10000)
    result=connT.Execute(img)
    writer=sitk.ImageFileWriter()
    writer.SetFileName(dest+'/imageG.nii.gz')
    writer.Execute(result)

    #problem might be intestinal air below lungs - need to remove this?
    binInvert=sitk.InvertIntensityImageFilter()
    resultH=binInvert.Execute(result)
    writer=sitk.ImageFileWriter()
    writer.SetFileName(dest+'/imageH.nii.gz')
    writer.Execute(resultH)

    #not sure this is necessary
    binDilate=sitk.BinaryDilateImageFilter()
    binDilate.SetKernelRadius(3)
    resultJ=binDilate.Execute(resultH)
    writer=sitk.ImageFileWriter()
    writer.SetFileName(dest+'/imageJ.nii.gz')
    writer.Execute(resultJ)

    #image j is ok (might need dilation)
    #image f is fine as well (just a body contour)
    AND=sitk.AddImageFilter()
    resultFinal=AND.Execute(resultJ,resultF)
    writer=sitk.ImageFileWriter()
    writer.SetFileName(dest+'/imageK.nii.gz')
    writer.Execute(resultFinal)

    resultFinal=binInvert.Execute(resultFinal)
    writer=sitk.ImageFileWriter()
    writer.SetFileName(dest+'/imageK2H.nii.gz')
    writer.Execute(resultFinal)

    #Just change 255 to 1 for binary label
    BT=sitk.BinaryThresholdImageFilter()
    BT.SetLowerThreshold(255)
    BT.SetUpperThreshold(255)
    resultFinal=BT.Execute(resultFinal)
    writer=sitk.ImageFileWriter()
    writer.SetFileName(dest+'/imageFinal.nii.gz')
    writer.Execute(resultFinal)

    print('Generating Lung Mask... Done')

    return resultFinal

def GenAirwayMask(dest,img,lungMask):
    """Generate final bronchus segmentation .

    Args:
        image: path for CT image and mask destination

    Returns:
        None
    """

    iLungMaskHUValues = [-750, -775, -800, -825, -850, -900, -700, -950, -650]
    iDistanceFromSuPSliceValues = [3, 10, 20]
    expectedPhysicalSizeRange = [22000, 75000]

    zSize=img.GetDepth()

    #Identify airway start on superior slice
    LABELSHAPE=sitk.LabelIntensityStatisticsImageFilter()
    connected_component=sitk.ConnectedComponentImageFilter()
    connT=sitk.ConnectedThresholdImageFilter()

    print ('-------------------------------' )

    LABELSHAPE.Execute(lungMask, img)
    for l in LABELSHAPE.GetLabels():
        print ('summary of lung intensities.  Mean: ' + str(LABELSHAPE.GetMean(l)) + ' sd: ' + str(LABELSHAPE.GetStandardDeviation(l)) + ' median: ' + str(LABELSHAPE.GetMedian(l))    )
        iLungMaskMean=int(LABELSHAPE.GetMean(l))

    #iDistanceFromSuPSlice=3
    iLoopCount=0
    bProcessedCorrectly=False
    #iLungMaskHU=-800
    #bMedianSmooth=0
    best_result = None
    best_result_sim = 100000000
    best_lung_mask_hu = 0

    for k in range(2):

        if bProcessedCorrectly and fast_mode:
            break

        if k == 1:
            #Try smoothing the lung mask - effects all tests below
            medianfilter=sitk.MedianImageFilter()
            medianfilter.SetRadius(1)
            lungMask=medianfilter.Execute(lungMask)

        for j in range(len(iDistanceFromSuPSliceValues)):

            if bProcessedCorrectly and fast_mode:
                break

            iDistanceFromSuPSlice=iDistanceFromSuPSliceValues[j]

            labelSlice= lungMask[:,:,zSize-iDistanceFromSuPSlice-10:zSize-iDistanceFromSuPSlice]   #works for both cases 22 and 17
            imgSlice= img[:,:,zSize-iDistanceFromSuPSlice-10:zSize-iDistanceFromSuPSlice]

            connected = connected_component.Execute(labelSlice)
            nda_connected = sitk.GetArrayFromImage(connected)
            # In case there are multiple air regions at the top, select the region with
            # the max elongation as the airway seed
            # Also check that the region has a minimum physical size, to filter out other
            # unexpected regions
            max_elong = 0
            airwayOpen = [0,0,0]
            LABELSHAPE.Execute(connected, imgSlice)
            for l in LABELSHAPE.GetLabels():
                #print('Elong: ' + str(LABELSHAPE.GetElongation(l)) + ' Volume: ' + str(LABELSHAPE.GetPhysicalSize(l)))
                if LABELSHAPE.GetElongation(l) > max_elong and LABELSHAPE.GetPhysicalSize(l) > 2000:
                    centre=img.TransformPhysicalPointToIndex(LABELSHAPE.GetCentroid(l))
                    max_elong = LABELSHAPE.GetElongation(l)
                    airwayOpen=[int(centre[0]),int(centre[1]),int(centre[2])]

            #just check the opening is at the right location
            centroidMaskVal=lungMask.GetPixel(airwayOpen[0],airwayOpen[1],airwayOpen[2])

            if (centroidMaskVal==0):
                print ('Error locating trachea centroid.  Usually because of additional air features on this slice')

                # No point in doing the airway segmentation here as airway opening wasn't detected
                continue

            print ('*Airway opening: ' + str(airwayOpen))
            print ('*Voxel HU at opening: ' + str(lungMask.GetPixel(airwayOpen[0],airwayOpen[1],airwayOpen[2])))


            for i in range(len(iLungMaskHUValues)):

                iLungMaskHU=iLungMaskHUValues[i]

                print ('--------------------------------------------' )
                print ('Extracting airways.  Iteration: ' + str(iLoopCount) )
                print ('*Lung Mask HU: ' + str(iLungMaskHU) )
                print ('*Slices from sup for airway opening: ' + str(iDistanceFromSuPSlice) )
                if (k==1):
                    print ('*Mask median smoothing on')
                iLoopCount=iLoopCount+1

                connT.SetSeed(airwayOpen)
                connT.SetLower(-2000)
                connT.SetUpper(iLungMaskHU)
                result=connT.Execute(img)

                writer=sitk.ImageFileWriter()
                writer.SetFileName(dest+'/airwaysMask.nii.gz')
                writer.Execute(result)

                #process in 2D - check label elongation and size.
                corinaSlice=0
                for idxSlice in range(zSize):
                    labelSlice = result[:,:,idxSlice]
                    imgSlice=img[:,:,idxSlice]
                    connected = connected_component.Execute(labelSlice)
                    nda_connected = sitk.GetArrayFromImage(connected)
                    num_regions = nda_connected.max()

                    # Make sure the airway has branched
                    if num_regions < 2:
                        continue

                    LABELSHAPE.Execute(labelSlice, imgSlice)
                    for l in LABELSHAPE.GetLabels():
                        if (LABELSHAPE.GetElongation(l) > 5 and LABELSHAPE.GetPhysicalSize(l) > 30):
                            corinaSlice=idxSlice

                #crop from corinaSlice + 20 slices (~4cm)
                if (corinaSlice==0):
                    print ('Failed to located carina.  Adjusting parameters ')

                else:
                    print (' Cropping from slice: ' + str(corinaSlice) + ' + 20 slices and dilating' )
                    result=FastMask(result,corinaSlice+iExtendFromCarina,zSize)
                    result = sitk.Cast ( result, lungMask.GetPixelIDValue() )

                    # Dilate and check if the output is in the expected range
                    binDilate=sitk.BinaryDilateImageFilter()
                    binDilate.SetKernelRadius(2)
                    result=binDilate.Execute(result)

                    #check size of label - if it's too large the lungs have been included..
                    LABELSHAPE.Execute(result, img)
                    for l in LABELSHAPE.GetLabels():
                        iAirwayMaskPhysicalSize=int(LABELSHAPE.GetPhysicalSize(l))
                        fRoundness=float(LABELSHAPE.GetRoundness(l))
                        fElongation=float(LABELSHAPE.GetElongation(l))

                    bThisProcessedCorrectly = False
                    if (iAirwayMaskPhysicalSize > expectedPhysicalSizeRange[1]):
                        print (' Airway Mask size failed (> '+str(expectedPhysicalSizeRange[1])+'): ' + str(iAirwayMaskPhysicalSize) )
                    elif (iAirwayMaskPhysicalSize < expectedPhysicalSizeRange[0]):
                        print (' Airway Mask size failed (< '+str(expectedPhysicalSizeRange[0])+'): ' + str(iAirwayMaskPhysicalSize) )
                    else:
                        print (' Airway Mask size passed: ' + str(iAirwayMaskPhysicalSize) )
                        bProcessedCorrectly = True
                        bThisProcessedCorrectly = True

                    print (' Roundness: ' + str(fRoundness) )
                    print (' Elongation: ' + str(fElongation) )
                    target_size = (expectedPhysicalSizeRange[1] + expectedPhysicalSizeRange[0]) / 2
                    size_sim = abs(iAirwayMaskPhysicalSize - target_size)

                    if fRoundness < best_result_sim and bThisProcessedCorrectly:
                        best_result_sim = fRoundness
                        best_result = result
                        best_lung_mask_hu = iLungMaskHU

    if not bProcessedCorrectly:
        print(' Unable to process correctly!!!')

    print('Selected Lung Mask HU: ' + str(best_lung_mask_hu))

    return best_result
