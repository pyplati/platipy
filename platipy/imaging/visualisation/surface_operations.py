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
Script name: Surface Operations
Author:      Robert Finnegan
Date:        21/03/2019
Description
-----------
Operations for evaluating surfaces of segmented volumes
"""

import SimpleITK as sitk

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable, AxesGrid
from mpl_toolkits.basemap import Basemap

from scipy.ndimage import measurements
from scipy.interpolate import griddata
from scipy.interpolate import RectSphereBivariateSpline

import os, sys, functools, math, random, datetime

##########################################################################################################################

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

def vectorisedTransformPhysicalPointToIndex(image, pointArr):
    """
    Transforms a set of points from real-space to array indices
    """
    spacing = image.GetSpacing()
    origin = image.GetOrigin()

    return (pointArr-origin)/spacing


def evaluateDistanceOnSurface(referenceVolume, testVolume, debug=False, absDistance=True, referenceAsDistanceMap=False):
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

    if debug:
        print("Distance measures: {0:.2f},{1:.2f},{2:.2f}".format(distanceArray.min(),(distanceArray[distanceArray!=0.0]).mean(),distanceArray.max()))

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
    phi =  np.arctan2(ptsDiff.T[2],-1.0*ptsDiff.T[1])

    # Convert to spherical polar coordinates - base at 0Lat0Lon
    # rho = np.sqrt((ptsDiff**2).sum(axis=1))
    # theta = np.pi/2.-np.arccos(ptsDiff.T[2]/rho)
    # phi = np.arctan2(ptsDiff.T[1],ptsDiff.T[0])

    # Extract values
    values = distanceArray[testSurfaceLocations]
    if debug:
        print('    {0}'.format(values.shape))

    return theta, phi, values


def evaluateDistanceToSurface(testVolume, debug=False):
    """
    Evaluates the distance from the origin of a volume to the surface
    Input: referenceVolume: binary volume SimpleITK image, or alternatively a distance map
           testVolume: binary volume SimpleITK image
    Output: theta, phi, values
    """
    centre = np.array(measurements.center_of_mass(sitk.GetArrayFromImage(testVolume)))[::-1]
    centre_int = tuple(int(i) for i in centre)
    blank_volume = (0*testVolume)
    blank_volume[centre_int] = 1

    referenceDistanceMap = sitk.Abs(sitk.SignedMaurerDistanceMap(blank_volume, squaredDistance=False, useImageSpacing=True))

    testSurface = sitk.LabelContour(testVolume)
    distanceImage = sitk.Multiply(referenceDistanceMap, sitk.Cast(testSurface, sitk.sitkFloat32))
    distanceArray = sitk.GetArrayFromImage(distanceImage)

    if debug:
        print("Distance measures: {0:.2f},{1:.2f},{2:.2f}".format(distanceArray.min(),(distanceArray[distanceArray!=0.0]).mean(),distanceArray.max()))

    # Calculate centre of mass in real coordinates
    testSurfaceArray = sitk.GetArrayFromImage(testSurface)
    testSurfaceLocations = np.where(testSurfaceArray==1)
    testSurfaceLocationsArray = np.array(testSurfaceLocations)

    # Calculate each point on the surface in real coordinates
    pts = testSurfaceLocationsArray.T
    ptsReal = vectorisedTransformIndexToPhysicalPoint(testSurface, pts)
    centreReal = vectorisedTransformIndexToPhysicalPoint(testSurface, centre[::-1])

    ptsDiff = ptsReal - centreReal

    # Convert to spherical polar coordinates - base at north pole
    rho = np.sqrt((ptsDiff*ptsDiff).sum(axis=1))
    theta = np.pi/2.-np.arccos(ptsDiff.T[0]/rho)
    phi =  np.arctan2(ptsDiff.T[2],-1.0*ptsDiff.T[1])

    # Convert to spherical polar coordinates - base at 0Lat0Lon
    # rho = np.sqrt((ptsDiff**2).sum(axis=1))
    # theta = np.pi/2.-np.arccos(ptsDiff.T[2]/rho)
    # phi = np.arctan2(ptsDiff.T[1],ptsDiff.T[0])

    # Extract values
    values = distanceArray[testSurfaceLocations]
    if debug:
        print('    {0}'.format(values.shape))

    return theta, phi, values


def regridSphericalData(theta, phi, values, resolution, plotFig=False, saveFig=False, figName='Figure_{0}.png'.format(datetime.datetime.now()), vmin=-20, vmax=20):
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

    if plotFig:
        fig = plt.figure(figsize=(12,4))
        ax = fig.add_subplot(121)
        im = ax.scatter(phi, theta, c=values, s=2, cmap=plt.cm.Spectral_r, clim=(0,1), vmin=vmin, vmax=vmax)
        ax.set_ylabel(r'Elevation $\theta$')
        ax.set_xlabel(r'Azimuth $\phi$')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
        cax.xaxis.tick_top()
        ax = fig.add_subplot(122)
        im = ax.scatter(pLong*180/np.pi, pLat*180/np.pi, c=gridValues, cmap=plt.cm.Spectral_r, clim=(0,1), vmin=vmin, vmax=vmax)
        ax.set_ylabel('Latitude (deg)')
        ax.set_xlabel('Longitude (deg)')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
        cax.xaxis.tick_top()
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)
        if saveFig:
            fig.savefig(figName, dpi=300)
        else:
            return pLat, pLong, gridValues, fig

    return pLat, pLong, gridValues

def plotSufaceEvaluationEckIV(pLat, pLong, gridValues, saveFig=False, figName='Figure_{0}.png'.format(datetime.datetime.now()), title='', vmin=-20, vmax=20, annotate=True, scaleName='Surface Distance [mm]', userFontSize=14):
    """
    Plots a gridded value on a spherical projection (Eckert IV)
    Input: pLat, pLong, gridValues
    Options: figName, saveFig
    Output: fig
    """
    fig = plt.figure(figsize=(11,5))
    ax = fig.add_subplot(1,1,1)
    m = Basemap(projection='eck4', lon_0=0, ax=ax)
    uncPlot = m.pcolormesh(pLong*180/np.pi, pLat*180/np.pi, gridValues, cmap=plt.cm.Spectral_r, vmin=vmin, vmax=vmax, clim=(0,1), latlon=True)
    ax.grid()
    fig.suptitle(title)

    parallels = [-80,-60,-40,-20,0,20,40,60,80]
    m.drawparallels(parallels,labels=[True,True,False,False], labelstyle='+/-', fontsize=userFontSize)
    meridians = [-180, -120, -60,0,60,120,180]
    m.drawmeridians(meridians,labels=[False,False,False,True], labelstyle='+/-', fontsize=userFontSize)

    if annotate:
        plt.annotate('Anterior', xy=m(0,0),  xycoords='data', verticalalignment='center', horizontalalignment='center', fontsize=20)
        plt.annotate('Left', xy=m(90,0),  xycoords='data', verticalalignment='center', horizontalalignment='center', fontsize=20)
        plt.annotate('Right', xy=m(-90,0),  xycoords='data', verticalalignment='center', horizontalalignment='center', fontsize=20)
        plt.annotate('Cranial', xy=m(0,80),  xycoords='data', verticalalignment='top', horizontalalignment='center', fontsize=20)
        plt.annotate('Caudal', xy=m(0,-80),  xycoords='data', verticalalignment='bottom', horizontalalignment='center', fontsize=20)
        plt.annotate('Base', xy=m(-80,50),  xycoords='data', verticalalignment='center', horizontalalignment='center', fontsize=20)
        plt.annotate('Apex', xy=m(80,-50),  xycoords='data', verticalalignment='center', horizontalalignment='center', fontsize=20)
        plt.annotate('Post', xy=m(-175,0),  xycoords='data', verticalalignment='center', horizontalalignment='left', fontsize=20)
        plt.annotate('Post', xy=m(175,0),  xycoords='data', verticalalignment='center', horizontalalignment='right', fontsize=20)

    ax_blank = fig.add_subplot(1,2,2)
    im = ax_blank.scatter(pLong*180/np.pi, pLat*180/np.pi, c=gridValues, s=2, cmap=uncPlot.cmap, alpha=0, vmin=vmin, vmax=vmax, clim=(0,1))
    ax_blank.axis('off')
    divider = make_axes_locatable(ax_blank)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, orientation='vertical')
    cax.set_ylabel(scaleName, fontsize=userFontSize)
    cbar.solids.set_alpha(1)
    cax.xaxis.tick_top()
    # labels = cax.get_yticklabels()
    # print([i for i in labels])
    # cax.set_yticklabels(labels, fontsize=userFontSize)

    # fig.subplots_adjust(left=0, bottom=0.06, right=0.92, top=0.97)
    fig.tight_layout()
    fig.axes[2].set_yticklabels(fig.axes[2].get_yticklabels(), fontsize=userFontSize)


    if saveFig:
        fig.savefig("{0}".format(figName), dpi=300)

    return fig

def computeSurfaceDistance(labelImageReference, labelImageMeasure):
    """
    Calculate the surface-to-surface distance from a reference label to test label
    Returns an array of surface distances
     - This can be used to calculate mean surface distance, Haussdorf distance, etc.
    """
    labelImageReference = sitk.Cast(labelImageReference, sitk.sitkUInt8)
    labelImageMeasure = sitk.Cast(labelImageMeasure, sitk.sitkUInt8)

    referenceDistanceMap = sitk.GetArrayFromImage(sitk.Abs(sitk.SignedMaurerDistanceMap(labelImageReference, squaredDistance=False, useImageSpacing=True)))
    labelImageMeasureContour = sitk.GetArrayFromImage(sitk.LabelContour(labelImageMeasure))

    surfaceDistanceArr = referenceDistanceMap[np.where(labelImageMeasureContour==1)]
    return surfaceDistanceArr

def main(arguments):
    return True

if __name__ == '__main__':
    main()
