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

import numpy as np
import SimpleITK as sitk
import os

import matplotlib.pyplot as plt
from mayavi import mlab

def vectorisedTransformIndexToPhysicalPoint(image, pointArr, correct=True):
    if correct:
        spacing = image.GetSpacing()[::-1]
        origin = image.GetOrigin()[::-1]
    else:
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
    return pointArr*spacing + origin

def cFunc(i, m, cmap=plt.cm.Spectral):
    index = int(255.0*i/m)
    return cmap(index)[:3]

idList = [i[5:] for i in os.listdir('../Data') if i[:4]=='Case']

# change atlas name here
atlas = 'AustralianAtlas' #ABASAtlas AustralianAtlas
structDict = {'AustralianAtlas':['WHOLEHEART','LANTDESCARTERY_SPLINE']}

for id in idList:
    print('Processing {0}'.format(id))

    mlab.figure(size = (1024,1024), bgcolor = (1,1,1))

    # Add automatic mesh - heart
    im = sitk.ReadImage("../Processing/{0}/LabelFusion/LeaveOut{1}/Case_{1}_{2}_BINARY_CORR.nii.gz".format(atlas, id, structDict[atlas][0]))
    arr = sitk.GetArrayFromImage(im)[::-1]
    m = mlab.contour3d(arr, contours=[1], color=(0,0,1), transparent=True, opacity=0.15)
    m.actor.actor.scale = (3,1,1)
    COM = np.mean(np.where(arr), axis=1)*(3,1,1)

    # Add manual mesh - heart (vote)
    im = sitk.ReadImage("../Data/Case_{0}/Structures/Case_{0}_COR_MANUAL_VOTE_CROP.nii.gz".format(id))
    arr = sitk.GetArrayFromImage(im)[::-1]
    m = mlab.contour3d(arr, contours=[1], color=(1,0,0), transparent=True, opacity=0.15)
    m.actor.actor.scale = (3,1,1)

    # Add manual meshes - ladca
    colors = plt.cm.hot(np.linspace(0.2,0.8, 9))
    for ii in range(9):
        im = sitk.ReadImage("../Data/Case_{0}/Structures/Case_{0}_LAD_{1}_CROP.nii.gz".format(id, ii))
        arr = sitk.GetArrayFromImage(im)[::-1]
        m = mlab.contour3d(arr, contours=[1], color=tuple(colors[:,0:3][ii]), transparent=True, opacity=0.5)
        m.actor.actor.scale = (3,1,1)

    # Add automatic splined vessel
    im = sitk.ReadImage("../Processing/{0}/LabelFusion/LeaveOut{1}/Case_{1}_{2}_CORR.nii.gz".format(atlas, id, structDict[atlas][1]))
    arr = sitk.GetArrayFromImage(im)[::-1]
    m = mlab.contour3d(arr, contours=[1], color=(0,0,1), transparent=True, opacity=0.5)
    m.actor.actor.scale = (3,1,1)

    mlab.view(-110, 41, 317, COM, -80)

    mlab.savefig('PythonMeshImages/{0}/Case_{1}.png'.format(atlas, id))
    mlab.close()
