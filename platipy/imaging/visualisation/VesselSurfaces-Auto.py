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
organList = ['ASCENDINGAORTA','DESCENDINGAORTA','LEFTATRIUM','LEFTVENTRICLE','RIGHTATRIUM','RIGHTVENTRICLE']

for id in idList:
    print('Processing {0}'.format(id))

    mlab.figure(size = (1024,1024), bgcolor = (1,1,1))

    for index, organ in enumerate(organList):
        im = sitk.ReadImage("../Processing/AustralianAtlas/LabelFusion/LeaveOut{0}/Case_{0}_{1}_optimal.nii.gz".format(id, organ))
        arr = sitk.GetArrayFromImage(im)[::-1]

        m = mlab.contour3d(arr, contours=[1], color=cFunc(index, len(organList), plt.cm.YlOrRd), transparent=True, opacity=1)
        m.actor.actor.scale = (3,1,1)

    im = sitk.ReadImage("../Processing/AustralianAtlas/LabelFusion/LeaveOut{0}/Case_{0}_WHOLEHEART_optimal.nii.gz".format(id, organ))
    arr = sitk.GetArrayFromImage(im)[::-1]
    COM = np.mean(np.where(arr), axis=1)*(3,1,1)

    m = mlab.contour3d(arr, contours=[1], color=(0,0,0), transparent=True, opacity=0.2)
    m.actor.actor.scale = (3,1,1)

    autoDir = '../Processing/ABASAtlas/Registration/Demons/LeaveOut{0}/Structures/'.format(id)
    autoStructNames = [autoDir+i for i in os.listdir(autoDir) if 'LAD' in i]
    for index, autoStructName in enumerate(autoStructNames):
        im = sitk.ReadImage(autoStructName)
        arr = sitk.GetArrayFromImage(im)[::-1]
        arr = 1.0*(arr>0.3)

        m = mlab.contour3d(arr, contours=[1], color=cFunc(index, len(autoStructNames), plt.cm.Greens), transparent=True, opacity=1)
        m.actor.actor.scale = (3,1,1)

    # autoSplineName = "../Processing/ABASAtlas/LabelFusion/LeaveOut{0}/Case_{0}_LAD_SPLINE.nii.gz".format(id)
    # im = sitk.ReadImage(autoSplineName)
    # arr = sitk.GetArrayFromImage(im)[::-1]
    #
    # m = mlab.contour3d(arr, contours=[1], color=plt.cm.Greens(200)[:3], transparent=True, opacity=1)
    # m.actor.actor.scale = (3,1,1)


    mlab.view(-110, 41, 317, COM, -80)

    mlab.savefig('VesselSurfaces/AutoOnly/Case_{0}.png'.format(id))
    mlab.close()
