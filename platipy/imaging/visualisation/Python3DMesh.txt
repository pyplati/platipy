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


for case_id in ['101','102','103']:
    print('Processing {0}'.format(case_id))

    mlab.figure(size = (512,512), bgcolor = (1,1,1))

    # Add structures
    for index, structure in enumerate(['HEART','LUNG_L','LUNG_R','ESOPHAGUS','SPINALCORD']):

        filename = f"/home/robbie/Work/3_ResearchProjects/CardiacAtlasSets/NSCLC-Radiomics/NIFTI_CONVERTED/Test-S1-{case_id}/Structures/Test-S1-{case_id}_{structure}.nii.gz"

        im = sitk.ReadImage(filename)
        arr = sitk.GetArrayFromImage(im)[::-1]
        m = mlab.contour3d(arr, contours=[1], color=cFunc(index, 5, plt.cm.magma), transparent=True, opacity=0.5)

        im_spacing = im.GetSpacing()[::-1]
        m.actor.actor.scale = im_spacing

    COM = np.mean(np.where(arr), axis=1)*im_spacing
    mlab.view(-110, 41, 850, COM, -80)

    mlab.savefig(f'./Case_{case_id}.png')
    mlab.close()
