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
from skimage import measure

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
        im_structure = sitk.ReadImage(filename, sitk.sitkFloat32)

        filename = f"/home/robbie/Work/3_ResearchProjects/CardiacAtlasSets/NSCLC-Radiomics/NIFTI_CONVERTED/Test-S1-{case_id}/Images/Test-S1-{case_id}.nii.gz"
        im_ct = sitk.ReadImage(filename, sitk.sitkFloat32)

        arr_structure = sitk.GetArrayFromImage(im_structure)[::-1]
        arr_ct = sitk.GetArrayFromImage(im_ct)[::-1]

        verts, faces, normals, values = measure.marching_cubes_lewiner(arr_structure)
        vert_x, vert_y, vert_z = verts.T

        values_ct = arr_ct[vert_x.astype(int), vert_y.astype(int), vert_z.astype(int)]

        mesh = mlab.triangular_mesh(vert_x, vert_y, vert_z, faces, opacity=0.9, colormap='Greys', vmin=-250, vmax=300)

        mesh.mlab_source.scalars = values_ct
        mesh.actor.mapper.scalar_visibility = True
        mesh.actor.property.frontface_culling = True

        im_spacing = im_ct.GetSpacing()[::-1]
        mesh.actor.actor.scale = im_spacing

    COM = np.mean(np.where(arr_structure), axis=1)*im_spacing
    mlab.view(-110, 41, 850, COM, -80)

    mlab.savefig(f'./Case_{case_id}.png')
    mlab.close()
