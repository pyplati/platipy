# Copyright 2020 CSIRO

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import yaml
import numpy
import itk


def gen_contours(mask_file):

    image = itk.Image[itk.SS, 3]
    image2 = itk.Image[itk.SS, 2]

    reader = itk.ImageFileReader[image].New()
    reader.SetFileName(mask_file)
    reader.Update()

    region = itk.ImageRegion[3]()  # reader.GetOutput().GetLargestPossibleRegion()
    r2 = reader.GetOutput().GetLargestPossibleRegion()
    sz2 = r2.GetSize()
    indx2 = r2.GetIndex()

    sz = region.GetSize()
    sz[2] = 0
    sz[1] = sz2[1]
    sz[0] = sz2[0]
    region.SetSize(sz)

    indx = region.GetIndex()
    indx[0] = 0
    indx[1] = 0

    slice_ext = itk.ExtractImageFilter[image, image2].New()
    slice_ext.SetInput(reader.GetOutput())
    slice_ext.SetDirectionCollapseToIdentity()

    eps = sys.float_info.epsilon
    offset = reader.GetOutput().GetOrigin()
    space = reader.GetOutput().GetSpacing()
    conts = {"Contours": []}
    for zz in range(0, sz2[2]):

        indx[2] = zz
        region.SetIndex(indx)
        slice_ext.SetExtractionRegion(region)
        slice_ext.Update()

        contour = itk.ContourExtractor2DImageFilter[image2].New()
        contour.SetInput(slice_ext.GetOutput())
        contour.SetContourValue(100.0)
        contour.ReverseContourOrientationOn()
        contour.SetContourValue(0.5)
        contour.Update()

        aa = contour.GetOutput()

        if aa == None:
            continue
        for cn in range(0, contour.GetNumberOfOutputs()):
            cont = contour.GetOutput(cn).GetVertexList()
            numVert = cont.Size()
            first = cont.ElementAt(0)
            last = cont.ElementAt(numVert - 1)

            contType = None
            data = []

            if abs(first[0] - last[0]) < eps and abs(first[1] - last[1]) < eps:

                contType = "CLOSED_PLANAR"
                numVert = numVert - 1
            else:
                contType = "OPEN_PLANAR"

            for vv in range(0, numVert):
                vert = cont.ElementAt(vv)
                data.append(
                    [
                        vert[0] * space[0] + offset[0],
                        vert[1] * space[1] + offset[1],
                        zz * space[2] + offset[2],
                    ]
                )
            aa = {"SliceNumber": zz, "GeometricType": contType, "Data": data}
            conts["Contours"].append(aa)

    return conts
