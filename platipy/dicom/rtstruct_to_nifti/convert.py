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
import click
import pydicom
import numpy as np
import SimpleITK as sitk
from skimage.draw import polygon


def readDICOMImage(filename):
    s_img_list = sitk.ImageSeriesReader().GetGDCMSeriesFileNames(filename)
    return sitk.ReadImage(s_img_list)


def readDICOMStructFile(filename):
    dicomStructFile = pydicom.read_file(filename, force=True)
    return dicomStructFile


def fixMissingData(SS):
    contourData = np.array(SS)
    if contourData.any() == "":
        print("Missing values detected.")
        missingVals = np.where(contourData == "")[0]
        if missingVals.shape[0] > 1:
            print("More than one value missing, fixing this isn't implemented yet...")
        else:
            print("Only one value missing.")
            missingIndex = missingVals[0]
            missingAxis = missingIndex % 3
            if missingAxis == 0:
                print("Missing value in x axis: interpolating.")
                if missingIndex > len(contourData) - 3:
                    lowerVal = contourData[missingIndex - 3]
                    upperVal = contourData[0]
                elif missingIndex == 0:
                    lowerVal = contourData[-3]
                    upperVal = contourData[3]
                else:
                    lowerVal = contourData[missingIndex - 3]
                    upperVal = contourData[missingIndex + 3]
                contourData[missingIndex] = 0.5 * (lowerVal + upperVal)
            elif missingAxis == 1:
                print("Missing value in y axis: interpolating.")
                if missingIndex > len(contourData) - 2:
                    lowerVal = contourData[missingIndex - 3]
                    upperVal = contourData[1]
                elif missingIndex == 0:
                    lowerVal = contourData[-2]
                    upperVal = contourData[4]
                else:
                    lowerVal = contourData[missingIndex - 3]
                    upperVal = contourData[missingIndex + 3]
                contourData[missingIndex] = 0.5 * (lowerVal + upperVal)
            else:
                print("Missing value in z axis: taking slice value")
                temp = contourData[2::3].tolist()
                temp.remove("")
                contourData[missingIndex] = np.min(np.array(temp, dtype=np.double))
    return contourData


def transformPointSetFromDICOMStruct(
    DICOMImage, DICOMStruct, writeImage, imageOutputName, spacingOverride
):

    if spacingOverride:
        currentSpacing = list(DICOMImage.GetSpacing())
        newSpacing = tuple(
            [
                currentSpacing[k] if spacingOverride[k] == 0 else spacingOverride[k]
                for k in range(3)
            ]
        )
        DICOMImage.SetSpacing(newSpacing)

    transformPhysicalPointToIndex = DICOMImage.TransformPhysicalPointToIndex

    structPointSequence = DICOMStruct.ROIContourSequence
    structNameSequence = [
        "_".join(i.ROIName.split()) for i in DICOMStruct.StructureSetROISequence
    ]

    structList = []
    finalStructNameSequence = []

    for structIndex, structName in enumerate(structNameSequence):
        imageBlank = np.zeros(DICOMImage.GetSize()[::-1], dtype=np.uint8)
        print("Converting structure {0} with name: {1}".format(structIndex, structName))

        if not hasattr(structPointSequence[structIndex], "ContourSequence"):
            print("No contour sequence found for this structure, skipping.")
            continue

        if len(structPointSequence[structIndex].ContourSequence) == 0:
            print("Contour sequence empty for this structure, skipping.")
            continue

        if len(structPointSequence[structIndex].ContourSequence) == 0:
            print("Contour sequence empty for this structure, skipping.")
            continue

        if (
            not structPointSequence[structIndex].ContourSequence[0].ContourGeometricType
            == "CLOSED_PLANAR"
        ):
            print("This is not a closed planar structure, skipping.")
            continue

        for sl in range(len(structPointSequence[structIndex].ContourSequence)):

            contourData = fixMissingData(
                structPointSequence[structIndex].ContourSequence[sl].ContourData
            )

            structSliceContourData = np.array(contourData, dtype=np.double)
            vertexArr_physical = structSliceContourData.reshape(
                structSliceContourData.shape[0] // 3, 3
            )

            pointArr = np.array(
                [transformPhysicalPointToIndex(i) for i in vertexArr_physical]
            ).T

            [xVertexArr_image, yVertexArr_image] = pointArr[[0, 1]]
            zIndex = pointArr[2][0]
            if np.any(pointArr[2] != zIndex):
                print("Error: axial slice index varies in contour. Quitting now.")
                print("Structure:   {0}".format(structName))
                print("Slice index: {0}".format(zIndex))
                quit()

            if zIndex >= DICOMImage.GetSize()[2]:
                print("Warning: Slice index greater than image size. Skipping slice.")
                print("Structure:   {0}".format(structName))
                print("Slice index: {0}".format(zIndex))
                continue

            # sliceArr = np.zeros(DICOMImage.GetSize()[:2], dtype=np.uint8)
            # PC: Correct this for non-square slices
            sliceArr = np.zeros(imageBlank.shape[-2:], dtype=np.uint8)

            filledIndicesX, filledIndicesY = polygon(
                xVertexArr_image, yVertexArr_image, shape=sliceArr.shape
            )
            sliceArr[filledIndicesY, filledIndicesX] = 1

            imageBlank[zIndex] += sliceArr

        structImage = sitk.GetImageFromArray(1 * (imageBlank > 0))
        structImage.CopyInformation(DICOMImage)
        structList.append(sitk.Cast(structImage, sitk.sitkUInt8))
        finalStructNameSequence.append(structName)

    if writeImage:
        print("Saving DICOM image.")
        sitk.WriteImage(DICOMImage, imageOutputName)

    return structList, finalStructNameSequence


def convert_rtstruct(
    dcm_img,
    dcm_rt_file,
    prefix="Struct_",
    output_dir=".",
    output_img=None,
    spacing=None,
):

    print("Converting RTStruct: {0}".format(dcm_rt_file))
    print("Using image series: {0}".format(dcm_img))
    print("Output file prefix: {0}".format(prefix))
    print("Output directory: {0}".format(output_dir))

    prefix = prefix + "{0}"

    DICOMImageFile = readDICOMImage(dcm_img)
    DICOMStructFile = readDICOMStructFile(dcm_rt_file)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if output_img:
        if output_img.endswith(".nii.gz"):
            imageOutputName = output_img
        else:
            imageOutputName = output_img + ".nii.gz"
        imageOutputName = os.path.join(output_dir, imageOutputName)
        writeImage = True

        print("Image series to be converted to: {0}".format(imageOutputName))
    else:
        imageOutputName = None
        writeImage = False

    if spacing:

        if type(spacing) == str:
            spacing = [float(i) for i in spacing.split(",")]

        print("Overriding image spacing with: {0}".format(spacing))

    structList, structNameSequence = transformPointSetFromDICOMStruct(
        DICOMImageFile, DICOMStructFile, writeImage, imageOutputName, spacing
    )
    print("Converted all structures. Writing output.")
    for structIndex, structImage in enumerate(structList):
        outName = "{0}.nii.gz".format(prefix.format(structNameSequence[structIndex]))
        outName = os.path.join(output_dir, outName)
        print("Writing file to: {0}".format(outName))
        sitk.WriteImage(structImage, outName)

    print("Finished")
