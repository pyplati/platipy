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


import warnings

from loguru import logger

import numpy as np
import SimpleITK as sitk
import vtk

from vtk.util.numpy_support import vtk_to_numpy


def com_from_image_list(
    sitk_image_list,
    condition_type="count",
    condition_value=0,
    scan_direction="z",
):
    """
    Input: list of SimpleITK images
           minimum total slice area required for the tube to be inserted at that slice
           scan direction: x = sagittal, y=coronal, z=axial
    Output: mean centre of mass positions, with shape (NumSlices, 2)
    Note: positions are converted into image space by default
    """
    if scan_direction.lower() == "x":
        logger.debug("Scanning in sagittal direction")
        com_z_list = []
        com_y_list = []
        weight_list = []
        count_list = []

        reference_image = sitk_image_list[0]
        reference_array = sitk.GetArrayFromImage(reference_image)
        z, y = np.mgrid[0 : reference_array.shape[0] : 1, 0 : reference_array.shape[1] : 1]

        with np.errstate(divide="ignore", invalid="ignore"):
            for sitk_image in sitk_image_list:
                volume_array = sitk.GetArrayFromImage(sitk_image)
                com_z = 1.0 * (z[:, :, np.newaxis] * volume_array).sum(axis=(1, 0))
                com_y = 1.0 * (y[:, :, np.newaxis] * volume_array).sum(axis=(1, 0))

                weights = np.sum(volume_array, axis=(1, 0))
                weight_list.append(weights)

                count_list.append(np.any(volume_array, axis=(1, 0)))

                com_z /= 1.0 * weights
                com_y /= 1.0 * weights

                com_z_list.append(com_z)
                com_y_list.append(com_y)

        with warnings.catch_warnings():
            """
            It's fairly likely some slices have just np.NaN values
            It raises a warning but we can suppress it here
            """
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_com_z_list = np.nanmean(com_z_list, axis=0)
            mean_com_y_list = np.nanmean(com_y_list, axis=0)
            if condition_type.lower() == "area":
                mean_com = (
                    np.dstack((mean_com_z_list, mean_com_y_list))[0]
                    * np.array((np.sum(weight_list, axis=0) > (condition_value),) * 2).T
                )
            elif condition_type.lower() == "count":
                mean_com = (
                    np.dstack((mean_com_z_list, mean_com_y_list))[0]
                    * np.array((np.sum(count_list, axis=0) > (condition_value),) * 2).T
                )
            else:
                raise ValueError("Invalid condition type, please select from 'area' or 'count'.")

        point_array = []
        for index, com in enumerate(mean_com):
            if np.all(np.isfinite(com)):
                if np.all(com > 0):
                    point_array.append(
                        reference_image.TransformIndexToPhysicalPoint(
                            (index, int(com[1]), int(com[0]))
                        )
                    )

        return point_array

    elif scan_direction.lower() == "z":
        logger.debug("Scanning in axial direction")
        com_x_list = []
        com_y_list = []
        weight_list = []
        count_list = []

        reference_image = sitk_image_list[0]
        reference_array = sitk.GetArrayFromImage(reference_image)
        x, y = np.mgrid[0 : reference_array.shape[1] : 1, 0 : reference_array.shape[2] : 1]

        with np.errstate(divide="ignore", invalid="ignore"):
            for sitk_image in sitk_image_list:
                volume_array = sitk.GetArrayFromImage(sitk_image)
                com_x = 1.0 * (x * volume_array).sum(axis=(1, 2))
                com_y = 1.0 * (y * volume_array).sum(axis=(1, 2))

                weights = np.sum(volume_array, axis=(1, 2))
                weight_list.append(weights)

                count_list.append(np.any(volume_array, axis=(1, 2)))

                com_x /= 1.0 * weights
                com_y /= 1.0 * weights

                com_x_list.append(com_x)
                com_y_list.append(com_y)

        with warnings.catch_warnings():
            """
            It's fairly likely some slices have just np.NaN values - it raises a warning but we
            can suppress it here
            """
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_com_x_list = np.nanmean(com_x_list, axis=0)
            mean_com_y_list = np.nanmean(com_y_list, axis=0)
            if condition_type.lower() == "area":
                mean_com = (
                    np.dstack((mean_com_x_list, mean_com_y_list))[0]
                    * np.array((np.sum(weight_list, axis=0) > (condition_value),) * 2).T
                )
            elif condition_type.lower() == "count":
                mean_com = (
                    np.dstack((mean_com_x_list, mean_com_y_list))[0]
                    * np.array((np.sum(count_list, axis=0) > (condition_value),) * 2).T
                )
            else:
                print("Invalid condition type, please select from 'area' or 'count'.")
                quit()

        point_array = []
        for index, com in enumerate(mean_com):
            if np.all(np.isfinite(com)):
                if np.all(com > 0):
                    point_array.append(
                        reference_image.TransformIndexToPhysicalPoint(
                            (int(com[1]), int(com[0]), index)
                        )
                    )

        return point_array


def tube_from_com_list(com_list, radius):
    """
    Input: image-space positions along the tube centreline.
    Output: VTK tube
    Note: positions do not have to be continuous - the tube is interpolated in real space
    """
    points = vtk.vtkPoints()
    for i, pt in enumerate(com_list):
        points.InsertPoint(i, pt[0], pt[1], pt[2])

    # Fit a spline to the points
    logger.debug("Fitting spline")
    spline = vtk.vtkParametricSpline()
    spline.SetPoints(points)

    functionSource = vtk.vtkParametricFunctionSource()
    functionSource.SetParametricFunction(spline)
    functionSource.SetUResolution(10 * points.GetNumberOfPoints())
    functionSource.Update()

    # Generate the radius scalars
    tube_radius = vtk.vtkDoubleArray()
    n = functionSource.GetOutput().GetNumberOfPoints()
    tube_radius.SetNumberOfTuples(n)
    tube_radius.SetName("TubeRadius")
    for i in range(n):
        # We can set the radius based on the given propagated segmentations in that slice?
        # Typically segmentations are elliptical, this could be an issue so for now a constant
        # radius is used
        tube_radius.SetTuple1(i, radius)

    # Add the scalars to the polydata
    tubePolyData = vtk.vtkPolyData()
    tubePolyData = functionSource.GetOutput()
    tubePolyData.GetPointData().AddArray(tube_radius)
    tubePolyData.GetPointData().SetActiveScalars("TubeRadius")

    # Create the tubes
    tuber = vtk.vtkTubeFilter()
    tuber.SetInputData(tubePolyData)
    tuber.SetNumberOfSides(50)
    tuber.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
    tuber.Update()

    return tuber


def write_vtk_tube_to_file(tube, filename):
    """
    Input: VTK tube
    Output: exit success
    Note: format is XML VTP
    """
    print("Writing tube to polydata file (VTP)")
    poly_data_writer = vtk.vtkXMLPolyDataWriter()
    poly_data_writer.SetInputData(tube.GetOutput())

    poly_data_writer.SetFileName(filename)
    poly_data_writer.SetCompressorTypeToNone()
    poly_data_writer.SetDataModeToAscii()
    s = poly_data_writer.Write()

    return s


def simpleitk_image_from_vtk_tube(tube, sitk_reference_image):
    """
    Input: VTK tube, referenceImage (used for spacing, etc.)
    Output: SimpleITK image
    Note: Uses binary output (background 0, foreground 1)
    """
    size = list(sitk_reference_image.GetSize())
    origin = list(sitk_reference_image.GetOrigin())
    spacing = list(sitk_reference_image.GetSpacing())
    ncomp = sitk_reference_image.GetNumberOfComponentsPerPixel()

    # convert the SimpleITK image to a numpy array
    arr = sitk.GetArrayFromImage(sitk_reference_image).transpose(2, 1, 0).flatten()

    # send the numpy array to VTK with a vtkImageImport object
    data_importer = vtk.vtkImageImport()

    data_importer.CopyImportVoidPointer(arr, len(arr))
    data_importer.SetDataScalarTypeToUnsignedChar()
    data_importer.SetNumberOfScalarComponents(ncomp)

    # Set the new VTK image's parameters
    data_importer.SetDataExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)
    data_importer.SetWholeExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)
    data_importer.SetDataOrigin(origin)
    data_importer.SetDataSpacing(spacing)

    data_importer.Update()

    vtk_reference_image = data_importer.GetOutput()

    # fill the image with foreground voxels:
    inval = 1
    outval = 0
    vtk_reference_image.GetPointData().GetScalars().Fill(inval)

    logger.debug("Using polydaya to generate stencil.")
    pol2stenc = vtk.vtkPolyDataToImageStencil()
    pol2stenc.SetTolerance(0.5)  # points within 0.5 voxels are included
    pol2stenc.SetInputConnection(tube.GetOutputPort())
    pol2stenc.SetOutputOrigin(vtk_reference_image.GetOrigin())
    pol2stenc.SetOutputSpacing(vtk_reference_image.GetSpacing())
    pol2stenc.SetOutputWholeExtent(vtk_reference_image.GetExtent())
    pol2stenc.Update()

    logger.debug("using stencil to generate image.")
    imgstenc = vtk.vtkImageStencil()
    imgstenc.SetInputData(vtk_reference_image)
    imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
    imgstenc.ReverseStencilOff()
    imgstenc.SetBackgroundValue(outval)
    imgstenc.Update()

    logger.debug("Generating SimpleITK image.")
    finalImage = imgstenc.GetOutput()
    finalArray = finalImage.GetPointData().GetScalars()
    finalArray = vtk_to_numpy(finalArray).reshape(sitk_reference_image.GetSize()[::-1])
    logger.debug(f"Volume = {finalArray.sum()*sum(spacing):.3f} mm^3")
    finalImageSITK = sitk.GetImageFromArray(finalArray)
    finalImageSITK.CopyInformation(sitk_reference_image)

    return finalImageSITK


def convert_simpleitk_to_vtk(img):
    """"""
    size = list(img.GetSize())
    origin = list(img.GetOrigin())
    spacing = list(img.GetSpacing())
    ncomp = img.GetNumberOfComponentsPerPixel()

    # convert the SimpleITK image to a numpy array
    arr = sitk.GetArrayFromImage(img).transpose(2, 1, 0).flatten()
    arr_string = arr.tostring()

    # send the numpy array to VTK with a vtkImageImport object
    dataImporter = vtk.vtkImageImport()

    dataImporter.CopyImportVoidPointer(arr_string, len(arr_string))
    dataImporter.SetDataScalarTypeToUnsignedChar()
    dataImporter.SetNumberOfScalarComponents(ncomp)

    # Set the new VTK image's parameters
    dataImporter.SetDataExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)
    dataImporter.SetWholeExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)
    dataImporter.SetDataOrigin(origin)
    dataImporter.SetDataSpacing(spacing)

    dataImporter.Update()

    vtk_image = dataImporter.GetOutput()
    return vtk_image


def vessel_spline_generation(
    reference_image,
    atlas_set,
    vessel_name_list,
    vessel_radius_mm_dict,
    stop_condition_type_dict,
    stop_condition_value_dict,
    scan_direction_dict,
    atlas_label="DIR",
):
    """"""
    splinedVessels = {}

    if isinstance(vessel_name_list, str):
        vessel_name_list = [vessel_name_list]

    for vesselName in vessel_name_list:

        # We must set the image direction to identity
        # This is because it is not possible to modify VTK Image directions
        # This may get fixed in a future VTK version

        initial_image_direction = reference_image.GetDirection()

        imageList = [atlas_set[i][atlas_label][vesselName] for i in atlas_set.keys()]
        for im in imageList:
            im.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))

        vesselRadius = vessel_radius_mm_dict[vesselName]
        stopcondition_type = stop_condition_type_dict[vesselName]
        stopcondition_value = stop_condition_value_dict[vesselName]
        scan_direction = scan_direction_dict[vesselName]

        pointArray = com_from_image_list(
            imageList,
            condition_type=stopcondition_type,
            condition_value=stopcondition_value,
            scan_direction=scan_direction,
        )
        tube = tube_from_com_list(pointArray, radius=vesselRadius)

        SITKReferenceImage = imageList[0]

        vessel_delineation = simpleitk_image_from_vtk_tube(tube, SITKReferenceImage)

        vessel_delineation.SetDirection(initial_image_direction)

        splinedVessels[vesselName] = vessel_delineation

        # We also have to reset the direction to whatever it was
        # This is because SimpleITK doesn't use deep copying
        # And it isn't necessary here as we can save some sweet, sweet memory
        for im in imageList:
            im.SetDirection(initial_image_direction)

    return splinedVessels