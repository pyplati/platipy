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

from pathlib import Path

import pydicom
import numpy as np
import SimpleITK as sitk

from loguru import logger
from skimage.draw import polygon


def read_dicom_image(dicom_path):
    """Read a DICOM image series

    Args:
        dicom_path (str|pathlib.Path): Path to the DICOM series to read

    Returns:
        sitk.Image: The image as a SimpleITK Image
    """
    dicom_images = sitk.ImageSeriesReader().GetGDCMSeriesFileNames(str(dicom_path))
    return sitk.ReadImage(dicom_images)


def read_dicom_struct_file(filename):
    """Read a DICOM RTSTRUCT file

    Args:
        filename (str|pathlib.Path): Path to the RTSTRUCT to read

    Returns:
        pydicom.Dataset: The RTSTRUCT as a DICOM Dataset
    """
    dicom_struct_file = pydicom.read_file(filename, force=True)
    return dicom_struct_file


def fix_missing_data(contour_data):
    """Fixed a set of contour data if there are values missing

    Args:
        contour_data (pydicom.Sequence): The contour sequence from the DICOM object

    Returns:
        np.array: The array of contour data with missing values fixed
    """
    contour_data = np.array(contour_data)
    if contour_data.any() == "":
        logger.debug("Missing values detected.")
        missing_values = np.where(contour_data == "")[0]
        if missing_values.shape[0] > 1:
            logger.debug("More than one value missing, fixing this isn't implemented yet...")
        else:
            logger.debug("Only one value missing.")
            missing_index = missing_values[0]
            missing_axis = missing_index % 3
            if missing_axis == 0:
                logger.debug("Missing value in x axis: interpolating.")
                if missing_index > len(contour_data) - 3:
                    lower_value = contour_data[missing_index - 3]
                    upper_value = contour_data[0]
                elif missing_index == 0:
                    lower_value = contour_data[-3]
                    upper_value = contour_data[3]
                else:
                    lower_value = contour_data[missing_index - 3]
                    upper_value = contour_data[missing_index + 3]
                contour_data[missing_index] = 0.5 * (lower_value + upper_value)
            elif missing_axis == 1:
                logger.debug("Missing value in y axis: interpolating.")
                if missing_index > len(contour_data) - 2:
                    lower_value = contour_data[missing_index - 3]
                    upper_value = contour_data[1]
                elif missing_index == 0:
                    lower_value = contour_data[-2]
                    upper_value = contour_data[4]
                else:
                    lower_value = contour_data[missing_index - 3]
                    upper_value = contour_data[missing_index + 3]
                contour_data[missing_index] = 0.5 * (lower_value + upper_value)
            else:
                logger.debug("Missing value in z axis: taking slice value")
                temp = contour_data[2::3].tolist()
                temp.remove("")
                contour_data[missing_index] = np.min(np.array(temp, dtype=np.double))
    return contour_data


def transform_point_set_from_dicom_struct(dicom_image, dicom_struct, spacing_override=None):
    """Converts a set of points from a DICOM RTSTRUCT into a mask array

    Args:
        dicom_image (sitk.Image): The reference image
        dicom_struct (pydicom.Dataset): The DICOM RTSTRUCT
        spacing_override (list): The spacing to override. Defaults to None

    Returns:
        tuple: Returns a list of masks and a list of structure names
    """

    if spacing_override:
        current_spacing = list(dicom_image.GetSpacing())
        new_spacing = tuple(
            [
                current_spacing[k] if spacing_override[k] == 0 else spacing_override[k]
                for k in range(3)
            ]
        )
        dicom_image.SetSpacing(new_spacing)

    struct_point_sequence = dicom_struct.ROIContourSequence
    struct_name_sequence = [
        "_".join(i.ROIName.split()) for i in dicom_struct.StructureSetROISequence
    ]

    struct_list = []
    final_struct_name_sequence = []

    for struct_index, struct_name in enumerate(struct_name_sequence):
        image_blank = np.zeros(dicom_image.GetSize()[::-1], dtype=np.uint8)
        logger.debug("Converting structure {0} with name: {1}".format(struct_index, struct_name))

        if not hasattr(struct_point_sequence[struct_index], "ContourSequence"):
            logger.debug("No contour sequence found for this structure, skipping.")
            continue

        if len(struct_point_sequence[struct_index].ContourSequence) == 0:
            logger.debug("Contour sequence empty for this structure, skipping.")
            continue

        if len(struct_point_sequence[struct_index].ContourSequence) == 0:
            logger.debug("Contour sequence empty for this structure, skipping.")
            continue

        if (
            not struct_point_sequence[struct_index].ContourSequence[0].ContourGeometricType
            == "CLOSED_PLANAR"
        ):
            logger.debug("This is not a closed planar structure, skipping.")
            continue

        for sl in range(len(struct_point_sequence[struct_index].ContourSequence)):

            contour_data = fix_missing_data(
                struct_point_sequence[struct_index].ContourSequence[sl].ContourData
            )

            struct_slice_contour_data = np.array(contour_data, dtype=np.double)
            vertex_arr_physical = struct_slice_contour_data.reshape(
                struct_slice_contour_data.shape[0] // 3, 3
            )

            point_arr = np.array(
                [dicom_image.TransformPhysicalPointToIndex(i) for i in vertex_arr_physical]
            ).T

            [x_vertex_arr_image, y_vertex_arr_image] = point_arr[[0, 1]]
            z_index = point_arr[2][0]
            if np.any(point_arr[2] != z_index):
                logger.debug("Error: axial slice index varies in contour. Quitting now.")
                logger.debug("Structure:   {0}".format(struct_name))
                logger.debug("Slice index: {0}".format(z_index))
                quit()

            if z_index >= dicom_image.GetSize()[2]:
                logger.debug("Warning: Slice index greater than image size. Skipping slice.")
                logger.debug("Structure:   {0}".format(struct_name))
                logger.debug("Slice index: {0}".format(z_index))
                continue

            slice_arr = np.zeros(image_blank.shape[-2:], dtype=np.uint8)

            filled_indices_x, filled_indices_y = polygon(
                x_vertex_arr_image, y_vertex_arr_image, shape=slice_arr.shape
            )
            slice_arr[filled_indices_y, filled_indices_x] = 1

            image_blank[z_index] += slice_arr

        struct_image = sitk.GetImageFromArray(1 * (image_blank > 0))
        struct_image.CopyInformation(dicom_image)
        struct_list.append(sitk.Cast(struct_image, sitk.sitkUInt8))
        final_struct_name_sequence.append(struct_name)

    return struct_list, final_struct_name_sequence


def convert_rtstruct(
    dcm_img,
    dcm_rt_file,
    prefix="Struct_",
    output_dir=".",
    output_img=None,
    spacing=None,
):
    """Convert a DICOM RTSTRUCT to NIFTI masks.

    The masks are stored as NIFTI files in the output directory

    Args:
        dcm_img (str|pathlib.Path): Path to the reference DICOM image series
        dcm_rt_file (str|pathlib.Path): Path to the DICOM RTSTRUCT file
        prefix (str, optional): The prefix to give the output files. Defaults to "Struct_".
        output_dir (str|pathlib.Path, optional): Path to the output directory. Defaults to ".".
        output_img (str|pathlib.Path, optional): If set, write the reference image to this file as
                                                 in NIFTI format. Defaults to None.
        spacing (list, optional): Values of image spacing to override. Defaults to None.
    """

    logger.debug("Converting RTStruct: {0}".format(dcm_rt_file))
    logger.debug("Using image series: {0}".format(dcm_img))
    logger.debug("Output file prefix: {0}".format(prefix))
    logger.debug("Output directory: {0}".format(output_dir))

    prefix = prefix + "{0}"

    dicom_image = read_dicom_image(dcm_img)
    dicom_struct = read_dicom_struct_file(dcm_rt_file)

    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)

    if output_dir.exists():
        output_dir.mkdir(exist_ok=True, parents=True)

    image_output_path = None
    if output_img is not None:

        if not isinstance(output_img, Path):
            if not output_img.endswith(".nii.gz"):
                output_img = f"{output_img}.nii.gz"
            output_img = output_dir.joinpath(output_img)

        image_output_path = output_img

        logger.debug("Image series to be converted to: {0}".format(image_output_path))

    if spacing:

        if isinstance(spacing, str):
            spacing = [float(i) for i in spacing.split(",")]

        logger.debug("Overriding image spacing with: {0}".format(spacing))

    struct_list, struct_name_sequence = transform_point_set_from_dicom_struct(
        dicom_image, dicom_struct, spacing
    )
    logger.debug("Converted all structures. Writing output.")
    for struct_index, struct_image in enumerate(struct_list):
        out_name = "{0}.nii.gz".format(prefix.format(struct_name_sequence[struct_index]))
        out_name = output_dir.joinpath(out_name)
        logger.debug(f"Writing file to: {output_dir}")
        sitk.WriteImage(struct_image, str(out_name))

    if image_output_path is not None:
        sitk.WriteImage(dicom_image, str(image_output_path))
