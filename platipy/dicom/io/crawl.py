# Copyright 2020 University of New South Wales, University of Sydney

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import re
import sys

import pathlib
import pydicom
import numpy as np
import SimpleITK as sitk

from skimage.draw import polygon
from loguru import logger

from datetime import datetime


def flatten(itr):
    if type(itr) in (str, bytes, sitk.Image):
        yield itr
    else:
        for x in itr:
            try:
                yield from flatten(x)
            except TypeError:
                yield x


def get_suv_bw_scale_factor(ds):
    # Modified from
    # https://qibawiki.rsna.org/images/6/62/SUV_vendorneutral_pseudocode_happypathonly_20180626_DAC.pdf

    if ds.Units == "CNTS":
        # Try to find the Philips private scale factor")
        return float(ds[0x7053, 0x1000].value)

    assert ds.Modality == "PT"
    assert "DECY" in ds.CorrectedImage
    assert "ATTN" in ds.CorrectedImage
    assert "START" in ds.DecayCorrection
    assert ds.Units == "BQML"

    half_life = float(ds.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife)

    if "SeriesTime" in ds:
        series_date_time = ds.SeriesDate + "_" + ds.SeriesTime
    if "." in series_date_time:
        series_date_time = series_date_time[
            : -(len(series_date_time) - series_date_time.index("."))
        ]
    series_date_time = datetime.strptime(series_date_time, "%Y%m%d_%H%M%S")

    if "SeriesTime" in ds:
        start_time = (
            ds.SeriesDate
            + "_"
            + ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime
        )
    if "." in start_time:
        start_time = start_time[: -(len(start_time) - start_time.index("."))]
    start_time = datetime.strptime(start_time, "%Y%m%d_%H%M%S")

    decay_time = (series_date_time - start_time).seconds
    injected_dose = float(ds.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose)
    decayed_dose = injected_dose * pow(2, -decay_time / half_life)
    patient_weight = float(ds.PatientWeight)
    suv_bw_scale_factor = patient_weight * 1000 / decayed_dose

    return suv_bw_scale_factor


def get_dicom_info_from_description(dicom_object, return_extra=False, sop_class_name="UNKNOWN"):
    """
    Attempts to return some information from a DICOM
    This is typically used for naming converted NIFTI files

    Args:
        dicom_object (pydicom.dataset.FileDataset): The DICOM object
        return_extra (bool, optional): return information that is usually not required

    Returns:
        info (str): Some extracted information
    """
    try:
        dicom_sop_class_name = dicom_object.SOPClassUID.name
    except AttributeError:
        logger.warning(f"Could not find DICOM SOP Class UID, using {sop_class_name}.")
        dicom_sop_class_name = sop_class_name

    if "Image" in dicom_sop_class_name:
        # Get the modality
        image_modality = dicom_object.Modality
        logger.info(f"    Image modality: {image_modality}")

        if image_modality == "CT":
            # There is typically not much extra information
            # At the moment, we do not return anything for CT imaging
            if return_extra:
                try:
                    protocol_name = dicom_object.ProtocolName

                    if protocol_name != "":
                        return re.sub(r"[^\w]", "_", protocol_name).upper()
                except AttributeError:
                    logger.warning("    Could not find ProtocolName")

            return ""

        elif image_modality == "MR":
            # Not much consistency, but we can get the protocol name
            try:
                protocol_name = re.sub(r"[^\w]", "_", dicom_object.ProtocolName).upper()
            except AttributeError:
                logger.warning("    Could not find ProtocolName")
                protocol_name = ""

            try:
                sequence_name = re.sub(r"[^\w]", "_", dicom_object.SequenceName).upper()
            except AttributeError:
                logger.warning("    Could not find SequenceName")
                sequence_name = ""

            try:
                series_description = re.sub(r"[^\w]", "_", dicom_object.SeriesDescription).upper()
            except AttributeError:
                logger.warning("    Could not find SequenceName")
                series_description = ""

            combined_name = "_".join([protocol_name, sequence_name, series_description])

            while "__" in combined_name:
                combined_name = combined_name.replace("__", "_")

            if protocol_name != "" and not return_extra:
                return protocol_name

            else:
                return combined_name

        elif image_modality == "PT":
            # Not much experience with this
            # We can search through the corrections applied
            # Return whether or not attentuation is applied

            try:
                corrections = dicom_object.CorrectedImage
            except AttributeError:
                corrections = "NONE"

            if "ATTN" in corrections:
                return "AC"
            else:
                return "NAC"


def safe_sort_dicom_image_list(dicom_image_list):
    """
    Sorts a list of DICOM image files based on a DICOM tag value.
    This is a much safer method than reading SliceLocation.
    It takes mandatory DICOM fields (Image Position [Patient]) and (Image Orientation [Patient]).
    The list of DICOM files is sorted by projecting the image position onto the axis normal to the
    place defined by the image orientation.

    This accounts for differences in patient position (e.g. HFS/FFS).

    Args:
        dicom_image_list (list): [description]
    """
    sorted_dict = {}
    for dicom_file in dicom_image_list:
        dcm = pydicom.read_file(dicom_file, force=True)

        image_position = np.array(dcm.ImagePositionPatient, dtype=float)
        image_orientation = np.array(dcm.ImageOrientationPatient, dtype=float)

        image_plane_normal = np.cross(image_orientation[:3], image_orientation[3:])

        slice_location = (image_position * image_plane_normal)[2]

        sorted_dict[dicom_file] = slice_location

    sorter_safe = lambda dcm_file: sorted_dict[dcm_file]

    return sorted(dicom_image_list, key=sorter_safe)


def fix_missing_data(contour_data_list):
    """
    Fixes missing points in contouring using simple linear interpolation


    Args:
        contour_data_list (list): The contour data for each slice

    Returns:
        contour_data (numpy array): Interpolated contour data
    """
    contour_data = np.array(contour_data_list)
    if contour_data.any() == "":
        logger.warning("    Missing values detected.")
        missing_values = np.where(contour_data == "")[0]
        if missing_values.shape[0] > 1:
            logger.warning("    More than one value missing, fixing this isn't implemented yet...")
        else:
            logger.warning("    Only one value missing.")
            missing_index = missing_values[0]
            missing_axis = missing_index % 3
            if missing_axis == 0:
                logger.warning("    Missing value in x axis: interpolating.")
                if missing_index > len(contour_data) - 3:
                    lower_val = contour_data[missing_index - 3]
                    upper_val = contour_data[0]
                elif missing_index == 0:
                    lower_val = contour_data[-3]
                    upper_val = contour_data[3]
                else:
                    lower_val = contour_data[missing_index - 3]
                    upper_val = contour_data[missing_index + 3]
                contour_data[missing_index] = 0.5 * (lower_val + upper_val)
            elif missing_axis == 1:
                logger.warning("    Missing value in y axis: interpolating.")
                if missing_index > len(contour_data) - 2:
                    lower_val = contour_data[missing_index - 3]
                    upper_val = contour_data[1]
                elif missing_index == 0:
                    lower_val = contour_data[-2]
                    upper_val = contour_data[4]
                else:
                    lower_val = contour_data[missing_index - 3]
                    upper_val = contour_data[missing_index + 3]
                contour_data[missing_index] = 0.5 * (lower_val + upper_val)
            else:
                logger.warning("    Missing value in z axis: taking slice value")
                temp = contour_data[2::3].tolist()
                temp.remove("")
                contour_data[missing_index] = np.min(np.array(temp, dtype=np.double))
    return contour_data


def transform_point_set_from_dicom_struct(image, dicom_struct, spacing_override=False):
    """
    This function is used to generate a binary mask from a set of vertices.
    This allows us to convert from DICOM-RTStruct format to any imaging format.

    Args:
        image ([SimpleITK.Image]): The image, used to copy imaging information
            (e.g. resolution, spacing)
        dicom_struct ([pydicom.Dataset]): The DICOM-RTStruct file
        spacing_override (bool | tuple, optional): Overwrite the spacing.
            Set with (axial_spacing, coronal_spacing, sagittal spacing). Defaults to False.

    Returns:
        list, list : final_struct_name_sequence, structure_list
    """
    if spacing_override:
        current_spacing = list(image.GetSpacing())
        new_spacing = tuple(
            [
                current_spacing[k] if spacing_override[k] == 0 else spacing_override[k]
                for k in range(3)
            ]
        )
        image.SetSpacing(new_spacing)

    struct_point_sequence = dicom_struct.ROIContourSequence
    struct_name_sequence = [
        "_".join(i.ROIName.split()) for i in dicom_struct.StructureSetROISequence
    ]

    structure_list = []
    final_struct_name_sequence = []

    for structIndex, structure_name in enumerate(struct_name_sequence):
        image_blank = np.zeros(image.GetSize()[::-1], dtype=np.uint8)
        logger.info(
            "    Converting structure {0} with name: {1}".format(structIndex, structure_name)
        )

        if structIndex >= len(struct_point_sequence):
            logger.warning("    Contour sequence is missing, skipping.")
            continue

        if not hasattr(struct_point_sequence[structIndex], "ContourSequence"):
            logger.warning("    No contour sequence found for this structure, skipping.")
            continue

        if len(struct_point_sequence[structIndex].ContourSequence) == 0:
            logger.warning("    Contour sequence is empty, skipping.")
            continue

        if (
            not struct_point_sequence[structIndex].ContourSequence[0].ContourGeometricType
            == "CLOSED_PLANAR"
        ):
            logger.warning("    This is not a closed planar structure, skipping.")
            continue

        for sl in range(len(struct_point_sequence[structIndex].ContourSequence)):

            contour_data = fix_missing_data(
                struct_point_sequence[structIndex].ContourSequence[sl].ContourData
            )

            struct_slice_contour_data = np.array(contour_data, dtype=np.double)
            vertexArr_physical = struct_slice_contour_data.reshape(
                struct_slice_contour_data.shape[0] // 3, 3
            )

            point_arr = np.array(
                [image.TransformPhysicalPointToIndex(i) for i in vertexArr_physical]
            ).T

            [xVertexArr_image, yVertexArr_image] = point_arr[[0, 1]]
            zIndex = point_arr[2][0]

            if np.any(point_arr[2] != zIndex):
                logger.error("    Axial slice index varies in contour. Quitting now.")
                logger.error("    Structure:   {0}".format(structure_name))
                logger.error("    Slice index: {0}".format(zIndex))
                quit()

            if zIndex >= image.GetSize()[2]:
                logger.warning("    Slice index greater than image size. Skipping slice.")
                logger.warning("    Structure:   {0}".format(structure_name))
                logger.warning("    Slice index: {0}".format(zIndex))
                continue

            sliceArr = np.zeros(image.GetSize()[:2], dtype=np.uint8)
            filledIndicesX, filledIndicesY = polygon(
                xVertexArr_image, yVertexArr_image, shape=sliceArr.shape
            )
            sliceArr[filledIndicesX, filledIndicesY] = 1
            image_blank[zIndex] += sliceArr.T

        struct_image = sitk.GetImageFromArray(1 * (image_blank > 0))
        struct_image.CopyInformation(image)
        structure_list.append(sitk.Cast(struct_image, sitk.sitkUInt8))
        structure_name_clean = re.sub(r"[^\w]", "_", structure_name).upper()
        while "__" in structure_name_clean:
            structure_name_clean = structure_name_clean.replace("__", "_")
        final_struct_name_sequence.append(structure_name_clean)

    return final_struct_name_sequence, structure_list


def process_dicom_file_list(dicom_file_list, parent_sorting_field="PatientName", verbose=False):

    """
    Organise the DICOM files by the series UID
    """
    dicom_series_dict_parent = {}

    for i, dicom_file in enumerate(sorted(dicom_file_list)):
        if verbose is True:
            logger.debug(f"  Sorting file {i}")

        dicom_file = dicom_file.as_posix()

        if "dicomdir" in dicom_file.lower():
            logger.warning(
                "DICOMDIR is not supported in this tool, images are read directly. Skipping."
            )
            continue

        dicom_object = pydicom.read_file(dicom_file, force=True)

        parent_sorting_field_data = dicom_object[parent_sorting_field].value

        if parent_sorting_field_data not in dicom_series_dict_parent.keys():
            dicom_series_dict_parent[parent_sorting_field_data] = {}

        series_uid = dicom_object.SeriesInstanceUID

        if series_uid not in dicom_series_dict_parent[parent_sorting_field_data].keys():
            dicom_series_dict_parent[parent_sorting_field_data][series_uid] = [dicom_file]

        else:
            dicom_series_dict_parent[parent_sorting_field_data][series_uid].append(dicom_file)

    return dicom_series_dict_parent


def process_dicom_series(
    dicom_series_dict,
    series_uid,
    parent_sorting_field="PatientName",
    return_extra=True,
    individual_file=False,
    initial_sop_class_name_default="UNKNOWN",
):
    if not individual_file:
        logger.info(f"  Processing series UID: {series_uid}")
        dicom_file_list = dicom_series_dict[series_uid]
    else:
        logger.info(f"  Processing individual file: {individual_file}")
        dicom_file_list = [individual_file]

    logger.info(f"  Number of DICOM files: {len(dicom_file_list)}")

    initial_dicom = pydicom.read_file(dicom_file_list[0])

    # Get the data in the parent sorting field, clean with RegEx
    parent_sorting_data = re.sub(
        r"[^\w]", "_", str(initial_dicom[parent_sorting_field].value)
    ).upper()

    if parent_sorting_data == "":
        logger.error(
            f"Could not find any data in {parent_sorting_field}. This is very bad, the data cannot be sorted properly."
        )
        """
        ! TO DO
        Implement a routine to let a user correlate a root directory with a name
        """
        parent_sorting_data = "TEMP"

    try:
        initial_dicom_sop_class_name = initial_dicom.SOPClassUID.name
    except AttributeError:
        logger.warning(
            f"Could not find DICOM SOP Class UID, using {initial_sop_class_name_default}."
        )
        initial_dicom_sop_class_name = initial_sop_class_name_default

    try:
        study_uid = initial_dicom.StudyInstanceUID
    except AttributeError:
        study_uid = "00001"

    """
    ! TO DO
    Need to check for secondary capture image storage
    This can include JPEGs with written information on them
    This is typically not very useful
    We can dump it to file
    Or just save the DICOM file in the folder of interest

    Not a big problem, sort out another day
    """

    # Check the potential types of DICOM files
    if (
        "Image" in initial_dicom_sop_class_name
        and initial_dicom_sop_class_name != "Secondary Capture Image Storage"
    ):
        # Load as an primary image

        sorted_file_list = safe_sort_dicom_image_list(dicom_file_list)

        try:
            image = sitk.ReadImage(sorted_file_list)
        except RuntimeError:
            logger.warning("  Could not read image into SimpleITK.")
            logger.info("  Processing files individually.")

            for dicom_file in dicom_file_list:
                return process_dicom_series(
                    dicom_series_dict,
                    series_uid,
                    parent_sorting_field=parent_sorting_field,
                    return_extra=return_extra,
                    individual_file=dicom_file,
                    initial_sop_class_name_default=initial_sop_class_name_default,
                )

        dicom_file_metadata = {
            "parent_sorting_data": parent_sorting_data,
            "study_uid": study_uid,
        }

        """
        ! TO DO - integrity check
            Read in all the files here, check the slice location and determine if any are missing
        """
        if initial_dicom.Modality == "PT":

            # scaling_factor = get_suv_bw_scale_factor(initial_dicom)
            # image *= scaling_factor

            # !TO DO
            # Work on PET SUV conversion
            None

        """
        ! CHECKPOINT
        Some DCE MRI sequences have the same series UID
        Here we check the sequence name, and split if necessary
        """

        if initial_dicom.Modality == "MR":

            try:
                sequence_names = np.unique(
                    [pydicom.read_file(x).SequenceName for x in dicom_file_list]
                )

                sequence_dict = {}
                for dcm_name in dicom_file_list:
                    dcm_obj = pydicom.read_file(dcm_name)
                    var = dcm_obj.SequenceName
                    if var not in sequence_dict.keys():
                        sequence_dict[var] = [dcm_name]
                    else:
                        sequence_dict[var].append(dcm_name)

            except AttributeError:
                try:
                    logger.warning(
                        "    MRI sequence name not found. The SeriesDescription will be used instead."
                    )

                    sequence_names = np.unique(
                        [pydicom.read_file(x).SeriesDescription for x in dicom_file_list]
                    )

                    sequence_dict = {}
                    for dcm_name in dicom_file_list:
                        dcm_obj = pydicom.read_file(dcm_name)
                        var = dcm_obj.SeriesDescription
                        if var not in sequence_dict.keys():
                            sequence_dict[var] = [dcm_name]
                        else:
                            sequence_dict[var].append(dcm_name)

                except AttributeError:
                    logger.warning(
                        "    MRI SeriesDescription not found. The AcquisitionComments will be used instead."
                    )

                    sequence_names = np.unique(
                        [pydicom.read_file(x).AcquisitionComments for x in dicom_file_list]
                    )

                    sequence_dict = {}
                    for dcm_name in dicom_file_list:
                        dcm_obj = pydicom.read_file(dcm_name)
                        var = dcm_obj.AcquisitionComments
                        if var not in sequence_dict.keys():
                            sequence_dict[var] = [dcm_name]
                        else:
                            sequence_dict[var].append(dcm_name)

            if initial_dicom.Manufacturer == "GE MEDICAL SYSTEMS":
                # GE use the DICOM tag (0019, 10a2) [Raw data run number]
                # in Diffusion weighted MRI sequences
                # We need to separate this out to get the difference sequences

                if initial_dicom.SeriesDescription == "Diffusion Weighted":

                    # num_sequences = int( (initial_dicom[(0x0025, 0x1007)]) / (initial_dicom[(0x0021, 0x104f)]) )
                    # number_of_images / images_per_seq
                    num_images_per_seq = initial_dicom[(0x0021, 0x104F)].value

                    sequence_names = np.unique(
                        [
                            f"DWI_{str( ( pydicom.read_file(x)['InstanceNumber'].value - 1) // num_images_per_seq )}"
                            for x in dicom_file_list
                        ]
                    )

                    sequence_name_index_dict = {
                        name: index for index, name in enumerate(sequence_names)
                    }

                    sequence_dict = {}
                    for dcm_name in dicom_file_list:
                        dcm_obj = pydicom.read_file(dcm_name)
                        var = f"DWI_{str( ( dcm_obj['InstanceNumber'].value - 1) // num_images_per_seq )}"
                        var_to_index = sequence_name_index_dict[var]

                        if var_to_index not in sequence_dict.keys():
                            sequence_dict[var_to_index] = [dcm_name]
                        else:
                            sequence_dict[var_to_index].append(dcm_name)

                    sequence_names = sorted(sequence_dict.keys())

            if np.alen(sequence_names) > 1:
                logger.warning("  Two MR sequences were found under a single series UID.")
                logger.warning("  These will be split into separate images.")

                # Split up the DICOM file list by sequence name
                for sequence_name in sequence_names:

                    dicom_file_list_by_sequence = sequence_dict[sequence_name]

                    logger.info(sequence_name)
                    logger.info(len(dicom_file_list_by_sequence))

                    sorted_file_list = safe_sort_dicom_image_list(dicom_file_list_by_sequence)

                    initial_dicom = pydicom.read_file(sorted_file_list[0], force=True)

                    image_by_sequence = sitk.ReadImage(sorted_file_list)

                    dicom_file_metadata_by_sequence = {
                        "parent_sorting_data": parent_sorting_data,
                        "study_uid": study_uid,
                    }

                    yield "IMAGES", dicom_file_metadata_by_sequence, initial_dicom, image_by_sequence
                return  # Stop iteration

        yield "IMAGES", dicom_file_metadata, initial_dicom, image

    if "Structure" in initial_dicom_sop_class_name:
        # Load as an RT structure set
        # This should be done individually for each file

        logger.info(f"      Number of files: {len(dicom_file_list)}")
        for index, dicom_file in enumerate(dicom_file_list):
            dicom_object = pydicom.read_file(dicom_file, force=True)

            # We must also read in the corresponding DICOM image
            # This can be found by matching the references series UID to the series UID

            """
            ! TO DO
            What happens if there is an RT structure set with different referenced sequences?
            """

            # Get the "ReferencedFrameOfReferenceSequence", first item
            referenced_frame_of_reference_item = dicom_object.ReferencedFrameOfReferenceSequence[0]

            # Get the "RTReferencedStudySequence", first item
            # This retrieves the study UID
            # This might be useful, but would typically match the actual StudyInstanceUID in the
            # DICOM object
            rt_referenced_series_item = (
                referenced_frame_of_reference_item.RTReferencedStudySequence[0]
            )

            # Get the "RTReferencedSeriesSequence", first item
            # This retreives the actual referenced series UID, which we need to match imaging
            # parameters
            rt_referenced_series_again_item = rt_referenced_series_item.RTReferencedSeriesSequence[
                0
            ]

            # Get the appropriate series instance UID
            image_series_uid = rt_referenced_series_again_item.SeriesInstanceUID
            logger.info(f"      Item {index}: Matched SeriesInstanceUID = {image_series_uid}")

            # Read in the corresponding image
            sorted_file_list = safe_sort_dicom_image_list(dicom_series_dict[image_series_uid])
            image = sitk.ReadImage(sorted_file_list)

            initial_dicom = pydicom.read_file(sorted_file_list[0], force=True)

            (
                structure_name_list,
                structure_image_list,
            ) = transform_point_set_from_dicom_struct(image, dicom_object)

            dicom_file_metadata = {
                "parent_sorting_data": parent_sorting_data,
                "study_uid": study_uid,
                "structure_name_list": structure_name_list,
            }

            yield "STRUCTURES", dicom_file_metadata, dicom_object, structure_image_list

    if "Dose" in initial_dicom_sop_class_name:
        # Load as an RT Dose distribution
        # This should be done individually for each file

        logger.info(f"      Number of files: {len(dicom_file_list)}")
        for index, dicom_file in enumerate(dicom_file_list):
            dicom_object = pydicom.read_file(dicom_file, force=True)

            """
            ! CHECKPOINT
            There should only be a single RT dose file (with each series UID)
            If there are more, yield each
            """

            initial_dicom = pydicom.read_file(dicom_file, force=True)

            dicom_file_metadata = {
                "parent_sorting_data": parent_sorting_data,
                "study_uid": study_uid,
            }

            # We must read in as a float otherwise when we multiply by one later it will not work!
            raw_dose_image = sitk.ReadImage(dicom_file, sitk.sitkFloat32)

            dose_grid_scaling = dicom_object.DoseGridScaling

            logger.debug(f"  Dose grid scaling: {dose_grid_scaling} Gy")

            scaled_dose_image = raw_dose_image * dose_grid_scaling

            yield "DOSES", dicom_file_metadata, dicom_object, scaled_dose_image

        """
        ! TO DO
        1. (DONE) Implement conversion of dose files (to NIFTI images)
        2. Implement conversion of RT plan files to text dump
        3. Do something with other files (e.g. Deformable Image Registration stuff)
        """

    return


def write_output_data_to_disk(
    output_data_dict,
    output_directory="./",
    output_file_suffix=".nii.gz",
    overwrite_existing_files=False,
):
    """
    Write output to disk
    """
    if output_data_dict is None:
        return

    filename_fields = [i for i in output_data_dict.keys() if i != "parent_sorting_data"]
    parent_sorting_data = output_data_dict["parent_sorting_data"]

    files_written = {}

    """
    Write the the converted images to disk

    ! CONSIDER
    We could simply write as we go?
    Pro: save memory, important if processing very large files
    Con: Reading as we go allows proper indexing

    """

    for field in filename_fields:
        logger.info(f"  Writing files for field: {field}")
        p = pathlib.Path(output_directory) / parent_sorting_data / field
        p.mkdir(parents=True, exist_ok=True)
        files_written[field] = []

        for field_filename_base, field_list in output_data_dict[field].items():
            # Check if there is a list of images with matching names
            # This will depend on the name format chosen
            # If there is a list, we append an index as we write to disk

            if isinstance(field_list, (tuple, list)):
                # Flatten
                field_list_flat = list(flatten(field_list))

                # Iterate
                for suffix, file_to_write in enumerate(field_list_flat):
                    field_filename = field_filename_base + f"_{suffix}"

                    # Some cleaning
                    while "__" in field_filename:
                        field_filename = field_filename.replace("__", "_")

                    while field_filename[-1] == "_":
                        field_filename = field_filename[:-1]

                    # Save image!
                    output_name = (
                        pathlib.Path(output_directory)
                        / parent_sorting_data
                        / field
                        / (field_filename + output_file_suffix)
                    )
                    files_written[field].append(output_name)

                    if output_name.is_file():
                        logger.warning(f"  File exists: {output_name}")

                        if overwrite_existing_files:
                            logger.warning("  You have selected to overwrite existing files.")

                        else:
                            logger.info(
                                "  You have selected to NOT overwrite existing files. Continuing."
                            )
                            continue

                    sitk.WriteImage(file_to_write, output_name.as_posix())

            else:
                field_filename = field_filename_base
                file_to_write = field_list

                # Some cleaning
                while "__" in field_filename:
                    field_filename = field_filename.replace("__", "_")

                while field_filename[-1] == "_":
                    field_filename = field_filename[:-1]

                # Save image!
                """
                ! TO DO
                Use pathlib, and perform some checks so we don"t overwrite anything!
                """
                output_name = (
                    pathlib.Path(output_directory)
                    / parent_sorting_data
                    / field
                    / (field_filename + output_file_suffix)
                )
                files_written[field].append(output_name)

                if output_name.is_file():
                    logger.warning(f"  File exists: {output_name}")

                    if overwrite_existing_files:
                        logger.warning("  You have selected to overwrite existing files.")

                    else:
                        logger.info(
                            "  You have selected to NOT overwrite existing files. Continuing."
                        )
                        continue

                sitk.WriteImage(file_to_write, output_name.as_posix())

    return files_written


def process_dicom_directory(
    dicom_directory,
    parent_sorting_field="PatientName",
    output_image_name_format="{parent_sorting_data}_{study_uid_index}_{Modality}_{image_desc}_{SeriesNumber}",
    output_structure_name_format="{parent_sorting_data}_{study_uid_index}_{Modality}_{structure_name}",
    output_dose_name_format="{parent_sorting_data}_{study_uid_index}_{DoseSummationType}",
    return_extra=True,
    output_directory="./",
    output_file_suffix=".nii.gz",
    overwrite_existing_files=False,
    write_to_disk=True,
    verbose=False,
    initial_sop_class_name_default="UNKNOWN",
):

    # Check dicom_directory type
    if isinstance(dicom_directory, str) or isinstance(dicom_directory, pathlib.Path):
        # Get all the DICOM files in the given directory
        root_path = pathlib.Path(dicom_directory)
        # Find files ending with .dcm, .dc3
        dicom_file_list = [
            p
            for p in root_path.glob("**/*")
            if p.name.lower().endswith(".dcm") or p.name.lower().endswith(".dc3")
        ]

    elif hasattr(dicom_directory, "__iter__"):
        dicom_file_list = []
        for dicom_dir in dicom_directory:
            # Get all the DICOM files in each directory
            root_path = pathlib.Path(dicom_dir)
            # Find files ending with .dcm, .dc3
            dicom_file_list += [
                p
                for p in root_path.glob("**/*")
                if p.name.lower().endswith(".dcm") or p.name.lower().endswith(".dc3")
            ]

    if len(dicom_file_list) == 0:
        logger.info("No DICOM files found in input directory. Exiting now.")
        return

    # Process the DICOM files
    # This returns a dictionary (of dictionaries):
    #   {parent_data (e.g. PatientName): {series_UID_1: [list_of_DICOM_files],
    #                                    {series_UID_2: [list_of_DICOM_files], ...
    #   parent_data_2                  : {series_UID_1: [list_of_DICOM_files],
    #                                    {series_UID_2: [list_of_DICOM_files], ...
    #   ...     }
    dicom_series_dict_parent = process_dicom_file_list(
        dicom_file_list, parent_sorting_field=parent_sorting_field, verbose=verbose
    )

    if dicom_series_dict_parent is None:
        logger.info("No valid DICOM files found. Ending.")
        return None

    output = {}

    for parent_data, dicom_series_dict in dicom_series_dict_parent.items():
        logger.info(f"Processing data for {parent_sorting_field} = {parent_data}.")
        logger.info(f"  Number of DICOM series = {len(dicom_series_dict.keys())}")

        # Set up the output data
        # This stores the SimpleITK images and file names
        output_data_dict = {}

        # Set up the study UID dict
        # This helps match structure sets to relevant images
        # And paired images to each other (e.g. PET/CT)
        study_uid_dict = {}

        # Give some user feedback
        logger.debug(f"  Output image name format: {output_image_name_format}")
        logger.debug(f"  Output structure name format: {output_structure_name_format}")
        logger.debug(f"  Output dose name format: {output_dose_name_format}")

        # For each unique series UID, process the DICOM files
        for series_uid in dicom_series_dict.keys():

            # This function returns four values
            # 1. dicom_type: This is IMAGES, STRUCTURES, DOSES, etc
            # 2. dicom_file_metadata: Some special metadata extracted from the DICOM header
            # 3. initial_dicom: The first DICOM in the series. For doses and structures there is
            #    (usually) only one DICOM anyway
            # 4. dicom_file_data: The actual SimpleITK image data

            for (
                dicom_type,
                dicom_file_metadata,
                initial_dicom,
                dicom_file_data,
            ) in process_dicom_series(
                dicom_series_dict=dicom_series_dict,
                series_uid=series_uid,
                parent_sorting_field=parent_sorting_field,
                return_extra=return_extra,
                initial_sop_class_name_default=initial_sop_class_name_default,
            ):

                # Step 1
                # Check the parent sorting field is consistent
                # This would usually be the PatientName, PatientID, or similar
                # Occasionally these will both be blank

                parent_sorting_data = dicom_file_metadata["parent_sorting_data"]

                if "parent_sorting_data" not in output_data_dict.keys():
                    output_data_dict["parent_sorting_data"] = parent_sorting_data

                else:
                    if parent_sorting_data != output_data_dict["parent_sorting_data"]:
                        logger.error(
                            f"A conflict was found for the parent sorting field "
                            f"({parent_sorting_field}): {parent_sorting_data}"
                        )
                        logger.error("Quitting now.")
                        print(dicom_series_dict_parent.keys())
                        sys.exit()
                    else:
                        logger.info(
                            f"  Parent sorting field ({parent_sorting_field}) match found: "
                            f"{parent_sorting_data}"
                        )

                # Step 2
                # Get the study UID
                # Used for indexing DICOM series

                study_uid = dicom_file_metadata["study_uid"]

                if study_uid not in study_uid_dict.keys():
                    try:
                        study_uid_index = max(study_uid_dict.values()) + 1
                    except AttributeError:
                        study_uid_index = 0  # Study UID dict might not exist
                    except ValueError:
                        study_uid_index = 0  # Study UID dict might be empty

                    logger.info(f"  Setting study instance UID index: {study_uid_index}")

                    study_uid_dict[study_uid] = study_uid_index

                else:
                    logger.info(
                        f"  Study instance UID index already exists: {study_uid_dict[study_uid]}"
                    )

                # Step 3
                # Generate names for output files

                # Special names
                # ! This can be defined once at the start of the function
                special_name_fields = [
                    "parent_sorting_data",
                    "study_uid_index",
                    "image_desc",
                    "structure_name",
                ]

                # Get the image description (other special names are already defined above)
                image_desc = get_dicom_info_from_description(
                    initial_dicom, return_extra=return_extra
                )

                # Get all the fields from the user-given name format
                if dicom_type == "IMAGES":
                    all_naming_fields = [
                        i[i.find("{") + 1 :]
                        for i in output_image_name_format.split("}")
                        if len(i) > 0
                    ]
                elif dicom_type == "STRUCTURES":
                    all_naming_fields = [
                        i[i.find("{") + 1 :]
                        for i in output_structure_name_format.split("}")
                        if len(i) > 0
                    ]
                elif dicom_type == "DOSES":
                    all_naming_fields = [
                        i[i.find("{") + 1 :]
                        for i in output_dose_name_format.split("}")
                        if len(i) > 0
                    ]

                # Now exclude those that aren't derived from the DICOM header
                dicom_header_tags = [i for i in all_naming_fields if i not in special_name_fields]

                naming_info_dict = {}
                for dicom_field in dicom_header_tags:
                    try:
                        dicom_field_value = initial_dicom[dicom_field].value
                    except (AttributeError, KeyError):
                        logger.warning(
                            f"  Could not find DICOM header {dicom_field}. Setting as 0 to "
                            f"preserve naming convention."
                        )
                        dicom_field_value = 0
                    naming_info_dict[dicom_field] = dicom_field_value

                if dicom_type == "IMAGES":

                    output_name = output_image_name_format.format(
                        parent_sorting_data=parent_sorting_data,
                        study_uid_index=study_uid_dict[study_uid],
                        image_desc=image_desc,
                        **naming_info_dict,
                    )

                    if "IMAGES" not in output_data_dict.keys():
                        # Make a new entry
                        output_data_dict["IMAGES"] = {output_name: dicom_file_data}

                    else:
                        # First check if there is another image of the same name

                        if output_name not in output_data_dict["IMAGES"].keys():
                            output_data_dict["IMAGES"][output_name] = dicom_file_data

                        else:
                            logger.info("      An image with this name exists, appending.")

                            if hasattr(output_data_dict["IMAGES"][output_name], "__iter__"):
                                output_data_dict["IMAGES"][output_name] = list(
                                    [output_data_dict["IMAGES"][output_name]]
                                )

                            output_data_dict["IMAGES"][output_name].append(dicom_file_data)

                elif dicom_type == "STRUCTURES":

                    for structure_name, structure_image in zip(
                        dicom_file_metadata["structure_name_list"], dicom_file_data
                    ):

                        output_name = output_structure_name_format.format(
                            parent_sorting_data=parent_sorting_data,
                            study_uid_index=study_uid_dict[study_uid],
                            image_desc=image_desc,
                            structure_name=structure_name,
                            **naming_info_dict,
                        )

                        if "STRUCTURES" not in output_data_dict.keys():
                            # Make a new entry
                            output_data_dict["STRUCTURES"] = {output_name: structure_image}

                        else:
                            # First check if there is another structure of the same name

                            if output_name not in output_data_dict["STRUCTURES"].keys():
                                output_data_dict["STRUCTURES"][output_name] = structure_image

                            else:
                                logger.info("      A structure with this name exists, appending.")
                                if hasattr(
                                    output_data_dict["STRUCTURES"][output_name], "__iter__"
                                ):
                                    output_data_dict["STRUCTURES"][output_name] = list(
                                        [output_data_dict["STRUCTURES"][output_name]]
                                    )

                                output_data_dict["STRUCTURES"][output_name].append(structure_image)

                elif dicom_type == "DOSES":

                    output_name = output_dose_name_format.format(
                        parent_sorting_data=parent_sorting_data,
                        study_uid_index=study_uid_dict[study_uid],
                        **naming_info_dict,
                    )

                    if "DOSES" not in output_data_dict.keys():
                        # Make a new entry
                        output_data_dict["DOSES"] = {output_name: dicom_file_data}

                    else:
                        # First check if there is another image of the same name

                        if output_name not in output_data_dict["DOSES"].keys():
                            output_data_dict["DOSES"][output_name] = dicom_file_data

                        else:
                            logger.info("      An image with this name exists, appending.")

                            if isinstance(output_data_dict["DOSES"][output_name], sitk.Image):
                                output_data_dict["DOSES"][output_name] = list(
                                    [output_data_dict["DOSES"][output_name]]
                                )

                            output_data_dict["DOSES"][output_name].append(dicom_file_data)

        if write_to_disk:
            output[str(parent_data)] = write_output_data_to_disk(
                output_data_dict=output_data_dict,
                output_directory=output_directory,
                output_file_suffix=output_file_suffix,
                overwrite_existing_files=overwrite_existing_files,
            )
        else:
            output[str(parent_data)] = output_data_dict

    """
    TO DO!
    Memory issue with output_data_dict
    Use in inner loop, reset output_data_dict
    """

    return output
