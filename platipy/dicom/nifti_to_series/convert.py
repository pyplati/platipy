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
import time

import pydicom
import SimpleITK as sitk


def convert_nifti_to_dicom_series(image, reference_dcm, tag_overrides=[], output_directory="."):
    """Converts a Nifti image to a Dicom image series

    Args:
        image (sitk.Image): A SimpleITK image object to convert
        reference_dcm (str): A directory path containing a reference Dicom series to use
        tag_overrides (list, optional): A list of tags to override containing tuples of
                                        (key, value). Defaults to [].
        output_directory (str, optional): The directory in which to place the generated Dicom
                                          files. Defaults to ".".
    """

    # Make the output directory if it doesn't already exist
    os.makedirs(output_directory, exist_ok=True)

    # Read in the reference Dicom series
    reference_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(reference_dcm)
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(reference_file_names)
    series_reader.MetaDataDictionaryArrayUpdateOn()
    series_reader.LoadPrivateTagsOn()
    series_reader.Execute()

    # Prepare the Dicom writer
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()

    # Copy relevant tags from the original meta-data dictionary (private tags are also
    # accessible).
    tags_to_copy = [
        "0010|0010",  # Patient Name
        "0010|0020",  # Patient ID
        "0010|0030",  # Patient Birth Date
        "0020|000D",  # Study Instance UID, for machine consumption
        "0020|0010",  # Study ID, for human consumption
        "0008|0020",  # Study Date
        "0008|0030",  # Study Time
        "0008|0050",  # Accession Number
        "0008|0060",  # Modality
        "0018|5100",  # Patient Position
        "0020|1041",  # Slice Location
        "0018|0022",  # Scan Options
        "0018|0060",  # KVP
        "0018|0070",  # Counts Accumulated
        "0018|0088",  # Spacing Between Slices
        "0018|0090",  # Data Collection Diameter
        "0018|1000",  # Device Serial Number
        "0018|1020",  # Software Version(s)
        "0018|1100",  # Reconstruction Diameter
        "0018|1120",  # Gantry/Detector Tilt
        "0018|1130",  # Table Height
        "0018|1140",  # Rotation Direction
        "0018|1150",  # Exposure Time
        "0018|1151",  # X-Ray Tube Current
        "0018|1152",  # Exposure
        "0018|1160",  # Filter Type
        "0018|1210",  # Convolution Kernel
        "0018|5100",  # Patient Position
        "0018|9323",  # Exposure Modulation Type
        "0018|9345",  # CTDIvol
        "0020|0010",  # Study ID
        "0020|0011",  # Series Number
        "0020|0012",  # Acquisition Number
        "0020|0013",  # Instance Number
        "0020|0032",  # Image Position (Patient)
        "0020|0037",  # Image Orientation (Patient)
        "0020|0052",  # Frame of Reference UID
        "0020|0060",  # Laterality
        "0020|1040",  # Position Reference Indicator
        "0020|1041",  # Slice Location
        "0028|1052",  # Rescale Intercept
        "0028|1053",  # Rescale Slope
        "0028|1054",  # Rescale Type
    ]

    # Prepare the modification date and time
    modification_date = time.strftime("%Y%m%d")
    modification_time = time.strftime("%H%M%S")

    # Generate some new UIDs
    for_uid = pydicom.uid.generate_uid()
    study_uid = pydicom.uid.generate_uid()
    series_uid = pydicom.uid.generate_uid()

    # Copy some of the tags and add the relevant tags indicating the change.
    direction = image.GetDirection()
    series_tag_values = [
        (k, series_reader.GetMetaData(0, k))
        for k in tags_to_copy
        if series_reader.HasMetaDataKey(0, k)
    ] + [
        ("0008|0031", modification_time),  # Series Time
        ("0008|0021", modification_date),  # Series Date
        ("0008|0008", "DERIVED\\SECONDARY"),  # Image Type
        ("0018|0050", str(image.GetSpacing()[2])),
        ("0018|0088", str(image.GetSpacing()[2])),
        ("0020|0052", for_uid),  # Frame of reference UID
        ("0020|000E", series_uid),  # Series UID
        ("0020|000D", study_uid),  # Study UID
        (
            "0020|0037",
            "\\".join(
                map(
                    str,
                    (
                        direction[0],
                        direction[3],
                        direction[6],  # Image Orientation (Patient)
                        direction[1],
                        direction[4],
                        direction[7],
                    ),
                )
            ),
        ),
    ]

    # Add the tag overrides to the list
    for tag in tag_overrides:
        dcm_tag = pydicom.tag.Tag(pydicom.datadict.tag_for_keyword(tag[0]))
        tag_string = str(dcm_tag).replace(", ", "|").replace("(", "").replace(")", "")
        series_tag_values.append((tag_string, tag[1]))

    # Write each image slice, preparing the header tags as needed
    for i in range(image.GetDepth()):

        image_slice = image[:, :, i]

        # Tags shared by the series.
        for tag, value in series_tag_values:
            image_slice.SetMetaData(tag, value)

        # Slice specific tags.
        image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))  # Instance Creation Date
        image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))  # Instance Creation Time
        image_slice.SetMetaData(
            "0020|0032", "\\".join(map(str, image.TransformIndexToPhysicalPoint((0, 0, i))))
        )  # Image Position (Patient)
        image_slice.SetMetaData("0020|0013", str(i))  # Instance Number
        image_slice.SetMetaData(
            "0020|1041", str(image.TransformIndexToPhysicalPoint((0, 0, i))[2])
        )

        # Write to the output directory and add the extension dcm, to force writing in DICOM
        # format.
        writer.SetFileName(os.path.join(output_directory, str(i) + ".dcm"))
        writer.Execute(image_slice)
