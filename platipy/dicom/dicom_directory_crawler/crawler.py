import re
import sys

import pydicom
import SimpleITK as sitk

from loguru import logger

import pathlib

from platipy.dicom.dicom_directory_crawler.conversion_utils import (
    process_dicom_directory,
    write_output_data_to_disk
)

"""
Settings
"""

parent_sorting_field = "PatientName"

output_image_name_format = "{parent_sorting_data}_{study_uid_index}_{image_modality}_{image_desc}_{series_num}"
output_structure_name_format = "{parent_sorting_data}_{study_uid_index}_{image_modality}_{structure_name}"

output_directory = "./"

overwrite_existing_files = False

output_file_suffix = ".nii.gz"

input_directory = "./temp/WES-005/"

return_extra = True

"""
Code
"""

logger.info("########################")
logger.info(" Running DICOM crawler ")
logger.info("########################")

output_data_dict = process_dicom_directory( input_directory,
                                            parent_sorting_field=parent_sorting_field,
                                            output_image_name_format = output_image_name_format,
                                            output_structure_name_format = output_structure_name_format,
                                            return_extra=return_extra)

write_output_data_to_disk(  output_data_dict,
                            output_directory = output_directory,
                            output_file_suffix = output_file_suffix,
                            overwrite_existing_files = overwrite_existing_files)

logger.info("########################")
logger.info(" DICOM crawler complete")
logger.info("########################")
