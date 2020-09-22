import pydicom
import SimpleITK as sitk

from loguru import logger
from skimage.draw import polygon
import numpy as np

import pathlib
import sys

logger.info('#######################')
logger.info(' Running DICOM crawler ')
logger.info('#######################')

"""
Search for all DICOM files on the user-given path
"""

root_path = pathlib.Path("./")
dicom_list = list(root_path.glob("**/*.dcm"))

logger.info(f'Found {len(dicom_list)} DICOM files, analysing.')
# for i, dicom_file in enumerate(dicom_list):
#     logger.debug(f'{i}, {dicom_file}')

"""
Organise the DICOM files by the series UID
"""
dicom_dict = {}
for i, dicom_file in enumerate(sorted(dicom_list)):
    logger.debug(f'  Sorting file {i}')
    # Get the patient name
    dicom_object = pydicom.read_file(dicom_file, force=True)
    patient_name = dicom_object.PatientName

    if patient_name == '':
        # We will set a default name
        patient_name = 'STUDY_001'

    study_uid = dicom_object.StudyInstanceUID
    series_uid = dicom_object.SeriesInstanceUID

    if series_uid not in dicom_dict.keys():
        dicom_dict[series_uid] = [dicom_file]

    else:
        dicom_dict[series_uid].append(dicom_file)

"""
Provide a summary
"""
logger.info('#######################')
logger.info(' Analysis complete:')
logger.info(f'  Number of uniques series UIDs: {len(dicom_dict.keys())}')


"""
Go through each unique series UID
    1. Determine the type of DICOM file (image, RT-struct, etc.), technically the SOP Class UID
    2. Generate an image where appropriate
        a. For CT/MR/PET, by stacking the individual DICOM slices
            i. Integrity check: ensure all slices are present
                If not, raise a warning and interpolate
        b. For RT-Struct, by generating images from the polygonal contours
            i. Integrity check: ensure contouring is performed continuously
                If not, raise a warning and interpolate contour position from surrounding slices
    3. 
"""

def sort_dicom_image_list(dicom_image_list, sort_by='SliceLocation'):
    """
    Sorts a list of DICOM image files based on a DICOM tag value

    Args:
        dicom_image_list (list): [description]
        sort_by (str, optional): [description]. Defaults to 'SliceLocation'.
    """
    sorter_float = lambda dcm_file: float(pydicom.read_file(dcm_file, force=True)[sort_by].value)

    return sorted(dicom_image_list, key=sorter_float)


output_data_dict = {}

for series_uid, dicom_file_list in sorted(dicom_dict.items()):
    logger.info(f'  Processing series UID: {series_uid}')

    initial_dicom = pydicom.read_file(dicom_file_list[0])
    initial_dicom_sop_class_uid = initial_dicom.SOPClassUID
    initial_dicom_sop_class_name = pydicom._uid_dict.UID_dictionary[initial_dicom_sop_class_uid][0]

    logger.info(f'    SOP Class name: {initial_dicom_sop_class_name}')

    # Check the potential types of DICOM files
    if 'Image' in initial_dicom_sop_class_name:
        # Load as an image

        sorted_file_list = [filename.as_posix() for filename in sort_dicom_image_list(dicom_file_list)]
        image = sitk.ReadImage(sorted_file_list)

        """
        ! TO DO - integrity check
            Read in all the files here, check the slice location and determine if any are missing
        """

        if 'IMAGES' not in output_data_dict.keys():
            output_data_dict['IMAGES'] = {series_uid:image}

        else:
            output_data_dict['IMAGES'][series_uid] = image

    if ''