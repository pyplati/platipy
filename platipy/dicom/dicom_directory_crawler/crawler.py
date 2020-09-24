import re

import pydicom
import SimpleITK as sitk

from loguru import logger

import pathlib

from platipy.dicom.dicom_directory_crawler.conversion_utils import (
    sort_dicom_image_list,
    transform_point_set_from_dicom_struct,
    get_dicom_info_from_description,
)

output_image_name_format = '{patient_name}_{index}_{image_modality}_{image_desc}'
output_structure_name_format = '{patient_name}_{index}_{image_modality}_{structure}'
study_name_from_uid = True # !TO DO implement this


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

    dicom_file = dicom_file.as_posix()

    dicom_object = pydicom.read_file(dicom_file, force=True)
    # Take out any non-alphanumeric characters for safety
    patient_name = re.sub(r'[^\w]', '_', str(dicom_object.PatientName)).upper()

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

        sorted_file_list = [filename for filename in sort_dicom_image_list(dicom_file_list)]
        image = sitk.ReadImage(sorted_file_list)

        """
        ! TO DO - integrity check
            Read in all the files here, check the slice location and determine if any are missing
        """

        """
        Retrieve some information
        This will be used to populate fields in the output name
        """

        image_modality = initial_dicom.Modality
        image_desc = get_dicom_info_from_description(initial_dicom)
        acq_date = initial_dicom.AcquisitionDate

        output_name = output_image_name_format.format(patient_name=patient_name,
                                                index=0,
                                                image_modality=image_modality,
                                                image_desc=image_desc,
                                                acq_date = acq_date)

        logger.info(f'      Image name: {output_name}')

        if 'IMAGES' not in output_data_dict.keys():
            # Make a new entry
            output_data_dict['IMAGES'] = {output_name:image}

        else:
            # First check if there is another image of the same name

            if output_name not in output_data_dict['IMAGES'].keys():
                output_data_dict['IMAGES'][output_name] = image

            else:
                logger.info('      An image with this name exists, appending.')
                if type(output_data_dict['IMAGES'][output_name]) != list:
                    output_data_dict['IMAGES'][output_name] = list(output_data_dict['IMAGES'][output_name])

                output_data_dict['IMAGES'][output_name].append(image)

    if 'Structure' in initial_dicom_sop_class_name:
        # Load as an RT structure set
        # This should be done individually for each file

        logger.info(f'      Number of files: {len(dicom_file_list)}')
        for index, dicom_file in enumerate(dicom_file_list):
            dicom_object = pydicom.read_file(dicom_file, force=True)

            # We must also read in the corresponding DICOM image
            # This can be found by matching the 

            """
            ! TO DO
            What happens if there is an RT structure set with different referenced sequences?
            """

            # Get the "ReferencedFrameOfReferenceSequence", first item
            referenced_frame_of_reference_item = dicom_object.ReferencedFrameOfReferenceSequence[0]

            # Get the "RTReferencedStudySequence", first item
            # This retrieves the study UID
            # This might be useful, but would typically match the actual StudyInstanceUID in the DICOM object
            rt_referenced_series_item = referenced_frame_of_reference_item.RTReferencedStudySequence[0]

            # Get the "RTReferencedSeriesSequence", first item
            # This retreives the actual referenced series UID, which we need to match imaging parameters
            rt_referenced_series_again_item = rt_referenced_series_item.RTReferencedSeriesSequence[0]

            # Get the appropriate series instance UID
            image_series_uid = rt_referenced_series_again_item.SeriesInstanceUID
            logger.info(f'      Item {index}: Matched SeriesInstanceUID = {image_series_uid}')

            # Read in the corresponding image
            sorted_file_list = sort_dicom_image_list( dicom_dict[image_series_uid] )
            image = sitk.ReadImage(sorted_file_list)

            structure_name_list, structure_image_list = transform_point_set_from_dicom_struct(image, dicom_object)

            print(structure_name_list)

            for structure_name, structure_image in zip(structure_name_list, structure_image_list)

            output_name = output_structure_name_format.format(patient_name=patient_name,
                                        index=0,
                                        image_modality=image_modality,
                                        acq_date = acq_date)