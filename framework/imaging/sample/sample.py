from impit.framework.imaging.app import web_app, celery
from impit.dicom.nifti_to_rtstruct.convert import convert_nifti

from loguru import logger

import SimpleITK as sitk
import pydicom
import random
import time
import os

body_settings_defaults = {
    'outputContourName': 'primitive_body_contour',
    'regionGrowingSetting': {
        'seed': [0, 0, 0],
        'numberOfIterations': 1,
        'multiplier': 2.5,
        'initialNeighborhoodRadius': 1,
        'replaceValue': 1
    },
    'vectorRadius': [1, 1, 1]
}


@web_app.register('Primitive Body Segmentation', default_settings=body_settings_defaults)
def primitive_body_segmentation(dicom_input_path, settings):
    logger.info('Running Primitive Body Segmentation on image series in: {0}'.format(
        dicom_input_path))

    logger.info('Using settings: ' + str(settings))

    # Read the image series
    s_img_list = sitk.ImageSeriesReader().GetGDCMSeriesFileNames(dicom_input_path)
    img = sitk.ReadImage(s_img_list)

    # Basic region growing from the first voxel
    region_growing_settings = settings['regionGrowingSetting']
    seg_conf = sitk.ConfidenceConnected(img, seedList=[tuple(region_growing_settings['seed'])],
                                        numberOfIterations=region_growing_settings['numberOfIterations'],
                                        multiplier=region_growing_settings['multiplier'],
                                        initialNeighborhoodRadius=region_growing_settings[
                                            'initialNeighborhoodRadius'],
                                        replaceValue=region_growing_settings['replaceValue'])

    # Clean up the segmentation
    vectorRadius = tuple(settings['vectorRadius'])
    kernel = sitk.sitkBall
    seg_clean = sitk.BinaryMorphologicalClosing(seg_conf,
                                                vectorRadius,
                                                kernel)

    mask = sitk.BinaryNot(seg_clean)

    # Write the mask
    mask_file = os.path.join(web_app.working_dir, 'tmp.nii.gz')
    sitk.WriteImage(mask, mask_file)

    # Find a dicom file to use for the conversion to RTStruct
    dicom_file = s_img_list[0]
    logger.info('Will write Dicom using file: {0}'.format(dicom_file))
    masks = {settings['outputContourName']: mask_file}

    suid = pydicom.dcmread(dicom_file).SeriesInstanceUID

    output_file = './data/RS.{0}.dcm'.format(suid)
    convert_nifti(dicom_file, masks, output_file)

    return output_file


@web_app.register('Another Segmentation')
def another_func(dicom_input_path):
    print('The other one!')


if __name__ == "__main__":
    web_app.run(debug=True, host="0.0.0.0", port=8000)
