from impit.framework.imaging.app import web_app, register, celery
from impit.dicom.nifti_to_rtstruct.convert import convert_nifti

from loguru import logger

import SimpleITK as sitk
import pydicom
import random
import time
import os

@register('Primitive Body Segmentation')
def my_func(dicom_input_path):
    logger.info('Running Primitive Body Segmentation on image series in: {0}'.format(dicom_input_path))

    logger.info(web_app.working_dir)

    # Read the image series
    s_img_list = sitk.ImageSeriesReader().GetGDCMSeriesFileNames(dicom_input_path)
    img = sitk.ReadImage(s_img_list)

    # Basic region growing from the first voxel
    seg_conf = sitk.ConfidenceConnected(img, seedList=[(0,0,0)],
                                    numberOfIterations=1,
                                    multiplier=2.5,
                                    initialNeighborhoodRadius=1,
                                    replaceValue=1)

    # Clean up the segmentation
    vectorRadius = (1, 1, 1)
    kernel = sitk.sitkBall
    seg_clean = sitk.BinaryMorphologicalClosing(seg_conf,
                                            vectorRadius,
                                            kernel)

    mask = sitk.BinaryNot(seg_clean)

    # Write the mask
    mask_file = os.path.join(web_app.working_dir,'tmp.nii.gz')
    sitk.WriteImage(mask, mask_file)

    # Find a dicom file to use for the conversion to RTStruct
    dicom_file = s_img_list[0]
    logger.info('Will write Dicom using file: {0}'.format(dicom_file))
    masks = {'primitive_body': mask_file}

    suid = pydicom.dcmread(dicom_file).SeriesInstanceUID

    output_file = './data/RS.{0}.dcm'.format(suid)
    convert_nifti(dicom_file, masks, output_file)

    return output_file


@register('Another Segmentation')
def another_func(dicom_input_path):
    print('The other one!')

if __name__== "__main__":
    web_app.run(debug=True, host="0.0.0.0", port=8000)
