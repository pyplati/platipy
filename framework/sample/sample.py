from impit.framework.app import web_app, DataObject
from impit.dicom.nifti_to_rtstruct.convert import convert_nifti

from loguru import logger

import SimpleITK as sitk
import pydicom
import os

body_settings_defaults = {
    'outputContourName': 'primitive_body_contour',
    'seed': [0, 0, 0],
    'lowerThreshold': -5000,
    'upperThreshold': -800,
    'vectorRadius': [1, 1, 1],
}


@web_app.register('Primitive Body Segmentation', default_settings=body_settings_defaults)
def primitive_body_segmentation(data_objects, working_dir, settings):

    logger.info('Running Primitive Body Segmentation')
    logger.info('Using settings: ' + str(settings))

    output_objects = []
    for d in data_objects:
        logger.info('Running on data object: ' + d.path)

        # Read the image series
        load_path = d.path
        if d.type == 'DICOM':
            load_path = sitk.ImageSeriesReader().GetGDCMSeriesFileNames(d.path)

        img = sitk.ReadImage(load_path)

        # Region growing using Connected Threshold Image Filter
        seg_con = sitk.ConnectedThreshold(img,
                                          seedList=[tuple(settings['seed'])],
                                          lower=settings['lowerThreshold'],
                                          upper=settings['upperThreshold'])

        # Clean up the segmentation
        vectorRadius = tuple(settings['vectorRadius'])
        kernel = sitk.sitkBall
        seg_clean = sitk.BinaryMorphologicalClosing(seg_con,
                                                    vectorRadius,
                                                    kernel)
        mask = sitk.BinaryNot(seg_clean)

        # Write the mask to a file in the working_dir
        mask_file = os.path.join(
            working_dir, '{0}.nii.gz'.format(settings['outputContourName']))
        sitk.WriteImage(mask, mask_file)

        # Create the output Data Object and add it to the list of output_objects
        do = DataObject(type='FILE', path=mask_file, parent=d)
        output_objects.append(do)

        # If the input was a DICOM, then we can use it to generate an output RTStruct
        if d.type == 'DICOM':

            dicom_file = load_path[0]
            logger.info('Will write Dicom using file: {0}'.format(dicom_file))
            masks = {settings['outputContourName']: mask_file}

            # Use the image series UID for the file of the RTStruct
            suid = pydicom.dcmread(dicom_file).SeriesInstanceUID
            output_file = os.path.join(working_dir, 'RS.{0}.dcm'.format(suid))

            # Use the convert nifti function to generate RTStruct from nifti masks
            convert_nifti(dicom_file, masks, output_file)

            # Create the Data Object for the RTStruct and add it to the list
            do = DataObject(type='DICOM', path=output_file, parent=d)
            output_objects.append(do)

            logger.info('RTStruct generated')

    return output_objects


if __name__ == "__main__":

    # Run app by calling "python sample.py" from the command line

    dicom_listener_port=7777
    dicom_listener_aetitle="SAMPLE_SERVICE"

    web_app.run(debug=True, host="0.0.0.0", port=8000,
        dicom_listener_port=dicom_listener_port,
        dicom_listener_aetitle=dicom_listener_aetitle)
