"""
Service to run bronchus segmentation.
"""
import os

# import pydicom

import SimpleITK as sitk
from loguru import logger

from impit.framework import app, DataObject, celery

# from impit.dicom.nifti_to_rtstruct.convert import convert_nifti

from impit.segmentation.bronchus.bronchus import (
    generate_lung_mask,
    generate_airway_mask,
    default_settings
)


BRONCHUS_SETTINGS_DEFAULTS = {
    "outputBronchusName": "Auto_Bronchus",
    "outputLungName": "Auto_Lung",
    "algorithmSettings": default_settings
}


@app.register("Bronchus Segmentation", default_settings=BRONCHUS_SETTINGS_DEFAULTS)
def bronchus_service(data_objects, working_dir, settings):
    """
    Implements the impit framework to provide bronchus segmentation.
    """

    logger.info("Running Bronchus Segmentation")

    output_objects = []
    for data_object in data_objects:
        logger.info("Running on data object: " + data_object.path)

        # Read the image series
        load_path = data_object.path
        if data_object.type == "DICOM":
            load_path = sitk.ImageSeriesReader().GetGDCMSeriesFileNames(
                data_object.path
            )

        img = sitk.ReadImage(load_path)

        # Compute the lung mask
        lung_mask = generate_lung_mask(img)

        lung_mask_file = os.path.join(
            working_dir, "{0}.nii.gz".format(settings["outputLungName"])
        )
        sitk.WriteImage(lung_mask, lung_mask_file)

        output_data_object = DataObject(
            type="FILE", path=lung_mask_file, parent=data_object
        )
        output_objects.append(output_data_object)

        bronchus_mask = generate_airway_mask(working_dir,
                                             img,
                                             lung_mask,
                                             config_dict=settings["algorithmSettings"])

        # If the bronchus mask counldn't be generated then skip it
        if not bronchus_mask:
            continue

        # Write the mask to a file in the working_dir
        bronchus_mask_file = os.path.join(
            working_dir, "{0}.nii.gz".format(settings["outputBronchusName"])
        )
        sitk.WriteImage(bronchus_mask, bronchus_mask_file)

        # Create the output Data Object and add it to the list of output_objects
        output_data_object = DataObject(
            type="FILE", path=bronchus_mask_file, parent=data_object
        )
        output_objects.append(output_data_object)

        # If the input was a DICOM, then we can use it to generate an output RTStruct
        # if d.type == 'DICOM':

        #     dicom_file = load_path[0]
        #     logger.info('Will write Dicom using file: {0}'.format(dicom_file))
        #     masks = {settings['outputContourName']: mask_file}

        #     # Use the image series UID for the file of the RTStruct
        #     suid = pydicom.dcmread(dicom_file).SeriesInstanceUID
        #     output_file = os.path.join(working_dir, 'RS.{0}.dcm'.format(suid))

        #     # Use the convert nifti function to generate RTStruct from nifti masks
        #     convert_nifti(dicom_file, masks, output_file)

        #     # Create the Data Object for the RTStruct and add it to the list
        #     do = DataObject(type='DICOM', path=output_file, parent=d)
        #     output_objects.append(do)

        #     logger.info('RTStruct generated')

    return output_objects


if __name__ == "__main__":

    # Run app by calling "python bronchus.py" from the command line

    DICOM_LISTENER_PORT = 7777
    DICOM_LISTENER_AETITLE = "BRONCHUS_SERVICE"

    app.run(
        debug=True,
        host="0.0.0.0",
        port=8000,
        dicom_listener_port=DICOM_LISTENER_PORT,
        dicom_listener_aetitle=DICOM_LISTENER_AETITLE,
    )
