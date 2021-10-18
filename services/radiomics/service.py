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

from loguru import logger

import SimpleITK as sitk
import numpy as np
import pandas as pd

from radiomics import firstorder, shape, glcm, glrlm, glszm, ngtdm, gldm, imageoperations
from radiomics_custom import RadiomicsCustom

from platipy.backend import app, DataObject, celery  # pylint: disable=unused-import
from platipy.backend.api import Resource, api

AVAILABLE_RADIOMICS = {
    "firstorder": firstorder.RadiomicsFirstOrder,
    "shape": shape.RadiomicsShape,
    "glcm": glcm.RadiomicsGLCM,
    "glrlm": glrlm.RadiomicsGLRLM,
    "glszm": glszm.RadiomicsGLSZM,
    "ngtdm": ngtdm.RadiomicsNGTDM,
    "gldm": gldm.RadiomicsGLDM,
    "custom": RadiomicsCustom,
}

RADIOMICS_SETTINGS = {
    "contours": [],  # If empty will extract radiomics for all contours
    "radiomics": {},  # If empty will extract all first order radiomics
    "pyradiomics_settings": {
        "binWidth": 25,
        "resampledPixelSpacing": None,
        "interpolator": "sitkNearestNeighbor",
        "verbose": True,
        "removeOutliers": 10000,
    },
    "resample_to_image": False,  # If true, mask will be resampled to spacing of image
    "append_histogram": False,  # If true, histogram will be appended to the end of each output row
    "histogram_bins": 256,  # Used only if append_histogram is true
}


def compute_histogram(image, mask, bins):
    """
    Computes histogram for image values within a given mask

    image: A SimpleITK image object
    mask: A SimpleITK mask object

    returns: Tuple of histogram values and bin edges
    """

    np_im = sitk.GetArrayFromImage(image)
    np_ma = sitk.GetArrayFromImage(mask)

    masked_im = np_im[np.where(np_ma)]

    return np.histogram(masked_im, bins=bins)


@app.register("PyRadiomics Extractor", default_settings=RADIOMICS_SETTINGS)
def pyradiomics_extractor(data_objects, working_dir, settings):
    """Run to extract radiomics from data objects

    Args:
        data_objects (list): List of data objects to process
        working_dir (str): Path to directory used for working
        settings ([type]): The settings to use for processing radiomics

    Returns:
        list: List of output data objects
    """

    logger.info("Running PyRadiomics Extract")
    logger.info("Using settings: " + str(settings))

    pyrad_settings = settings["pyradiomics_settings"]

    # If no Radiomics are supplied then extract for all first order radiomics
    if len(settings["radiomics"].keys()) == 0:
        features = firstorder.RadiomicsFirstOrder.getFeatureNames()
        settings["radiomics"] = {"firstorder": [f for f in features if not features[f]]}

    results = None
    meta_data_cols = [("", "Contour")]
    for data_obj in data_objects:

        try:
            if len(data_obj.children) > 0:

                logger.info("Running on data object: " + data_obj.path)

                # Read the image series
                load_path = data_obj.path
                if data_obj.type == "DICOM":
                    load_path = sitk.ImageSeriesReader().GetGDCMSeriesFileNames(data_obj.path)

                # Children of Image Data Object are masks, compute PyRadiomics for all of them!
                output_frame = pd.DataFrame()
                for child_obj in data_obj.children:

                    contour_name = child_obj.path.split("/")[-1].split(".")[0]
                    if len(settings["contours"]) > 0 and not contour_name in settings["contours"]:
                        # If a contour list is provided and this contour isn't in the list then
                        # skip it
                        logger.debug("Skipping Contour: ", contour_name)
                        continue

                    # Reload the image for each new contour in case resampling is occuring,
                    # should start fresh each time.
                    image = sitk.ReadImage(load_path)
                    mask = sitk.ReadImage(child_obj.path)

                    logger.debug("Image Origin: " + str(image.GetOrigin()))
                    logger.debug("Mask Origin: " + str(mask.GetOrigin()))
                    logger.debug("Image Direction: " + str(image.GetDirection()))
                    logger.debug("Mask Direction: " + str(mask.GetDirection()))
                    logger.debug("Image Size: " + str(image.GetSize()))
                    logger.debug("Mask Size: " + str(mask.GetSize()))

                    logger.info(child_obj.path)

                    interpolator = pyrad_settings.get("interpolator")
                    resample_pixel_spacing = pyrad_settings.get("resampledPixelSpacing")

                    if settings["resample_to_image"]:
                        logger.info("Will resample to spacing of image")
                        resample_pixel_spacing = list(image.GetSpacing())
                        pyrad_settings["resampledPixelSpacing"] = resample_pixel_spacing

                    if interpolator is not None and resample_pixel_spacing is not None:
                        logger.info("Resampling Image and Mask")
                        image, mask = imageoperations.resampleImage(image, mask, **pyrad_settings)

                    # output[contour_name] = {"Contour": contour_name}
                    df_contour = pd.DataFrame()

                    logger.info("Computing Radiomics for contour: {0}", contour_name)

                    for rad in settings["radiomics"].keys():

                        logger.info("Computing {0} radiomics".format(rad))

                        if rad not in AVAILABLE_RADIOMICS.keys():
                            logger.warning("Radiomic Class not found: {0}", rad)
                            continue

                        radiomics_obj = AVAILABLE_RADIOMICS[rad]

                        features = radiomics_obj(image, mask, **pyrad_settings)

                        features.disableAllFeatures()

                        # All features seem to be computed if all are disabled (possible
                        # pyradiomics bug?). Skip if all features in a class are disabled.
                        if len(settings["radiomics"][rad]) == 0:
                            continue

                        for feature in settings["radiomics"][rad]:
                            try:
                                features.enableFeatureByName(feature, True)
                            except LookupError:
                                # Feature not available in this set
                                logger.warning("Feature not found: {0}", feature)

                        feature_result = features.execute()
                        feature_result = dict(
                            ((rad, key), value) for (key, value) in feature_result.items()
                        )
                        df_feature_result = pd.DataFrame(feature_result, index=[contour_name])

                        # Merge the results
                        df_contour = pd.concat([df_contour, df_feature_result], axis=1)

                    df_contour[("", "Contour")] = contour_name
                    output_frame = pd.concat([output_frame, df_contour])

                    # Add the meta data for this contour if there is any
                    if child_obj.meta_data:
                        for key in child_obj.meta_data:

                            col_key = ("", key)

                            output_frame[col_key] = child_obj.meta_data[key]

                            if col_key not in meta_data_cols:
                                meta_data_cols.append(col_key)

                # Add Image Series Data Object's Meta Data to the table
                if data_obj.meta_data:
                    for key in data_obj.meta_data.keys():

                        col_key = ("", key)

                        output_frame[col_key] = pd.Series(
                            [data_obj.meta_data[key] for p in range(len(output_frame.index))],
                            index=output_frame.index,
                        )

                        if col_key not in meta_data_cols:
                            meta_data_cols.append(col_key)

                if results is None:
                    results = output_frame
                else:
                    results = results.append(output_frame)
        except Exception as exception:  # pylint: disable=broad-except
            logger.error("An Error occurred while computing the Radiomics: {0}", exception)

    # Set the order of the columns output
    cols = results.columns.tolist()
    new_cols = list(meta_data_cols)
    new_cols += [c for c in cols if not c in meta_data_cols]
    results = results[new_cols]

    # Write output to file
    output_file = os.path.join(working_dir, "output.csv")
    results = results.reset_index()
    results = results.drop(columns=["index"])
    results.to_csv(output_file)
    logger.info("Radiomics written to {0}".format(output_file))

    # Create the output Data Object and add it to output_objects
    data_object = DataObject(type="FILE", path=output_file)
    output_objects = [data_object]

    return output_objects


class RadiomicsEndpoint(Resource):
    def get(self):

        logger.info(firstorder.RadiomicsFirstOrder.getFeatureNames())

        result = {}
        for rad in AVAILABLE_RADIOMICS:
            features = AVAILABLE_RADIOMICS[rad].getFeatureNames()
            result[rad] = [f for f in features if not features[f]]

        return result


api.add_resource(RadiomicsEndpoint, "/api/radiomics")

if __name__ == "__main__":

    # Run app by calling "python service.py" from the command line

    DICOM_LISTENER_PORT = 7777
    DICOM_LISTENER_AETITLE = "RADIOMICS_EXTRACT_SERVICE"

    app.run(
        debug=True,
        host="0.0.0.0",
        port=8000,
        dicom_listener_port=DICOM_LISTENER_PORT,
        dicom_listener_aetitle=DICOM_LISTENER_AETITLE,
    )
