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
import subprocess

import logging
logger = logging.getLogger(__name__)
import SimpleITK as sitk
import pandas as pd

from platipy.backend import app, DataObject, celery


DIRQA_SETTINGS_DEFAULTS = {
    "includePointsMode": "CONTOUR",  # "CONTOUR" or "BOUNDINGBOX"
    "intensityRange": [-1024, -200],  # Range: low to high
    "contrastThreshold": 0.03,  # Param for plastimatch
    "curvatureThreshold": 172.3,  # Param for plastimatch
}


def crop_to_contour_bounding_box(img, mask):
    """
    Get the bounding box around a mask and return the image cropped to that mask
    """

    # Resample the mask (MIM send the masks at 2x the resolution)
    mask = sitk.Resample(
        mask,
        img.GetSize(),
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        img.GetOrigin(),
        img.GetSpacing(),
        img.GetDirection(),
        0,
        mask.GetPixelID(),
    )

    # Get the mask bounding box
    label_statistics_image_filter = sitk.LabelStatisticsImageFilter()
    label_statistics_image_filter.Execute(img, mask)
    bounding_box = label_statistics_image_filter.GetBoundingBox(True)

    # Crop the primary and secondary to the contour bounding box
    return img[
        bounding_box[0] : bounding_box[1] + 1,
        bounding_box[2] : bounding_box[3] + 1,
        bounding_box[4] : bounding_box[5] + 1,
    ]


@app.register("DIRQA Service", default_settings=DIRQA_SETTINGS_DEFAULTS)
def dirqa_service(data_objects, working_dir, settings):
    """
    Implements the platipy framework to provide a DIR QA service based on SIFT
    """

    logger.info("Running DIR QA")
    logger.info("Using settings: %s", settings)
    logger.info("Working Dir: %s", working_dir)

    # First figure out what data object is which
    primary = None
    secondary = None
    for data_object in data_objects:

        if "type" in data_object.meta_data:
            if data_object.meta_data["type"] == "primary":
                primary = data_object

            if data_object.meta_data["type"] == "secondary":
                secondary = data_object

    if not primary or not secondary:
        logger.error("Unable to find primary and secondary data object.")
        logger.error("Set the type on the data objects meta data.")
        return []

    logger.info("Primary: %s", primary.path)
    logger.info("Secondary: %s", secondary.path)

    # Compute SIFT point matches within each of the child contours
    # Contours with corresponding names set in metadata are expected in both the primary and
    # secondary child objects
    output_objects = []
    for primary_contour_object in primary.children:
        logger.info(f"Contour: %s", primary_contour_object.path)

        # Make sure that the 'name' is set in the meta data
        if not "name" in primary_contour_object.meta_data.keys():
            logger.error(
                "'name' not set in contour meta data. Set matching name in "
                "primary and secondary contours."
            )
            continue

        logger.info("Primary Contour: %s", primary_contour_object.meta_data['name'])

        secondary_contour_object = None
        for search_contour_object in secondary.children:

            if not "name" in search_contour_object.meta_data.keys():
                logger.error(
                    "'name' not set in contour meta data. Set matching name in "
                    "primary and secondary contours."
                )
                continue

            if search_contour_object.meta_data["name"] == primary_contour_object.meta_data["name"]:
                secondary_contour_object = search_contour_object

        if not secondary_contour_object:
            logger.error(
                "No matching contour found for %s", primary_contour_object.meta_data['name']
            )
            continue

        logger.info("Secondary Contour: %s", secondary_contour_object.meta_data['name'])

        # Read the images
        primary_path = primary.path
        if primary.type == "DICOM":
            primary_path = sitk.ImageSeriesReader().GetGDCMSeriesFileNames(primary.path)
        secondary_path = secondary.path
        if secondary.type == "DICOM":
            secondary_path = sitk.ImageSeriesReader().GetGDCMSeriesFileNames(secondary.path)
        primary_image = sitk.ReadImage(primary_path)
        secondary_image = sitk.ReadImage(secondary_path)

        # Read the contour masks
        primary_contour_mask = sitk.ReadImage(primary_contour_object.path)
        secondary_contour_mask = sitk.ReadImage(secondary_contour_object.path)

        # Crop to the contour bounding box
        primary_image = crop_to_contour_bounding_box(primary_image, primary_contour_mask)
        secondary_image = crop_to_contour_bounding_box(secondary_image, secondary_contour_mask)

        # Threshold intensities
        low_range = settings["intensityRange"][0]
        high_range = settings["intensityRange"][1]

        primary_image = sitk.Clamp(primary_image, lowerBound=low_range, upperBound=high_range)
        secondary_image = sitk.Clamp(secondary_image, lowerBound=low_range, upperBound=high_range)

        # Save cropped volumes and compute SIFT points
        primary_cropped_path = "cropped_primary.nii.gz"
        secondary_cropped_path = "cropped_secondary.nii.gz"
        sitk.WriteImage(primary_image, primary_cropped_path)
        sitk.WriteImage(secondary_image, secondary_cropped_path)

        primary_cropped_match = os.path.join(
            working_dir, "primary_{0}_match.csv".format(primary_contour_object.meta_data["name"]),
        )
        secondary_cropped_match = os.path.join(
            working_dir,
            "secondary_{0}_match.csv".format(secondary_contour_object.meta_data["name"]),
        )

        contrast_threshold = settings["contrastThreshold"]
        curvature_threshold = settings["curvatureThreshold"]

        subprocess.call(
            [
                "plastimatch",
                "sift",
                primary_cropped_path,
                secondary_cropped_path,
                "--contrast-threshold",
                str(contrast_threshold),
                "--curvature-threshold",
                str(curvature_threshold),
                "--output-match-1",
                primary_cropped_match,
                "--output-match-2",
                secondary_cropped_match,
            ]
        )

        if not os.path.exists(primary_cropped_match) or not os.path.exists(
            secondary_cropped_match
        ):
            logger.warning("No output from platimatch SIFT computation")
            continue

        # Need to negate values in dim 0 & 1 (not sure why plastimatch outputs these negated)
        primary_points = pd.read_csv(primary_cropped_match, header=None)
        secondary_points = pd.read_csv(secondary_cropped_match, header=None)

        primary_points[1] = -primary_points[1]
        primary_points[2] = -primary_points[2]
        secondary_points[1] = -secondary_points[1]
        secondary_points[2] = -secondary_points[2]

        # Prefix point names with structure name
        primary_points[0] = (
            primary_contour_object.meta_data["name"] + "_" + primary_points[0].astype(str)
        )
        secondary_points[0] = (
            secondary_contour_object.meta_data["name"] + "_" + secondary_points[0].astype(str)
        )

        if settings["includePointsMode"] == "CONTOUR":
            # Filter out points which fall outside of contour
            logger.info("Filtering out points outside the contour")

            remove_point_names = []
            for point in primary_points.iterrows():
                phys_point = list(point[1][1:4])
                mask_point = primary_contour_mask.TransformPhysicalPointToIndex(phys_point)
                is_in_contour = primary_contour_mask[mask_point]

                if not is_in_contour:
                    remove_point_names.append(point[1][0])

            for point in secondary_points.iterrows():
                phys_point = list(point[1][1:4])
                mask_point = secondary_contour_mask.TransformPhysicalPointToIndex(phys_point)
                is_in_contour = secondary_contour_mask[mask_point]

                if not is_in_contour:
                    remove_point_names.append(point[1][0])

            primary_points = primary_points[~primary_points[0].isin(remove_point_names)]
            secondary_points = secondary_points[~secondary_points[0].isin(remove_point_names)]

        # Save the updated points
        primary_points.to_csv(primary_cropped_match, index=False, header=None)
        secondary_points.to_csv(secondary_cropped_match, index=False, header=None)

        # Create the output Data Object and add it to output_objects
        primary_output_object = DataObject(type="FILE", path=primary_cropped_match, parent=primary)
        secondary_output_object = DataObject(
            type="FILE", path=secondary_cropped_match, parent=secondary
        )
        output_objects.append(primary_output_object)
        output_objects.append(secondary_output_object)

        os.remove(primary_cropped_path)
        os.remove(secondary_cropped_path)

    logger.info("Finished DIR QA")

    return output_objects


if __name__ == "__main__":

    # Run app by calling "python service.py" from the command line

    DICOM_LISTENER_PORT = 7777
    DICOM_LISTENER_AETITLE = "DIRQA_SERVICE"

    app.run(
        debug=True,
        host="0.0.0.0",
        port=8001,
        dicom_listener_port=DICOM_LISTENER_PORT,
        dicom_listener_aetitle=DICOM_LISTENER_AETITLE,
    )
