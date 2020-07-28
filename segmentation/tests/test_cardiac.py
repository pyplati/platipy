"""
Tests for cardiac atlas
"""
# pylint: disable=redefined-outer-name

import os
import tempfile

import pytest

import SimpleITK as sitk
import numpy as np

from impit.segmentation.tests.pull_data import get_lung_data

os.environ["ATLAS_PATH"] = os.path.join(os.path.dirname(__file__), "data", "lung")

from impit.segmentation.cardiac.service import cardiac_service, CARDIAC_SETTINGS_DEFAULTS
from impit.framework.models import DataObject


@pytest.fixture
def cardiac_data():
    """Gets the data needed for the cardiac tests

    Returns:
        dict -- Data for cardiac test
    """
    return get_lung_data(number_of_patients=5)


def test_cardiac_service(cardiac_data):
    """An end-to-end test to check that the cardiac service is running as expected
    """

    working_dir = tempfile.mkdtemp()

    patient_ids = list(cardiac_data.keys())

    # Create a data object to be segmented
    data_object = DataObject()
    data_object.id = 1
    data_object.path = os.path.join(cardiac_data[patient_ids[0]], "CT.nii.gz")
    data_object.type = "FILE"
    print(cardiac_data)

    test_settings = CARDIAC_SETTINGS_DEFAULTS
    test_settings["atlasSettings"]["atlasIdList"] = [
        os.path.basename(patient_ids[1]),
        os.path.basename(patient_ids[2]),
        os.path.basename(patient_ids[3]),
    ]
    test_settings["atlasSettings"]["atlasStructures"] = ["Heart"]
    test_settings["atlasSettings"]["atlasPath"] = os.path.dirname(cardiac_data[patient_ids[1]])

    # Run the service function
    result = cardiac_service([data_object], working_dir, CARDIAC_SETTINGS_DEFAULTS)
    print(result)
    # Should have returned two output objects
    assert len(result) == 2

    # lung_mask = sitk.ReadImage(result[0].path)
    # assert_lung_mask(lung_mask)


# CARDIAC_SETTINGS_DEFAULTS = {
#     "outputFormat": "Auto_{0}.nii.gz",
#     "atlasSettings": {
#         "atlasIdList": ["08", "11", "12", "13", "14"],
#         "atlasStructures": ["WHOLEHEART", "LANTDESCARTERY"],
#         # For development, run: 'export ATLAS_PATH=/atlas/path'
#         "atlasPath": os.environ["ATLAS_PATH"],
#     },
#     "lungMaskSettings": {
#         "coronalExpansion": 15,
#         "axialExpansion": 5,
#         "sagittalExpansion": 0,
#         "lowerNormalisedThreshold": -0.1,
#         "upperNormalisedThreshold": 0.4,
#         "voxelCountThreshold": 5e4,
#     },
#     "rigidSettings": {
#         "initialReg": "Affine",
#         "options": {
#             "shrinkFactors": [8, 4, 2, 1],
#             "smoothSigmas": [8, 4, 1, 0],
#             "samplingRate": 0.25,
#             "finalInterp": sitk.sitkBSpline,
#         },
#         "trace": True,
#         "guideStructure": False,
#     },
#     "deformableSettings": {
#         "resolutionStaging": [16, 4, 2, 1],
#         "iterationStaging": [20, 10, 10, 10],
#         "ncores": 8,
#         "trace": True,
#     },
#     "IARSettings": {
#         "referenceStructure": "WHOLEHEART",
#         "smoothDistanceMaps": True,
#         "smoothSigma": 1,
#         "zScoreStatistic": "MAD",
#         "outlierMethod": "IQR",
#         "outlierFactor": 1.5,
#         "minBestAtlases": 4,
#     },
#     "labelFusionSettings": {"voteType": "local", "optimalThreshold": {"WHOLEHEART": 0.44}},
#     "vesselSpliningSettings": {
#         "vesselNameList": ["LANTDESCARTERY"],
#         "vesselRadius_mm": {"LANTDESCARTERY": 2.2},
#         "spliningDirection": {"LANTDESCARTERY": "z"},
#         "stopCondition": {"LANTDESCARTERY": "count"},
#         "stopConditionValue": {"LANTDESCARTERY": 1},
#     },
# }
