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

# pylint: disable=redefined-outer-name

import os
import tempfile

import pytest

import SimpleITK as sitk

from platipy.imaging.tests.pull_data import fetch_data

# os.environ["ATLAS_PATH"] = os.path.join(os.path.dirname(__file__), "data", "lung")

from platipy.imaging.projects.cardiac.service import (
    cardiac_service,
    CARDIAC_SETTINGS_DEFAULTS,
)
from platipy.backend.models import DataObject


@pytest.fixture
def cardiac_data():
    """Gets the data needed for the cardiac tests

    Returns:
        dict -- Data for cardiac test
    """
    return fetch_data(
        "LCTSC",
        patient_ids=[
            "LCTSC-Train-S1-001",
            "LCTSC-Train-S1-002",
            "LCTSC-Train-S1-003",
            "LCTSC-Train-S1-004",
            "LCTSC-Train-S1-005",
        ],
    )


def test_cardiac_service(cardiac_data):
    """An end-to-end test to check that the cardiac service is running as expected
    """

    working_dir = tempfile.mkdtemp()

    patient_ids = list(cardiac_data.keys())

    # Create a data object to be segmented
    data_object = DataObject()
    data_object.id = 1
    data_object.path = os.path.join(cardiac_data[patient_ids[4]], "CT.nii.gz")
    data_object.type = "FILE"
    atlas_cases = list(cardiac_data.keys())[:4]

    atlas_dir = tempfile.mkdtemp()
    for atlas_case in atlas_cases:

        atlas_case_dir = os.path.join(atlas_dir, atlas_case)
        os.makedirs(atlas_case_dir)

        ct_image_file = os.path.join(cardiac_data[atlas_case], "CT.nii.gz")
        image = sitk.ReadImage(ct_image_file)

        left_lung_file = os.path.join(cardiac_data[atlas_case], "Struct_Lung_L.nii.gz")
        right_lung_file = os.path.join(cardiac_data[atlas_case], "Struct_Lung_R.nii.gz")
        left_lung = sitk.ReadImage(left_lung_file)
        right_lung = sitk.ReadImage(right_lung_file)

        lungs = left_lung + right_lung

        label_stats_image_filter = sitk.LabelStatisticsImageFilter()
        label_stats_image_filter.Execute(image, lungs)
        bounding_box = list(label_stats_image_filter.GetBoundingBox(1))
        index = [bounding_box[x * 2] for x in range(3)]
        size = [bounding_box[(x * 2) + 1] - bounding_box[x * 2] for x in range(3)]

        cropped_image = sitk.RegionOfInterest(image, size=size, index=index)

        sitk.WriteImage(cropped_image, os.path.join(atlas_case_dir, "CT.nii.gz"))

        for struct in os.listdir(cardiac_data[atlas_case]):

            if not struct.startswith("Struct_"):
                continue

            mask = sitk.ReadImage(os.path.join(cardiac_data[atlas_case], struct))

            cropped_mask = sitk.RegionOfInterest(mask, size=size, index=index)

            sitk.WriteImage(cropped_mask, os.path.join(atlas_case_dir, struct))

    test_settings = CARDIAC_SETTINGS_DEFAULTS
    test_settings["atlasSettings"]["atlasIdList"] = [
        os.path.basename(patient_ids[0]),
        os.path.basename(patient_ids[1]),
        os.path.basename(patient_ids[2]),
    ]

    test_settings["atlasSettings"]["atlasPath"] = atlas_dir
    test_settings["atlasSettings"]["atlasStructures"] = ["Heart", "Lung_L", "Lung_R"]
    test_settings["atlasSettings"]["atlasIdList"] = atlas_cases
    test_settings["atlasSettings"]["atlasImageFormat"] = "{0}/CT.nii.gz"
    test_settings["atlasSettings"]["atlasLabelFormat"] = "{0}/Struct_{1}.nii.gz"

    # Run the DIR a bit more than default
    test_settings["deformableSettings"]["iterationStaging"] = [15, 15, 15]

    # Run the IAR using the heart
    test_settings["IARSettings"]["referenceStructure"] = "Heart"

    # Set the threshold
    test_settings["labelFusionSettings"]["optimalThreshold"] = {
        "Heart": 0.5,
        "Lung_L": 0.5,
        "Lung_R": 0.5,
    }

    # No vessels
    test_settings["vesselSpliningSettings"]["vesselNameList"] = []

    # Run the service function
    results = cardiac_service([data_object], working_dir, CARDIAC_SETTINGS_DEFAULTS)

    # Should have returned three output objects (Heart, Lung_L, Lung_R)
    assert len(results) == 3

    # Check the results are somewhat similar to the ground truth for
    # this case. We don't expect good results here since the settings
    # are set so low for this to run relatively quickly
    for result in results:

        auto_file = result.path
        auto_mask = sitk.ReadImage(auto_file)
        struct = os.path.basename(auto_file)

        gt_file = os.path.join(cardiac_data[patient_ids[4]], f"Struct_{struct}")
        gt_mask = sitk.ReadImage(gt_file)

        label_overlap_filter = sitk.LabelOverlapMeasuresImageFilter()
        label_overlap_filter.Execute(auto_mask, gt_mask)
        print(f"{struct}: {label_overlap_filter.GetDiceCoefficient()}")

        # Let's just check that the DSC is resonable (> 0.8)
        assert label_overlap_filter.GetDiceCoefficient() > 0.8
