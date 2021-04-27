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

import tempfile
from pathlib import Path

import pytest

import SimpleITK as sitk
import numpy as np

from platipy.imaging.projects.cardiac.run import (
    run_cardiac_segmentation,
    CARDIAC_SETTINGS_DEFAULTS,
)
from platipy.imaging.projects.cardiac.run_structureguided import (
    run_cardiac_segmentation_structure_guided,
    CARDIAC_STRUCTURE_GUIDED_SETTINGS_DEFAULTS,
)


def insert_sphere(arr, sp_radius=4, sp_centre=(0, 0, 0)):
    """Insert a sphere into the give array

    Args:
        arr (np.array): Array in which to insert sphere
        sp_radius (int, optional): The radius of the sphere. Defaults to 4.
        sp_centre (tuple, optional): The position at which the sphere should be inserted. Defaults
                                     to (0, 0, 0).

    Returns:
        np.array: An array with the sphere inserted
    """

    arr_copy = arr[:]

    x, y, z = np.indices(arr.shape)

    arr_copy[
        (x - sp_centre[0]) ** 2 + (y - sp_centre[1]) ** 2 + (z - sp_centre[2]) ** 2
        <= sp_radius ** 2
    ] = 1

    return arr_copy


@pytest.fixture
def cardiac_data():
    """Generates the data needed for the cardiac tests

    Returns:
        dict -- Data for cardiac test
    """

    # Generate 5 pseudo CT images and the wholeheart masks
    data = {}
    for i in range(5):

        case_id = str(i + 1).zfill(3)

        ct_arr = np.ones((60, 128, 128)) * -1000
        mask_arr = np.zeros((60, 128, 128))
        submask_arr = np.zeros((60, 128, 128))

        ct_arr = insert_sphere(ct_arr, sp_radius=25, sp_centre=(30 + i, 64 + i, 64))
        mask_arr = insert_sphere(mask_arr, sp_radius=25, sp_centre=(30 + i, 64 + i, 64))
        submask_arr = insert_sphere(submask_arr, sp_radius=5, sp_centre=(30 + i, 60 + i, 60))

        ct = sitk.GetImageFromArray(ct_arr)
        ct.SetSpacing((0.9 + i * 0.01, 0.9 + i * 0.01, 2.5 + i * 0.01))
        ct.SetOrigin((320, -52, 60))

        mask = sitk.GetImageFromArray(mask_arr)
        mask.CopyInformation(ct)
        mask = sitk.Cast(mask, sitk.sitkUInt8)

        submask = sitk.GetImageFromArray(submask_arr)
        submask.CopyInformation(ct)
        submask = sitk.Cast(submask, sitk.sitkUInt8)

        data[case_id] = {"CT": ct, "WHOLEHEART": mask, "SUBSTRUCTURE": submask}

    return data


def test_cardiac_service(cardiac_data):
    """An end-to-end test to check that the cardiac service is running as expected"""

    with tempfile.TemporaryDirectory() as working_dir:

        working_path = Path(working_dir)
        working_path = Path(".")

        # Save off data
        cases = list(cardiac_data.keys())
        for case in cardiac_data:

            ct_path = working_path.joinpath(f"Case_{case}", "Images", f"Case_{case}_CROP.nii.gz")
            ct_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path = working_path.joinpath(
                f"Case_{case}", "Structures", f"Case_{case}_WHOLEHEART_CROP.nii.gz"
            )
            mask_path.parent.mkdir(parents=True, exist_ok=True)

            sitk.WriteImage(cardiac_data[case]["CT"], str(ct_path))
            sitk.WriteImage(cardiac_data[case]["WHOLEHEART"], str(mask_path))

        # Prepare algorithm settings
        test_settings = CARDIAC_SETTINGS_DEFAULTS
        test_settings["atlasSettings"]["atlasIdList"] = cases[:-1]
        test_settings["atlasSettings"]["atlasPath"] = str(working_path)
        test_settings["atlasSettings"]["atlasStructures"] = ["WHOLEHEART"]
        test_settings["atlasSettings"]["autoCropAtlas"] = False
        test_settings["deformableSettings"]["iterationStaging"] = [5, 5, 5]
        test_settings["IARSettings"]["referenceStructure"] = "WHOLEHEART"
        test_settings["labelFusionSettings"]["optimalThreshold"] = {"WHOLEHEART": 0.5}
        test_settings["vesselSpliningSettings"]["vesselNameList"] = []

        test_settings["rigidSettings"]["options"] = {
            "shrink_factors": [2, 1],
            "smooth_sigmas": [0, 0],
            "sampling_rate": 0.75,
            "default_value": -1024,
            "number_of_iterations": 5,
            "final_interp": sitk.sitkBSpline,
            "metric": "mean_squares",
            "optimiser": "gradient_descent_line_search",
        }

        test_settings["IARSettings"]["referenceStructure"] = None

        # Run the service function.
        infer_case = cases[-1]

        output = run_cardiac_segmentation(cardiac_data[infer_case]["CT"], settings=test_settings)

        # Check we have a WHOLEHEART structure
        assert "WHOLEHEART" in output

        # Check the result is similar to the GT

        label_overlap_filter = sitk.LabelOverlapMeasuresImageFilter()
        auto_mask = output["WHOLEHEART"]
        gt_mask = sitk.Cast(cardiac_data[infer_case]["WHOLEHEART"], auto_mask.GetPixelID())
        label_overlap_filter.Execute(auto_mask, gt_mask)
        assert label_overlap_filter.GetDiceCoefficient() > 0.99


def test_cardiac_structure_guided_service(cardiac_data):
    """
    An end-to-end test to check that the cardiac structure guided service is running as expected
    """

    with tempfile.TemporaryDirectory() as working_dir:

        working_path = Path(working_dir)

        # Save off data
        cases = list(cardiac_data.keys())
        for case in cardiac_data:

            ct_path = working_path.joinpath(f"Case_{case}", "Images", f"Case_{case}_CROP.nii.gz")
            ct_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path = working_path.joinpath(
                f"Case_{case}", "Structures", f"Case_{case}_WHOLEHEART_CROP.nii.gz"
            )
            mask_path.parent.mkdir(parents=True, exist_ok=True)

            substructure_path = working_path.joinpath(
                f"Case_{case}", "Structures", f"Case_{case}_SUBSTRUCTURE_CROP.nii.gz"
            )

            sitk.WriteImage(cardiac_data[case]["CT"], str(ct_path))
            sitk.WriteImage(cardiac_data[case]["WHOLEHEART"], str(mask_path))
            sitk.WriteImage(cardiac_data[case]["SUBSTRUCTURE"], str(substructure_path))

        # Prepare algorithm settings
        test_settings = CARDIAC_STRUCTURE_GUIDED_SETTINGS_DEFAULTS
        test_settings["atlasSettings"]["atlasIdList"] = cases[:-1]
        test_settings["atlasSettings"]["atlasPath"] = str(working_path)
        test_settings["atlasSettings"]["atlasStructures"] = ["WHOLEHEART", "SUBSTRUCTURE"]
        test_settings["atlasSettings"]["autoCropAtlas"] = False
        test_settings["deformableSettingsImage"]["resolution_staging"] = [6, 3, 1.5]
        test_settings["deformableSettingsImage"]["iteration_staging"] = [5, 5, 5]
        test_settings["IARSettings"]["referenceStructure"] = "WHOLEHEART"
        test_settings["labelFusionSettings"]["optimalThreshold"] = {
            "WHOLEHEART": 0.5,
            "SUBSTRUCTURE": 0.5,
        }
        test_settings["vesselSpliningSettings"]["vesselNameList"] = []

        test_settings["rigidSettings"] = {
            "reg_method": "Translation",
            "shrink_factors": [2, 1],
            "smooth_sigmas": [0, 0],
            "sampling_rate": 0.75,
            "default_value": 0,
            "number_of_iterations": 5,
            "final_interp": sitk.sitkLinear,
            "metric": "mean_squares",
            "optimiser": "gradient_descent_line_search",
        }

        test_settings["IARSettings"]["referenceStructure"] = None

        # Run the service function.
        infer_case = cases[-1]

        output = run_cardiac_segmentation_structure_guided(
            cardiac_data[infer_case]["CT"],
            cardiac_data[infer_case]["WHOLEHEART"],
            settings=test_settings,
        )

        # Check we have a WHOLEHEART structure
        assert "WHOLEHEART" in output

        # Check we have a SUBSTRUCTURE structure
        assert "SUBSTRUCTURE" in output

        # Check the result is similar to the GT

        label_overlap_filter = sitk.LabelOverlapMeasuresImageFilter()
        auto_mask = output["WHOLEHEART"]
        gt_mask = sitk.Cast(cardiac_data[infer_case]["WHOLEHEART"], auto_mask.GetPixelID())
        label_overlap_filter.Execute(auto_mask, gt_mask)
        print(label_overlap_filter.GetDiceCoefficient())
        assert label_overlap_filter.GetDiceCoefficient() > 0.9

        auto_mask = output["SUBSTRUCTURE"]
        gt_mask = sitk.Cast(cardiac_data[infer_case]["SUBSTRUCTURE"], auto_mask.GetPixelID())
        label_overlap_filter.Execute(auto_mask, gt_mask)
        print(label_overlap_filter.GetDiceCoefficient())
        assert 0
        assert label_overlap_filter.GetDiceCoefficient() > 0.9
