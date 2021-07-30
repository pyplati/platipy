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
import shutil

import pytest

import SimpleITK as sitk
import numpy as np

from platipy.imaging.tests.data import get_lung_nifti

from platipy.imaging.projects.bronchus.bronchus import (
    generate_lung_mask,
    generate_airway_mask,
)


@pytest.fixture
def bronchus_data():
    """Gets the data needed for the bronchus tests

    Returns:
        pathlib.Path: Path to Data for bronchus test
    """
    return get_lung_nifti()


def assert_lung_mask(lung_mask):
    """Checks that the lung mask looks as expected for the tests in this file"""

    label_shape_statistics_image_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_statistics_image_filter.Execute(lung_mask)

    assert np.allclose(label_shape_statistics_image_filter.GetPhysicalSize(1), 2480246, atol=100)

    assert np.allclose(label_shape_statistics_image_filter.GetElongation(1), 1.48, atol=0.01)
    assert np.allclose(label_shape_statistics_image_filter.GetRoundness(1), 0.46, atol=0.01)

    centroid = label_shape_statistics_image_filter.GetCentroid(1)
    assert np.allclose(centroid[0], -1, atol=1)
    assert np.allclose(centroid[1], -169, atol=1)
    assert np.allclose(centroid[2], -476, atol=1)


def assert_bronchus_mask(bronchus_mask):
    """Checks that the bronchus mask looks as expected for the tests in this file"""

    label_shape_statistics_image_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_statistics_image_filter.Execute(bronchus_mask)

    print(label_shape_statistics_image_filter.GetPhysicalSize(1))
    print(label_shape_statistics_image_filter.GetElongation(1))
    print(label_shape_statistics_image_filter.GetRoundness(1))
    print(label_shape_statistics_image_filter.GetCentroid(1))
    assert np.allclose(label_shape_statistics_image_filter.GetPhysicalSize(1), 42823, atol=100)

    assert np.allclose(label_shape_statistics_image_filter.GetElongation(1), 1.41, atol=0.01)
    assert np.allclose(label_shape_statistics_image_filter.GetRoundness(1), 0.55, atol=0.01)

    centroid = label_shape_statistics_image_filter.GetCentroid(1)
    assert np.allclose(centroid[0], 8.85, atol=1)
    assert np.allclose(centroid[1], -160, atol=1)
    assert np.allclose(centroid[2], -457, atol=1)


def test_lung_segmentation(bronchus_data):
    """Tests the lung segmentation used as an initial step of bronchus segmentation"""

    patient_path = bronchus_data.joinpath("LCTSC-Test-S1-201")

    ct_path = next(patient_path.glob("IMAGES/*.nii.gz"))
    img = sitk.ReadImage(str(ct_path))

    lung_mask = generate_lung_mask(img)

    assert_lung_mask(lung_mask)


def test_bronchus_segmentation(bronchus_data):
    """Tests the bronchus segmentation algorithm"""

    patient_path = bronchus_data.joinpath("LCTSC-Test-S1-201")

    ct_path = next(patient_path.glob("IMAGES/*.nii.gz"))
    img = sitk.ReadImage(str(ct_path))

    working_dir = tempfile.mkdtemp()

    lung_mask = generate_lung_mask(img)
    bronchus_mask = generate_airway_mask(working_dir, img, lung_mask)

    assert_bronchus_mask(bronchus_mask)

    shutil.rmtree(working_dir)


# def test_bronchus_service(bronchus_data):
#     """An end-to-end test to check that the bronchus service is running as expected"""

#     working_dir = tempfile.mkdtemp()

#     patient_path = bronchus_data.joinpath("LCTSC-Test-S1-201")
#     ct_path = next(patient_path.glob("IMAGES/*.nii.gz"))

#     # Create a data object to be segmented
#     data_object = DataObject()
#     data_object.id = 1
#     data_object.path = str(ct_path)
#     data_object.type = "FILE"

#     # Run the service function
#     result = bronchus_service([data_object], working_dir, BRONCHUS_SETTINGS_DEFAULTS)

#     # Should have returned two output objects
#     assert len(result) == 2

#     lung_mask = sitk.ReadImage(result[0].path)
#     assert_lung_mask(lung_mask)

#     bronchus_mask = sitk.ReadImage(result[1].path)
#     assert_bronchus_mask(bronchus_mask)
