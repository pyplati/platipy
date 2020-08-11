"""
Tests for bronchus segmentation
"""
# pylint: disable=redefined-outer-name

import tempfile
import shutil
import os

import pytest

import SimpleITK as sitk
import numpy as np

from platipy.segmentation.tests.pull_data import get_lung_data

from platipy.segmentation.bronchus.bronchus import (
    generate_lung_mask,
    generate_airway_mask,
)

from platipy.segmentation.bronchus.service import (
    bronchus_service,
    BRONCHUS_SETTINGS_DEFAULTS,
)
from platipy.framework.models import DataObject


@pytest.fixture
def bronchus_data():
    """Gets the data needed for the bronchus tests

    Returns:
        dict -- Data for broncus test
    """
    return get_lung_data(number_of_patients=1)


def assert_lung_mask(lung_mask):
    """Checks that the lung mask looks as expected for the tests in this file
    """

    label_shape_statistics_image_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_statistics_image_filter.Execute(lung_mask)

    assert np.allclose(
        label_shape_statistics_image_filter.GetPhysicalSize(1), 4138240, atol=100
    )

    assert np.allclose(
        label_shape_statistics_image_filter.GetElongation(1), 1.52, atol=0.01
    )
    assert np.allclose(
        label_shape_statistics_image_filter.GetRoundness(1), 0.46, atol=0.01
    )

    centroid = label_shape_statistics_image_filter.GetCentroid(1)
    assert np.allclose(centroid[0], 11, atol=1)
    assert np.allclose(centroid[1], -200, atol=1)
    assert np.allclose(centroid[2], -448, atol=1)


def assert_bronchus_mask(bronchus_mask):
    """Checks that the bronchus mask looks as expected for the tests in this file
    """

    label_shape_statistics_image_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_statistics_image_filter.Execute(bronchus_mask)

    assert np.allclose(
        label_shape_statistics_image_filter.GetPhysicalSize(1), 51700, atol=100
    )

    assert np.allclose(
        label_shape_statistics_image_filter.GetElongation(1), 1.39, atol=0.01
    )
    assert np.allclose(
        label_shape_statistics_image_filter.GetRoundness(1), 0.55, atol=0.01
    )

    centroid = label_shape_statistics_image_filter.GetCentroid(1)
    assert np.allclose(centroid[0], 18, atol=1)
    assert np.allclose(centroid[1], -188, atol=1)
    assert np.allclose(centroid[2], -446, atol=1)


def test_lung_segmentation(bronchus_data):
    """Tests the lung segmentation used as an initial step of bronchus segmentation
    """

    img_file = os.path.join(bronchus_data["LCTSC-Train-S1-001"], "CT.nii.gz")
    img = sitk.ReadImage(img_file)

    lung_mask = generate_lung_mask(img)

    assert_lung_mask(lung_mask)


def test_bronchus_segmentation(bronchus_data):
    """Tests the bronchus segmentation algorithm
    """

    img_file = os.path.join(bronchus_data["LCTSC-Train-S1-001"], "CT.nii.gz")
    img = sitk.ReadImage(img_file)

    working_dir = tempfile.mkdtemp()

    lung_mask = generate_lung_mask(img)
    bronchus_mask = generate_airway_mask(working_dir, img, lung_mask)

    assert_bronchus_mask(bronchus_mask)

    shutil.rmtree(working_dir)


def test_bronchus_service(bronchus_data):
    """An end-to-end test to check that the bronchus service is running as expected
    """

    working_dir = tempfile.mkdtemp()

    # Create a data object to be segmented
    data_object = DataObject()
    data_object.id = 1
    data_object.path = os.path.join(bronchus_data["LCTSC-Train-S1-001"], "CT.nii.gz")
    data_object.type = "FILE"

    # Run the service function
    result = bronchus_service([data_object], working_dir, BRONCHUS_SETTINGS_DEFAULTS)

    # Should have returned two output objects
    assert len(result) == 2

    lung_mask = sitk.ReadImage(result[0].path)
    assert_lung_mask(lung_mask)

    bronchus_mask = sitk.ReadImage(result[1].path)
    assert_bronchus_mask(bronchus_mask)
