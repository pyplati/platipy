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
import pydicom
import numpy as np
import SimpleITK as sitk

from platipy.imaging.tests.data import get_lung_dicom, get_lung_nifti
from platipy.dicom.io.rtstruct_to_nifti import convert_rtstruct
from platipy.dicom.io.nifti_to_rtstruct import convert_nifti


@pytest.fixture
def dicom_data():
    """Gets the data needed for converting DICOM

    Returns:
        pathlib.Path: Path to DICOM data
    """
    return get_lung_dicom()


@pytest.fixture
def nifti_data():
    """Gets the data needed for converting NIFTI

    Returns:
        pathlib.Path: Path to NIFTI data
    """
    return get_lung_nifti()


def test_convert_dicom_to_nifti(dicom_data):

    test_pat_id = "LCTSC-Test-S1-101"
    ct_uid = "1.3.6.1.4.1.14519.5.2.1.7014.4598.106943890850011666503487579262"
    rtstruct_uid = "1.3.6.1.4.1.14519.5.2.1.7014.4598.280355341349691222365783556597"

    pat_path = dicom_data.joinpath(test_pat_id)
    ct_path = pat_path.joinpath(ct_uid)
    rtstruct_path = next(pat_path.joinpath(rtstruct_uid).glob("*.dcm"))

    pre = "Test_"

    output_img = "img.nii.gz"
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)

        # Run the function
        convert_rtstruct(
            ct_path, rtstruct_path, prefix=pre, output_dir=output_dir, output_img=output_img
        )

        # Check some of the output files for sanity
        assert len(list(output_dir.glob("*.nii.gz"))) == 6

        # Check the converted image series
        ct_output = next(output_dir.glob("img.nii.gz"))
        im = sitk.ReadImage(str(ct_output))

        assert np.allclose(im.GetOrigin(), (-249.51, -483.01, -640.2), atol=0.01)
        assert np.allclose(im.GetSize(), (512, 512, 130), atol=0.01)
        assert np.allclose(im.GetSpacing(), (0.97, 0.97, 3.0), atol=0.01)

        nda = sitk.GetArrayFromImage(im)
        assert nda.sum() == -23952778432

        # Check a converted contour mask
        heart_output = next(output_dir.glob("Test_Heart.nii.gz"))
        mask = sitk.ReadImage(str(heart_output))
        assert np.allclose(mask.GetOrigin(), (-249.51, -483.01, -640.2), atol=0.01)
        assert np.allclose(mask.GetSize(), (512, 512, 130), atol=0.01)
        assert np.allclose(mask.GetSpacing(), (0.97, 0.97, 3.0), atol=0.01)
        nda = sitk.GetArrayFromImage(mask)

        assert nda.sum() == 263237


def test_convert_nifti_to_dicom(nifti_data, dicom_data):

    test_pat_id = "LCTSC-Test-S1-101"
    ct_uid = "1.3.6.1.4.1.14519.5.2.1.7014.4598.106943890850011666503487579262"
    rtstruct_uid = "1.3.6.1.4.1.14519.5.2.1.7014.4598.280355341349691222365783556597"

    pat_path = dicom_data.joinpath(test_pat_id)
    ct_path = pat_path.joinpath(ct_uid)
    rtstruct_path = next(pat_path.joinpath(rtstruct_uid).glob("*.dcm"))

    # Generate dict of masks in the masks directory
    masks = {}
    for mask_path in nifti_data.joinpath(test_pat_id).glob("STRUCTURES/*.nii.gz"):
        name = mask_path.name.split(".")[0].split("RTSTRUCT_")[1]
        masks[name] = str(mask_path)

    with tempfile.TemporaryDirectory() as temp_dir:

        output_file = Path(temp_dir).joinpath("test.dcm")

        # Run the function
        convert_nifti(ct_path, masks, output_file)

        # Check some of the output files for sanity
        original = pydicom.read_file(rtstruct_path)
        rts = pydicom.read_file(output_file)

        assert rts.Modality == original.Modality
        assert rts.PatientName == original.PatientName
        assert rts.PatientID == original.PatientID

        # Check that all the contour names match
        contour_map = {}
        for i in original.StructureSetROISequence:
            for j in rts.StructureSetROISequence:
                if j.ROIName.upper() == i.ROIName.upper():
                    contour_map[int(i.ROINumber)] = int(j.ROINumber)

        assert len(contour_map.keys()) == 5

        # Confirm that the min/max contour points land within 1 voxel size of the original
        i = 1
        for j in original.ROIContourSequence[i - 1].ContourSequence:

            for k in rts.ROIContourSequence[contour_map[i] - 1].ContourSequence:

                if j.ContourData[2] == k.ContourData[2]:
                    j_points = np.array(j.ContourData)
                    j_points = j_points.reshape(j.NumberOfContourPoints, 3)

                    k_points = np.array(k.ContourData)
                    k_points = k_points.reshape(k.NumberOfContourPoints, 3)
                    assert (max(abs(k_points.min(axis=0) - j_points.min(axis=0)))) < 0.83
                    assert (max(abs(k_points.max(axis=0) - j_points.max(axis=0)))) < 0.83
