# Copyright 2021 University of New South Wales, University of Sydney, Ingham Institute

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=redefined-outer-name,missing-function-docstring

import pytest

import SimpleITK as sitk
import numpy as np

from platipy.imaging.tests.data import get_hn_nifti

from platipy.imaging.dose.dvh import calculate_dvh_for_labels, calculate_d_x, calculate_v_x


@pytest.fixture
def nifti_data():

    return get_hn_nifti()


def compute_dvh_for_data(data_path):

    test_pat_path = data_path.joinpath("TCGA_CV_5977")

    ct_image = sitk.ReadImage(
        str(test_pat_path.joinpath("IMAGES/TCGA_CV_5977_1_CT_ONC_NECK_NECK_4.nii.gz"))
    )

    dose = sitk.ReadImage(str(test_pat_path.joinpath("DOSES/TCGA_CV_5977_1_PLAN.nii.gz")))
    dose = sitk.Resample(dose, ct_image)

    structure_names = [
        "BRAINSTEM",
        "MANDIBLE",
        "CTV_60_GY",
        "PTV60",
        "CORD",
        "L_PAROTID",
        "R_PAROTID",
    ]
    structures = {
        s: sitk.ReadImage(
            str(test_pat_path.joinpath("STRUCTURES", f"TCGA_CV_5977_1_RTSTRUCT_{s}.nii.gz"))
        )
        for s in structure_names
    }

    return calculate_dvh_for_labels(dose, structures)


def test_compute_dvh(nifti_data):

    dvh = compute_dvh_for_data(nifti_data)
    assert len(dvh) == 7

    # Check the values for one bin
    assert np.allclose(
        dvh[60.0],
        [
            0.0,
            0.2022032,
            0.9675792,
            0.8746213,
            0.0,
            0.0003158,
            0.0,
        ],
        atol=1e-4,
    )


def test_compute_d_metric(nifti_data):

    dvh = compute_dvh_for_data(nifti_data)
    df_metrics = calculate_d_x(dvh, 95)
    assert np.allclose(list(df_metrics.value), [0.0, 36.5, 60.2, 58.9, 6.5, 9.1, 7.3])


def test_compute_v_metric(nifti_data):

    dvh = compute_dvh_for_data(nifti_data)
    df_metrics = calculate_v_x(dvh, 40)
    assert np.allclose(
        list(df_metrics.value),
        [
            0.0,
            59.137344,
            190.98043,
            280.18474,
            0.0,
            0.9465217,
            1.3923645,
        ],
        atol=1e-4,
    )
