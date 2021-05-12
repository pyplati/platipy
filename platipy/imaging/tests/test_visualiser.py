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

import pytest

import SimpleITK as sitk

from platipy.imaging.tests.data import get_lung_nifti

from platipy.imaging import ImageVisualiser
from platipy.imaging.label.utils import get_com


@pytest.fixture
def nifti_data():

    return get_lung_nifti()


def test_contour_visualisation(nifti_data):

    patient_path = nifti_data.joinpath("LCTSC-Test-S1-201")

    ct_path = next(patient_path.glob("IMAGES/*.nii.gz"))

    structures = {
        struct.name.split(".nii.gz")[0].split("RTSTRUCT_")[-1]: sitk.ReadImage(str(struct))
        for struct in patient_path.glob("STRUCTURES/*.nii.gz")
    }
    img = sitk.ReadImage(str(ct_path))

    vis = ImageVisualiser(img, cut=get_com(structures["HEART"]))
    vis.add_contour(structures)
    fig = vis.show()

    assert len(fig.axes[0].collections) == 5  # Check correct number of contours

    for idx, structure in enumerate(structures.keys()):
        assert fig.axes[0].legend().get_texts()[idx].get_text() == structure
