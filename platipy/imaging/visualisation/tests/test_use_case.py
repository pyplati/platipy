# Copyright 2022 Radiotherapy AI Pty Ltd

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import pathlib
import shutil
import tempfile
import urllib.request

import matplotlib.pyplot as plt
import SimpleITK as sitk

from platipy.dicom.io.rtstruct_to_nifti import convert_rtstruct
from platipy.dicom.io.rtstruct_to_nifti import read_dicom_image
from platipy.imaging import ImageVisualiser


def teardown_function():
    plt.close()


def test_image_visualiser_use_case():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = pathlib.Path(temp_dir)

        zip_url = (
            "https://github.com/RadiotherapyAI/test-data-public/"
            "releases/download/deepmind-dicom/0522c0768.zip"
        )
        zip_filepath = temp_dir / "data.zip"

        urllib.request.urlretrieve(zip_url, zip_filepath)
        shutil.unpack_archive(zip_filepath, temp_dir)

        patient_dir = temp_dir / "0522c0768"
        structure_path = patient_dir / "RS.dcm"

        ct_images = read_dicom_image(patient_dir)
        prefix = "structure_masks_"

        convert_rtstruct(
            patient_dir,
            structure_path,
            prefix=prefix,
            output_dir=patient_dir,
            spacing=None,
        )

        structure_paths = patient_dir.glob(f"{prefix}*.nii.gz")

        contours = {}
        for path in structure_paths:
            filename = _name_with_all_suffixes_removed(path)
            contour_name = removeprefix(filename, prefix)

            contours[contour_name] = sitk.ReadImage(str(path))


    image_visualiser = ImageVisualiser(ct_images)
    image_visualiser.add_contour(contours, color="red", linewidth=1)
    fig = image_visualiser.show()

    assert fig is not None

    return fig


def _name_with_all_suffixes_removed(path: pathlib.Path):
    while path.suffix:
        path = path.with_suffix("")

    return path.name


# https://peps.python.org/pep-0616/#specification
def removeprefix(string: str, prefix: str) -> str:
    if string.startswith(prefix):
        return string[len(prefix):]
    else:
        return string[:]
