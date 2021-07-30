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

import pydicom
import SimpleITK as sitk


def convert_rtdose(dcm_dose, dose_output_path=None):
    """Convert DICOM RTDose to SimpleITK image, saving as NiFTI if needed.

    Args:
        dcm_dose (str|path): Path to DICOM dose file
        dose_output_path (str|path, optional): If set, NiFTI file will be written. Defaults to
            None.

    Returns:
        SimpleITK.Image: The dose grid as a SimpleITK image
    """

    ds = pydicom.read_file(dcm_dose)
    dose = sitk.ReadImage(str(dcm_dose))
    dose = sitk.Cast(dose, sitk.sitkFloat32)
    dose = dose * ds.DoseGridScaling

    if dose_output_path is not None:
        sitk.WriteImage(dose, str(dose_output_path))

    return dose
