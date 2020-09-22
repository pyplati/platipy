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

import os
import pydicom
import numpy as np

from platipy.dicom.nifti_to_rtstruct.convert import convert_nifti


def test_convert_nifti():

    phantom_dir = os.path.dirname(__file__)
    dicom_file = os.path.join(
        phantom_dir,
        r"../data/phantom/CT/2.16.840.1.114362.1.11775105.22396782581.502959996.700.3.dcm",
    )
    masks_dir = os.path.join(phantom_dir, r"../data/phantom/masks/")
    output_file = "test_output.dcm"

    # Generate dict of masks in the masks directory
    masks = {}
    for m in os.listdir(masks_dir):
        name = m.split(".")[0].split("Test_")[1]
        mask_path = os.path.join(masks_dir, m)
        masks[name] = mask_path

    # Run the function
    convert_nifti(dicom_file, masks, output_file)

    # Check some of the output files for sanity
    original_file = os.path.join(phantom_dir, r"../data/phantom/RTStruct.dcm")
    original = pydicom.read_file(original_file)
    rts = pydicom.read_file(output_file)

    assert rts.Modality == original.Modality
    assert rts.PatientName == original.PatientName
    assert rts.PatientID == original.PatientID

    # Check that all the contour names match
    contour_map = {}
    for i in original.StructureSetROISequence:
        for j in rts.StructureSetROISequence:
            if j.ROIName == i.ROIName:
                contour_map[int(i.ROINumber)] = int(j.ROINumber)
    assert len(contour_map.keys()) == 11

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
