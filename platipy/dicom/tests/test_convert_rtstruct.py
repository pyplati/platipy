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
import SimpleITK as sitk

from platipy.dicom.rtstruct_to_nifti.convert import convert_rtstruct


def test_convert_rtstruct():

    phantom_dir = os.path.dirname(__file__)
    rtstruct_in = os.path.join(
        phantom_dir, r"../data/phantom/RTStruct.dcm"
    )  # Path to RTStruct file
    ct_in = os.path.join(phantom_dir, r"../data/phantom/CT")  # Path to CT directory

    pre = "Test_"
    output_dir = "test_output_nifti"
    output_img = "img.nii.gz"

    # Run the function
    convert_rtstruct(
        ct_in, rtstruct_in, prefix=pre, output_dir=output_dir, output_img=output_img
    )

    # Check some of the output files for sanity
    assert len(os.listdir(output_dir)) == 12

    # Check the converted image series
    im = sitk.ReadImage(os.path.join(output_dir, output_img), sitk.sitkInt64)
    print(os.path.join(output_dir, output_img))
    assert im.GetOrigin() == (-211.12600708007812, -422.1260070800781, -974.5)
    assert im.GetSize() == (512, 512, 88)
    assert im.GetSpacing() == (0.8263229727745056, 0.8263229727745056, 3.0)
    nda = sitk.GetArrayFromImage(im)
    print(nda.sum())
    assert nda.sum() == -19933669253

    # Check a converted contour mask
    mask = sitk.ReadImage(os.path.join(output_dir, "Test_BRAINSTEM_PRI.nii.gz"))
    assert mask.GetOrigin() == (-211.12600708007812, -422.1260070800781, -974.5)
    assert mask.GetSize() == (512, 512, 88)
    assert mask.GetSpacing() == (0.8263229727745056, 0.8263229727745056, 3.0)
    nda = sitk.GetArrayFromImage(mask)
    assert nda.sum() == 13606
