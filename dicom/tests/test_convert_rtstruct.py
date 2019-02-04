import os
import SimpleITK as sitk

from impit.dicom.rtstruct_to_nifti.convert import convert_rtstruct

def test_convert_rtstruct():

    phantom_dir = os.path.dirname(__file__)
    rtstruct_in = os.path.join(phantom_dir, r'../data/phantom/RTStruct.dcm') # Path to RTStruct file
    ct_in = os.path.join(phantom_dir, r'../data/phantom/CT') # Path to CT directory

    pre = 'Test_'
    output_dir = 'test_output_nifti'
    output_img = 'img.nii.gz'

    # Run the function
    convert_rtstruct(ct_in, rtstruct_in, prefix=pre, output_dir=output_dir, output_img=output_img)

    # Check some of the output files for sanity
    assert len(os.listdir(output_dir)) == 12

    # Check the converted image series
    im = sitk.ReadImage(os.path.join(output_dir, output_img))
    assert im.GetOrigin() == (-211.12600708007812, -422.1260070800781, -974.5)
    assert im.GetSize() == (512, 512, 88)
    assert im.GetSpacing() == (0.8263229727745056, 0.8263229727745056, 3.0)
    nda = sitk.GetArrayFromImage(im)
    print(nda.sum())
    assert nda.sum() == 1541167227

    # Check a converted contour mask
    mask = sitk.ReadImage(os.path.join(output_dir, 'Test_BRAINSTEM_PRI.nii.gz'))
    assert mask.GetOrigin() == (-211.12600708007812, -422.1260070800781, -974.5)
    assert mask.GetSize() == (512, 512, 88)
    assert mask.GetSpacing() == (0.8263229727745056, 0.8263229727745056, 3.0)
    nda = sitk.GetArrayFromImage(mask)
    assert nda.sum() == 13606
