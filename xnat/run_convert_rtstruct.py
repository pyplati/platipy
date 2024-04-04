import sys
from platipy.dicom.io.rtstruct_to_nifti import convert_rtstruct


INPUT_DCM_DIRNAME = sys.argv[1]
INPUT_RT_FILENAME = sys.argv[2]
OUTPUT_NII_DIRNAME = sys.argv[3]
OUTPUT_NII_FILENAME = sys.argv[4]
OUTPUT_FILE_PREFIX = sys.argv[5]

convert_rtstruct(
    dcm_img=INPUT_DCM_DIRNAME,
    dcm_rt_file=INPUT_RT_FILENAME,
    output_dir=OUTPUT_NII_DIRNAME,
    output_img=OUTPUT_NII_FILENAME,
    prefix=OUTPUT_FILE_PREFIX,  # applies to contour files only, not image file
)