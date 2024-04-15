#!/usr/bin/env python

print("Entered run_convert_rtstruct.py ...")

import sys
import os
import glob
from platipy.dicom.io.rtstruct_to_nifti import convert_rtstruct

# Logging for debugging
print(f"Supplied sys.argv: {sys.argv}")
print(f"Current directory: {os.getcwd()}")
print(f"Directory content: {os.listdir()}")

# Check mount directories exist
if os.path.isdir(sys.argv[1]):
    print(f"{sys.argv[1]} content: {os.listdir(sys.argv[1])}")
else:
    raise Warning(f"Directory not found for sys.argv: {sys.argv[1]} ")
if os.path.isdir(sys.argv[2]):
    print(f"{sys.argv[2]} content: {os.listdir(sys.argv[2])}")
else:
    raise Warning(f"Directory not found for sys.argv: {sys.argv[2]} ")
if os.path.isdir(sys.argv[3]):
    print(f"{sys.argv[3]} content: {os.listdir(sys.argv[3])}")
else:
    raise Warning(f"Directory not found for sys.argv: {sys.argv[3]} ")

# Construct arguments
INPUT_DCM_DIRNAME = sys.argv[1]

# get RTSTRUCT filename from mount folder
# print(glob.glob(os.listdir(sys.argv[2]) + '*'))
rt_files = []
for file in os.listdir(sys.argv[2]):
    if file.endswith(".dcm"):
        rt_files.append(file)
if len(rt_files) > 1:
    raise Exception(f"More than one file found in {sys.argv[2]} directory.")
elif len(rt_files) == 1:
    INPUT_RT_FILENAME = os.path.join(sys.argv[2], rt_files[0])

OUTPUT_NII_DIRNAME = sys.argv[3]

print("INPUT_DCM_DIRNAME:", INPUT_DCM_DIRNAME)
print("INPUT_RT_FILENAME:", INPUT_RT_FILENAME)
print("OUTPUT_NII_DIRNAME:", OUTPUT_NII_DIRNAME)

# Run RTSTRUCT to NIFTI conversion
print("Executing convert_rtstruct ...")

if len(sys.argv) == 4:

    convert_rtstruct(
        dcm_img=INPUT_DCM_DIRNAME,
        dcm_rt_file=INPUT_RT_FILENAME,
        output_dir=OUTPUT_NII_DIRNAME
    )

# TODO: Extend XNAT Commands to enable customisable output_img arguments (currently hard-coded to image-data.nii.gz)
if len(sys.argv) == 5:
    OUTPUT_NII_IMG_FILENAME = sys.argv[4]

    print("OUTPUT_NII_IMG_FILENAME:", OUTPUT_NII_IMG_FILENAME)

    convert_rtstruct(
        dcm_img=INPUT_DCM_DIRNAME,
        dcm_rt_file=INPUT_RT_FILENAME,
        output_dir=OUTPUT_NII_DIRNAME,
        output_img=OUTPUT_NII_IMG_FILENAME
    )

# TODO: Extend XNAT Commands to enable customisable NIFTI file prefix
# if len(sys.argv) == 6:
#     OUTPUT_NII_IMG_FILENAME = sys.argv[4]
#     OUTPUT_FILE_PREFIX = sys.argv[5]
    
#     convert_rtstruct(
#         dcm_img=INPUT_DCM_DIRNAME,
#         dcm_rt_file=INPUT_RT_FILENAME,
#         output_dir=OUTPUT_NII_DIRNAME,
#         output_img=OUTPUT_NII_IMG_FILENAME,
#         prefix=OUTPUT_FILE_PREFIX,  # applies to contour files only, not image file
#     )

print("Completed convert_rtstruct ...")

# Log content of output dir
print(f"{sys.argv[3]} content: {os.listdir(sys.argv[3])}")

print("Exiting run_convert_rtstruct.py ...")