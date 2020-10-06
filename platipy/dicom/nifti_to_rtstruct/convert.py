# Copyright 2020 CSIRO

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
import yaml
from collections import OrderedDict
import pydicom
import SimpleITK as sitk
from loguru import logger
from platipy.dicom.nifti_to_rtstruct.export_to_rtstruct import export_to_rtstruct
from platipy.dicom.nifti_to_rtstruct.gen_contours import gen_contours


def convert_nifti(dcm_file, mask, out_rt_filename, debug=False):

    logger.info("Will convert the following Nifti masks to RTStruct:")

    masks = {}  # Dict stores key value pairs for masks
    if type(mask) == dict:
        # Dict was already passed in
        masks = mask
    else:
        # Otherwise convert list of comma separated masks
        for m in mask:
            s = m.split(",")
            masks[s[0]] = s[1]

    for m in masks:
        logger.info(" - {0}".format(m))

    logger.info("Will use the following Dicom file as reference: {0}".format(dcm_file))

    dd = {}

    name, ext = os.path.splitext(out_rt_filename)

    ## check if dicom filename has an extention
    if ext == "":
        out_rt_filename = name + ".dcm"

    out_param_name = name + "_param.yaml"

    dd["StartSliceNum"] = 0

    mask_img = sitk.ReadImage(masks[list(masks)[0]])
    dd["NumSlices"] = mask_img.GetSize()[0]
    dat_ct = pydicom.dcmread(dcm_file)
    dd["UIDPrefix"] = ".".join(dat_ct.SOPInstanceUID.split(".")[:-1]) + "."

    rois = OrderedDict()
    dd["ROIS"] = rois

    for ii, key in enumerate(masks.keys()):
        temp = {}
        temp["InterpretedType"] = "ORGAN"
        iCol = 100 + (20 * ii)
        if iCol > 250:
            iCol = 250

        temp["Color"] = [iCol, 0, iCol]
        conts = gen_contours(masks[key])
        temp["ContourFile"] = key + ".yaml"
        if debug:
            open(key + ".yaml", "w").write(yaml.dump(conts))
        temp["Contours"] = conts["Contours"]
        rois[key] = temp

    rt_struct = export_to_rtstruct(dd, dat_ct, debug=debug)
    logger.info("Writing RTStruct to: {0}".format(out_rt_filename))
    rt_struct.save_as(out_rt_filename)

    if debug:
        for roi in dd["ROIS"]:
            dd["ROIS"][roi]["Contours"] = None

        open(out_param_name, "w").write(yaml.dump(dd))

    logger.info("Finished")
