import click
import yaml
import sys
import argparse
import os
from platipy.dicom.nifti_to_rtstruct.export_to_rtstruct import export_to_rtstruct
from platipy.dicom.nifti_to_rtstruct.gen_contours import gen_contours
import pydicom
from collections import OrderedDict
from loguru import logger

import SimpleITK as sitk


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


@click.command()
@click.option(
    "--dcm_file",
    "-d",
    required=True,
    help="Reference DICOM file from which header tags will be copied",
)
@click.option(
    "--debug",
    default=False,
    is_flag=True,
    help="Whether intermediate debug info is written",
)
@click.option(
    "--mask", "-m", multiple=True, required=True, help="Mask pairs with name,filename"
)
@click.option("--out_rt_filename", "-o", required=True, help="Name of RT struct output")
def click_command(dcm_file, debug, mask, out_rt_filename):
    """
    Convert Nifti masks to Dicom RTStruct
    """

    convert_nifti(dcm_file, mask, out_rt_filename, debug=debug)


if __name__ == "__main__":
    click_command()  # pylint: disable=no-value-for-parameter
