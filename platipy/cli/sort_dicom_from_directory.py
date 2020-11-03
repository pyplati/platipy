#!/usr/bin/env python
import sys
import click
import shutil
import pathlib

from loguru import logger
from platipy.dicom.dicom_directory_crawler.conversion_utils import safe_sort_dicom_image_list

logger.remove()
logger.add(sys.stderr, level="DEBUG")

@click.command()
@click.option(
    "--input_dir",
    "-i",
    required=True,
    type=click.Path(),
    help="Input DICOM directory.",
)
@click.option(
    "--backup_dir",
    "-o",
    default="./original_naming",
    show_default=True,
    required=False,
    type=click.Path(),
    help="Backup directory. Original images will be copied here.",
)
@click.option(
    "--reverse",
    "-r",
    is_flag=True,
    default=False,
    show_default=True,
    help="Reverse sorting.",
)
def click_command(
    input_dir,
    backup_dir,
    reverse
):
    """
    DICOM DIRECTORY SORTER

    This tool sorts DICOM directories based on the slice location.


    """
    logger.info("########################")
    logger.info(" Running DICOM sorter ")
    logger.info("########################")

    input_dir = pathlib.Path(input_dir)
    backup_dir = pathlib.Path(backup_dir)

    backup_dir.mkdir(parents=True, exist_ok=True)

    # Find files ending with .dcm, .dc3
    dicom_file_list = (
        list(input_dir.glob("**/*.dcm"))
        + list(input_dir.glob("**/*.DCM*"))
        + list(input_dir.glob("**/*.dc3"))
        + list(input_dir.glob("**/*.DC3"))
    )

    sorted_dicom_file_list = safe_sort_dicom_image_list(dicom_file_list)
    if reverse:
        sorted_dicom_file_list = sorted_dicom_file_list[::-1]

    logger.info(f" {len(sorted_dicom_file_list)} DICOM files have been sorted.")

    for index, dicom_file in enumerate(sorted_dicom_file_list):

        dicom_name = dicom_file.name

        input_name = (
            pathlib.Path(input_dir / dicom_name)
        )
        
        output_name = (
            pathlib.Path(input_dir / f'MR.{str(index+1).zfill(4)}.dcm')
        )

        backup_name = (
            pathlib.Path(backup_dir / dicom_name)
        )

        shutil.copy2(input_name, backup_name)
        shutil.move(input_name, output_name)


    logger.info("########################")
    logger.info(" DICOM sorter complete")
    logger.info("########################")


if __name__ == "__main__":
    click_command()  # pylint: disable=no-value-for-parameter
