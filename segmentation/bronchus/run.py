"""
Service to run bronchus segmentation.
"""
import tempfile

from loguru import logger

from impit.segmentation.bronchus.bronchus import (
    generate_lung_mask,
    generate_airway_mask,
    default_settings,
)

BRONCHUS_SETTINGS_DEFAULTS = {
    "outputBronchusName": "Auto_Bronchus",
    "outputLungName": "Auto_Lung",
    "algorithmSettings": default_settings,
}


def run_bronchus_segmentation(input_image, settings=BRONCHUS_SETTINGS_DEFAULTS):
    """Runs the Proximal Bronchial Tree segmentation

    Args:
        input_image (str): Path to the input image on which to run the segmentation
        output_directory (str, optional): Path to output the segmentations. Defaults to ".".
        settings (dict, optional): Dictionary containing settings for algorithm. 
                                   Defaults to default_settings.
    
    Returns:
        dict: Dictionary containing output of segmentation
    """

    working_directory = tempfile.mkdtemp()
    results = {}

    # Compute the lung mask
    lung_mask = generate_lung_mask(input_image)
    results[settings["outputLungName"]] = lung_mask

    bronchus_mask = generate_airway_mask(
        working_directory, input_image, lung_mask, config_dict=settings["algorithmSettings"]
    )

    # If the bronchus mask counldn't be generated then skip it
    if not bronchus_mask:
        logger.error("Unable to generate bronchus mask")
        return results

    results[settings["outputBronchusName"]] = bronchus_mask

    return results
