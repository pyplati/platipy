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
import numpy as np

from loguru import logger

from platipy.imaging.registration.utils import apply_transform

from platipy.imaging.registration.linear import (
    linear_registration,
)

from platipy.imaging.registration.deformable import (
    fast_symmetric_forces_demons_registration,
)

from platipy.imaging.label.fusion import (
    process_probability_image,
    compute_weight_map,
    combine_labels,
)

from platipy.imaging.utils.crop import label_to_roi, crop_to_roi

from platipy.imaging.label.utils import correct_volume_overlap

ATLAS_PATH = "/atlas"
if "ATLAS_PATH" in os.environ:
    ATLAS_PATH = os.environ["ATLAS_PATH"]

MUTLIATLAS_SETTINGS_DEFAULTS = {
    "atlas_settings": {
        "atlas_id_list": [
            "03",
        ],
        "atlas_structure_list": [
            "WHOLEHEART",
        ],
        "atlas_path": ATLAS_PATH,
        "atlas_image_format": "Case_{0}/Images/Case_{0}_CROP.nii.gz",
        "atlas_label_format": "Case_{0}/Structures/Case_{0}_{1}_CROP.nii.gz",
        "crop_atlas_to_structures": False,
        "crop_atlas_expansion_mm": (20, 20, 40),
    },
    "auto_crop_target_image_settings": {
        "expansion_mm": [20, 20, 40],
    },
    "linear_registration_settings": {
        "reg_method": "affine",
        "shrink_factors": [16, 8, 4],
        "smooth_sigmas": [0, 0, 0],
        "sampling_rate": 0.75,
        "default_value": None,
        "number_of_iterations": 50,
        "metric": "mean_squares",
        "optimiser": "gradient_descent_line_search",
        "verbose": False,
    },
    "deformable_registration_settings": {
        "isotropic_resample": True,
        "resolution_staging": [
            6,
            3,
            1.5,
        ],  # specify voxel size (mm) since isotropic_resample is set
        "iteration_staging": [150, 125, 100],
        "smoothing_sigmas": [
            0,
            0,
            0,
        ],
        "ncores": 8,
        "default_value": None,
        "verbose": False,
    },
    "label_fusion_settings": {
        "vote_type": "unweighted",
        "vote_params": None,
        "optimal_threshold": {},
    },
    "postprocessing_settings": {
        "run_postprocessing": True,
        "binaryfillhole_mm": 3,
        "structures_for_binaryfillhole": [],
        "structures_for_overlap_correction": [],
    },
}


def run_segmentation(img, settings=MUTLIATLAS_SETTINGS_DEFAULTS):
    """Runs the atlas-based segmentation algorithm

    Args:
        img (sitk.Image):
        settings (dict, optional): Dictionary containing settings for algorithm.
                                   Defaults to default_settings.

    Returns:
        dict: Dictionary containing output of segmentation
    """

    results = {}
    results_prob = {}

    """
    Initialisation - Read in atlases
    - image files
    - structure files

        Atlas structure:
        'ID': 'Original': 'CT Image'    : sitk.Image
                            'Struct A'    : sitk.Image
                            'Struct B'    : sitk.Image
                'RIR'     : 'CT Image'    : sitk.Image
                            'Transform'   : transform parameter map
                            'Struct A'    : sitk.Image
                            'Struct B'    : sitk.Image
                'DIR'     : 'CT Image'    : sitk.Image
                            'Transform'   : displacement field transform
                            'Weight Map'  : sitk.Image
                            'Struct A'    : sitk.Image
                            'Struct B'    : sitk.Image


    """

    logger.info("")
    # Settings
    atlas_path = settings["atlas_settings"]["atlas_path"]
    atlas_id_list = settings["atlas_settings"]["atlas_id_list"]
    atlas_structure_list = settings["atlas_settings"]["atlas_structure_list"]

    atlas_image_format = settings["atlas_settings"]["atlas_image_format"]
    atlas_label_format = settings["atlas_settings"]["atlas_label_format"]

    crop_atlas_to_structures = settings["atlas_settings"]["crop_atlas_to_structures"]
    crop_atlas_expansion_mm = settings["atlas_settings"]["crop_atlas_expansion_mm"]

    atlas_set = {}
    for atlas_id in atlas_id_list:
        atlas_set[atlas_id] = {}
        atlas_set[atlas_id]["Original"] = {}

        image = sitk.ReadImage(f"{atlas_path}/{atlas_image_format.format(atlas_id)}")

        structures = {
            struct: sitk.ReadImage(f"{atlas_path}/{atlas_label_format.format(atlas_id, struct)}")
            for struct in atlas_structure_list
        }

        if crop_atlas_to_structures:
            logger.info(f"Automatically cropping atlas: {atlas_id}")

            original_volume = np.product(image.GetSize())

            crop_box_size, crop_box_index = label_to_roi(
                structures.values(), expansion_mm=crop_atlas_expansion_mm
            )

            image = crop_to_roi(image, size=crop_box_size, index=crop_box_index)

            final_volume = np.product(image.GetSize())

            logger.info(f"  > Volume reduced by factor {original_volume/final_volume:.2f}")

            for struct in atlas_structure_list:
                structures[struct] = crop_to_roi(
                    structures[struct], size=crop_box_size, index=crop_box_index
                )

        atlas_set[atlas_id]["Original"]["CT Image"] = image

        for struct in atlas_structure_list:
            atlas_set[atlas_id]["Original"][struct] = structures[struct]

    """
    Step 1 - Automatic cropping
    If we have a guide structure:
        - use structure to crop target image

    Otherwise:
        - using a quick registration to register each atlas
        - expansion of the bounding box to ensure entire volume of interest is enclosed
        - target image is cropped
    """

    expansion_mm = settings["auto_crop_target_image_settings"]["expansion_mm"]

    quick_reg_settings = {
        "reg_method": "similarity",
        "shrink_factors": [8],
        "smooth_sigmas": [0],
        "sampling_rate": 0.75,
        "default_value": -1000,
        "number_of_iterations": 25,
        "final_interp": sitk.sitkLinear,
        "metric": "mean_squares",
        "optimiser": "gradient_descent_line_search",
    }

    registered_crop_images = []

    logger.info("Running initial Translation tranform to crop image volume")

    for atlas_id in atlas_id_list[: min([8, len(atlas_id_list)])]:

        logger.info(f"  > atlas {atlas_id}")

        # Register the atlases
        atlas_set[atlas_id]["RIR"] = {}
        atlas_image = atlas_set[atlas_id]["Original"]["CT Image"]

        reg_image, _ = linear_registration(
            img,
            atlas_image,
            **quick_reg_settings,
        )

        registered_crop_images.append(sitk.Cast(reg_image, sitk.sitkFloat32))

        del reg_image

    combined_image = sum(registered_crop_images) / len(registered_crop_images) > -1000

    crop_box_size, crop_box_index = label_to_roi(combined_image, expansion_mm=expansion_mm)

    img_crop = crop_to_roi(img, crop_box_size, crop_box_index)

    logger.info("Calculated crop box:")
    logger.info(f"  > {crop_box_index}")
    logger.info(f"  > {crop_box_size}")
    logger.info(f"  > Vol reduction = {np.product(img.GetSize())/np.product(crop_box_size):.2f}")

    """
    Step 2 - Rigid registration of target images
    - Individual atlas images are registered to the target
    - The transformation is used to propagate the labels onto the target
    """
    linear_registration_settings = settings["linear_registration_settings"]

    logger.info(
        f"Running {linear_registration_settings['reg_method']} tranform to align atlas images"
    )

    for atlas_id in atlas_id_list:
        # Register the atlases

        logger.info(f"  > atlas {atlas_id}")

        atlas_set[atlas_id]["RIR"] = {}

        target_reg_image = img_crop
        atlas_reg_image = atlas_set[atlas_id]["Original"]["CT Image"]

        _, initial_tfm = linear_registration(
            target_reg_image,
            atlas_reg_image,
            **linear_registration_settings,
        )

        # Save in the atlas dict
        atlas_set[atlas_id]["RIR"]["Transform"] = initial_tfm

        atlas_set[atlas_id]["RIR"]["CT Image"] = apply_transform(
            input_image=atlas_set[atlas_id]["Original"]["CT Image"],
            reference_image=img_crop,
            transform=initial_tfm,
            default_value=-1000,
            interpolator=sitk.sitkLinear,
        )

        # sitk.WriteImage(rigid_image, f"./RR_{atlas_id}.nii.gz")

        for struct in atlas_structure_list:
            input_struct = atlas_set[atlas_id]["Original"][struct]
            atlas_set[atlas_id]["RIR"][struct] = apply_transform(
                input_image=input_struct,
                reference_image=img_crop,
                transform=initial_tfm,
                default_value=0,
                interpolator=sitk.sitkNearestNeighbor,
            )

        atlas_set[atlas_id]["Original"] = None

    """
    Step 3 - Deformable image registration
    - Using Fast Symmetric Diffeomorphic Demons
    """

    # Settings
    deformable_registration_settings = settings["deformable_registration_settings"]

    logger.info("Running DIR to refine atlas image registration")

    for atlas_id in atlas_id_list:

        logger.info(f"  > atlas {atlas_id}")

        # Register the atlases
        atlas_set[atlas_id]["DIR"] = {}

        atlas_reg_image = atlas_set[atlas_id]["RIR"]["CT Image"]
        target_reg_image = img_crop

        _, dir_tfm, _ = fast_symmetric_forces_demons_registration(
            target_reg_image,
            atlas_reg_image,
            **deformable_registration_settings,
        )

        # Save in the atlas dict
        atlas_set[atlas_id]["DIR"]["Transform"] = dir_tfm

        atlas_set[atlas_id]["DIR"]["CT Image"] = apply_transform(
            input_image=atlas_set[atlas_id]["RIR"]["CT Image"],
            transform=dir_tfm,
            default_value=-1000,
            interpolator=sitk.sitkLinear,
        )

        for struct in atlas_structure_list:
            input_struct = atlas_set[atlas_id]["RIR"][struct]
            atlas_set[atlas_id]["DIR"][struct] = apply_transform(
                input_image=input_struct,
                transform=dir_tfm,
                default_value=0,
                interpolator=sitk.sitkNearestNeighbor,
            )

        atlas_set[atlas_id]["RIR"] = None

    """
    Step 4 - Label Fusion
    """
    # Compute weight maps
    vote_type = settings["label_fusion_settings"]["vote_type"]
    vote_params = settings["label_fusion_settings"]["vote_params"]

    # Compute weight maps
    for atlas_id in list(atlas_set.keys()):
        atlas_image = atlas_set[atlas_id]["DIR"]["CT Image"]
        weight_map = compute_weight_map(
            img_crop, atlas_image, vote_type=vote_type, vote_params=vote_params
        )
        atlas_set[atlas_id]["DIR"]["Weight Map"] = weight_map

    combined_label_dict = combine_labels(atlas_set, atlas_structure_list)

    """
    Step 6 - Paste the cropped structure into the original image space
    """
    logger.info("Generating binary segmentations.")
    template_img_binary = sitk.Cast((img * 0), sitk.sitkUInt8)
    template_img_prob = sitk.Cast((img * 0), sitk.sitkFloat64)

    for structure_name in atlas_structure_list:

        probability_map = combined_label_dict[structure_name]

        if structure_name in settings["label_fusion_settings"]["optimal_threshold"]:
            optimal_threshold = settings["label_fusion_settings"]["optimal_threshold"][
                structure_name
            ]
        else:
            optimal_threshold = 0.5

        binary_struct = process_probability_image(probability_map, optimal_threshold)

        # Un-crop binary structure
        paste_img_binary = sitk.Paste(
            template_img_binary,
            binary_struct,
            binary_struct.GetSize(),
            (0, 0, 0),
            crop_box_index,
        )
        results[structure_name] = paste_img_binary

        # Un-crop probability map
        paste_prob_img = sitk.Paste(
            template_img_prob,
            probability_map,
            probability_map.GetSize(),
            (0, 0, 0),
            crop_box_index,
        )
        results_prob[structure_name] = paste_prob_img

    """
    Step 8 - Post-processing
    """
    postprocessing_settings = settings["postprocessing_settings"]

    if postprocessing_settings["run_postprocessing"]:
        logger.info("Running post-processing.")

        # Remove any smaller components and perform morphological closing (hole filling)
        binaryfillhole_img = [
            int(postprocessing_settings["binaryfillhole_mm"] / sp) for sp in img.GetSpacing()
        ]

        for structure_name in postprocessing_settings["structures_for_binaryfillhole"]:

            if structure_name not in results.keys():
                continue

            contour_s = results[structure_name]
            contour_s = sitk.RelabelComponent(sitk.ConnectedComponent(contour_s)) == 1
            contour_s = sitk.BinaryMorphologicalClosing(contour_s, binaryfillhole_img)
            results[structure_name] = contour_s

        if len(postprocessing_settings["structures_for_overlap_correction"]) >= 2:
            # Remove any overlaps
            input_overlap = {
                s: results[s] for s in postprocessing_settings["structures_for_overlap_correction"]
            }
            output_overlap = correct_volume_overlap(input_overlap)

            for s in postprocessing_settings["structures_for_overlap_correction"]:
                results[s] = output_overlap[s]

    logger.info("Done!")

    return results, results_prob
