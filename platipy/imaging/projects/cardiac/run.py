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

from platipy.imaging.registration.utils import apply_transform, convert_mask_to_reg_structure

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
from platipy.imaging.label.iar import run_iar

from platipy.imaging.utils.vessel import vessel_spline_generation

from platipy.imaging.utils.crop import label_to_roi, crop_to_roi

from platipy.imaging.generation.mask import extend_mask

from platipy.imaging.label.utils import binary_encode_structure_list

ATLAS_PATH = "/atlas"
if "ATLAS_PATH" in os.environ:
    ATLAS_PATH = os.environ["ATLAS_PATH"]

CARDIAC_SETTINGS_DEFAULTS = {
    "atlas_settings": {
        "atlas_id_list": ["13", "17", "33", "12", "16", "22", "27"],
        "atlas_structure_list": ["WHOLEHEART", "LANTDESCARTERY_SPLINE"],
        "atlas_path": ATLAS_PATH,
        "atlas_image_format": "Case_{0}/Images/Case_{0}_CROP.nii.gz",
        "atlas_label_format": "Case_{0}/Structures/Case_{0}_{1}_CROP.nii.gz",
        "crop_atlas_to_structures": False,
        "crop_atlas_expansion_mm": (10, 10, 10),
        "guide_structure_name": "WHOLEHEART",
        "superior_extension": 30,
    },
    "auto_crop_target_image_settings": {
        "expansion_mm": [2, 2, 2],
    },
    "linear_registration_settings": {
        "reg_method": "similarity",
        "shrink_factors": [16, 8, 4],
        "smooth_sigmas": [0, 0, 0],
        "sampling_rate": 0.75,
        "default_value": -1024,
        "number_of_iterations": 50,
        "metric": "mean_squares",
        "optimiser": "gradient_descent_line_search",
        "verbose": False,
    },
    "deformable_registration_settings": {
        "isotropic_resample": True,
        "resolution_staging": [
            16,
            8,
            2,
        ],  # specify voxel size (mm) since isotropic_resample is set
        "iteration_staging": [5, 5, 5],
        "smoothing_sigmas": [0, 0, 0],
        "ncores": 8,
        "default_value": -1000,
        "verbose": False,
    },
    "structure_guided_registration_settings": {
        "isotropic_resample": True,
        "resolution_staging": [
            16,
            8,
            2,
        ],  # specify voxel size (mm) since isotropic_resample is set
        "iteration_staging": [25, 25, 25],
        "smoothing_sigmas": [0, 0, 0],
        "ncores": 8,
        "default_value": 0,
        "verbose": False,
    },
    "iar_settings": {
        "reference_structure": "WHOLEHEART",
        "smooth_distance_maps": True,
        "smooth_sigma": 1,
        "z_score_statistic": "mad",
        "outlier_method": "iqr",
        "outlier_factor": 1.5,
        "min_best_atlases": 5,
        "project_on_sphere": False,
    },
    "label_fusion_settings": {
        "vote_type": "unweighted",
        "vote_params": {},  # No parameters needed for majority voting
        "optimal_threshold": {"WHOLEHEART": 0.5},
    },
    "vessel_spline_settings": {
        "vessel_name_list": ["LANTDESCARTERY_SPLINE"],
        "vessel_radius_mm_dict": {"LANTDESCARTERY_SPLINE": 2},
        "scan_direction_dict": {"LANTDESCARTERY_SPLINE": "z"},
        "stop_condition_type_dict": {"LANTDESCARTERY_SPLINE": "count"},
        "stop_condition_value_dict": {"LANTDESCARTERY_SPLINE": 1},
    },
    "return_as_cropped": False,
}


def run_cardiac_segmentation(img, guide_structure=None, settings=CARDIAC_SETTINGS_DEFAULTS):
    """Runs the atlas-based cardiac segmentation

    Args:
        img (sitk.Image):
        settings (dict, optional): Dictionary containing settings for algorithm.
                                   Defaults to default_settings.

    Returns:
        dict: Dictionary containing output of segmentation
    """

    results = {}
    results_prob = {}

    return_as_cropped = settings["return_as_cropped"]

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

    if guide_structure:

        crop_box_size, crop_box_index = label_to_roi(guide_structure, expansion_mm=expansion_mm)
        img_crop = crop_to_roi(img, crop_box_size, crop_box_index)

        guide_structure = crop_to_roi(guide_structure, crop_box_size, crop_box_index)
        target_reg_structure = convert_mask_to_reg_structure(guide_structure, expansion=0)

    else:
        quick_reg_settings = {
            "reg_method": "similarity",
            "shrink_factors": [8],
            "smooth_sigmas": [0],
            "sampling_rate": 0.75,
            "default_value": -1024,
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

    logger.info(
        f"Calculated crop box\n\
                {crop_box_index}\n\
                {crop_box_size}\n\n\
                Volume reduced by factor {np.product(img.GetSize())/np.product(crop_box_size)}"
    )

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

        if guide_structure:
            guide_structure_name = settings["atlas_settings"]["guide_structure_name"]
            target_reg_image = target_reg_structure
            atlas_reg_image = convert_mask_to_reg_structure(
                atlas_set[atlas_id]["Original"][guide_structure_name], expansion=0
            )

        else:
            target_reg_image = img_crop
            atlas_reg_image = atlas_set[atlas_id]["Original"]["CT Image"]

        _, initial_tfm = linear_registration(
            target_reg_image,
            atlas_reg_image,
            **linear_registration_settings,
        )

        # Save in the atlas dict
        atlas_set[atlas_id]["RIR"]["Transform"] = initial_tfm

        if guide_structure:
            atlas_set[atlas_id]["RIR"]["Reg Mask"] = apply_transform(
                input_image=atlas_reg_image,
                reference_image=img_crop,
                transform=initial_tfm,
                default_value=0,
                interpolator=sitk.sitkNearestNeighbor,
            )

            expanded_atlas_guide_structure = extend_mask(
                atlas_set[atlas_id]["Original"][guide_structure_name],
                direction=("ax", "sup"),
                extension_mm=settings["atlas_settings"]["superior_extension"],
                interior_mm_shape=settings["atlas_settings"]["superior_extension"] / 2,
            )

            atlas_set[atlas_id]["RIR"][guide_structure_name + "EXPANDED"] = apply_transform(
                input_image=expanded_atlas_guide_structure,
                reference_image=img_crop,
                transform=initial_tfm,
                default_value=0,
                interpolator=sitk.sitkNearestNeighbor,
            )

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
    if guide_structure:
        structure_guided_registration_settings = settings["structure_guided_registration_settings"]

        logger.info("Running structure-guided deformable registration on atlas labels")

        for atlas_id in atlas_id_list:

            logger.info(f"  > atlas {atlas_id}")

            # Register the atlases
            atlas_set[atlas_id]["DIR_STRUCT"] = {}

            deform_image, struct_guided_tfm, _ = fast_symmetric_forces_demons_registration(
                target_reg_structure,
                atlas_set[atlas_id]["RIR"]["Reg Mask"],
                **structure_guided_registration_settings,
            )

            # Save in the atlas dict
            atlas_set[atlas_id]["DIR_STRUCT"]["Reg Mask"] = deform_image
            atlas_set[atlas_id]["DIR_STRUCT"]["Transform"] = struct_guided_tfm

            atlas_set[atlas_id]["DIR_STRUCT"]["CT Image"] = apply_transform(
                input_image=atlas_set[atlas_id]["RIR"]["CT Image"],
                transform=struct_guided_tfm,
                default_value=-1000,
                interpolator=sitk.sitkLinear,
            )

            atlas_set[atlas_id]["DIR_STRUCT"][guide_structure_name + "EXPANDED"] = apply_transform(
                input_image=atlas_set[atlas_id]["RIR"][guide_structure_name + "EXPANDED"],
                reference_image=img_crop,
                transform=struct_guided_tfm,
                default_value=0,
                interpolator=sitk.sitkNearestNeighbor,
            )

            # sitk.WriteImage(deform_image, f"./DIR_STRUCT_{atlas_id}.nii.gz")

            for struct in atlas_structure_list:
                input_struct = atlas_set[atlas_id]["RIR"][struct]
                atlas_set[atlas_id]["DIR_STRUCT"][struct] = apply_transform(
                    input_image=input_struct,
                    transform=struct_guided_tfm,
                    default_value=0,
                    interpolator=sitk.sitkNearestNeighbor,
                )

            atlas_set[atlas_id]["RIR"] = None

    # Settings
    deformable_registration_settings = settings["deformable_registration_settings"]

    logger.info("Running DIR to refine atlas image registration")

    for atlas_id in atlas_id_list:

        logger.info(f"  > atlas {atlas_id}")

        # Register the atlases
        atlas_set[atlas_id]["DIR"] = {}

        if guide_structure:
            label = "DIR_STRUCT"
        else:
            label = "RIR"

        atlas_reg_image = atlas_set[atlas_id][label]["CT Image"]
        target_reg_image = img_crop

        if guide_structure:
            expanded_atlas_mask = atlas_set[atlas_id]["DIR_STRUCT"][
                guide_structure_name + "EXPANDED"
            ]
            expanded_target_mask = extend_mask(
                guide_structure,
                direction=("ax", "sup"),
                extension_mm=settings["atlas_settings"]["superior_extension"],
                interior_mm_shape=settings["atlas_settings"]["superior_extension"] / 2,
            )

            combined_mask = sitk.Maximum(expanded_atlas_mask, expanded_target_mask)

            atlas_reg_image = sitk.Mask(atlas_reg_image, combined_mask, outsideValue=-1000)
            atlas_reg_image = sitk.Mask(
                atlas_reg_image, atlas_reg_image > -400, outsideValue=-1000
            )

            target_reg_image = sitk.Mask(target_reg_image, combined_mask, outsideValue=-1000)
            target_reg_image = sitk.Mask(
                target_reg_image, atlas_reg_image > -400, outsideValue=-1000
            )

        deform_image, dir_tfm, _ = fast_symmetric_forces_demons_registration(
            target_reg_image,
            atlas_reg_image,
            **deformable_registration_settings,
        )

        # Save in the atlas dict
        atlas_set[atlas_id]["DIR"]["Transform"] = dir_tfm

        atlas_set[atlas_id]["DIR"]["CT Image"] = apply_transform(
            input_image=atlas_set[atlas_id][label]["CT Image"],
            transform=dir_tfm,
            default_value=-1000,
            interpolator=sitk.sitkLinear,
        )

        for struct in atlas_structure_list:
            input_struct = atlas_set[atlas_id][label][struct]
            atlas_set[atlas_id]["DIR"][struct] = apply_transform(
                input_image=input_struct,
                transform=dir_tfm,
                default_value=0,
                interpolator=sitk.sitkNearestNeighbor,
            )

        atlas_set[atlas_id][label] = None

    """
    Step 4 - Iterative atlas removal
    - This is an automatic process that will attempt to remove inconsistent atlases from the entire set

    """
    # Compute weight maps
    # Here we use simple GWV as this minises the potentially negative influence of mis-registered
    # atlases
    iar_settings = settings["iar_settings"]

    if iar_settings["reference_structure"]:

        for atlas_id in atlas_id_list:
            atlas_image = atlas_set[atlas_id]["DIR"]["CT Image"]
            weight_map = compute_weight_map(img_crop, atlas_image, vote_type="global")
            atlas_set[atlas_id]["DIR"]["Weight Map"] = weight_map

        atlas_set = run_iar(atlas_set=atlas_set, **iar_settings)

    else:
        logger.info("IAR: No reference structure, skipping iterative atlas removal.")

    """
    Step 4 - Vessel Splining

    """
    vessel_spline_settings = settings["vessel_spline_settings"]

    if len(vessel_spline_settings["vessel_name_list"]) > 0:

        segmented_vessel_dict = vessel_spline_generation(
            img_crop, atlas_set, **vessel_spline_settings
        )
    else:
        logger.info("No vessel splining required, continue.")

    """
    Step 5 - Label Fusion
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

    template_img_binary = sitk.Cast((img * 0), sitk.sitkUInt8)
    template_img_prob = sitk.Cast((img * 0), sitk.sitkFloat64)

    vote_structures = settings["label_fusion_settings"]["optimal_threshold"].keys()

    for structure_name in vote_structures:

        probability_map = combined_label_dict[structure_name]

        optimal_threshold = settings["label_fusion_settings"]["optimal_threshold"][structure_name]

        binary_struct = process_probability_image(probability_map, optimal_threshold)

        if return_as_cropped:
            results[structure_name] = binary_struct
            results_prob[structure_name] = probability_map

        else:
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

    for structure_name in vessel_spline_settings["vessel_name_list"]:
        binary_struct = segmented_vessel_dict[structure_name]

        if return_as_cropped:
            results[structure_name] = binary_struct

            results_prob[structure_name] = [
                atlas_set[atlas_id]["DIR"][structure_name] for atlas_id in list(atlas_set.keys())
            ]

        else:
            # Un-crop binary vessel
            paste_img_binary = sitk.Paste(
                template_img_binary,
                binary_struct,
                binary_struct.GetSize(),
                (0, 0, 0),
                crop_box_index,
            )
            results[structure_name] = paste_img_binary

            vessel_list = []
            for atlas_id in list(atlas_set.keys()):
                paste_img_binary = sitk.Paste(
                    template_img_binary,
                    atlas_set[atlas_id]["DIR"][structure_name],
                    atlas_set[atlas_id]["DIR"][structure_name].GetSize(),
                    (0, 0, 0),
                    crop_box_index,
                )
                vessel_list.append(paste_img_binary)

            # Encode list of vessels
            encoded_vessels = binary_encode_structure_list(vessel_list)
            results_prob[structure_name] = encoded_vessels

    if return_as_cropped:
        results["CROP_IMAGE"] = img_crop

    return results, results_prob
