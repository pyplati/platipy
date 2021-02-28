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


import tempfile

import SimpleITK as sitk
import numpy as np
import os

from loguru import logger

from platipy.imaging.registration.registration import (
    initial_registration,
    transform_propagation,
    fast_symmetric_forces_demons_registration,
    apply_field,
)

from platipy.imaging.label.label_operations import process_probability_image

from platipy.imaging.atlas.label_fusion import (
    compute_weight_map,
    combine_labels,
)

from platipy.imaging.atlas.iterative_atlas_removal import run_iar

from platipy.imaging.projects.cardiac.utils import vesselSplineGeneration

from platipy.imaging.utils.tools import label_to_roi, crop_to_roi

ATLAS_PATH = "/atlas"
if "ATLAS_PATH" in os.environ:
    ATLAS_PATH = os.environ["ATLAS_PATH"]

CARDIAC_SETTINGS_DEFAULTS = {
    "outputFormat": "Auto_{0}.nii.gz",
    "atlasSettings": {
        "atlasIdList": ["13", "17", "33", "12", "16", "22", "27"],
        "atlasStructures": ["WHOLEHEART", "LANTDESCARTERY_SPLINE"],
        "atlasPath": ATLAS_PATH,
        "atlasImageFormat": "Case_{0}/Images/Case_{0}_CROP.nii.gz",
        "atlasLabelFormat": "Case_{0}/Structures/Case_{0}_{1}_CROP.nii.gz",
        "autoCropAtlas": True,
    },
    "autoCropSettings": {
        "expansion": [2, 2, 2],
    },
    "rigidSettings": {
        "initialReg": "Similarity",
        "options": {
            "shrink_factors": [16, 8, 4],
            "smooth_sigmas": [0, 0, 0],
            "sampling_rate": 0.75,
            "default_value": -1024,
            "number_of_iterations": 50,
            "final_interp": sitk.sitkBSpline,
            "metric": "mean_squares",
            "optimiser": "gradient_descent_line_search",
        },
        "trace": False,
        "guideStructure": False,
    },
    "deformableSettings": {
        "isotropicResample": True,
        "resolutionStaging": [
            16,
            8,
            2,
        ],  # specify voxel size (mm) since isotropic_resample is set
        "iterationStaging": [5, 5, 5],
        "smoothingSigmas": [0, 0, 0],
        "ncores": 8,
        "trace": False,
    },
    "IARSettings": {
        "referenceStructure": "WHOLEHEART",
        "smoothDistanceMaps": True,
        "smoothSigma": 1,
        "zScoreStatistic": "MAD",
        "outlierMethod": "IQR",
        "outlierFactor": 1.5,
        "minBestAtlases": 5,
        "project_on_sphere": False,
    },
    "labelFusionSettings": {
        "voteType": "unweighted",
        "voteParams": {},  # No parameters needed for majority voting
        "optimalThreshold": {"WHOLEHEART": 0.44},
    },
    "vesselSpliningSettings": {
        "vesselNameList": ["LANTDESCARTERY_SPLINE"],
        "vesselRadius_mm": {"LANTDESCARTERY_SPLINE": 2},
        "spliningDirection": {"LANTDESCARTERY_SPLINE": "z"},
        "stopCondition": {"LANTDESCARTERY_SPLINE": "count"},
        "stopConditionValue": {"LANTDESCARTERY_SPLINE": 1},
    },
    "returnAsCropped": False,
}


def run_cardiac_segmentation(img, settings=CARDIAC_SETTINGS_DEFAULTS):
    """Runs the atlas-based cardiac segmentation

    Args:
        img (sitk.Image):
        settings (dict, optional): Dictionary containing settings for algorithm.
                                   Defaults to default_settings.

    Returns:
        dict: Dictionary containing output of segmentation
    """

    results = {}
    return_as_cropped = settings["returnAsCropped"]

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
    atlas_path = settings["atlasSettings"]["atlasPath"]
    atlas_id_list = settings["atlasSettings"]["atlasIdList"]
    atlas_structures = settings["atlasSettings"]["atlasStructures"]

    atlas_image_format = settings["atlasSettings"]["atlasImageFormat"]
    atlas_label_format = settings["atlasSettings"]["atlasLabelFormat"]

    auto_crop_atlas = settings["atlasSettings"]["autoCropAtlas"]

    atlas_set = {}
    for atlas_id in atlas_id_list:
        atlas_set[atlas_id] = {}
        atlas_set[atlas_id]["Original"] = {}

        image = sitk.ReadImage(f"{atlas_path}/{atlas_image_format.format(atlas_id)}")

        structures = {
            struct: sitk.ReadImage(f"{atlas_path}/{atlas_label_format.format(atlas_id, struct)}")
            for struct in atlas_structures
        }

        if auto_crop_atlas:
            logger.info(f"Automatically cropping atlas: {atlas_id}")

            original_volume = np.product(image.GetSize())

            label_stats_image_filter = sitk.LabelStatisticsImageFilter()
            label_stats_image_filter.Execute(image, sum(structures.values()) > 0)
            bounding_box = list(label_stats_image_filter.GetBoundingBox(1))
            index = [bounding_box[x * 2] for x in range(3)]
            size = [bounding_box[(x * 2) + 1] - bounding_box[x * 2] for x in range(3)]

            image = sitk.RegionOfInterest(image, size=size, index=index)

            final_volume = np.product(image.GetSize())
            logger.info(f"  > Volume reduced by factor {original_volume/final_volume:.2f}")

            for struct in atlas_structures:
                structures[struct] = sitk.RegionOfInterest(
                    structures[struct], size=size, index=index
                )

        atlas_set[atlas_id]["Original"]["CT Image"] = image

        for struct in atlas_structures:
            atlas_set[atlas_id]["Original"][struct] = structures[struct]

    """
    Step 1 - Automatic cropping using a translation transform
    - Registration of atlas images (maximum 5)
    - Potential expansion of the bounding box to ensure entire volume of interest is enclosed
    - Target image is cropped
    """
    # Settings
    quick_reg_settings = {
        "shrink_factors": [8],
        "smooth_sigmas": [0],
        "sampling_rate": 0.75,
        "default_value": -1024,
        "number_of_iterations": 25,
        "final_interp": 3,
        "metric": "mean_squares",
        "optimiser": "gradient_descent_line_search",
    }

    registered_crop_images = []

    logger.info(f"Running initial Translation tranform to crop image volume")

    for atlas_id in atlas_id_list[: min([8, len(atlas_id_list)])]:

        logger.info(f"  > atlas {atlas_id}")

        # Register the atlases
        atlas_set[atlas_id]["RIR"] = {}
        atlas_image = atlas_set[atlas_id]["Original"]["CT Image"]

        reg_image, _ = initial_registration(
            img,
            atlas_image,
            moving_structure=False,
            fixed_structure=False,
            options=quick_reg_settings,
            trace=False,
            reg_method="Similarity",
        )

        registered_crop_images.append(sitk.Cast(reg_image, sitk.sitkFloat32))

        del reg_image

    combined_image_extent = sum(registered_crop_images) / len(registered_crop_images) > -1000

    shape_filter = sitk.LabelShapeStatisticsImageFilter()
    shape_filter.Execute(combined_image_extent)
    bounding_box = np.array(shape_filter.GetBoundingBox(1))

    """
    Crop image to region of interest (ROI)
    --> Defined by images
    """

    expansion = settings["autoCropSettings"]["expansion"]
    expansion_array = expansion * np.array(img.GetSpacing())

    crop_box_size, crop_box_index = label_to_roi(
        img, combined_image_extent, expansion=expansion_array
    )
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
    initial_reg = settings["rigidSettings"]["initialReg"]
    rigid_options = settings["rigidSettings"]["options"]
    trace = settings["rigidSettings"]["trace"]
    guide_structure = settings["rigidSettings"]["guideStructure"]

    logger.info(f"Running {initial_reg} tranform to align atlas images")

    for atlas_id in atlas_id_list:
        # Register the atlases

        logger.info(f"  > atlas {atlas_id}")

        atlas_set[atlas_id]["RIR"] = {}
        atlas_image = atlas_set[atlas_id]["Original"]["CT Image"]

        if guide_structure:
            atlas_struct = atlas_set[atlas_id]["Original"][guide_structure]
        else:
            atlas_struct = False

        rigid_image, initial_tfm = initial_registration(
            img_crop,
            atlas_image,
            moving_structure=atlas_struct,
            options=rigid_options,
            trace=trace,
            reg_method=initial_reg,
        )

        # Save in the atlas dict
        atlas_set[atlas_id]["RIR"]["CT Image"] = rigid_image
        atlas_set[atlas_id]["RIR"]["Transform"] = initial_tfm

        # sitk.WriteImage(rigidImage, f'./RR_{atlas_id}.nii.gz')

        for struct in atlas_structures:
            input_struct = atlas_set[atlas_id]["Original"][struct]
            atlas_set[atlas_id]["RIR"][struct] = transform_propagation(
                img_crop, input_struct, initial_tfm, structure=True, interp=sitk.sitkLinear,
            )

    """
    Step 3 - Deformable image registration
    - Using Fast Symmetric Diffeomorphic Demons
    """
    # Settings
    isotropic_resample = settings["deformableSettings"]["isotropicResample"]
    resolution_staging = settings["deformableSettings"]["resolutionStaging"]
    iteration_staging = settings["deformableSettings"]["iterationStaging"]
    smoothing_sigmas = settings["deformableSettings"]["smoothingSigmas"]
    ncores = settings["deformableSettings"]["ncores"]
    trace = settings["deformableSettings"]["trace"]

    logger.info(f"Running DIR to register atlas images")

    for atlas_id in atlas_id_list:

        logger.info(f"  > atlas {atlas_id}")

        # Register the atlases
        atlas_set[atlas_id]["DIR"] = {}
        atlas_image = atlas_set[atlas_id]["RIR"]["CT Image"]

        cleaned_img_crop = sitk.Mask(img_crop, atlas_image > -1023, outsideValue=-1024)

        deform_image, deform_field = fast_symmetric_forces_demons_registration(
            cleaned_img_crop,
            atlas_image,
            resolution_staging=resolution_staging,
            iteration_staging=iteration_staging,
            isotropic_resample=isotropic_resample,
            smoothing_sigmas=smoothing_sigmas,
            ncores=ncores,
            trace=trace,
        )

        # Save in the atlas dict
        atlas_set[atlas_id]["DIR"]["CT Image"] = deform_image
        atlas_set[atlas_id]["DIR"]["Transform"] = deform_field

        # sitk.WriteImage(deformImage, f'./DIR_{atlas_id}.nii.gz')

        for struct in atlas_structures:
            input_struct = atlas_set[atlas_id]["RIR"][struct]
            atlas_set[atlas_id]["DIR"][struct] = apply_field(
                input_struct, deform_field, structure=True, interp=sitk.sitkLinear
            )

    """
    Step 4 - Iterative atlas removal
    - This is an automatic process that will attempt to remove inconsistent atlases from the entire set

    """
    # Compute weight maps
    # Here we use simple GWV as this minises the potentially negative influence of mis-registered atlases
    reference_structure = settings["IARSettings"]["referenceStructure"]

    if reference_structure:

        smooth_distance_maps = settings["IARSettings"]["smoothDistanceMaps"]
        smooth_sigma = settings["IARSettings"]["smoothSigma"]
        z_score_statistic = settings["IARSettings"]["zScoreStatistic"]
        outlier_method = settings["IARSettings"]["outlierMethod"]
        outlier_factor = settings["IARSettings"]["outlierFactor"]
        min_best_atlases = settings["IARSettings"]["minBestAtlases"]
        project_on_sphere = settings["IARSettings"]["project_on_sphere"]

        for atlas_id in atlas_id_list:
            atlas_image = atlas_set[atlas_id]["DIR"]["CT Image"]
            weight_map = compute_weight_map(img_crop, atlas_image, vote_type="global")
            atlas_set[atlas_id]["DIR"]["Weight Map"] = weight_map

        atlas_set = run_iar(
            atlas_set=atlas_set,
            structure_name=reference_structure,
            smooth_maps=smooth_distance_maps,
            smooth_sigma=smooth_sigma,
            z_score=z_score_statistic,
            outlier_method=outlier_method,
            min_best_atlases=min_best_atlases,
            n_factor=outlier_factor,
            iteration=0,
            single_step=False,
            project_on_sphere=project_on_sphere,
        )

    else:
        logger.info("IAR: No reference structure, skipping iterative atlas removal.")

    """
    Step 4 - Vessel Splining

    """

    vessel_name_list = settings["vesselSpliningSettings"]["vesselNameList"]

    if len(vessel_name_list) > 0:

        vessel_radius_mm = settings["vesselSpliningSettings"]["vesselRadius_mm"]
        splining_direction = settings["vesselSpliningSettings"]["spliningDirection"]
        stop_condition = settings["vesselSpliningSettings"]["stopCondition"]
        stop_condition_value = settings["vesselSpliningSettings"]["stopConditionValue"]

        segmented_vessel_dict = vesselSplineGeneration(
            img_crop,
            atlas_set,
            vessel_name_list,
            vessel_radius_mm,
            stop_condition,
            stop_condition_value,
            splining_direction,
        )
    else:
        logger.info("No vessel splining required, continue.")

    """
    Step 5 - Label Fusion
    """
    # Compute weight maps
    vote_type = settings["labelFusionSettings"]["voteType"]
    vote_params = settings["labelFusionSettings"]["voteParams"]

    # Compute weight maps
    for atlas_id in list(atlas_set.keys()):
        atlas_image = atlas_set[atlas_id]["DIR"]["CT Image"]
        weight_map = compute_weight_map(
            img_crop, atlas_image, vote_type=vote_type, vote_params=vote_params
        )
        atlas_set[atlas_id]["DIR"]["Weight Map"] = weight_map

    combined_label_dict = combine_labels(atlas_set, atlas_structures)

    """
    Step 6 - Paste the cropped structure into the original image space
    """

    template_img_binary = sitk.Cast((img * 0), sitk.sitkUInt8)

    vote_structures = settings["labelFusionSettings"]["optimalThreshold"].keys()

    for structure_name in vote_structures:

        probability_map = combined_label_dict[structure_name]

        optimal_threshold = settings["labelFusionSettings"]["optimalThreshold"][structure_name]

        binary_struct = process_probability_image(probability_map, optimal_threshold)

        if return_as_cropped:
            results[structure_name] = binary_struct

        else:
            paste_binary_img = sitk.Paste(
                template_img_binary,
                binary_struct,
                binary_struct.GetSize(),
                (0, 0, 0),
                crop_box_index,
            )

            results[structure_name] = paste_binary_img

    for structure_name in vessel_name_list:
        binary_struct = segmented_vessel_dict[structure_name]

        if return_as_cropped:
            results[structure_name] = binary_struct

        else:
            paste_img_binary = sitk.Paste(
                template_img_binary,
                binary_struct,
                binary_struct.GetSize(),
                (0, 0, 0),
                crop_box_index,
            )

            results[structure_name] = paste_img_binary

    if return_as_cropped:
        results["CROP_IMAGE"] = img_crop

    return results
