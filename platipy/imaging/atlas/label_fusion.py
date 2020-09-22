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


from functools import reduce

import numpy as np
import SimpleITK as sitk

def compute_weight_map(
    target_image,
    moving_image,
    vote_type="unweighted",
    vote_params={
        "sigma": 2.0,
        "epsilon": 1e-5,
        "factor": 1e12,
        "gain": 6,
        "blockSize": 5,
    },
):
    """
    Computes the weight map
    """

    # Cast to floating point representation, if necessary
    if target_image.GetPixelID() != 6:
        target_image = sitk.Cast(target_image, sitk.sitkFloat32)
    if moving_image.GetPixelID() != 6:
        moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)

    square_difference_image = sitk.SquaredDifference(
        target_image, moving_image)
    square_difference_image = sitk.Cast(
        square_difference_image, sitk.sitkFloat32)

    if vote_type.lower() == "unweighted":
        weight_map = target_image * 0.0 + 1.0

    elif vote_type.lower() == "global":
        factor = vote_params["factor"]
        sum_squared_difference = sitk.GetArrayFromImage(square_difference_image).sum(
            dtype=np.float
        )
        global_weight = factor / sum_squared_difference

        weight_map = target_image * 0.0 + global_weight

    elif vote_type.lower() == "local":
        sigma = vote_params["sigma"]
        epsilon = vote_params["epsilon"]

        raw_map = sitk.DiscreteGaussian(square_difference_image, sigma * sigma)
        weight_map = sitk.Pow(raw_map + epsilon, -1.0)

    elif vote_type.lower() == "block":
        factor = vote_params["factor"]
        gain = vote_params["gain"]
        block_size = vote_params["blockSize"]
        if isinstance(block_size, int):
            block_size = (block_size,) * target_image.GetDimension()

        # rawMap = sitk.Mean(square_difference_image, blockSize)
        raw_map = sitk.BoxMean(square_difference_image, block_size)
        weight_map = factor * sitk.Pow(raw_map, -1.0) ** abs(gain / 2.0)
        # Note: we divide gain by 2 to account for using the squared difference image
        #       which raises the power by 2 already.

    else:
        raise ValueError("Weighting scheme not valid.")

    return sitk.Cast(weight_map, sitk.sitkFloat32)


def combine_labels_staple(label_list_dict, threshold=1e-4):
    """
    Combine labels using STAPLE
    """

    combined_label_dict = {}

    structure_name_list = [list(i.keys()) for i in label_list_dict.values()]
    structure_name_list = np.unique(
        [item for sublist in structure_name_list for item in sublist]
    )

    for structure_name in structure_name_list:
        # Ensure all labels are binarised
        binary_labels = [
            sitk.BinaryThreshold(
                label_list_dict[i][structure_name], lowerThreshold=0.5)
            for i in label_list_dict
        ]

        # Perform STAPLE
        combined_label = sitk.STAPLE(binary_labels)

        # Normalise
        combined_label = sitk.RescaleIntensity(combined_label, 0, 1)

        # Threshold - grants vastly improved compression performance
        if threshold:
            combined_label = sitk.Threshold(
                combined_label, lower=threshold, upper=1, outsideValue=0.0
            )

        combined_label_dict[structure_name] = combined_label

    return combined_label_dict


def combine_labels(atlas_set, structure_name, label='DIR', threshold=1e-4, smooth_sigma=1.0):
    """
    Combine labels using weight maps
    """

    case_id_list = list(atlas_set.keys())

    if isinstance(structure_name, str):
        structure_name_list = [structure_name]
    elif isinstance(structure_name, list):
        structure_name_list = structure_name

    combined_label_dict = {}

    for structure_name in structure_name_list:
        # Find the cases which have the strucure (in case some cases do not)
        valid_case_id_list = [
            i for i in case_id_list if structure_name in atlas_set[i][label].keys()
        ]

        # Get valid weight images
        weight_image_list = [
            atlas_set[caseId][label]["Weight Map"] for caseId in valid_case_id_list
        ]

        # Sum the weight images
        weight_sum_image = reduce(lambda x, y: x + y, weight_image_list)
        weight_sum_image = sitk.Mask(
            weight_sum_image, weight_sum_image == 0, maskingValue=1, outsideValue=1
        )

        # Combine weight map with each label
        weighted_labels = [
            atlas_set[caseId][label]["Weight Map"]
            * sitk.Cast(atlas_set[caseId][label][structure_name], sitk.sitkFloat32)
            for caseId in valid_case_id_list
        ]

        # Combine all the weighted labels
        combined_label = reduce(
            lambda x, y: x + y, weighted_labels) / weight_sum_image

        # Smooth combined label
        combined_label = sitk.DiscreteGaussian(
            combined_label, smooth_sigma * smooth_sigma
        )

        # Normalise
        combined_label = sitk.RescaleIntensity(combined_label, 0, 1)

        # Threshold - grants vastly improved compression performance
        if threshold:
            combined_label = sitk.Threshold(
                combined_label, lower=threshold, upper=1, outsideValue=0.0
            )

        combined_label_dict[structure_name] = combined_label

    return combined_label_dict
