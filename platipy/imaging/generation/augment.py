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

from abc import ABC, abstractmethod
from collections.abc import Iterable
import random
import logging
import sys

from pathlib import Path

from argparse import ArgumentParser

import SimpleITK as sitk
import numpy as np

import matplotlib.pyplot as plt
from platipy.imaging import ImageVisualiser

from platipy.imaging.generation.dvf import (
    generate_field_shift,
    generate_field_expand,
)

from platipy.imaging.generation.mask import (
    get_bone_mask,
)

from platipy.imaging.registration.utils import apply_transform

from platipy.imaging.utils.lung import detect_holes
from platipy.imaging.label.utils import get_union_mask
from platipy.imaging.utils.crop import label_to_roi, crop_to_roi

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logger = logging.getLogger(__name__)


def apply_augmentation(image, augmentation, context_map=None, masks=[]):
    if not isinstance(image, sitk.Image):
        raise AttributeError("image should be a SimpleITK.Image")

    if isinstance(augmentation, DeformableAugment):
        augmentation = [augmentation]

    if not isinstance(augmentation, Iterable):
        raise AttributeError(
            "augmentation must be a DeformableAugment or an iterable (such as list) of"
            "DeformableAugment's"
        )

    # transforms = []
    transform = None
    dvf = None
    for aug in augmentation:
        if not isinstance(aug, DeformableAugment):
            raise AttributeError("Each augmentation must be of type DeformableAugment")

        logger.debug(str(aug))
        tfm, field = aug.augment()
        # transforms.append(tfm)

        if dvf is None:
            dvf = field
            transform = tfm
        else:
            dvf += field
            transform = sitk.CompositeTransform([transform, tfm])

    # transform = sitk.CompositeTransform(transforms)
    # del transforms

    image_deformed = apply_transform(
        image,
        transform=transform,
        default_value=int(sitk.GetArrayViewFromImage(image).min()),
        interpolator=sitk.sitkLinear,
    )

    masks_deformed = []
    for mask in masks:
        def_mask = apply_transform(
            mask,
            transform=transform,
            default_value=0,
            interpolator=sitk.sitkNearestNeighbor,
        )

        def_mask = sitk.BinaryMorphologicalClosing(def_mask, [3, 3, 3])

        masks_deformed.append(def_mask)

    cmap_deformed = None
    if context_map is not None:
        cmap_deformed = apply_transform(
            context_map,
            transform=transform,
            default_value=0,
            interpolator=sitk.sitkNearestNeighbor,
        )

    if masks:
        return image_deformed, cmap_deformed, masks_deformed, dvf

    return image_deformed, cmap_deformed, dvf


def generate_random_augmentation(ct_image, masks, augmentation_types):
    augmentation = []

    probabilities = [a["probability"] for a in augmentation_types]
    prob_total = sum(probabilities)
    prob_none = 1.0 - prob_total
    if prob_none < 0:
        prob_none = 0

    for mask in masks:
        aug = random.choices(
            augmentation_types + [None], weights=probabilities + [prob_none]
        )[0]

        if aug is None:
            continue

        aug_class = aug["class"]
        aug_args = {}
        for arg in aug["args"]:
            value = aug["args"][arg]
            if isinstance(value, list):
                # Randomly sample for each dim
                result = []
                for rng in value:
                    result.append(random.randint(rng[0], rng[1]))
                value = result
            elif isinstance(value, tuple):
                # Randomly sample a value in range
                value = random.randint(value[0], value[1])

            if arg == "bone_mask" and aug["args"][arg]:
                value = get_bone_mask(ct_image)

            aug_args[arg] = value
        augmentation.append(aug_class(mask, **aug_args))
    return augmentation


class DeformableAugment(ABC):
    @abstractmethod
    def augment(self):
        # return deformation
        pass


class ShiftAugment(DeformableAugment):
    def __init__(self, mask, vector_shift=(10, 10, 10), gaussian_smooth=5):
        self.mask = mask
        self.vector_shift = vector_shift
        self.gaussian_smooth = gaussian_smooth

    def augment(self):
        _, transform, dvf = generate_field_shift(
            self.mask,
            self.vector_shift,
            self.gaussian_smooth,
        )
        return transform, dvf

    def __str__(self):
        return f"Shift with vector: {self.vector_shift}, gauss: {self.gaussian_smooth}"


class ExpandAugment(DeformableAugment):
    def __init__(
        self, mask, vector_expand=(10, 10, 10), gaussian_smooth=5, bone_mask=False
    ):
        self.mask = mask
        self.vector_expand = vector_expand
        self.gaussian_smooth = gaussian_smooth
        self.bone_mask = bone_mask

    def augment(self):
        _, transform, dvf = generate_field_expand(
            self.mask,
            bone_mask=self.bone_mask,
            expand=self.vector_expand,
            gaussian_smooth=self.gaussian_smooth,
        )

        return transform, dvf

    def __str__(self):
        return (
            f"Expand with vector: {self.vector_expand}, smooth: {self.gaussian_smooth}"
        )


class ContractAugment(DeformableAugment):
    def __init__(
        self, mask, vector_contract=(10, 10, 10), gaussian_smooth=5, bone_mask=False
    ):
        self.mask = mask
        self.vector_contract = [
            int(-x / s) for x, s in zip(vector_contract, mask.GetSpacing())
        ]
        self.gaussian_smooth = gaussian_smooth
        self.bone_mask = bone_mask

    def augment(self):
        _, transform, dvf = generate_field_expand(
            self.mask,
            bone_mask=self.bone_mask,
            expand=self.vector_contract,
            gaussian_smooth=self.gaussian_smooth,
        )
        return transform, dvf

    def __str__(self):
        return f"Contract with vector: {self.vector_contract}, smooth: {self.gaussian_smooth}"


def augment_data(args):
    random.seed(args.seed)

    augmentation_types = []

    if args.enable_shift:
        augmentation_types.append(
            {
                "class": ShiftAugment,
                "args": {
                    "vector_shift": [
                        tuple(args.shift_x_range),
                        tuple(args.shift_y_range),
                        tuple(args.shift_z_range),
                    ],
                    "gaussian_smooth": tuple(args.shift_smooth_range),
                },
                "probability": args.shift_probability,
            }
        )

    if args.enable_contract:
        augmentation_types.append(
            {
                "class": ContractAugment,
                "args": {
                    "vector_contract": [
                        tuple(args.contract_x_range),
                        tuple(args.contract_y_range),
                        tuple(args.contract_z_range),
                    ],
                    "gaussian_smooth": tuple(args.contract_smooth_range),
                    "bone_mask": args.contract_bone_mask,
                },
                "probability": args.contract_probability,
            }
        )

    if args.enable_expand:
        augmentation_types.append(
            {
                "class": ExpandAugment,
                "args": {
                    "vector_expand": [
                        tuple(args.expand_x_range),
                        tuple(args.expand_y_range),
                        tuple(args.expand_z_range),
                    ],
                    "gaussian_smooth": tuple(args.expand_smooth_range),
                    "bone_mask": args.expand_bone_mask,
                },
                "probability": args.expand_probability,
            }
        )

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    cases = [
        p.name.replace(".nii.gz", "")
        for p in data_dir.glob(args.case_glob)
        if not p.name.startswith(".")
    ]
    cases.sort()

    data = {
        case: {
            "image": data_dir.joinpath(args.image_glob.format(case=case)),
            "context_map": data_dir.joinpath(args.context_map_glob.format(case=case)),
            "label": [
                i
                for sl in [
                    list(data_dir.glob(lg.format(case=case))) for lg in args.label_glob
                ]
                for i in sl
            ],
        }
        for case in cases
    }

    for case in cases:
        logger.info(f"Augmenting for case: {case}")

        ct_image_original = sitk.ReadImage(str(data[case]["image"]))

        cmap_original = None
        if data[case]["context_map"]:
            cmap_original = sitk.ReadImage(str(data[case]["context_map"]))

        # Get list of structures to generate augmentations off
        logger.debug("Collecting structures")
        all_masks = []
        all_names = []
        for structure_path in data[case]["label"]:
            mask = sitk.ReadImage(str(structure_path))

            all_masks.append(mask)
            all_names.append(structure_path.name.replace(".nii.gz", ""))

        logger.debug("Cropping to regions around all structures")
        union_mask = get_union_mask(all_masks)
        size, index = label_to_roi(union_mask, expansion_mm=[25, 25, 25])
        ct_image = crop_to_roi(ct_image_original, size, index)

        for m, mask in enumerate(all_masks):
            all_masks[m] = crop_to_roi(mask, size, index)

        if args.enable_fill_holes:
            logger.debug("Finding holes")
            label_image, labels = detect_holes(ct_image)

        # Generate x random augmentations per case
        for i in range(args.augmentations_per_case):
            logger.debug(f"Generating augmentation {i}")

            ct_image = sitk.ReadImage(str(data[case]["image"]))
            ct_image = crop_to_roi(ct_image, size, index)

            cmap = None
            if data[case]["context_map"]:
                cmap = sitk.ReadImage(str(data[case]["context_map"]))
                cmap = crop_to_roi(cmap, size, index)

            if args.enable_fill_holes:
                logger.debug("Filling holes")

                for label in labels[1:]:  # Skip first hole since likely air around body
                    if random.random() > args.fill_probability:
                        continue

                    hole = label_image == label["label"]
                    hole_dilate = sitk.BinaryDilate(hole, (2, 2, 2), sitk.sitkBall)
                    contour_points = sitk.BinaryContour(hole_dilate)
                    fill_value = np.median(
                        sitk.GetArrayFromImage(ct_image)[
                            sitk.GetArrayFromImage(contour_points) == 1
                        ]
                    )

                    ct_arr = sitk.GetArrayFromImage(ct_image)
                    ct_arr[sitk.GetArrayFromImage(hole_dilate) == 1] = fill_value
                    ct_filled = sitk.GetImageFromArray(ct_arr)
                    ct_filled.CopyInformation(ct_image)

                    ct_image = ct_filled

            augmented_case_path = output_dir.joinpath(case, f"augment_{i}")
            augmented_case_path.mkdir(exist_ok=True, parents=True)

            logger.debug("Generating random augmentations")
            augmentation = generate_random_augmentation(
                ct_image, all_masks, augmentation_types
            )

            dvf = None
            augmented_cmap = None

            if len(augmentation) == 0:
                logger.debug(
                    "No augmentations generated, generated image won't differ from original"
                )

                augmented_image = ct_image
                augmented_masks = all_masks
            else:
                logger.debug("Applying augmentation")
                (
                    augmented_image,
                    augmented_cmap,
                    augmented_masks,
                    dvf,
                ) = apply_augmentation(
                    ct_image, augmentation, context_map=cmap, masks=all_masks
                )

            # Save off image
            augmented_image_path = augmented_case_path.joinpath("CT.nii.gz")
            ct_image_original[
                index[0] : index[0] + size[0],
                index[1] : index[1] + size[1],
                index[2] : index[2] + size[2],
            ] = augmented_image
            sitk.WriteImage(ct_image_original, str(augmented_image_path))

            # Save off context map if we have one
            if augmented_cmap:
                augmented_cmap_path = augmented_case_path.joinpath("context_map.nii.gz")
                cmap_original[
                    index[0] : index[0] + size[0],
                    index[1] : index[1] + size[1],
                    index[2] : index[2] + size[2],
                ] = augmented_cmap
                sitk.WriteImage(cmap_original, str(augmented_cmap_path))

            vis = ImageVisualiser(image=ct_image, figure_size_in=6)
            vis.add_comparison_overlay(augmented_image)
            if dvf is not None:
                vis.add_vector_overlay(dvf, arrow_scale=1, subsample=(4, 12, 12))
            for mask_name, mask, augmented_mask in zip(
                all_names, all_masks, augmented_masks
            ):
                vis.add_contour(
                    {f"{mask_name}": mask, f"{mask_name} (augmented)": augmented_mask}
                )

                logger.debug(f"Applying augmentation to mask: {mask_name}")
                augmented_mask_path = augmented_case_path.joinpath(
                    f"{mask_name}.nii.gz"
                )
                augmented_mask = sitk.Resample(
                    augmented_mask,
                    ct_image_original,
                    sitk.Transform(),
                    sitk.sitkNearestNeighbor,
                )
                sitk.WriteImage(augmented_mask, str(augmented_mask_path))

            fig = vis.show()

            figure_path = augmented_case_path.joinpath("aug.png")
            fig.savefig(figure_path, bbox_inches="tight")
            plt.close()


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--seed", type=int, default=42, help="an integer to use as seed"
    )
    arg_parser.add_argument("--data_dir", type=str, default="./data")
    arg_parser.add_argument("--output_dir", type=str, default="./augment")
    arg_parser.add_argument("--case_glob", type=str, default="images/*.nii.gz")
    arg_parser.add_argument("--image_glob", type=str, default="images/{case}.nii.gz")
    arg_parser.add_argument(
        "--label_glob", nargs="+", type=str, default="labels/{case}_*.nii.gz"
    )
    arg_parser.add_argument("--context_map_glob", type=str, default=None)
    arg_parser.add_argument(
        "--augmentations_per_case",
        type=int,
        default=10,
        help="How many augmented images per case to generate",
    )

    arg_parser.add_argument("--enable_shift", type=bool, default=True)
    arg_parser.add_argument("--shift_x_range", nargs="+", type=int, default=[-10, 10])
    arg_parser.add_argument("--shift_y_range", nargs="+", type=int, default=[-10, 10])
    arg_parser.add_argument("--shift_z_range", nargs="+", type=int, default=[-10, 10])
    arg_parser.add_argument("--shift_smooth_range", nargs="+", type=int, default=[3, 5])
    arg_parser.add_argument("--shift_probability", type=float, default=0.5)

    arg_parser.add_argument("--enable_expand", type=bool, default=True)
    arg_parser.add_argument("--expand_x_range", nargs="+", type=int, default=[0, 10])
    arg_parser.add_argument("--expand_y_range", nargs="+", type=int, default=[0, 10])
    arg_parser.add_argument("--expand_z_range", nargs="+", type=int, default=[0, 10])
    arg_parser.add_argument(
        "--expand_smooth_range", nargs="+", type=int, default=[3, 5]
    )
    arg_parser.add_argument("--expand_bone_mask", type=bool, default=True)
    arg_parser.add_argument("--expand_probability", type=float, default=0.5)

    arg_parser.add_argument("--enable_contract", type=bool, default=True)
    arg_parser.add_argument("--contract_x_range", nargs="+", type=int, default=[0, 10])
    arg_parser.add_argument("--contract_y_range", nargs="+", type=int, default=[0, 10])
    arg_parser.add_argument("--contract_z_range", nargs="+", type=int, default=[0, 10])
    arg_parser.add_argument(
        "--contract_smooth_range", nargs="+", type=int, default=[3, 5]
    )
    arg_parser.add_argument("--contract_bone_mask", type=bool, default=True)
    arg_parser.add_argument("--contract_probability", type=float, default=0.5)

    arg_parser.add_argument("--enable_fill_holes", type=bool, default=True)
    arg_parser.add_argument("--fill_probability", type=float, default=0.2)

    augment_data(arg_parser.parse_args())
