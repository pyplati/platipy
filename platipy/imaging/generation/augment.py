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

from pathlib import Path

from argparse import ArgumentParser

import SimpleITK as sitk

from loguru import logger

from platipy.imaging import ImageVisualiser

from platipy.imaging.generation.dvf import (
    generate_field_shift,
    generate_field_expand,
)

from platipy.imaging.generation.mask import (
    get_bone_mask,
)

from platipy.imaging.registration.utils import apply_transform


def apply_augmentation(image, augmentation, masks=[]):

    if not isinstance(image, sitk.Image):
        raise AttributeError("image should be a SimpleITK.Image")

    if isinstance(augmentation, DeformableAugment):
        augmentation = [augmentation]

    if not isinstance(augmentation, Iterable):
        raise AttributeError(
            "augmentation must be a DeformableAugment or an iterable (such as list) of"
            "DeformableAugment's"
        )

    transforms = []
    dvf = None
    for aug in augmentation:

        if not isinstance(aug, DeformableAugment):
            raise AttributeError("Each augmentation must be of type DeformableAugment")

        tfm, field = aug.augment()
        transforms.append(tfm)

        if dvf is None:
            dvf = field
        else:
            dvf += field

    transform = sitk.CompositeTransform(transforms)
    del transforms

    image_deformed = apply_transform(
        image,
        transform=transform,
        default_value=int(sitk.GetArrayViewFromImage(image).min()),
        interpolator=sitk.sitkLinear,
    )

    masks_deformed = []
    for mask in masks:
        masks_deformed.append(
            apply_transform(
                mask, transform=transform, default_value=0, interpolator=sitk.sitkNearestNeighbor
            )
        )

    if masks:
        return image_deformed, masks_deformed, dvf

    return image_deformed, dvf


def generate_random_augmentation(ct_image, masks, augmentation_types):

    augmentation = []

    probabilities = [a["probability"] for a in augmentation_types]
    prob_total = sum(probabilities)
    prob_none = 1.0 - prob_total
    if prob_none < 0:
        prob_none = 0

    for mask in masks:
        aug = random.choices(augmentation_types + [None], weights=probabilities + [prob_none])[0]

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


class ExpandAugment(DeformableAugment):
    def __init__(self, mask, vector_expand=(10, 10, 10), gaussian_smooth=5, bone_mask=False):

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


class ContractAugment(DeformableAugment):
    def __init__(self, mask, vector_contract=(10, 10, 10), gaussian_smooth=5, bone_mask=False):

        self.mask = mask
        self.contract = [int(-x / s) for x, s in zip(vector_contract, mask.GetSpacing())]
        self.gaussian_smooth = gaussian_smooth
        self.bone_mask = bone_mask

    def augment(self):

        _, transform, dvf = generate_field_expand(
            self.mask,
            bone_mask=self.bone_mask,
            expand=self.contract,
            gaussian_smooth=self.gaussian_smooth,
        )
        return transform, dvf


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
            "label": [p for p in data_dir.glob(args.label_glob.format(case=case))],
        }
        for case in cases
    }

    for case in cases:

        logger.info(f"Augmenting for case: {case}")

        ct_image = sitk.ReadImage(str(data[case]["image"]))

        # Get list of structures to generate augmentations off
        all_masks = []
        all_names = []
        for structure_path in data[case]["label"]:

            mask = sitk.ReadImage(str(structure_path))

            all_masks.append(mask)
            all_names.append(structure_path.name.replace(".nii.gz", ""))

        # Generate 10 random augmentations per case
        for i in range(args.augmentations_per_case):

            logger.debug("Generating augmentation")

            augmented_case_path = output_dir.joinpath(case, f"augment_{i}")
            augmented_case_path.mkdir(exist_ok=True, parents=True)

            augmentation = generate_random_augmentation(ct_image, all_masks, augmentation_types)

            dvf = None

            if len(augmentation) == 0:
                logger.debug(
                    "No augmentations generated, generated image won't differ from original"
                )

                augmented_image = ct_image
                augmented_masks = all_masks
            else:

                logger.debug("Applying augmentation")
                augmented_image, augmented_masks, dvf = apply_augmentation(
                    ct_image, augmentation, masks=all_masks
                )

            augmented_image_path = augmented_case_path.joinpath("CT.nii.gz")
            sitk.WriteImage(augmented_image, str(augmented_image_path))

            vis = ImageVisualiser(image=ct_image, figure_size_in=6)
            vis.add_comparison_overlay(augmented_image)
            if dvf is not None:
                vis.add_vector_overlay(dvf, arrow_scale=1, subsample=(4, 12, 12))
            for mask_name, mask, augmented_mask in zip(all_names, all_masks, augmented_masks):
                vis.add_contour({f"{mask_name}": mask, f"{mask_name} (augmented)": augmented_mask})

                logger.debug(f"Applying augmentation to mask: {mask_name}")
                augmented_mask_path = augmented_case_path.joinpath(f"{mask_name}.nii.gz")
                sitk.WriteImage(augmented_mask, str(augmented_mask_path))

            fig = vis.show()

            figure_path = augmented_case_path.joinpath("aug.png")
            fig.savefig(figure_path, bbox_inches="tight")


if __name__ == "__main__":

    arg_parser = ArgumentParser()
    arg_parser.add_argument("--seed", type=int, default=42, help="an integer to use as seed")
    arg_parser.add_argument("--data_dir", type=str, default="./data")
    arg_parser.add_argument("--output_dir", type=str, default="./augment")
    arg_parser.add_argument("--case_glob", type=str, default="images/*.nii.gz")
    arg_parser.add_argument("--image_glob", type=str, default="images/{case}.nii.gz")
    arg_parser.add_argument("--label_glob", type=str, default="labels/{case}_*.nii.gz")
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
    arg_parser.add_argument("--expand_smooth_range", nargs="+", type=int, default=[3, 5])
    arg_parser.add_argument("--expand_bone_mask", type=bool, default=True)
    arg_parser.add_argument("--expand_probability", type=float, default=0.5)

    arg_parser.add_argument("--enable_contract", type=bool, default=True)
    arg_parser.add_argument("--contract_x_range", nargs="+", type=int, default=[0, 10])
    arg_parser.add_argument("--contract_y_range", nargs="+", type=int, default=[0, 10])
    arg_parser.add_argument("--contract_z_range", nargs="+", type=int, default=[0, 10])
    arg_parser.add_argument("--contract_smooth_range", nargs="+", type=int, default=[3, 5])
    arg_parser.add_argument("--contract_bone_mask", type=bool, default=True)
    arg_parser.add_argument("--contract_probability", type=float, default=0.5)

    augment_data(arg_parser.parse_args())
