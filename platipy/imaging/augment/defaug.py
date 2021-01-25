from abc import ABC, abstractmethod
from collections.abc import Iterable
import random

import SimpleITK as sitk

from platipy.imaging.deformation_fields.deformation_field_operations import (
    generate_field_shift,
    generate_field_expand,
    get_bone_mask,
)

from platipy.imaging.registration.registration import apply_field


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

    image_deformed = apply_field(
        image,
        transform,
        structure=False,
        default_value=int(sitk.GetArrayViewFromImage(image).min()),
    )

    masks_deformed = []
    for mask in masks:
        masks_deformed.append(apply_field(mask, transform=transform, structure=True, interp=1))

    if masks:
        return image_deformed, masks_deformed, dvf

    return image_deformed, dvf


def generate_random_augmentation(ct_image, masks):

    random.shuffle(masks)
    # mask_count = len(masks)
    # masks = masks[: random.randint(2, 5)]

    # print(len(masks))
    augmentation_types = [
        {
            "class": ShiftAugment,
            "args": {"vector_shift": [(-10, 10), (10, 10), (-10, 10)], "gaussian_smooth": (3, 5)},
        },
        {
            "class": ContractAugment,
            "args": {
                "vector_contract": [(0, 10), (0, 10), (0, 10)],
                "gaussian_smooth": (3, 5),
                "bone_mask": True,
            },
        },
        {
            "class": ExpandAugment,
            "args": {
                "vector_expand": [(0, 10), (0, 10), (0, 10)],
                "gaussian_smooth": (3, 5),
                "bone_mask": True,
            },
        },
    ]

    augmentation = []
    for mask in masks:
        aug = random.choice(augmentation_types)

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
            self.mask, self.vector_shift, self.gaussian_smooth,
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
