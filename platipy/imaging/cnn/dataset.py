import re
from pathlib import Path

import numpy as np

import torch

import SimpleITK as sitk

from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

from loguru import logger

from platipy.imaging.cnn.utils import preprocess_image, resample_mask_to_image, get_contour_mask
from platipy.imaging.label.utils import get_union_mask, get_intersection_mask


def prepare_transforms():

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    seq = iaa.Sequential(
        [
            sometimes(
                iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    rotate=(-15, 15),
                    shear=(-8, 8),
                    cval=-1,
                )
            ),
            # execute 0 to 2 of the following (less important) augmenters per image
            iaa.SomeOf(
                (0, 2),
                [
                    iaa.OneOf(
                        [
                            iaa.GaussianBlur((0, 1.5)),
                            iaa.AverageBlur(k=(3, 5)),
                        ]
                    ),
                    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1))),
                ],
                random_order=True,
            ),
            sometimes(iaa.CoarseDropout(
                        (0.03, 0.15), size_percent=(0.02, 0.1)
                    ))
        ],
        random_order=True,
    )

    return seq


class NiftiDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for processing Nifti data"""

    def __init__(
        self,
        data,
        working_dir,
        augment_on_fly=True,
        spacing=[1, 1, 1],
        crop_using_localise_model=None,
        crop_to_grid_size=128,
        contour_mask_kernel=5,
        combine_observers=None,
        intensity_scaling="window",
        intensity_window=[-500, 500],
        ndims=2,
    ):
        """Prepare a dataset from Nifti images/labels

        Args:
            data (list): List of dict's where each item contains keys: "image" and "label". Values
                are paths to the Nifti file. "label" may be a list where each item is a path to one
                observer.
            working_dir (str|path): Working directory where to write prepared files.
        """

        self.data = data
        self.transforms = None
        if augment_on_fly:
            self.transforms = prepare_transforms()
        self.slices = []
        self.working_dir = Path(working_dir)
        self.ndims = ndims

        self.img_dir = working_dir.joinpath("img")
        self.label_dir = working_dir.joinpath("label")
        self.contour_mask_dir = working_dir.joinpath("contour_mask")

        self.img_dir.mkdir(exist_ok=True, parents=True)
        self.label_dir.mkdir(exist_ok=True, parents=True)
        self.contour_mask_dir.mkdir(exist_ok=True, parents=True)

        for case in data:
            case_id = case["id"]
            img_path = str(case["image"])

            structure_paths = case["label"]
            if isinstance(structure_paths, (str, Path)):
                structure_paths = [structure_paths]

            existing_images = [i for i in self.img_dir.glob(f"{case_id}_*.npy")]
            if len(existing_images) > 0:
                logger.debug(f"Image for case already exist: {case_id}")

                for img_path in existing_images:
                    z_matches = re.findall(fr"{case_id}_([0-9]*)\.npy", img_path.name)
                    if len(z_matches) == 0:
                        continue
                    z_slice = int(z_matches[0])

                    img_file = self.img_dir.joinpath(f"{case_id}_{z_slice}.npy")
                    assert img_file.exists()

                    contour_mask_file = self.contour_mask_dir.joinpath(f"{case_id}_{z_slice}.npy")
                    assert contour_mask_file.exists()

                    for obs in range(len(structure_paths)):
                        label_file = self.label_dir.joinpath(f"{case_id}_{obs}_{z_slice}.npy")
                        assert label_file.exists()
                        self.slices.append(
                            {
                                "z": z_slice,
                                "image": img_file,
                                "label": label_file,
                                "contour_mask": contour_mask_file,
                                "case": case_id,
                                "observer": obs,
                            }
                        )

                continue

            logger.debug(f"Generating images for case: {case_id}")
            img = sitk.ReadImage(img_path)

            if crop_using_localise_model:
                crop_using_localise_model(
                    img,
                    crop_using_localise_model,
                    spacing=spacing,
                    crop_to_grid_size=crop_to_grid_size,
                )
            else:
                img = preprocess_image(
                    img,
                    spacing=spacing,
                    crop_to_grid_size_xy=crop_to_grid_size,
                    intensity_scaling=intensity_scaling,
                    intensity_window=intensity_window,
                )

            observers = []
            for obs, structure_path in enumerate(structure_paths):
                structure_path = str(structure_path)
                label = sitk.ReadImage(structure_path)
                label = resample_mask_to_image(img, label)
                observers.append(label)

            contour_mask = get_contour_mask(observers, kernel=contour_mask_kernel)

            if combine_observers == "union":
                observers = [get_union_mask(observers)]

            if combine_observers == "intersection":
                observers = [get_intersection_mask(observers)]

            z_range = range(img.GetSize()[2])
            if ndims == 3:
                z_range = range(1)
            for z_slice in z_range:

                # Save the image slice
                if ndims == 2:
                    img_slice = img[:, :, z_slice]
                else:
                    img_slice = img

                img_file = self.img_dir.joinpath(f"{case_id}_{z_slice}.npy")
                np.save(img_file, sitk.GetArrayFromImage(img_slice))

                # Save the contour mask slice
                if ndims == 2:
                    contour_mask_slice = contour_mask[:, :, z_slice]
                else:
                    contour_mask_slice = contour_mask
                contour_mask_file = self.contour_mask_dir.joinpath(f"{case_id}_{z_slice}.npy")
                np.save(contour_mask_file, sitk.GetArrayFromImage(contour_mask_slice))

                for obs, label in enumerate(observers):

                    if ndims == 2:
                        label_slice = label[:, :, z_slice]
                    else:
                        label_slice = label
                    label_file = self.label_dir.joinpath(f"{case_id}_{obs}_{z_slice}.npy")
                    np.save(label_file, sitk.GetArrayFromImage(label_slice).astype(np.int8))
                    self.slices.append(
                        {
                            "z": z_slice,
                            "image": img_file,
                            "label": label_file,
                            "contour_mask": contour_mask_file,
                            "case": case_id,
                            "observer": obs,
                        }
                    )

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):

        img = np.load(self.slices[index]["image"])
        label = np.load(self.slices[index]["label"])
        contour_mask = np.load(self.slices[index]["contour_mask"])

        if self.transforms:
            seg_arr = np.concatenate(
                (np.expand_dims(label, 2), np.expand_dims(contour_mask, 2)), 2
            )
            segmap = SegmentationMapsOnImage(seg_arr, shape=label.shape)
            img, seg = self.transforms(image=img, segmentation_maps=segmap)
            label = seg.get_arr()[:, :, 0].squeeze()
            contour_mask = seg.get_arr()[:, :, 1].squeeze()

        img = torch.FloatTensor(img)
        label = torch.IntTensor(label)
        contour_mask = torch.FloatTensor(contour_mask)

        return (
            img.unsqueeze(0),
            label,
            contour_mask,
            {
                "case": self.slices[index]["case"],
                "observer": self.slices[index]["observer"],
                "z": self.slices[index]["z"],
            },
        )
