import random
import math
from pathlib import Path

import torch

import pytorch_lightning as pl

from platipy.imaging.cnn.dataset import NiftiDataset
from platipy.imaging.cnn.sampler import ObserverSampler


class UNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        augmented_dir: str = None,
        working_dir: str = "./working",
        case_glob="images/*.nii.gz",
        image_glob="images/{case}.nii.gz",
        label_glob="labels/{case}_*.nii.gz",
        augmented_case_glob="{case}/*",
        augmented_image_glob="images/{augmented_case}.nii.gz",
        augmented_label_glob="labels/{augmented_case}_*.nii.gz",
        augment_on_fly=True,
        fold=0,
        k_folds=5,
        batch_size=5,
        num_workers=4,
        crop_to_mm=128,
        num_observers=5,
        spacing=[1, 1, 1],
        contour_mask_kernel=3,
        ndims=2,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.augmented_dir = augmented_dir
        self.working_dir = Path(working_dir)

        self.case_glob = case_glob
        self.image_glob = image_glob
        self.label_glob = label_glob
        self.augmented_case_glob = augmented_case_glob
        self.augmented_image_glob = augmented_image_glob
        self.augmented_label_glob = augmented_label_glob

        self.augment_on_fly = augment_on_fly
        self.fold = fold
        self.k_folds = k_folds

        self.train_cases = []
        self.validation_cases = []

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.crop_to_mm = crop_to_mm
        self.num_observers = num_observers
        self.spacing = spacing
        self.contour_mask_kernel = contour_mask_kernel

        print(self.spacing)

        self.training_set = None
        self.validation_set = None

        self.ndims = ndims

        print(f"Training fold {self.fold}")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data Loader")
        parser.add_argument("--data_dir", type=str, default="./data")
        parser.add_argument("--augmented_dir", type=str, default=None)
        parser.add_argument("--augment_onfly", type=bool, default=True)
        parser.add_argument("--fold", type=int, default=0)
        parser.add_argument("--k_folds", type=int, default=5)
        parser.add_argument("--batch_size", type=int, default=5)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--case_glob", type=str, default="images/*.nii.gz")
        parser.add_argument("--image_glob", type=str, default="images/{case}.nii.gz")
        parser.add_argument("--label_glob", type=str, default="labels/{case}_*.nii.gz")
        parser.add_argument("--augmented_case_glob", type=str, default=None)
        parser.add_argument("--augmented_image_glob", type=str, default=None)
        parser.add_argument("--augmented_label_glob", type=str, default=None)
        parser.add_argument("--crop_to_mm", type=int, default=128)
        parser.add_argument("--contour_mask_kernel", type=int, default=5)
        parser.add_argument("--ndims", type=int, default=2)

        return parent_parser

    def setup(self, stage=None):

        cases = [
            p.name.replace(".nii.gz", "")
            for p in self.data_dir.glob(self.case_glob)
            if not p.name.startswith(".")
        ]
        cases.sort()
        random.shuffle(cases)  # will be consistent for same value of 'seed everything'
        cases_per_fold = math.ceil(len(cases) / self.k_folds)
        for f in range(self.k_folds):

            if self.fold == f:
                self.validation_cases = cases[f * cases_per_fold : (f + 1) * cases_per_fold]
            else:
                self.train_cases += cases[f * cases_per_fold : (f + 1) * cases_per_fold]

        print(f"Training cases: {self.train_cases}")
        print(f"Validation cases: {self.validation_cases}")

        train_data = [
            {
                "id": case,
                "image": self.data_dir.joinpath(self.image_glob.format(case=case)),
                "label": [p for p in self.data_dir.glob(self.label_glob.format(case=case))],
            }
            for case in self.train_cases
        ]

        # If a directory with augmented data is specified, use that for training as well
        if self.augmented_dir is not None:

            for case in self.train_cases:

                case_aug_dir = Path(self.augmented_dir.format(case=case))
                augmented_cases = [
                    p.name.replace(".nii.gz", "")
                    for p in case_aug_dir.glob(self.augmented_case_glob.format(case=case))
                    if not p.name.startswith(".")
                ]

                train_data += [
                    {
                        "id": f"{case}_{augmented_case}",
                        "image": case_aug_dir.joinpath(
                            self.augmented_image_glob.format(
                                case=case, augmented_case=augmented_case
                            )
                        ),
                        "label": [
                            p
                            for p in case_aug_dir.glob(
                                self.augmented_label_glob.format(
                                    case=case, augmented_case=augmented_case
                                )
                            )
                        ],
                    }
                    for augmented_case in augmented_cases
                ]

        validation_data = [
            {
                "id": case,
                "image": self.data_dir.joinpath(self.image_glob.format(case=case)),
                "label": [
                    p
                    for p in self.data_dir.glob(self.label_glob.format(case=case))
                    if not "edited" in p.name
                ],
            }
            for case in self.validation_cases
        ]

        self.training_set = NiftiDataset(
            train_data,
            self.working_dir,
            augment_on_the_fly=self.augment_on_fly,
            spacing=self.spacing,
            crop_to_mm=self.crop_to_mm,
            contour_mask_kernel=self.contour_mask_kernel,
            ndims=self.ndims,
        )
        self.validation_set = NiftiDataset(
            validation_data,
            self.working_dir,
            augment_on_the_fly=False,
            spacing=self.spacing,
            crop_to_mm=self.crop_to_mm,
            contour_mask_kernel=self.contour_mask_kernel,
            ndims=self.ndims,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.training_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.validation_set,
            batch_sampler=torch.utils.data.BatchSampler(
                ObserverSampler(self.validation_set, self.num_observers),
                batch_size=self.num_observers,
                drop_last=False,
            ),
            num_workers=self.num_workers,
        )
