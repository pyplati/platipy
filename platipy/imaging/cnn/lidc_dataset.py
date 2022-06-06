# Copyright 2022 University of New South Wales, University of Sydney, Ingham Institute

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
import pickle
import os

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

import torch

from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

from platipy.imaging import ImageVisualiser

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
            sometimes(iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.1))),
        ],
        random_order=True,
    )

    return seq


class LIDCDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for processing LIDC data"""

    def __init__(
        self,
        working_dir,
        case_ids=None,
        pickle_path="lidc.pickle",
        augment_on_fly=True,
    ):
        """Prepare a dataset from Nifti images/labels

        Args:
            data (list): List of dict's where each item contains keys: "image" and "label". Values
                are paths to the Nifti file. "label" may be a list where each item is a path to one
                observer.
            working_dir (str|path): Working directory where to write prepared files.
        """

        self.transforms = None
        if augment_on_fly:
            self.transforms = prepare_transforms()
        self.slices = []
        self.working_dir = Path(working_dir)

        self.img_dir = self.working_dir.joinpath("img")
        self.label_dir = self.working_dir.joinpath("label")
        self.contour_mask_dir = self.working_dir.joinpath("contour_mask")
        self.snap_dir = self.working_dir.joinpath("snapshots")

        self.img_dir.mkdir(exist_ok=True, parents=True)
        self.label_dir.mkdir(exist_ok=True, parents=True)
        self.contour_mask_dir.mkdir(exist_ok=True, parents=True)
        self.snap_dir.mkdir(exist_ok=True, parents=True)

        # If data doesn't already exist, unpickle data and place into directory
        if len(list(self.img_dir.glob("*"))) == 0:
            pickle_path = Path(pickle_path)

            max_bytes = 2**31 - 1
            data = {}

            print("Loading file", pickle_path)
            bytes_in = bytearray(0)
            input_size = os.path.getsize(pickle_path)
            with open(pickle_path, 'rb') as f_in:
                for _ in range(0, input_size, max_bytes):
                    bytes_in += f_in.read(max_bytes)
            new_data = pickle.loads(bytes_in)
            data.update(new_data)

            for k,i in data.items():

                pat_id = k.split("_")[0]
                slice_id = k.split("_")[1].replace("slice", "")

                i["pixel_spacing"] = [float(a) for a in i["pixel_spacing"]]

                img_file = self.img_dir.joinpath(f"{pat_id}_{slice_id}.npy")
                np.save(img_file, i["image"])

                intersection = None
                union = None
                vis = ImageVisualiser(sitk.GetImageFromArray(np.expand_dims(i["image"], axis=0)), axis="z", window=[0,1])
                for obs, mask in enumerate(i["masks"]):

                    vis.add_contour(sitk.GetImageFromArray(np.expand_dims(mask, axis=0)), name=f"{obs}")
                    label_file = self.label_dir.joinpath(f"{pat_id}_{slice_id}_{obs}.npy")
                    np.save(label_file, mask)

                    mask = mask.astype(int)

                    if intersection is None:
                        intersection = np.copy(mask)
                    else:
                        intersection += mask

                    if union is None:
                        union = np.copy(mask)
                    else:
                        union += mask

                intersection[intersection>1] = 1
                union[union<len(i["masks"])] = 0
                union[union==len(i["masks"])] = 1

                intersection = sitk.GetImageFromArray(np.expand_dims(intersection, axis=0))
                intersection = sitk.BinaryDilate(intersection, kernelRadius=(2,2,2))
                union = sitk.GetImageFromArray(np.expand_dims(union, axis=0))
                union = sitk.BinaryErode(union, kernelRadius=(2,2,2))

            #     vis.add_contour(intersection, name="intersection")
            #     vis.add_contour(union, name="union")

                contour_mask = intersection - union
                vis.add_contour(contour_mask, name="contour_mask")
                contour_arr = sitk.GetArrayFromImage(contour_mask)
                contour_mask_file = self.contour_mask_dir.joinpath(f"{pat_id}_{slice_id}.npy")
                np.save(contour_mask_file, contour_arr)

                fig = vis.show()

                fig_path = self.snap_dir.joinpath(f"{pat_id}_{slice_id}.png")

                plt.savefig(fig_path)
                plt.close(fig)

        for img in self.img_dir.glob("*"):
            case_and_slice = img.name.replace(".npy", "")
            contour_mask = self.contour_mask_dir.joinpath(img.name)

            for label in self.label_dir.glob(f"{case_and_slice}_*.npy"):
                case_id, z_slice, obs = label.name.replace(".npy", "").split("_")

                if case_ids is not None and not case_id in case_ids:
                    continue

                self.slices.append(
                    {
                        "z": z_slice,
                        "image": img,
                        "label": label,
                        "contour_mask": contour_mask,
                        "case": case_id,
                        "observer": obs,
                    }
                )

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):

        img = np.load(self.slices[index]["image"]).astype("float32")
        label = np.load(self.slices[index]["label"]).astype("int8")
        contour_mask = np.load(self.slices[index]["contour_mask"]).squeeze().astype("int8")

        if self.transforms:
            masks = [label, contour_mask]
            seg_arr = np.concatenate([np.expand_dims(m, 2) for m in masks], 2)
            segmap = SegmentationMapsOnImage(seg_arr, shape=label.shape)
            img, seg = self.transforms(image=img, segmentation_maps=segmap)
            label = seg.get_arr()[:, :, 0]
            contour_mask = seg.get_arr()[:, :, 1]
        img = torch.FloatTensor(img)
        label = torch.IntTensor(np.expand_dims(label, 0))
        contour_mask = torch.IntTensor(np.expand_dims(contour_mask, 0))
        #label_present = [label is not None for label in self.slices[index]["labels"]]

        return (
            img.unsqueeze(0),
            label,
            contour_mask,
            {
                "case": str(self.slices[index]["case"]),
                "observer": str(self.slices[index]["observer"]),
                #"label_present": label_present,
                "z": self.slices[index]["z"],
            },
        )
