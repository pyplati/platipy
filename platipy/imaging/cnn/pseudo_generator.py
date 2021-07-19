from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

import random

from platipy.imaging.generation.image import insert_sphere
from platipy.imaging import ImageVisualiser


def generate_pseudo_data(data_dir="data"):

    test_data_directory = Path(data_dir)
    image_directory = test_data_directory.joinpath("images")
    label_directory = test_data_directory.joinpath("labels")
    slice_directory = test_data_directory.joinpath("slices")

    image_directory.mkdir(parents=True, exist_ok=True)
    label_directory.mkdir(parents=True, exist_ok=True)
    slice_directory.mkdir(parents=True, exist_ok=True)

    for case, sphere_rad in enumerate(range(10, 30)):

        xpos = random.randint(50, 80)
        ypos = random.randint(50, 80)

        mask_arr = np.zeros((80, 128, 128))
        mask_arr = insert_sphere(mask_arr, sp_radius=sphere_rad, sp_centre=(30, ypos, xpos))

        mask = sitk.GetImageFromArray(mask_arr)
        mask = sitk.Cast(mask, sitk.sitkUInt8)
        mask = sitk.BinaryNot(mask)

        ct = sitk.SignedMaurerDistanceMap(mask)

        ct_arr = sitk.GetArrayFromImage(ct)
        ct_arr[ct_arr < -10] = -1000
        ct_arr[ct_arr > 20] = 100

        ct = sitk.GetImageFromArray(ct_arr)

        sitk.WriteImage(ct, str(image_directory.joinpath(f"{case}.nii.gz")))

        vis = ImageVisualiser(ct, cut=(30, ypos, xpos))
        masks = {}

        for obs_id, obs in enumerate(range(-4, 5, 2)):
            obs_rad = sphere_rad + obs

            mask_arr = np.zeros((80, 128, 128))
            mask_arr = insert_sphere(mask_arr, sp_radius=obs_rad, sp_centre=(30, ypos, xpos))

            mask = sitk.GetImageFromArray(mask_arr)
            mask.CopyInformation(ct)
            mask = sitk.Cast(mask, sitk.sitkUInt8)
            sitk.WriteImage(mask, str(label_directory.joinpath(f"{case}_{obs_id}.nii.gz")))

            masks[f"obs_{obs_id}_{obs_rad}"] = mask

        vis.add_contour(masks)
        vis.show()
        plt.savefig(slice_directory.joinpath(f"{case}.png"))
        plt.close()


if __name__ == "__main__":
    generate_pseudo_data()