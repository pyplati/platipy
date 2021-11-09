from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

import random

from platipy.imaging.generation.image import insert_sphere
from platipy.imaging import ImageVisualiser


def generate_pseudo_data(data_dir="data", cases=5, size=(24, 32, 32), structures=["a", "b", "c"]):
    """Generates some Pseudo data to use for testing the CNN code

    Args:
        data_dir (str, optional): Directory in which to store pseudo data. Defaults to "data".
        cases (int, optional): Number of cases to generate data for. Defaults to 5.
        size (tuple, optional): The size of the generated images. Defaults to (24, 32, 32).
        structures (list, optional): A list of structure names to generate. Defaults to
          ["a", "b", "c"].
    """

    test_data_directory = Path(data_dir)

    if test_data_directory.exists():
        print("Data directory already exists, won't regenerate")
        return

    image_directory = test_data_directory.joinpath("images")
    label_directory = test_data_directory.joinpath("labels")
    slice_directory = test_data_directory.joinpath("slices")

    image_directory.mkdir(parents=True, exist_ok=True)
    label_directory.mkdir(parents=True, exist_ok=True)
    slice_directory.mkdir(parents=True, exist_ok=True)

    for case, sphere_rad in enumerate(range(5, 5 + cases)):

        xpos = random.randint(6, 24)
        ypos = random.randint(6, 24)

        mask_arr = np.zeros(size)
        mask_arr = insert_sphere(
            mask_arr, sp_radius=sphere_rad, sp_centre=(int(size[0] / 2), ypos, xpos)
        )

        mask = sitk.GetImageFromArray(mask_arr)
        mask = sitk.Cast(mask, sitk.sitkUInt8)
        mask = sitk.BinaryNot(mask)

        ct = sitk.SignedMaurerDistanceMap(mask)

        ct_arr = sitk.GetArrayFromImage(ct)
        ct_arr[ct_arr < -10] = -1000
        ct_arr[ct_arr > 20] = 100

        ct = sitk.GetImageFromArray(ct_arr)

        sitk.WriteImage(ct, str(image_directory.joinpath(f"{case}.nii.gz")))

        vis = ImageVisualiser(ct, cut=(int(size[0] / 2), ypos, xpos))
        masks = {}

        for struct_id, structure in enumerate(structures):

            x_shift = y_shift = 0
            if struct_id > 0:
                if struct_id % 2 == 0:
                    x_shift = struct_id
                else:
                    y_shift = struct_id

            for obs_id, obs in enumerate(range(-4, 5, 2)):
                obs_rad = sphere_rad + obs

                mask_arr = np.zeros(size)
                mask_arr = insert_sphere(
                    mask_arr,
                    sp_radius=obs_rad,
                    sp_centre=(int(size[0] / 2), ypos + y_shift, xpos + x_shift),
                )

                mask = sitk.GetImageFromArray(mask_arr)
                mask.CopyInformation(ct)
                mask = sitk.Cast(mask, sitk.sitkUInt8)
                sitk.WriteImage(
                    mask, str(label_directory.joinpath(f"{case}_{structure}_{obs_id}.nii.gz"))
                )

                masks[f"struct_{structure}_obs_{obs_id}_{obs_rad}"] = mask

        vis.add_contour(masks)
        vis.show()
        plt.savefig(slice_directory.joinpath(f"{case}.png"))
        plt.close()


if __name__ == "__main__":
    generate_pseudo_data()
