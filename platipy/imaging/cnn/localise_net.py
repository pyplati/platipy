# Copyright 2021 University of New South Wales, University of Sydney, Ingham Institute

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

from pathlib import Path
import SimpleITK as sitk
import numpy as np

import comet_ml  # pylint: disable=unused-import

import torch
import pytorch_lightning as pl

import matplotlib.pyplot as plt

from platipy.imaging.cnn.unet import UNet
from platipy.imaging.cnn.utils import preprocess_image, postprocess_mask, get_metrics

from platipy.imaging import ImageVisualiser
from platipy.imaging.label.utils import get_com


class LocaliseUNet(pl.LightningModule):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.unet = UNet(
            self.hparams.input_channels,
            self.hparams.num_classes,
            filters_per_layer=[32, 64, 128],
            final_layer=True,
        )

        self.validation_directory = None

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Localize UNet")
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        parser.add_argument("--input_channels", type=int, default=1)
        parser.add_argument("--num_classes", type=int, default=2)

        return parent_parser

    def forward(self, x):
        return self.unet.forward(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=0
        )

        return optimizer

    def infer(self, img):

        pp_img = preprocess_image(
            img, spacing=self.hparams.spacing, crop_to_grid_size_xy=self.hparams.crop_to_mm
        )

        preds = []
        for z in range(pp_img.GetSize()[2]):
            x = sitk.GetArrayFromImage(pp_img[:, :, z])
            x = torch.Tensor(x)
            x = x.unsqueeze(0)
            x = x.unsqueeze(0)
            y = self(x)
            y = y.squeeze(0)
            y = np.argmax(y.cpu().detach().numpy(), axis=0)
            preds.append(y)

        pred = sitk.GetImageFromArray(np.stack(preds))
        pred = sitk.Cast(pred, sitk.sitkUInt8)

        pred.CopyInformation(pp_img)
        pred = postprocess_mask(pred)
        pred = sitk.Resample(pred, img, sitk.Transform(), sitk.sitkNearestNeighbor)

        return pred

    def training_step(self, batch, _):

        x, y, _, _ = batch

        pred = self.unet.forward(x)

        criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")

        y = torch.unsqueeze(y, dim=1)
        not_y = y.logical_not()
        y = torch.cat((not_y, y), dim=1).float()

        loss = criterion(input=pred, target=y)

        return loss

    def validation_step(self, batch, _):

        if self.validation_directory is None:
            self.validation_directory = Path(tempfile.mkdtemp())
            print(self.validation_directory)

        with torch.set_grad_enabled(False):
            x, y, _, info = batch

            for s in range(y.shape[0]):

                img_file = self.validation_directory.joinpath(
                    f"img_{info['case'][s]}_{info['z'][s]}.npy"
                )
                np.save(img_file, x[0].squeeze(0).cpu().numpy())

                mask_file = self.validation_directory.joinpath(
                    f"mask_{info['case'][s]}_{info['z'][s]}_{info['observer'][s]}.npy"
                )
                np.save(mask_file, y[s].squeeze(0).cpu().numpy())

                pred = self.unet.forward(x[s].unsqueeze(0))
                pred_file = self.validation_directory.joinpath(
                    f"pred_{info['case'][s]}_{info['z'][s]}.npy"
                )
                pred = np.argmax(pred.squeeze(0).cpu().numpy(), axis=0)
                np.save(pred_file, pred)

        return info

    def validation_epoch_end(self, validation_step_outputs):

        cases = {}
        for info in validation_step_outputs:

            for case, z, observer in zip(info["case"], info["z"], info["observer"]):

                if not case in cases:
                    cases[case] = {"slices": z.item(), "observers": [observer.item()]}
                else:
                    if z.item() > cases[case]["slices"]:
                        cases[case]["slices"] = z.item()
                    if not observer in cases[case]["observers"]:
                        cases[case]["observers"].append(observer.item())

        metrics = {"JI": [], "DSC": [], "HD": [], "ASD": []}
        for case in cases:

            img_arrs = []
            pred_arrs = []
            slices = []
            for z in range(cases[case]["slices"] + 1):
                img_file = self.validation_directory.joinpath(f"img_{case}_{z}.npy")
                pred_file = self.validation_directory.joinpath(f"pred_{case}_{z}.npy")
                if img_file.exists():
                    img_arrs.append(np.load(img_file))
                    pred_arrs.append(np.load(pred_file))
                    slices.append(z)

            if len(slices) < 5:
                # Likely initial sanity check
                continue

            img_arr = np.stack(img_arrs)
            img = sitk.GetImageFromArray(img_arr)
            img.SetSpacing(self.hparams.spacing)

            pred_arr = np.stack(pred_arrs)
            pred = sitk.GetImageFromArray(pred_arr)
            pred = sitk.Cast(pred, sitk.sitkUInt8)
            pred = postprocess_mask(pred)
            pred.CopyInformation(img)
            sitk.WriteImage(pred, f"val_pred_{case}.nii.gz")

            color_dict = {}
            obs_dict = {}

            for _, observer in enumerate(cases[case]["observers"]):
                mask_arrs = []
                for z in slices:
                    mask_file = self.validation_directory.joinpath(
                        f"mask_{case}_{z}_{observer}.npy"
                    )

                    mask_arrs.append(np.load(mask_file))

                mask_arr = np.stack(mask_arrs)
                mask = sitk.GetImageFromArray(mask_arr)
                mask = sitk.Cast(mask, sitk.sitkUInt8)
                mask.CopyInformation(img)
                obs_dict[f"manual_{observer}"] = mask
                color_dict[f"manual_{observer}"] = [0.7, 0.2, 0.2]

            com = None
            try:
                com = get_com(mask)
            except:
                com = [int(i / 2) for i in mask.GetSize()]

            img_vis = ImageVisualiser(img, cut=com, figure_size_in=16)
            img_vis.set_limits_from_label(mask, expansion=[0, 0, 0])

            contour_dict = {**obs_dict}
            contour_dict["pred"] = pred
            color_dict["pred"] = [0.2, 0.4, 0.8]

            img_vis.add_contour(contour_dict, color=color_dict)
            fig = img_vis.show()
            figure_path = f"valid_{case}.png"
            fig.savefig(figure_path, dpi=300)
            plt.close("all")

            try:
                self.logger.experiment.log_image(figure_path)
            except AttributeError:
                # Likely offline mode
                pass

            case_metrics = get_metrics(pred, mask)
            for m in case_metrics:
                metrics[m].append(case_metrics[m])

        for m in metrics:
            self.log(
                m,
                np.array(metrics[m]).mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
