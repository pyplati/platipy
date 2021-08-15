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

import sys
import os
import tempfile
import json

from pathlib import Path
import SimpleITK as sitk
import numpy as np

import comet_ml  # pylint: disable=unused-import
from pytorch_lightning.loggers import CometLogger

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from argparse import ArgumentParser

import matplotlib.pyplot as plt

from platipy.imaging.cnn.unet import UNet
from platipy.imaging.cnn.dataload import UNetDataModule
from platipy.imaging.cnn.dataset import preprocess_image

from platipy.imaging import ImageVisualiser
from platipy.imaging.label.utils import get_com


def post_process(pred):

    # Take only the largest componenet
    labelled_image = sitk.ConnectedComponent(pred)
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(labelled_image)
    label_indices = label_shape_filter.GetLabels()
    voxel_counts = [label_shape_filter.GetNumberOfPixels(i) for i in label_indices]
    if len(voxel_counts) > 0:
        largest_component_label = label_indices[np.argmax(voxel_counts)]
        largest_component_image = labelled_image == largest_component_label
        pred = sitk.Cast(largest_component_image, sitk.sitkUInt8)

    # Fill any holes in the structure
    pred = sitk.BinaryMorphologicalClosing(pred, (5, 5, 5))
    pred = sitk.BinaryFillhole(pred)

    return pred


def get_metrics(target, pred):

    result = {}
    lomif = sitk.LabelOverlapMeasuresImageFilter()
    lomif.Execute(target, pred)
    result["JI"] = lomif.GetJaccardCoefficient()
    result["DSC"] = lomif.GetDiceCoefficient()

    if sitk.GetArrayFromImage(pred).sum() == 0:
        result["HD"] = 1000
        result["ASD"] = 100
    else:
        hdif = sitk.HausdorffDistanceImageFilter()
        hdif.Execute(target, pred)
        result["HD"] = hdif.GetHausdorffDistance()
        result["ASD"] = hdif.GetAverageHausdorffDistance()

    return result


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

        # scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer, lr_lambda=[lambda epoch: self.hparams.lr_lambda ** (epoch)]
        # )

        return optimizer

    def infer(self, img):

        pp_img = preprocess_image(img, spacing=self.hparams.spacing, crop_to_mm=self.hparams.crop_to_mm)

        preds = []
        for z in range(pp_img.GetSize()[2]):
            x = sitk.GetArrayFromImage(pp_img[:,:, z])
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
        pred = post_process(pred)
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

    def validation_step(self, batch, batch_idx):

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
            pred = post_process(pred)
            pred.CopyInformation(img)
            sitk.WriteImage(pred, f"val_pred_{case}.nii.gz")

            color_dict = {}
            obs_dict = {}

            try:
                get_com(pred)
            except:
                continue
            img_vis = ImageVisualiser(
                img, cut=get_com(pred), figure_size_in=16, window=[-1.0, 1.0]
            )

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


def main(args, config_json_path=None):

    pl.seed_everything(args.seed, workers=True)

    args.working_dir = Path(args.working_dir)
    args.working_dir = args.working_dir.joinpath(args.experiment)
    args.fold_dir = args.working_dir.joinpath(f"fold_{args.fold}")
    args.default_root_dir = str(args.fold_dir)

    comet_api_key = None
    comet_workspace = None
    comet_project = None

    if args.comet_api_key:
        comet_api_key = args.comet_api_key
        comet_workspace = args.comet_workspace
        comet_project = args.comet_project

    if comet_api_key is None:
        if "COMET_API_KEY" in os.environ:
            comet_api_key = os.environ["COMET_API_KEY"]
        if "COMET_WORKSPACE" in os.environ:
            comet_workspace = os.environ["COMET_WORKSPACE"]
        if "COMET_PROJECT" in os.environ:
            comet_project = os.environ["COMET_PROJECT"]

    if comet_api_key is not None:
        comet_logger = CometLogger(
            api_key=comet_api_key,
            workspace=comet_workspace,
            project_name=comet_project,
            experiment_name=args.experiment,
            save_dir=args.working_dir,
            offline=args.offline,
        )
        if config_json_path:
            comet_logger.experiment.log_code(config_json_path)

    dict_args = vars(args)

    data_module = UNetDataModule(**dict_args)

    prob_unet = LocaliseUNet(**dict_args)

    if args.resume_from is not None:
        trainer = pl.Trainer(resume_from_checkpoint=args.resume_from)
    else:
        trainer = pl.Trainer.from_argparse_args(args)

    if comet_api_key is not None:
        trainer.logger = comet_logger

    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer.callbacks.append(lr_monitor)

    # Save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor="DSC",
        dirpath=args.default_root_dir,
        filename="localise-{epoch:02d}-{DSC:.2f}",
        save_top_k=1,
        mode="max",
    )

    trainer.callbacks.append(checkpoint_callback)

    trainer.fit(prob_unet, data_module)


if __name__ == "__main__":

    args = None
    config_json_path = None
    if len(sys.argv) == 2:
        # Check if JSON file parsed, if so read arguments from there...
        if sys.argv[-1].endswith(".json"):
            config_json_path = sys.argv[-1]
            with open(config_json_path, "r") as f:
                params = json.load(f)
                args = []
                for key in params:
                    args.append(f"--{key}")

                    if isinstance(params[key], list):
                        for s in params[key]:
                            args.append(str(s))
                    else:
                        args.append(str(params[key]))

    arg_parser = ArgumentParser()
    arg_parser = LocaliseUNet.add_model_specific_args(arg_parser)
    arg_parser = UNetDataModule.add_model_specific_args(arg_parser)
    arg_parser = pl.Trainer.add_argparse_args(arg_parser)
    arg_parser.add_argument(
        "--config", type=str, default=None, help="JSON file with parameters to load"
    )
    arg_parser.add_argument("--seed", type=int, default=42, help="an integer to use as seed")
    arg_parser.add_argument("--experiment", type=str, default="default", help="Name of experiment")
    arg_parser.add_argument("--working_dir", type=str, default="./working")
    arg_parser.add_argument("--num_observers", type=int, default=5)
    arg_parser.add_argument("--spacing", nargs="+", type=int, default=[3, 3, 3])
    arg_parser.add_argument("--offline", type=bool, default=False)
    arg_parser.add_argument("--comet_api_key", type=str, default=None)
    arg_parser.add_argument("--comet_workspace", type=str, default=None)
    arg_parser.add_argument("--comet_project", type=str, default=None)
    arg_parser.add_argument("--resume_from", type=str, default=None)
    arg_parser.add_argument("--combine_observers", type=str, default="union")

    main(arg_parser.parse_args(args), config_json_path=config_json_path)
