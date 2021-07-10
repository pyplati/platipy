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
import math
import tempfile
import random
import json

from pathlib import Path
import SimpleITK as sitk
import numpy as np
from scipy.optimize import linear_sum_assignment

import comet_ml
from pytorch_lightning.loggers import CometLogger

import torch
import pytorch_lightning as pl

from argparse import ArgumentParser

import matplotlib.pyplot as plt

from platipy.imaging.cnn.prob_unet import ProbabilisticUnet
from platipy.imaging.cnn.unet import l2_regularisation
from platipy.imaging.cnn.dataset import NiftiDataset
from platipy.imaging.cnn.sampler import ObserverSampler

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


class ProbUNet(pl.LightningModule):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        loss_params = None

        if self.hparams.loss_type == "elbo":
            loss_params = {"beta": self.hparams.beta}

        if self.hparams.loss_type == "geco":
            loss_params = {
                "kappa": self.hparams.kappa,
                "clamp_rec": self.hparams.clamp_rec,
                "top_k_percentage": self.hparams.top_k_percentage,
            }

        self.prob_unet = ProbabilisticUnet(
            self.hparams.input_channels,
            self.hparams.num_classes,
            self.hparams.filters_per_layer,
            self.hparams.latent_dim,
            self.hparams.no_convs_fcomb,
            self.hparams.loss_type,
            loss_params,
        )

        self.validation_directory = None

        self.stddevs = np.linspace(-2, 2, 5)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Probabilistic UNet")
        parser.add_argument("--learning_rate", type=float, default=1e-5)
        parser.add_argument("--input_channels", type=int, default=1)
        parser.add_argument("--num_classes", type=int, default=2)
        parser.add_argument(
            "--filters_per_layer", nargs="+", type=int, default=[64 * (2 ** x) for x in range(5)]
        )
        parser.add_argument("--latent_dim", type=int, default=6)
        parser.add_argument("--no_convs_fcomb", type=int, default=4)
        parser.add_argument("--loss_type", type=str, default="elbo")
        parser.add_argument("--beta", type=float, default=1.0)
        parser.add_argument("--kappa", type=float, default=0.02)
        parser.add_argument("--clamp_rec", nargs="+", type=float, default=[1e-5, 1e5])
        parser.add_argument("--top_k_percentage", type=float, default=None)

        return parent_parser

    def forward(self, x):
        self.prob_unet.forward(x, None, False)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=0
        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=[lambda epoch: 0.99 ** (epoch)]
        )

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        self.prob_unet.forward(x, y, training=True)
        loss = self.prob_unet.loss(y, analytic_kl=True)
        reg_loss = (
            l2_regularisation(self.prob_unet.posterior)
            + l2_regularisation(self.prob_unet.prior)
            + l2_regularisation(self.prob_unet.fcomb.layers)
        )
        training_loss = loss["loss"] + 1e-5 * reg_loss
        self.log(
            "training_loss",
            training_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )

        for k in loss:
            self.log(
                k,
                loss[k],
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True,
            )
        return loss

    def validation_step(self, batch, batch_idx):

        if self.validation_directory is None:
            self.validation_directory = Path(tempfile.mkdtemp())
            print(self.validation_directory)

        with torch.set_grad_enabled(False):
            x, y, info = batch

            for s in range(y.shape[0]):

                img_file = self.validation_directory.joinpath(
                    f"img_{info['case'][s]}_{info['z'][s]}.npy"
                )
                np.save(img_file, x[0].squeeze(0).cpu().numpy())

                mask_file = self.validation_directory.joinpath(
                    f"mask_{info['case'][s]}_{info['z'][s]}_{info['observer'][s]}.npy"
                )
                np.save(mask_file, y[s].squeeze(0).cpu().numpy())

                self.prob_unet.forward(x[s].unsqueeze(0))
                sample = self.prob_unet.sample(
                    testing=True,
                    use_mean=False,
                    sample_x_stddev_from_mean=self.stddevs[info["observer"][s]],
                )
                sample_file = self.validation_directory.joinpath(
                    f"sample_{info['case'][s]}_{info['z'][s]}_{info['observer'][s]}.npy"
                )
                sample = np.argmax(sample.squeeze(0).cpu().numpy(), axis=0)
                np.save(sample_file, sample)

                mean = self.prob_unet.sample(testing=True, use_mean=True)
                mean_file = self.validation_directory.joinpath(
                    f"mean_{info['case'][s]}_{info['z'][s]}.npy"
                )
                mean = np.argmax(mean.squeeze(0).cpu().numpy(), axis=0)
                np.save(mean_file, mean)

        return info

    def validation_epoch_end(self, validation_step_outputs):

        cases = {}
        cmap = plt.cm.get_cmap("Set2")
        for info in validation_step_outputs:

            for case, z, observer in zip(info["case"], info["z"], info["observer"]):

                if not case in cases:
                    cases[case] = {"slices": z.item(), "observers": [observer.item()]}
                else:
                    if z.item() > cases[case]["slices"]:
                        cases[case]["slices"] = z.item()
                    if not observer in cases[case]["observers"]:
                        cases[case]["observers"].append(observer.item())

        metrics = ["JI", "DSC", "HD", "ASD"]
        computed_metrics = {
            **{f"probnet_{m}": [] for m in metrics},
            **{f"unet_{m}": [] for m in metrics},
        }
        for case in cases:

            img_arrs = []
            mean_arrs = []
            slices = []
            for z in range(cases[case]["slices"] + 1):
                img_file = self.validation_directory.joinpath(f"img_{case}_{z}.npy")
                mean_file = self.validation_directory.joinpath(f"mean_{case}_{z}.npy")
                if img_file.exists():
                    img_arrs.append(np.load(img_file))
                    mean_arrs.append(np.load(mean_file))
                    slices.append(z)

            if len(slices) < 5:
                # Likely initial sanity check
                continue

            img_arr = np.stack(img_arrs)
            img = sitk.GetImageFromArray(img_arr)
            # sitk.WriteImage(img, f"test_{case}.nii.gz")

            mean_arr = np.stack(mean_arrs)
            mean = sitk.GetImageFromArray(mean_arr)
            mean = sitk.Cast(mean, sitk.sitkUInt8)
            mean = post_process(mean)
            # sitk.WriteImage(mean, f"val_mean_{case}_mean.nii.gz")

            obs_dict = {}
            pred_dict = {}
            color_dict = {}
            observers = []
            samples = []
            for observer in cases[case]["observers"]:
                mask_arrs = []
                sample_arrs = []
                for z in slices:
                    mask_file = self.validation_directory.joinpath(
                        f"mask_{case}_{z}_{observer}.npy"
                    )
                    sample_file = self.validation_directory.joinpath(
                        f"sample_{case}_{z}_{observer}.npy"
                    )

                    mask_arrs.append(np.load(mask_file))
                    sample_arrs.append(np.load(sample_file))

                mask_arr = np.stack(mask_arrs)
                mask = sitk.GetImageFromArray(mask_arr)
                mask = sitk.Cast(mask, sitk.sitkUInt8)
                # sitk.WriteImage(mask, f"val_mask_{case}_{observer}.nii.gz")
                observers.append(mask)
                obs_dict[f"manual_{observer}"] = mask
                color_dict[f"manual_{observer}"] = [0.5, 0.5, 0.5]

                sample_arr = np.stack(sample_arrs)
                sample = sitk.GetImageFromArray(sample_arr)
                sample = sitk.Cast(sample, sitk.sitkUInt8)
                sample = post_process(sample)
                # sitk.WriteImage(sample, f"val_sample_{case}_{observer}.nii.gz")
                samples.append(sample)
                pred_dict[f"auto_{self.stddevs[observer]}"] = sample
                color_dict[f"auto_{self.stddevs[observer]}"] = cmap(observer / 5)

            img_vis = ImageVisualiser(
                img, cut=get_com(mask), figure_size_in=16, window=[-1.0, 1.0]
            )

            contour_dict = {**obs_dict, **pred_dict}
            contour_dict["auto_mean"] = mean
            color_dict["auto_mean"] = [0.0, 0.0, 0.0]

            img_vis.add_contour(contour_dict, color=color_dict)
            fig = img_vis.show()
            figure_path = f"valid_{case}.png"
            fig.savefig(figure_path, dpi=300)
            plt.close("all")

            self.logger.experiment.log_image(figure_path)

            sim = {k: np.zeros((len(observers), len(samples))) for k in metrics}
            msim = {k: np.zeros((len(observers), len(samples))) for k in metrics}
            for sid, samp in enumerate(samples):
                for oid, obs in enumerate(observers):
                    sample_metrics = get_metrics(obs, samp)
                    mean_metrics = get_metrics(obs, mean)

                    for k in sample_metrics:
                        sim[k][sid, oid] = sample_metrics[k]
                        msim[k][sid, oid] = mean_metrics[k]

            result = {"probnet": {k: [] for k in metrics}, "unet": {k: [] for k in metrics}}
            for k in sim:

                val = sim[k]
                if not k.endswith("D"):
                    val = -val
                row_idx, col_idx = linear_sum_assignment(val)
                prob_unet_mean = sim[k][row_idx, col_idx].mean()
                result["probnet"][k].append(prob_unet_mean)

                val = msim[k]
                if not k.endswith("D"):
                    val = -val
                row_idx, col_idx = linear_sum_assignment(val)
                unet_mean = msim[k][row_idx, col_idx].mean()
                result["unet"][k].append(unet_mean)

            for t in result:
                for m in result[t]:
                    computed_metrics[f"{t}_{m}"].append(np.array(result[t][m]).mean())

        for cm in computed_metrics:
            self.log(
                cm,
                np.array(computed_metrics[cm]).mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

        # shutil.rmtree(self.validation_directory)


class ProbUNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        working_dir: str = "./working",
        case_glob="images/*.nii.gz",
        image_glob="images/{case}.nii.gz",
        label_glob="labels/{case}_*.nii.gz",
        fold=0,
        k_folds=5,
        batch_size=5,
        num_workers=4,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.working_dir = Path(working_dir)

        self.case_glob = case_glob
        self.image_glob = image_glob
        self.label_glob = label_glob

        self.fold = fold
        self.k_folds = k_folds

        self.train_cases = []
        self.validation_cases = []

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.training_set = None
        self.validation_set = None

        print(f"Training fold {self.fold}")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data Loader")
        parser.add_argument("--data_dir", type=str, default="./data")
        parser.add_argument("--fold", type=int, default=0)
        parser.add_argument("--k_folds", type=int, default=5)
        parser.add_argument("--batch_size", type=int, default=5)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--case_glob", type=str, default="images/*.nii.gz")
        parser.add_argument("--image_glob", type=str, default="images/{case}.nii.gz")
        parser.add_argument("--label_glob", type=str, default="labels/{case}_*.nii.gz")

        return parent_parser

    def setup(self, stage=None):

        cases = [p.name.replace(".nii.gz", "") for p in self.data_dir.glob(self.case_glob)]
        cases.sort()
        random.shuffle(cases)  # will be consistent for same value of 'seed everything'
        cases_per_fold = math.ceil(len(cases) / self.k_folds)
        for f in range(self.k_folds):

            if self.fold == f:
                self.validation_cases = cases[f * cases_per_fold : (f + 1) * cases_per_fold]
            else:
                self.train_cases += cases[f * cases_per_fold : (f + 1) * cases_per_fold]

        train_data = [
            {
                "id": case,
                "image": self.data_dir.joinpath(self.image_glob.format(case=case)),
                "label": [p for p in self.data_dir.glob(self.label_glob.format(case=case))],
            }
            for case in self.train_cases
        ]
        print(train_data)

        validation_data = [
            {
                "id": case,
                "image": self.data_dir.joinpath(self.image_glob.format(case=case)),
                "label": [p for p in self.data_dir.glob(self.label_glob.format(case=case))],
            }
            for case in self.validation_cases
        ]
        print(validation_data)

        self.training_set = NiftiDataset(train_data, self.working_dir.joinpath("train"))
        self.validation_set = NiftiDataset(
            validation_data, self.working_dir.joinpath("validation"), False
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.training_set,
            # batch_sampler=BatchSampler(
            #    ObserverSampler(train_set, 5), batch_size=params["batch_size"], drop_last=False
            # ),
            # num_workers=params["num_workers"],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.validation_set,
            batch_sampler=torch.utils.data.BatchSampler(
                ObserverSampler(self.validation_set, 5), batch_size=5, drop_last=False
            ),
            num_workers=self.num_workers,
        )


def main(args):

    pl.seed_everything(args.seed, workers=True)

    args.working_dir = Path(args.working_dir)
    args.working_dir = args.working_dir.joinpath(args.experiment)
    args.default_root_dir = str(args.working_dir)

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

    dict_args = vars(args)

    data_module = ProbUNetDataModule(**dict_args)

    prob_unet = ProbUNet(**dict_args)
    trainer = pl.Trainer.from_argparse_args(args)

    if comet_api_key is not None:
        trainer.logger = comet_logger

    trainer.fit(prob_unet, data_module)  # pylint: disable=no-member


if __name__ == "__main__":

    args = None
    if len(sys.argv) == 2:
        # Check if JSON file parsed, if so read arguments from there...
        if sys.argv[-1].endswith(".json"):
            with open(sys.argv[-1], "r") as f:
                params = json.load(f)
                args = []
                for k in params:
                    args.append(f"--{k}")
                    args.append(params[k])

    arg_parser = ArgumentParser()
    arg_parser = ProbUNet.add_model_specific_args(arg_parser)
    arg_parser = ProbUNetDataModule.add_model_specific_args(arg_parser)
    arg_parser = pl.Trainer.add_argparse_args(arg_parser)
    arg_parser.add_argument("--seed", type=int, default=42, help="an integer to use as seed")
    arg_parser.add_argument("--experiment", type=str, default="default", help="Name of experiment")
    arg_parser.add_argument("--working_dir", type=str, default="./working")
    arg_parser.add_argument("--offline", type=bool, default=False)
    arg_parser.add_argument("--comet_api_key", type=str, default=None)
    arg_parser.add_argument("--comet_workspace", type=str, default=None)
    arg_parser.add_argument("--comet_project", type=str, default=None)

    main(arg_parser.parse_args(args))
