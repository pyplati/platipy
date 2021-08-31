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
from scipy.optimize import linear_sum_assignment

import comet_ml  # pylint: disable=unused-import
from pytorch_lightning.loggers import CometLogger

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from argparse import ArgumentParser

import matplotlib.pyplot as plt

from platipy.imaging.cnn.prob_unet import ProbabilisticUnet
from platipy.imaging.cnn.unet import l2_regularisation
from platipy.imaging.cnn.dataload import UNetDataModule
from platipy.imaging.cnn.dataset import crop_img_using_localise_model
from platipy.imaging.cnn.utils import (
    preprocess_image,
    postprocess_mask,
    get_metrics
)

from platipy.imaging import ImageVisualiser
from platipy.imaging.label.utils import get_com, get_union_mask, get_intersection_mask


class ProbUNet(pl.LightningModule):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        loss_params = None

        if self.hparams.loss_type == "elbo":
            loss_params = {
                "beta": self.hparams.beta,
            }

        if self.hparams.loss_type == "geco":
            loss_params = {
                "kappa": self.hparams.kappa,
                "clamp_rec": self.hparams.clamp_rec,
            }

        loss_params["top_k_percentage"] = self.hparams.top_k_percentage
        loss_params["contour_loss_lambda_threshold"] = self.hparams.contour_loss_lambda_threshold
        loss_params["contour_loss_weight"] = self.hparams.contour_loss_weight

        self.prob_unet = ProbabilisticUnet(
            self.hparams.input_channels,
            self.hparams.num_classes,
            self.hparams.filters_per_layer,
            self.hparams.latent_dim,
            self.hparams.no_convs_fcomb,
            self.hparams.loss_type,
            loss_params,
            self.hparams.ndims,
        )

        self.validation_directory = None

        self.stddevs = np.linspace(-2, 2, self.hparams.num_observers)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Probabilistic UNet")
        parser.add_argument("--learning_rate", type=float, default=1e-5)
        parser.add_argument("--lr_lambda", type=float, default=0.99)
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
        parser.add_argument("--contour_loss_lambda_threshold", type=float, default=None)
        parser.add_argument("--contour_loss_weight", type=float, default=0.0)
        parser.add_argument("--epochs_all_rec", type=int, default=0)

        return parent_parser

    def forward(self, x):
        self.prob_unet.forward(x, None, False)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=0
        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=[lambda epoch: self.hparams.lr_lambda ** (epoch)]
        )

        return [optimizer], [scheduler]

    def infer(
        self,
        img,
        num_samples=1,
        sample_strategy="mean",
        latent_dim=True,
        spaced_range=[-1.5, 1.5],
        preprocess=True,
    ):
        # sample strategy in "mean", "random", "spaced"

        if not hasattr(latent_dim, "__iter__"):
            latent_dim = [
                latent_dim,
            ] * self.hparams.latent_dim

        if sample_strategy == "mean":
            samples = [{"name": "mean", "std_dev_from_mean": torch.Tensor([0.0] * len(latent_dim)).to(self.device), "preds": []}]
        elif sample_strategy == "random":
            samples = [
                {
                    "name": f"random_{i}",
                    "std_dev_from_mean": torch.Tensor(
                        [np.random.normal(0, 1.0, 1)[0] if d else 0.0 for d in latent_dim]
                    ).to(self.device),
                    "preds": [],
                }
                for i in range(num_samples)
            ]
        elif sample_strategy == "spaced":
            samples = [
                {
                    "name": f"spaced_{s}",
                    "std_dev_from_mean": torch.Tensor([s if d else 0.0 for d in latent_dim]).to(self.device),
                    "preds": [],
                }
                for s in np.linspace(spaced_range[0], spaced_range[1], num_samples)
            ]

        with torch.no_grad():

            if preprocess:
                if self.hparams.crop_using_localise_model:
                    localise_path = self.hparams.crop_using_localise_model.format(
                        fold=self.hparams.fold
                    )
                    img = crop_img_using_localise_model(
                        img,
                        localise_path,
                        spacing=self.hparams.spacing,
                        crop_to_grid_size=self.hparams.localise_voxel_grid_size,
                    )
                else:
                    img = preprocess_image(
                        img,
                        spacing=self.hparams.spacing,
                        crop_to_grid_size_xy=self.hparams.crop_to_grid_size,
                        intensity_scaling=self.hparams.intensity_scaling,
                        intensity_window=self.hparams.intensity_window,
                    )

            img_arr = sitk.GetArrayFromImage(img)
            if self.hparams.ndims == 2:
               slices = [img_arr[z, :, :] for z in range(img_arr.shape[0])]
            else:
               slices = [img_arr]
            for i in slices:

                x = torch.Tensor(i).to(self.device)
                x = x.unsqueeze(0)
                x = x.unsqueeze(0)
                self.prob_unet.forward(x)

                for sample in samples:
                    if sample["name"] == "mean":
                        y = self.prob_unet.sample(testing=True, use_mean=True)
                    else:
                        y = self.prob_unet.sample(
                            testing=True,
                            use_mean=False,
                            sample_x_stddev_from_mean=sample["std_dev_from_mean"],
                        )

                    y = y.squeeze(0)
                    y = np.argmax(y.cpu().detach().numpy(), axis=0)
                    sample["preds"].append(y)

        result = {}
        for sample in samples:

            pred_arr = sample["preds"][0]
            if len(sample["preds"]) > 1:
                pred_arr = np.stack(sample["preds"])

            pred = sitk.GetImageFromArray(pred_arr)
            pred = sitk.Cast(pred, sitk.sitkUInt8)

            pred.CopyInformation(img)
            pred = postprocess_mask(pred)
            pred = sitk.Resample(pred, img, sitk.Transform(), sitk.sitkNearestNeighbor)

            result[sample["name"]] = pred

        return result

    def validate(self, img, manual_observers, samples, mean, matching_type="best", window=[-0.5,1.0]):

        metrics = {"DSC": "max", "HD": "min", "ASD": "min"}

        contour_cmap = "coolwarm"

        intersection_mask = get_intersection_mask(manual_observers)
        union_mask = get_union_mask(manual_observers)
        vis = ImageVisualiser(img, cut=get_com(union_mask), figure_size_in=16, window=window)

        vis.add_contour(
            mean, color=plt.cm.get_cmap(contour_cmap)(0.5), linewidth=3, show_legend=False
        )
        vis.add_contour(
            manual_observers, color=[0.13, 0.67, 0.275], linewidth=0.5, show_legend=False
        )

        vis.add_contour(
            intersection_mask, name="intersection", color=[0.13, 0.67, 0.275], linewidth=3
        )
        vis.add_contour(union_mask, name="union", color=[0.13, 0.67, 0.275], linewidth=3)
        vis.add_contour(
            samples,
            linewidth=1.5,
            color={
                s: c
                for s, c in zip(
                    samples, plt.cm.get_cmap(contour_cmap)(np.linspace(0, 1, len(samples)))
                )
            },
        )

        vis.set_limits_from_label(union_mask, expansion=30)

        fig = vis.show()

        sim = {k: np.zeros((len(samples), len(manual_observers))) for k in metrics}
        msim = {k: np.zeros((len(samples), len(manual_observers))) for k in metrics}
        for sid, samp in enumerate(samples):
            for oid, obs in enumerate(manual_observers):
                sample_metrics = get_metrics(manual_observers[obs], samples[samp])
                mean_metrics = get_metrics(manual_observers[obs], mean["mean"])

                for k in sample_metrics:
                    sim[k][sid, oid] = sample_metrics[k]
                    msim[k][sid, oid] = mean_metrics[k]

        result = {"probnet": {k: [] for k in metrics}, "unet": {k: [] for k in metrics}}
        for k in sim:

            val = sim[k]
            if matching_type == "hungarian":
                if metrics[k] == "max":
                    val = -val
                row_idx, col_idx = linear_sum_assignment(val)
                prob_unet_mean = sim[k][row_idx, col_idx].mean()
            else:
                if metrics[k] == "max":
                    prob_unet_mean = val.max()
                else:
                    prob_unet_mean = val.min()
            result["probnet"][k].append(prob_unet_mean)

            val = msim[k]
            if matching_type == "hungarian":
                if metrics[k] == "max":
                    val = -val
                row_idx, col_idx = linear_sum_assignment(val)
                unet_mean = msim[k][row_idx, col_idx].mean()
            else:
                if metrics[k] == "max":
                    unet_mean = val.max()
                else:
                    unet_mean = val.min()
            result["unet"][k].append(unet_mean)

        return result, fig

    def training_step(self, batch, _):

        x, y, m, _ = batch

        self.prob_unet.forward(x, y, training=True)

        use_max_lambda = self.current_epoch < self.hparams.epochs_all_rec

        loss = self.prob_unet.loss(y, analytic_kl=True, mask=m, use_max_lambda=use_max_lambda)
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
            if k == "loss":
                continue
            self.log(
                k,
                loss[k],
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True,
            )
        return loss

    def validation_step(self, batch, _):

        if self.validation_directory is None:
            self.validation_directory = Path(tempfile.mkdtemp())

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

        metrics = ["DSC", "HD", "ASD"]
        computed_metrics = {
            **{f"probnet_{m}": [] for m in metrics},
            **{f"unet_{m}": [] for m in metrics},
        }

        for case in cases:

            img_arrs = []
            slices = []

            if self.hparams.ndims == 2:
                for z in range(cases[case]["slices"] + 1):
                    img_file = self.validation_directory.joinpath(f"img_{case}_{z}.npy")
                    if img_file.exists():
                        img_arrs.append(np.load(img_file))
                        slices.append(z)

                img_arr = np.stack(img_arrs)

            else:
                img_file = self.validation_directory.joinpath(f"img_{case}_0.npy")
                img_arr = np.load(img_file)
            img = sitk.GetImageFromArray(img_arr)
            img.SetSpacing(self.hparams.spacing)

            mean = self.infer(img, num_samples=1, sample_strategy="mean", preprocess=False)
            samples = self.infer(
                img,
                sample_strategy="spaced",
                num_samples=5,
                spaced_range=[-1.5, 1.5],
                preprocess=False,
            )

            observers = {}
            for _, observer in enumerate(cases[case]["observers"]):

                if self.hparams.ndims == 2:
                    mask_arrs = []
                    for z in slices:
                        mask_file = self.validation_directory.joinpath(
                            f"mask_{case}_{z}_{observer}.npy"
                        )

                        mask_arrs.append(np.load(mask_file))

                    mask_arr = np.stack(mask_arrs)

                else:
                    mask_file = self.validation_directory.joinpath(
                        f"mask_{case}_{z}_{observer}.npy"
                    )
                    mask_arr = np.load(mask_file)

                mask = sitk.GetImageFromArray(mask_arr)
                mask = sitk.Cast(mask, sitk.sitkUInt8)
                mask.CopyInformation(img)
                observers[f"manual_{observer}"] = mask

            result, fig = self.validate(img, observers, samples, mean, matching_type="best")

            figure_path = f"valid_{case}.png"
            fig.savefig(figure_path, dpi=300)
            plt.close("all")

            try:
                self.logger.experiment.log_image(figure_path)
            except AttributeError:
                # Likely offline mode
                pass

            for t in result:
                for m in metrics:
                    computed_metrics[f"{t}_{m}"]+=result[t][m]

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


def main(args, config_json_path=None):

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
        if config_json_path:
            comet_logger.experiment.log_code(config_json_path)

    dict_args = vars(args)

    data_module = UNetDataModule(**dict_args)

    prob_unet = ProbUNet(**dict_args)

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
        monitor="probnet_DSC",
        dirpath=args.default_root_dir,
        filename="probunet-{fold}-{epoch:02d}-{probnet_DSC:.2f}",
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
                        for list_val in params[key]:
                            args.append(str(list_val))
                    else:
                        args.append(str(params[key]))

    arg_parser = ArgumentParser()
    arg_parser = ProbUNet.add_model_specific_args(arg_parser)
    arg_parser = UNetDataModule.add_model_specific_args(arg_parser)
    arg_parser = pl.Trainer.add_argparse_args(arg_parser)
    arg_parser.add_argument(
        "--config", type=str, default=None, help="JSON file with parameters to load"
    )
    arg_parser.add_argument("--seed", type=int, default=42, help="an integer to use as seed")
    arg_parser.add_argument("--experiment", type=str, default="default", help="Name of experiment")
    arg_parser.add_argument("--working_dir", type=str, default="./working")
    arg_parser.add_argument("--num_observers", type=int, default=5)
    arg_parser.add_argument("--spacing", nargs="+", type=float, default=[1, 1, 1])
    arg_parser.add_argument("--offline", type=bool, default=False)
    arg_parser.add_argument("--comet_api_key", type=str, default=None)
    arg_parser.add_argument("--comet_workspace", type=str, default=None)
    arg_parser.add_argument("--comet_project", type=str, default=None)
    arg_parser.add_argument("--resume_from", type=str, default=None)

    main(arg_parser.parse_args(args), config_json_path=config_json_path)
