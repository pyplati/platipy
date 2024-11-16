# 2021 University of New South Wales, University of Sydney, Ingham Institute

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
from argparse import ArgumentParser

from pathlib import Path
import matplotlib
import SimpleITK as sitk
import numpy as np
from scipy.optimize import linear_sum_assignment

import comet_ml  # pylint: disable=unused-import
from pytorch_lightning.loggers import CometLogger
from torchmetrics import JaccardIndex

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import matplotlib.pyplot as plt

from platipy.imaging.cnn.prob_unet import ProbabilisticUnet

# from platipy.imaging.cnn.hierarchical_prob_unet import HierarchicalProbabilisticUnet
from platipy.imaging.cnn.unet import l2_regularisation
from platipy.imaging.cnn.dataload import UNetDataModule
from platipy.imaging.cnn.dataset import crop_img_using_localise_model
from platipy.imaging.cnn.utils import (
    preprocess_image,
    postprocess_mask,
    get_metrics,
    resample_mask_to_image,
)
from platipy.imaging.cnn.metrics import probabilistic_dice

from platipy.imaging import ImageVisualiser
from platipy.imaging.label.utils import get_com, get_union_mask, get_intersection_mask


class GECOEarlyStopping(EarlyStopping):
    def on_validation_end(self, trainer, pl_module):
        # Make sure the GECO lambda metrics are below 0.1 before stopping
        logs = trainer.callback_metrics
        should_consider_early_stop = True

        if "lambda_rec" in logs and logs["lambda_rec"] >= 0.01:
            should_consider_early_stop = False

        if "lambda_contour" in logs and logs["lambda_contour"] >= 0.01:
            should_consider_early_stop = False

        if should_consider_early_stop:
            self._run_early_stopping_check(trainer)

    def on_train_epoch_end(self, trainer, pl_module):
        pass


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
                "clamp_contour": self.hparams.clamp_contour,
                "kappa_contour": self.hparams.kappa_contour,
                "rec_geco_step_size": self.hparams.rec_geco_step_size,
            }

        loss_params["top_k_percentage"] = self.hparams.top_k_percentage
        loss_params[
            "contour_loss_lambda_threshold"
        ] = self.hparams.contour_loss_lambda_threshold
        loss_params["contour_loss_weight"] = self.hparams.contour_loss_weight

        self.use_structure_context = self.hparams.use_structure_context

        if self.hparams.prob_type == "prob":
            self.prob_unet = ProbabilisticUnet(
                self.hparams.input_channels,
                len(self.hparams.structures)
                + 1,  # Add 1 to num classes for background class
                self.hparams.filters_per_layer,
                self.hparams.latent_dim,
                self.hparams.no_convs_fcomb,
                self.hparams.loss_type,
                loss_params,
                self.hparams.ndims,
                dropout_probability=self.hparams.dropout_probability,
                use_structure_context=self.use_structure_context,
            )
        elif self.hparams.prob_type == "hierarchical":
            raise NotImplementedError("Hierarchical Prob UNet current not working...")
            # self.prob_unet = HierarchicalProbabilisticUnet(
            #     input_channels=self.hparams.input_channels,
            #     num_classes=len(self.hparams.structures),
            #     filters_per_layer=self.hparams.filters_per_layer,
            #     down_channels_per_block=self.hparams.down_channels_per_block,
            #     latent_dims=[self.hparams.latent_dim] * (len(self.hparams.filters_per_layer) - 1),
            #     convs_per_block=self.hparams.convs_per_block,
            #     blocks_per_level=self.hparams.blocks_per_level,
            #     loss_type=self.hparams.loss_type,
            #     loss_params=loss_params,
            #     ndims=self.hparams.ndims,
            # )

        self.validation_directory = None
        self.kl_div = None

        self.stddevs = np.linspace(-2, 2, self.hparams.num_observers)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Probabilistic UNet")
        parser.add_argument("--prob_type", type=str, default="prob")
        parser.add_argument("--learning_rate", type=float, default=1e-5)
        parser.add_argument("--lr_lambda", type=float, default=0.99)
        parser.add_argument("--input_channels", type=int, default=1)
        parser.add_argument(
            "--filters_per_layer",
            nargs="+",
            type=int,
            default=[64 * (2**x) for x in range(5)],
        )
        parser.add_argument(
            "--down_channels_per_block", nargs="+", type=int, default=None
        )
        parser.add_argument("--latent_dim", type=int, default=6)
        parser.add_argument("--no_convs_fcomb", type=int, default=4)
        parser.add_argument("--convs_per_block", type=int, default=2)
        parser.add_argument("--blocks_per_level", type=int, default=1)
        parser.add_argument("--loss_type", type=str, default="elbo")
        parser.add_argument("--beta", type=float, default=1.0)
        parser.add_argument("--kappa", type=float, default=0.02)
        parser.add_argument("--kappa_contour", type=float, default=None)
        parser.add_argument("--rec_geco_step_size", type=float, default=1e-2)
        parser.add_argument("--weight_decay", type=float, default=1e-2)
        parser.add_argument("--clamp_rec", nargs="+", type=float, default=[1e-5, 1e5])
        parser.add_argument(
            "--clamp_contour", nargs="+", type=float, default=[1e-3, 1e3]
        )
        parser.add_argument("--top_k_percentage", type=float, default=None)
        parser.add_argument("--contour_loss_lambda_threshold", type=float, default=None)
        parser.add_argument(
            "--contour_loss_weight", type=float, default=0.0
        )  # no longer used
        parser.add_argument("--epochs_all_rec", type=int, default=0)  # no longer used
        parser.add_argument("--dropout_probability", type=float, default=0.0)
        parser.add_argument("--use_structure_context", type=int, default=0)

        return parent_parser

    def forward(self, x):
        self.prob_unet.forward(x, None, False)
        return x

    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(
        #    self.parameters(), lr=self.hparams.learning_rate, weight_decay=0
        #)
        #lr_lambda_unet = lambda epoch: self.hparams.lr_lambda ** (epoch)
#        scheduler = torch.optim.lr_scheduler.LambdaLR(
#           optimizer, lr_lambda=[lr_lambda_unet]
#        )

        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #    optimizer, 50, eta_min=1e-5, verbose=True
        #)

        #scheduler = torch.optim.lr_scheduler.CyclicLR(
        #   optimizer,
        #   base_lr=self.hparams.learning_rate / 10,
        #   max_lr=self.hparams.learning_rate * 10,
        #   step_size_up=50,
        #   mode="exp_range",
        #   gamma=0.9999,
        #   cycle_momentum=False
        #)

        #return [optimizer], [scheduler]
        params = [
            {
                "params": self.prob_unet.unet.parameters(),
                "weight_decay": self.hparams.weight_decay,
                "lr": 1e-5,
            }
        ]

        if self.prob_unet.prior is not None:
            param_list =[
                self.prob_unet.prior.parameters(),
                self.prob_unet.posterior.parameters(),
                self.prob_unet.fcomb.parameters(),
            ]
        else:
            param_list =[
                self.prob_unet.posterior.parameters(),
                self.prob_unet.fcomb.parameters(),
            ]
        for m in param_list:
            params += [
                {"params": m, "weight_decay": self.hparams.weight_decay, "lr": 1e-5}
            ]

        optimizer = torch.optim.Adam(params)

        lr_lambda_unet = lambda epoch: self.hparams.lr_lambda ** (epoch)
        lr_lambda_prob = lambda epoch: 0.99 ** (epoch)

        #        max_epochs = self.hparams.max_epochs
        #        lr_lambda = lambda x: np.interp(((np.sin(x/(max_epochs/8)) * np.sin(x/(max_epochs/4)))), np.array([-1,0,1]), np.array([0.1,1,10]))

        # scheduler = torch.optim.lr_scheduler.LambdaLR(
        #    optimizer, lr_lambda=[lr_lambda_unet, lr_lambda_prob, lr_lambda_prob, lr_lambda_prob]
        # )
        scheduler = torch.optim.lr_scheduler.CyclicLR(
           optimizer,
           base_lr=self.hparams.learning_rate,
           max_lr=self.hparams.learning_rate * 10,
           step_size_up=50,
           mode="exp_range",
           gamma=0.99,
          cycle_momentum=False
        )
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #    optimizer, 50, eta_min=1e-6, verbose=True
       #  )

        return [optimizer], [scheduler]

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    "max",
                    patience=20,
                    threshold=0.1e-2,
                    factor=0.75
                    #                     optimizer, "max", patience=200, threshold=0.75, factor=0.5
                ),
                "monitor": "probabilisticDice",
            },
        }

    def infer(
        self,
        img,
        context_map=None,
        seg=None,
        num_samples=1,
        sample_strategy="mean",
        latent_dim=True,
        spaced_range=[-1.5, 1.5],
        preprocess=True,
        return_latent_space=False
    ):
        # sample strategy in "mean", "random", "spaced"

        if not hasattr(latent_dim, "__iter__"):
            latent_dim = [
                latent_dim,
            ] * self.hparams.latent_dim

        if sample_strategy == "mean":
            samples = [
                {
                    "name": "mean",
                    "std_dev_from_mean": torch.Tensor([0.0] * len(latent_dim)).to(
                        self.device
                    ),
                    "preds": [],
                }
            ]
        elif sample_strategy == "random":
            samples = [
                {
                    "name": f"random_{i}",
                    "std_dev_from_mean": torch.Tensor(
                        [
                            np.random.normal(0, 1.0, 1)[0] if d else 0.0
                            for d in latent_dim
                        ]
                    ).to(self.device),
                    "preds": [],
                }
                for i in range(num_samples)
            ]
        elif sample_strategy == "spaced":
            if self.hparams.prob_type == "hierarchical":
                latent_dim = [True] * (len(self.hparams.filters_per_layer) - 1)
            samples = [
                {
                    "name": f"spaced_{s:.2f}",
                    "std_dev_from_mean": torch.Tensor(
                        [s if d else 0.0 for d in latent_dim]
                    ).to(self.device),
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
                        context_seg=seg
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

            if context_map is not None:
                context_map = resample_mask_to_image(img, context_map)
                cmap_arr = sitk.GetArrayFromImage(img)

            if seg is not None:
                seg = resample_mask_to_image(img, seg)
                seg_arr = sitk.GetArrayFromImage(seg)

            if self.hparams.ndims == 2:
                slices = [img_arr[z, :, :] for z in range(img_arr.shape[0])]

                if context_map is not None:
                    cmap_slices = [cmap_arr[z, :, :] for z in range(cmap_arr.shape[0])]

                if seg is not None:
                    seg_slices = [seg_arr[z, :, :] for z in range(seg_arr.shape[0])]
            else:
                slices = [img_arr]
                if context_map is not None:
                    cmap_slices = [cmap_arr]

                if seg is not None:
                    seg_slices = [seg_arr]

            for idx, i in enumerate(slices):
                x = torch.Tensor(i).to(self.device)
                x = x.unsqueeze(0)
                x = x.unsqueeze(0)

                if context_map is not None:
                    c = torch.Tensor(cmap_slices[idx]).to(self.device)
                    c = c.unsqueeze(0)
                    c = c.unsqueeze(0)

                    x = torch.cat((x, c), dim=1)

                if seg is not None:
                    s = torch.Tensor(seg_slices[idx]).to(self.device)
                    s = s.unsqueeze(0)
                    s = s.unsqueeze(0)

                    # Add in background channel
                    not_s = 1 - s.max(axis=1).values
                    not_s = torch.unsqueeze(not_s, dim=1)
                    s = torch.cat((not_s, s), dim=1).float()

                if self.hparams.prob_type == "prob":
                    if seg is not None:
                        self.prob_unet.forward(x, cseg=s)
                    else:
                        self.prob_unet.forward(x)

                if return_latent_space:
                    return self.prob_unet.prior_latent_space

                for sample in samples:
                    if self.hparams.prob_type == "prob":
                        if sample["name"] == "mean":
#                            if seg is None:
                            y = self.prob_unet.sample(testing=True, use_mean=True)
#                            else:
#                                y = self.prob_unet.reconstruct(use_posterior_mean=True)
                        else:
#                            if seg is None:
                            y = self.prob_unet.sample(
                                testing=True,
                                use_mean=False,
                                sample_x_stddev_from_mean=sample["std_dev_from_mean"],
                            )
 #                           else:
 #                               y = self.prob_unet.reconstruct(
 #                                   use_posterior_mean=False,
 #                                   sample_x_stddev_from_mean=sample["std_dev_from_mean"],
 #                               )

                    # else:
                    #     if sample["name"] == "mean":
                    #         y = self.prob_unet.sample(x, mean=True)
                    #     else:
                    #         y = self.prob_unet.sample(
                    #             x,
                    #             mean=True,
                    #             std_devs_from_mean=sample["std_dev_from_mean"],
                    # )

                    y = y.squeeze(0)
                    # y = np.argmax(y.cpu().detach().numpy(), axis=0)
                    y = torch.sigmoid(y)
                    sample["preds"].append(y.cpu().detach().numpy())

        result = {}
        for sample in samples:
            pred_arr = sample["preds"][0]

            if self.hparams.ndims == 2:
                pred_arr = np.expand_dims(pred_arr, 1)
            if len(sample["preds"]) > 1:
                pred_arr = np.stack(sample["preds"], axis=1)

            result[sample["name"]] = {}

            for idx, structure in enumerate(self.hparams.structures):
                pred = sitk.GetImageFromArray(pred_arr[idx + 1])  # Skip the background
                pred = pred > 0.5  # Threshold softmax at 0.5
                pred = sitk.Cast(pred, sitk.sitkUInt8)

                pred.CopyInformation(img)
                pred = postprocess_mask(pred)
                pred = sitk.Resample(
                    pred, img, sitk.Transform(), sitk.sitkNearestNeighbor
                )

                result[sample["name"]][structure] = pred

        return result

    def validate(
        self,
        img,
        manual_observers,
        samples,
        mean,
        matching_type="best",
        window=[-0.5, 1.0],
    ):
        metrics = {"DSC": "max", "HD": "min", "ASD": "min"}
        result = {}

        contour_cmaps = ["RdPu", "YlOrRd", "GnBu", "OrRd", "YlGn", "YlGnBu"]
        structures = self.hparams.structures

        try:
            cut = get_com(mean["mean"][structures[0]])
        except ValueError:
            cut = [int(i / 2) for i in img.GetSize()][::-1]

        vis = ImageVisualiser(img, cut=cut, figure_size_in=16, window=window)

        mean_contours = {}
        for idx, structure in enumerate(structures):
            color_map = matplotlib.colormaps.get_cmap(
                contour_cmaps[idx % len(structures)]
            )
            mean_contours[f"mean_{structure}"] = mean["mean"][structure]

            vis.add_contour(
                mean_contours, color=color_map(0.35), linewidth=3, show_legend=False
            )

            manual_color = color_map(0.9)

            manual_observers_struct = {
                f"{man_struct}_{structure}": manual_observers[man_struct][structure]
                for man_struct in manual_observers
            }

#            vis.add_contour(
#                manual_observers_struct,
#                color=manual_color,
#                linewidth=0.5,
#                show_legend=False,
#            )

            intersection_mask = get_intersection_mask(manual_observers_struct)
            union_mask = get_union_mask(manual_observers_struct)

            vis.add_contour(
                intersection_mask,
                name=f"intersection_{structure}",
                color=manual_color,
                linewidth=3,
            )
            vis.add_contour(
                union_mask, name=f"union_{structure}", color=manual_color, linewidth=3
            )

            samples_struct = {
                f"{sample_struct}_{structure}": samples[sample_struct][structure]
                for sample_struct in samples
            }
            vis.add_contour(
                samples_struct,
                linewidth=1.5,
                color={
                    s: c
                    for s, c in zip(
                        samples_struct,
                        color_map(np.linspace(0.1, 0.7, len(samples_struct))),
                    )
                },
            )

            # vis.set_limits_from_label(union_mask, expansion=30)

            sim = {
                k: np.zeros((len(samples_struct), len(manual_observers_struct)))
                for k in metrics
            }
            msim = {
                k: np.zeros((len(samples_struct), len(manual_observers_struct)))
                for k in metrics
            }
            for sid, samp in enumerate(samples_struct):
                for oid, obs in enumerate(manual_observers_struct):
                    sample_metrics = get_metrics(
                        manual_observers_struct[obs], samples_struct[samp]
                    )
                    mean_metrics = get_metrics(
                        manual_observers_struct[obs], mean_contours[f"mean_{structure}"]
                    )

                    for k in sample_metrics:
                        sim[k][sid, oid] = sample_metrics[k]
                        msim[k][sid, oid] = mean_metrics[k]

            result[f"probnet_{structure}"] = {k: [] for k in metrics}
            result[f"unet_{structure}"] = {k: [] for k in metrics}
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
                result[f"probnet_{structure}"][k].append(prob_unet_mean)

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
                result[f"unet_{structure}"][k].append(unet_mean)

        fig = vis.show()

        return result, fig

    def training_step(self, batch, _):
        x, c, y, cy, m, _ = batch

        # Add background layer for one-hot encoding
        not_y = 1 - y.max(axis=1).values
        not_y = torch.unsqueeze(not_y, dim=1)
        y = torch.cat((not_y, y), dim=1).float()

        not_cy = 1 - cy.max(axis=1).values
        not_cy = torch.unsqueeze(not_cy, dim=1)
        cy = torch.cat((not_cy, cy), dim=1).float()

        # Concat context map to image if we have one
        if c.numel() > 0:
            x = torch.cat((x, c), dim=1).float()

        print(f"{y.shape} {cy.shape}")

        # self.prob_unet.forward(x, y, training=True)
        if self.hparams.prob_type == "prob":
            self.prob_unet.forward(x, y, cy, training=True)
        # else:
        #     self.prob_unet.forward(x, y)

        if self.hparams.prob_type == "prob":
            loss = self.prob_unet.loss(y, mask=m)
        # else:
        #     loss = self.prob_unet.loss(x, y, mask=m)

        training_loss = loss["loss"]

        # Using weight decay instead
        # if self.hparams.prob_type == "prob":
        #     reg_loss = (
        #         l2_regularisation(self.prob_unet.posterior)
        #         + l2_regularisation(self.prob_unet.prior)
        #         + l2_regularisation(self.prob_unet.fcomb.layers)
        #     )
        #     training_loss = training_loss + 1e-5 * reg_loss
        self.log(
            "training_loss",
            training_loss.detach(),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )

        self.kl_div = loss["kl_div"].detach().cpu()

        for k in loss:
            if k == "loss":
                continue
            self.log(
                k,
                loss[k].detach() if isinstance(loss[k], torch.Tensor) else loss[k],
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True,
            )
        return training_loss

    def validation_step(self, batch, _):
        if self.validation_directory is None:
            self.validation_directory = Path(tempfile.mkdtemp())

        n = self.hparams.num_observers
        m = self.hparams.num_observers

        with torch.set_grad_enabled(False):
            x, c, y, cy, _, info = batch

            # Save off slices/volumes for analysis of entire structure in end of validation step
            for s in range(y.shape[0]):
                img_file = self.validation_directory.joinpath(
                    f"img_{info['case'][s]}_{info['z'][s]}.npy"
                )
                np.save(img_file, x[s].squeeze(0).cpu().numpy())

                if c.numel() > 0:
                    cmap_file = self.validation_directory.joinpath(
                        f"cmap_{info['case'][s]}_{info['z'][s]}.npy"
                    )
                    np.save(cmap_file, c[s].squeeze(0).cpu().numpy())

                mask_file = self.validation_directory.joinpath(
                    f"mask_{info['case'][s]}_{info['z'][s]}_{info['observer'][s]}.npy"
                )
                np.save(mask_file, y[s].cpu().numpy())

            # Image (and context map) will be same for all in batch
            x = x[0].unsqueeze(0)
            if c.numel() > 0:
                c = c[0].unsqueeze(0)
            if self.hparams.ndims == 2:
                vis = ImageVisualiser(sitk.GetImageFromArray(x.to("cpu")[0]), axis="z")
            else:
                vis = ImageVisualiser(sitk.GetImageFromArray(x.to("cpu")[0, 0]))

            if self.hparams.ndims == 2:
                x = x.repeat(m, 1, 1, 1)

                if c.numel() > 0:
                    c = c.repeat(m, 1, 1, 1)
            else:
                x = x.repeat(m, 1, 1, 1, 1)

                if c.numel() > 0:
                    c = c.repeat(m, 1, 1, 1, 1)

            if c.numel() > 0:
                x = torch.cat((x, c), dim=1)

            seg = None
            if self.use_structure_context:
                not_y = 1 - y.max(axis=1).values
                not_y = torch.unsqueeze(not_y, dim=1)
                seg = torch.cat((not_y, y), dim=1).float()

            self.prob_unet.forward(x, cseg=seg)
            # loss = self.prob_unet.loss(seg)
            # print(f"VAL LOSS: {loss}")

            py = self.prob_unet.sample(testing=True)
            py = py.to("cpu")

            pred_y = torch.zeros(py[:, 0, :].shape).int()
            for b in range(py.shape[0]):
                pred_y[b] = py[b, :].argmax(0).int()

            y = y.squeeze(1)
            y = y.int()
            y = y.to("cpu")


            cy = cy.squeeze(1)
            cy = cy.int()
            cy = cy.to("cpu")

            # TODO Make this work for multi class
            # Intersection over Union (also known as Jaccard Index)
            jaccard = JaccardIndex(num_classes=2)
            term_1 = 0
            for i in range(n):
                for j in range(m):
                    if pred_y[i].sum() + y[j].sum() == 0:
                        continue
                    iou = jaccard(pred_y[i], y[j])
                    term_1 += 1 - iou
            term_1 = term_1 * (2 / (m * n))

            term_2 = 0
            for i in range(n):
                for j in range(n):
                    if pred_y[i].sum() + pred_y[j].sum() == 0:
                        continue
                    iou = jaccard(pred_y[i], pred_y[j])
                    term_2 += 1 - iou
            term_2 = term_2 * (1 / (n * n))

            term_3 = 0
            for i in range(m):
                for j in range(m):
                    if y[i].sum() + y[j].sum() == 0:
                        continue
                    iou = jaccard(y[i], y[j])
                    term_3 += 1 - iou
            term_3 = term_3 * (1 / (m * m))

            D_ged = term_1 - term_2 - term_3

            contours = {}
            contour_colors = {}
            for o in range(n):
                obs_y = y[o].float()
                if self.hparams.ndims == 2:
                    obs_y = obs_y.unsqueeze(0)
                contours[f"obs_{o}"] = sitk.GetImageFromArray(obs_y)
                contour_colors[f"obs_{o}"] = (0.3, 0.6, 0.3)
            for mm in range(m):
                samp_pred = pred_y[mm].float()
                if self.hparams.ndims == 2:
                    samp_pred = samp_pred.unsqueeze(0)
                contours[f"sample_{mm}"] = sitk.GetImageFromArray(samp_pred)
                contour_colors[f"sample_{mm}"] = (0.1, 0.1, 0.8)

            if self.use_structure_context:
                for o in range(n):
                    obs_y = cy[o].float()
                    if self.hparams.ndims == 2:
                        obs_y = obs_y.unsqueeze(0)
                    contours[f"compobs_{o}"] = sitk.GetImageFromArray(obs_y)
                    contour_colors[f"compobs_{o}"] = (0.6, 0.3, 0.3)

            vis.add_contour(contours, color=contour_colors)
            vis.show()

            figure_path = f"ged_{info['z'][s]}.png"
            plt.savefig(figure_path, dpi=300)
            plt.close("all")

            try:
                self.logger.experiment.log_image(figure_path)
            except AttributeError:
                # Likely offline mode
                pass

        self.log("GED", D_ged)

        return info

    def validation_epoch_end(self, validation_step_outputs):
        cases = {}
        for info in validation_step_outputs:
            for case, z, observer in zip(info["case"], info["z"], info["observer"]):
                if not case in cases:
                    cases[case] = {"slices": z.item(), "observers": [observer]}
                else:
                    if z.item() > cases[case]["slices"]:
                        cases[case]["slices"] = z.item()
                    if not observer in cases[case]["observers"]:
                        cases[case]["observers"].append(observer)

        metrics = ["DSC", "HD", "ASD"]
        computed_metrics = {
            **{
                f"probnet_{s}_{m}": [] for m in metrics for s in self.hparams.structures
            },
            **{f"unet_{s}_{m}": [] for m in metrics for s in self.hparams.structures},
        }

        if len(cases) == 0:
            return

        prob_surface_dice = 0
        prob_dice = 0

        for case in cases:
            img_arrs = []
            cmap_arrs = []
            cmap_arr = None
            slices = []

            if self.hparams.ndims == 2:
                for z in range(cases[case]["slices"] + 1):
                    img_file = self.validation_directory.joinpath(f"img_{case}_{z}.npy")
                    if img_file.exists():
                        img_arrs.append(np.load(img_file))
                        slices.append(z)

                    cmap_file = self.validation_directory.joinpath(
                        f"cmap_{case}_{z}.npy"
                    )
                    if cmap_file.exists():
                        cmap_arrs.append(np.load(cmap_file))

                img_arr = np.stack(img_arrs)

                if len(cmap_arrs) > 0:
                    cmap_arr = np.stack(cmap_arr)

            else:
                img_file = self.validation_directory.joinpath(f"img_{case}_0.npy")
                img_arr = np.load(img_file)

                cmap_file = self.validation_directory.joinpath(f"cmap_{case}_0.npy")
                if cmap_file.exists():
                    cmap_arr = np.load(cmap_file)

            img = sitk.GetImageFromArray(img_arr)
            img.SetSpacing(self.hparams.spacing)

            observers = {}
            for _, observer in enumerate(cases[case]["observers"]):
                if self.hparams.ndims == 2:
                    mask_arrs = []
                    for z in slices:
                        mask_file = self.validation_directory.joinpath(
                            f"mask_{case}_{z}_{observer}.npy"
                        )

                        mask_arrs.append(np.load(mask_file))

                    mask_arr = np.stack(mask_arrs, axis=1)

                else:
                    mask_file = self.validation_directory.joinpath(
                        f"mask_{case}_{z}_{observer}.npy"
                    )
                    mask_arr = np.load(mask_file)

                observers[f"manual_{observer}"] = {}
                for idx, structure in enumerate(self.hparams.structures):
                    mask = sitk.GetImageFromArray(mask_arr[idx])
                    mask = sitk.Cast(mask, sitk.sitkUInt8)
                    mask.CopyInformation(img)
                    observers[f"manual_{observer}"][structure] = mask

            context_map = None
            if cmap_arr is not None:
                context_map = sitk.GetImageFromArray(cmap_arr)
                context_map.SetSpacing(self.hparams.spacing)

            seg = None
            if self.use_structure_context:
                # Staple the man observers to pass in as context seg
                masks = []
                for man_obs in observers:
                    masks.append(observers[man_obs][structure])

                stapled = sitk.STAPLE(masks)
                stapled = stapled > 0.5
                stapled = sitk.Cast(stapled, sitk.sitkUInt8)
                seg = stapled

#            try:
            mean = self.infer(
                img,
                context_map=context_map,
                seg=seg,
                num_samples=1,
                sample_strategy="mean",
                preprocess=False,
            )
            samples = self.infer(
                img,
                context_map=context_map,
                seg=seg,
                sample_strategy="spaced",
                num_samples=11,
                spaced_range=[-2, 2],
                preprocess=False,
            )
#            except Exception as e:
#                print(f"ERROR DURING VALIDATION INFERENCE: {e}")
#                return


            # try:
            result, fig = self.validate(
                img, observers, samples, mean, matching_type="best"
            )
            # except Exception as e:
            #    print(f"ERROR DURING VALIDATION VALIDATE: {e}")
            #    return

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
                    computed_metrics[f"{t}_{m}"] += result[t][m]

            # Compute the probabilistic (surface) dice
            for idx, structure in enumerate(self.hparams.structures):
                gt_labels = []
                for _, observer in enumerate(cases[case]["observers"]):
                    gt_labels.append(observers[f"manual_{observer}"][structure])

                sample_labels = []
                for rk in samples:
                    sample_labels.append(samples[rk][structure])

                prob_dice += probabilistic_dice(
                    gt_labels, sample_labels, dsc_type="dsc"
                )
                prob_surface_dice += probabilistic_dice(
                    gt_labels, sample_labels, dsc_type="sdsc", tau=3
                )

        prob_dice = prob_dice / len(cases)
        if np.isnan(prob_dice):
            prob_dice = 0
        self.log(
            "probabilisticDice",
            prob_dice,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        prob_surface_dice = prob_surface_dice / len(cases)
        if np.isnan(prob_surface_dice):
            prob_surface_dice = 0
        self.log(
            "probabilisticSurfaceDice",
            prob_surface_dice,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        if self.kl_div:
            p = u = 0
            for s in self.hparams.structures:
                p += np.array(computed_metrics[f"probnet_{s}_DSC"]).mean()
                u += np.array(computed_metrics[f"unet_{s}_DSC"]).mean()

            p /= len(self.hparams.structures)
            u /= len(self.hparams.structures)
            computed_metrics["scaled_DSC"] = ((p + u) / 2) + (p - u) - self.kl_div

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
    # args.default_root_dir = str(args.working_dir)
    args.fold_dir = args.working_dir.joinpath(f"fold_{args.fold}")
    args.default_root_dir = str(args.fold_dir)
    args.accumulate_grad_batches = {0: 5, 5: 10, 10: 15}
    #    args.precision = 16

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
    if args.checkpoint_var:
        checkpoint_callback = ModelCheckpoint(
            monitor=args.checkpoint_var,
            dirpath=args.default_root_dir,
            filename="probunet-{epoch:02d}-{" + args.checkpoint_var + ":.2f}",
            save_top_k=1,
            mode=args.checkpoint_mode,
        )
        trainer.callbacks.append(checkpoint_callback)

    if args.early_stopping_var:
        early_stop_callback = GECOEarlyStopping(
            monitor=args.early_stopping_var,
            min_delta=args.early_stopping_min_delta,
            patience=args.early_stopping_patience,
            verbose=True,
            mode=args.early_stopping_mode,
        )
        trainer.callbacks.append(early_stop_callback)

    trainer.fit(prob_unet, data_module)


def parse_config_file(config_json_path, args):
    with open(config_json_path, "r") as f:
        params = json.load(f)
        for key in params:
            args.append(f"--{key}")

            if isinstance(params[key], list):
                for list_val in params[key]:
                    args.append(str(list_val))
            else:
                args.append(str(params[key]))

    return args


if __name__ == "__main__":
    args = None
    config_json_path = None
    if len(sys.argv) == 2:
        # Check if JSON file parsed, if so read arguments from there...
        if sys.argv[-1].endswith(".json"):
            config_json_path = sys.argv[-1]
            args = parse_config_file(config_json_path, [])

    arg_parser = ArgumentParser()
    arg_parser = ProbUNet.add_model_specific_args(arg_parser)
    arg_parser = UNetDataModule.add_model_specific_args(arg_parser)
    arg_parser = pl.Trainer.add_argparse_args(arg_parser)
    arg_parser.add_argument(
        "--config", type=str, default=None, help="JSON file with parameters to load"
    )
    arg_parser.add_argument(
        "--seed", type=int, default=42, help="an integer to use as seed"
    )
    arg_parser.add_argument(
        "--experiment", type=str, default="default", help="Name of experiment"
    )
    arg_parser.add_argument("--working_dir", type=str, default="./working")
    arg_parser.add_argument("--num_observers", type=int, default=5)
    arg_parser.add_argument("--spacing", nargs="+", type=float, default=[1, 1, 1])
    arg_parser.add_argument("--offline", type=bool, default=False)
    arg_parser.add_argument("--comet_api_key", type=str, default=None)
    arg_parser.add_argument("--comet_workspace", type=str, default=None)
    arg_parser.add_argument("--comet_project", type=str, default=None)
    arg_parser.add_argument("--resume_from", type=str, default=None)
    arg_parser.add_argument("--early_stopping_var", type=str, default=None)
    arg_parser.add_argument("--early_stopping_min_delta", type=float, default=0.01)
    arg_parser.add_argument("--early_stopping_patience", type=int, default=50)
    arg_parser.add_argument("--early_stopping_mode", type=str, default="max")
    arg_parser.add_argument("--checkpoint_var", type=str, default=None)
    arg_parser.add_argument("--checkpoint_mode", type=str, default="max")

    parsed_args = arg_parser.parse_args(args)

    # Check if config arg parsed, if so take over values and reparse
    if parsed_args.config:
        print("parseing args")
        args = parse_config_file(parsed_args.config, sys.argv[1:])
        parsed_args = arg_parser.parse_args(args)

    main(parsed_args)
