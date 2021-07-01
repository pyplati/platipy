import os
import math

from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

from comet_ml import Experiment
from pytorch_lightning.loggers import CometLogger

import torch
import pytorch_lightning as pl

from argparse import ArgumentParser

from platipy.imaging.cnn.prob_unet import ProbabilisticUnet
from platipy.imaging.cnn.unet import l2_regularisation
from platipy.imaging.cnn.dataset import NiftiDataset
from platipy.imaging.cnn.sampler import ObserverSampler


class ProbUNet(pl.LightningModule):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        loss_params = {"beta": self.hparams.beta}

        if self.hparams.loss_type == "geco":
            loss_params = {"kappa": self.hparams.kappa, "clamp_rec": self.hparams.clamp_rec}

        self.prob_unet = ProbabilisticUnet(
            self.hparams.input_channels,
            self.hparams.num_classes,
            self.hparams.filters_per_layer,
            self.hparams.latent_dim,
            self.hparams.no_convs_fcomb,
            self.hparams.loss_type,
            loss_params,
        )

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
        return parent_parser

    def forward(self, x):
        self.prob_unet.forward(x, None, False)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=0
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        self.prob_unet.forward(x, y, training=True)
        loss = self.prob_unet.loss(y, analytic_kl=True)
        reg_loss = (
            l2_regularisation(self.prob_unet.posterior)
            + l2_regularisation(self.prob_unet.prior)
            + l2_regularisation(self.prob_unet.fcomb.layers)
        )
        training_loss = loss["loss"] + 1e-5 * reg_loss
        self.log(
            "training_loss", training_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        for k in loss:
            self.log(
                k,
                loss[k],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.set_grad_enabled(False):
            x, y = batch
            self.prob_unet.forward(x)
            sample = self.prob_unet.sample(testing=True)
            mean = self.prob_unet.sample(testing=True, use_mean=True)

            criterion = torch.nn.BCEWithLogitsLoss(
                size_average=False, reduce=False, reduction=None
            )

            onehot = torch.nn.functional.one_hot(y, 2).transpose(1, 3).float()

            observers = y.shape[0]
            sim_matrix = np.zeros((observers, observers))
            for i in range(observers):
                for j in range(observers):
                    rec_loss = criterion(input=sample[i], target=onehot[j])
                    rec_loss = torch.sum(rec_loss)
                    sim_matrix[i, j] = rec_loss.item()

            row_idx, col_idx = linear_sum_assignment(sim_matrix)

            matched_val = sim_matrix[row_idx, col_idx].mean()
            self.log(
                "matched_val", matched_val, on_step=True, on_epoch=True, prog_bar=True, logger=True
            )

            mean_val = criterion(input=mean, target=onehot)
            mean_val = torch.sum(rec_loss).item()
            self.log("mean_val", mean_val, on_step=True, on_epoch=True, prog_bar=True, logger=True)

            return matched_val


class ProbUNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        working_dir: str = "./working",
        fold=0,
        k_folds=5,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.working_dir = Path(working_dir)

        self.fold = fold
        self.k_folds = k_folds

        self.train_cases = []
        self.validation_cases = []

        print(f"Training fold {self.fold}")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data Loader")
        parser.add_argument("--data_dir", type=str, default="./data")
        parser.add_argument("--fold", type=int, default=0)
        parser.add_argument("--k_folds", type=int, default=5)

        return parent_parser

    def setup(self, stage=None):

        cases = [p.name.replace(".nii.gz", "") for p in self.data_dir.joinpath("images").glob("*")]
        cases.sort()
        cases_per_fold = math.ceil(len(cases) / self.k_folds)
        for f in range(self.k_folds):

            if self.fold == f:
                self.validation_cases = cases[f * cases_per_fold : (f + 1) * cases_per_fold]
            else:
                self.train_cases += cases[f * cases_per_fold : (f + 1) * cases_per_fold]

        train_data = [
            {
                "id": case,
                "image": self.data_dir.joinpath("images", f"{case}.nii.gz"),
                "label": [p for p in self.data_dir.joinpath("labels").glob(f"{case}_*.nii.gz")],
            }
            for case in self.train_cases
        ]

        validation_data = [
            {
                "id": case,
                "image": self.data_dir.joinpath("images", f"{case}.nii.gz"),
                "label": [p for p in self.data_dir.joinpath("labels").glob(f"{case}_*.nii.gz")],
            }
            for case in self.validation_cases
        ]

        self.training_set = NiftiDataset(train_data, self.working_dir.joinpath("train"))
        self.validation_set = NiftiDataset(
            validation_data, self.working_dir.joinpath("validation")
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.training_set,
            # batch_sampler=BatchSampler(
            #    ObserverSampler(train_set, 5), batch_size=params["batch_size"], drop_last=False
            # ),
            # num_workers=params["num_workers"],
            batch_size=5,
            shuffle=True,
            num_workers=4,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.validation_set,
            batch_sampler=torch.utils.data.BatchSampler(
                ObserverSampler(self.validation_set, 5), batch_size=5, drop_last=False
            ),
            num_workers=4,
        )


def main(args):

    pl.seed_everything(args.seed, workers=True)

    args.working_dir = Path(args.working_dir)
    args.working_dir = args.working_dir.joinpath(args.experiment)

    comet_logger = CometLogger(
        api_key=os.environ["COMET_API_KEY"],
        workspace=os.environ["COMET_WORKSPACE"],
        project_name=os.environ["COMET_PROJECT"],
        experiment_name=args.experiment,
        save_dir=args.working_dir,
    )

    dict_args = vars(args)

    data_module = ProbUNetDataModule(**dict_args)

    prob_unet = ProbUNet(**dict_args)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.logger = comet_logger
    trainer.fit(prob_unet, data_module)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = ProbUNet.add_model_specific_args(parser)
    parser = ProbUNetDataModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--seed", type=int, default=42, help="an integer to use as seed")
    parser.add_argument("--experiment", type=str, default="default", help="Name of experiment")
    parser.add_argument("--working_dir", type=str, default="./working")
    args = parser.parse_args()

    main(args)
