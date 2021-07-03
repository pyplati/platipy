import os
import math
import tempfile
import shutil

from pathlib import Path
import SimpleITK as sitk
import numpy as np
from scipy.optimize import linear_sum_assignment

import comet_ml
from pytorch_lightning.loggers import CometLogger

import torch
import pytorch_lightning as pl

from argparse import ArgumentParser

from torch._C import NoneType

from platipy.imaging.cnn.prob_unet import ProbabilisticUnet
from platipy.imaging.cnn.unet import l2_regularisation
from platipy.imaging.cnn.dataset import NiftiDataset
from platipy.imaging.cnn.sampler import ObserverSampler

from platipy.imaging import ImageVisualiser
from platipy.imaging.label.utils import get_com


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

        self.validation_directory = None

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
                sample = self.prob_unet.sample(testing=True)
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

        print(self.validation_directory)

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
            sitk.WriteImage(img, f"test_{case}.nii.gz")

            mean_arr = np.stack(mean_arrs)
            mean = sitk.GetImageFromArray(mean_arr)
            mean = sitk.Cast(mean, sitk.sitkUInt8)
            sitk.WriteImage(mean, f"test_mean_{case}_{observer}.nii.gz")

            obs_dict = {}
            pred_dict = {}
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
                sitk.WriteImage(mask, f"test_mask_{case}_{observer}.nii.gz")
                obs_dict[f"manual_{observer}"] = mask

                sample_arr = np.stack(sample_arrs)
                sample = sitk.GetImageFromArray(sample_arr)
                sample = sitk.Cast(sample, sitk.sitkUInt8)
                sitk.WriteImage(sample, f"test_sample_{case}_{observer}.nii.gz")
                pred_dict[f"auto_{observer}"] = sample

            img_vis = ImageVisualiser(
                img, cut=get_com(mask), figure_size_in=16, window=[img_arr.min(), img_arr.max()]
            )

            # color_dict = {str(i): [0.5, 0.5, 0.5] for i, m in enumerate(observers)}
            contour_dict = {**obs_dict, **pred_dict}

            img_vis.add_contour(contour_dict)  # , color=color_dict)
            fig = img_vis.show()
            figure_path = f"valid_{case}.png"
            fig.savefig(figure_path, dpi=300)

            self.logger.experiment.log_image(figure_path)

        # for pred in validation_step_outputs:
        #     # do something with a pred
        #     print(pred)

        #     break

        # criterion = torch.nn.BCEWithLogitsLoss(
        #     size_average=False, reduce=False, reduction=None
        # )

        # onehot = torch.nn.functional.one_hot(y, 2).transpose(1, 3).float()

        # observers = y.shape[0]
        # sim_matrix = np.zeros((observers, observers))
        # for i in range(observers):
        #     for j in range(observers):
        #         rec_loss = criterion(input=sample[i], target=onehot[j])
        #         rec_loss = torch.sum(rec_loss)
        #         sim_matrix[i, j] = rec_loss.item()

        # row_idx, col_idx = linear_sum_assignment(sim_matrix)

        # matched_val = sim_matrix[row_idx, col_idx].mean()
        # self.log(
        #     "matched_val",
        #     matched_val,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=False,
        #     logger=True,
        # )

        # mean_val = criterion(input=mean, target=onehot)
        # mean_val = torch.sum(rec_loss).item()
        # self.log(
        #     "mean_val", mean_val, on_step=False, on_epoch=True, prog_bar=False, logger=True
        # )

        # shutil.rmtree(self.validation_directory)


class ProbUNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        working_dir: str = "./working",
        fold=0,
        k_folds=5,
        batch_size=5,
        num_workers=4,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.working_dir = Path(working_dir)

        self.fold = fold
        self.k_folds = k_folds

        self.train_cases = []
        self.validation_cases = []

        self.batch_size = batch_size
        self.num_workers = num_workers

        print(f"Training fold {self.fold}")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data Loader")
        parser.add_argument("--data_dir", type=str, default="./data")
        parser.add_argument("--fold", type=int, default=0)
        parser.add_argument("--k_folds", type=int, default=5)
        parser.add_argument("--batch_size", type=int, default=5)
        parser.add_argument("--num_workers", type=int, default=4)

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
            offline=True,
        )

    dict_args = vars(args)

    data_module = ProbUNetDataModule(**dict_args)

    prob_unet = ProbUNet(**dict_args)
    trainer = pl.Trainer.from_argparse_args(args)

    if comet_api_key is not None:
        trainer.logger = comet_logger

    trainer.fit(prob_unet, data_module)  # pylint: disable=no-member


if __name__ == "__main__":
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

    main(arg_parser.parse_args())
