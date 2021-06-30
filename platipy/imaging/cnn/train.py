from pathlib import Path

import torch
import pytorch_lightning as pl

from argparse import ArgumentParser


from platipy.imaging.cnn.prob_unet import ProbabilisticUnet
from platipy.imaging.cnn.unet import l2_regularisation
from platipy.imaging.cnn.dataset import NiftiDataset


class ProbUNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.prob_unet = ProbabilisticUnet()

    def forward(self, x):
        self.prob_unet.forward(x, None, False)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5, weight_decay=0)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        self.prob_unet.forward(x, y, training=True)
        elbo = self.prob_unet.elbo(y, analytic_kl=True)
        reg_loss = (
            l2_regularisation(self.prob_unet.posterior)
            + l2_regularisation(self.prob_unet.prior)
            + l2_regularisation(self.prob_unet.fcomb.layers)
        )
        loss = -elbo["loss"] + 1e-5 * reg_loss

        print(loss)

        self.log("elbo_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "rec_loss", elbo["rec_loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "kl_loss", elbo["kl_div"], on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        # x, y = batch
        # y_hat = self(x)
        # val_loss = F.cross_entropy(y_hat, y)
        val_loss = 1
        return val_loss


class ProbUNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        working_dir: str = "./working",
        train_cases=[],
        validation_cases=[],
        test_cases=[],
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.working_dir = Path(working_dir)

        self.train_cases = train_cases
        self.validation_cases = validation_cases

    def prepare_data(self):
        pass

    def setup(self, stage=None):

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
        print(len(self.training_set))
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
            # batch_sampler=BatchSampler(
            #    ObserverSampler(train_set, 5), batch_size=params["batch_size"], drop_last=False
            # ),
            # num_workers=params["num_workers"],
            batch_size=5,
            shuffle=True,
            num_workers=4,
        )


def main(args):

    pl.seed_everything(args.seed, workers=True)

    data_module = ProbUNetDataModule(
        data_dir="./data",
        working_dir="./working",
        train_cases=[c for c in range(15)],
        validation_cases=[c for c in range(15, 20)],
    )

    prob_unet = ProbUNet()
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(prob_unet, data_module)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--seed", type=int, default=42, help="an integer to use as seed")
    args = parser.parse_args()

    main(args)
