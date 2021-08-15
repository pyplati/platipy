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
import json

from pathlib import Path

import comet_ml  # pylint: disable=unused-import
from pytorch_lightning.loggers import CometLogger

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from argparse import ArgumentParser

from platipy.imaging.cnn.localise_net import LocaliseUNet
from platipy.imaging.cnn.dataload import UNetDataModule


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
                        for list_val in params[key]:
                            args.append(str(list_val))
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
