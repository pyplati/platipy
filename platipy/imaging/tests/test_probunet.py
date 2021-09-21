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

# pylint: disable=redefined-outer-name,missing-function-docstring

from argparse import ArgumentParser

import pytest

import pytorch_lightning as pl
from platipy.imaging.cnn.train import main, ProbUNet, UNetDataModule
from platipy.imaging.cnn.pseudo_generator import generate_pseudo_data


@pytest.fixture
def trainer_arg_parser():

    generate_pseudo_data()

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
    return arg_parser


def test_prob_unet_2d_elbo(trainer_arg_parser):

    args = trainer_arg_parser.parse_args(
        [
            "--working_dir",
            "test_prob_unet_2d_elbo",
            "--num_workers",
            "1",
            "--limit_train_batches",
            "0.01",
            "--loss_type",
            "elbo",
            "--prob_type",
            "prob",
            "--max_epochs",
            "1",
            "--ndims",
            "2",
            "--filters_per_layer",
            "2",
            "4",
        ]
    )

    main(args)


def test_prob_unet_3d_elbo(trainer_arg_parser):

    args = trainer_arg_parser.parse_args(
        [
            "--working_dir",
            "test_prob_unet_3d_elbo",
            "--num_workers",
            "1",
            "--limit_train_batches",
            "0.05",
            "--loss_type",
            "elbo",
            "--prob_type",
            "prob",
            "--max_epochs",
            "1",
            "--ndims",
            "3",
            "--filters_per_layer",
            "2",
            "4",
            "--batch_size",
            "1",
        ]
    )

    main(args)


def test_prob_unet_2d_geco(trainer_arg_parser):

    args = trainer_arg_parser.parse_args(
        [
            "--working_dir",
            "test_prob_unet_2d_geco",
            "--num_workers",
            "1",
            "--limit_train_batches",
            "0.01",
            "--loss_type",
            "geco",
            "--prob_type",
            "prob",
            "--max_epochs",
            "1",
            "--ndims",
            "2",
            "--filters_per_layer",
            "2",
            "4",
        ]
    )

    main(args)


def test_prob_unet_3d_geco(trainer_arg_parser):

    args = trainer_arg_parser.parse_args(
        [
            "--working_dir",
            "test_prob_unet_3d_geco",
            "--num_workers",
            "1",
            "--limit_train_batches",
            "0.05",
            "--loss_type",
            "geco",
            "--prob_type",
            "prob",
            "--max_epochs",
            "1",
            "--ndims",
            "3",
            "--filters_per_layer",
            "2",
            "4",
            "--batch_size",
            "1",
        ]
    )

    main(args)


def test_hierarchical_prob_unet_2d_geco(trainer_arg_parser):

    args = trainer_arg_parser.parse_args(
        [
            "--working_dir",
            "test_prob_unet_2d_geco",
            "--num_workers",
            "1",
            "--limit_train_batches",
            "0.01",
            "--loss_type",
            "geco",
            "--prob_type",
            "hierarchical",
            "--max_epochs",
            "1",
            "--ndims",
            "2",
            "--filters_per_layer",
            "2",
            "4",
        ]
    )

    main(args)
