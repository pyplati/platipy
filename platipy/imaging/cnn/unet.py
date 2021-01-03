# Copyright 2020 University of New South Wales, University of Sydney, Ingham Institute

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Parts of this work are derived from:
# https://github.com/stefanknegt/Probabilistic-Unet-Pytorch
# which is released under the Apache Licence 2.0

# pylint: disable=invalid-name

import torch
from torch import nn


def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        truncated_normal_(m.bias, mean=0, std=0.001)


class Conv(torch.nn.Module):
    def __init__(self, input_channels, output_channels, up_down_sample=0):

        super(Conv, self).__init__()

        self.pre_op = None
        size_and_stride = abs(up_down_sample)
        if up_down_sample < 0:
            self.pre_op = nn.MaxPool2d(kernel_size=size_and_stride, stride=size_and_stride)
        elif up_down_sample > 0:
            self.pre_op = nn.ConvTranspose2d(
                input_channels,
                output_channels,
                kernel_size=size_and_stride,
                stride=size_and_stride,
            )

        layers = []
        layers.append(
            nn.Conv2d(
                in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1
            )
        )
        layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(
                in_channels=output_channels, out_channels=output_channels, kernel_size=3, padding=1
            )
        )
        layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

        self.layers.apply(init_weights)

    def forward(self, x, concat=None):

        if not self.pre_op is None:
            x = self.pre_op(x)

        if not concat is None:
            x = torch.cat([x, concat], 1)

        return self.layers(x)


class UNet(nn.Module):
    def __init__(
        self,
        input_channels=1,
        output_classes=2,
        filters_per_layer=[64 * (2 ** x) for x in range(5)],
        final_layer=True,
    ):

        super(UNet, self).__init__()

        self.encoder = nn.ModuleList()
        for idx, layer_filters in enumerate(filters_per_layer):
            input_filters = input_channels if idx == 0 else output_filters
            output_filters = layer_filters
            down_sample = 0 if idx == 0 else -2

            self.encoder.append(Conv(input_filters, output_filters, up_down_sample=down_sample))

        reversed_filters = list(reversed(filters_per_layer))
        self.decoder = nn.ModuleList()
        for idx, layer_filters in enumerate(reversed_filters):

            if idx == len(reversed_filters) - 1:
                continue

            input_filters = layer_filters
            output_filters = reversed_filters[idx + 1]

            self.decoder.append(Conv(input_filters, output_filters, up_down_sample=2))

        self.final = None
        if final_layer:
            self.final = nn.Conv2d(filters_per_layer[0], output_classes, kernel_size=1)

    def forward(self, x):

        blocks = []
        for idx, enc in enumerate(self.encoder):
            x = enc(x)
            if idx != len(self.encoder) - 1:
                blocks.append(x)

        for idx, dec in enumerate(self.decoder):
            x = dec(x, concat=blocks[-idx - 1])

        if self.final:
            return self.final(x)

        return x
