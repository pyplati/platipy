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


def conv_nd(ndims=2, **kwargs):
    """Generate a 2D or 3D convolution

    Args:
        ndims (int, optional): 2 or 3 dimensions. Defaults to 2.

    Raises:
        NotImplementedError: Raised if ndims is not in 2 or 3 dimensions.

    Returns:
        torch.nn.Conv: The convolution.
    """

    if ndims == 2:
        return torch.nn.Conv2d(**kwargs)
    elif ndims == 3:
        return torch.nn.Conv3d(**kwargs)

    raise NotImplementedError("Only 2 or 3 dimensions are supported")


def dropout_nd(ndims=2, **kwargs):
    """Get a 2D or 3D dropout layer

    Args:
        ndims (int, optional): 2 or 3 dimensions. Defaults to 2.

    Raises:
        NotImplementedError: Raised if ndims is not in 2 or 3 dimensions.

    Returns:
        torch.nn.Dropout: The dropout layer
    """

    if ndims == 2:
        return torch.nn.Dropout2d(**kwargs)
    elif ndims == 3:
        return torch.nn.Dropout3d(**kwargs)

    raise NotImplementedError("Only 2 or 3 dimensions are supported")


def init_weights(m):
    if (
        isinstance(m, torch.nn.Conv2d)
        or isinstance(m, torch.nn.ConvTranspose2d)
        or isinstance(m, torch.nn.Conv3d)
        or isinstance(m, torch.nn.ConvTranspose3d)
    ):
        torch.nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        truncated_normal_(m.bias, mean=0, std=0.001)

def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg


def init_zeros(m):
    if (
        isinstance(m, torch.nn.Conv2d)
        or isinstance(m, torch.nn.ConvTranspose2d)
        or isinstance(m, torch.nn.Conv3d)
        or isinstance(m, torch.nn.ConvTranspose3d)
    ):
        torch.nn.init.zeros_(m.weight)
        truncated_normal_(m.bias, mean=0, std=0.1)


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


def resize_down_func(scale=2, ndims=2):
    """Returns function to resize the input to downsample

    Args:
        scale (int, optional): The scale used to downsize. Defaults to 2.
        ndims (int, optional): Number of dimensions (2 or 3). Defaults to 2.

    Returns:
        function: The downsize function
    """
    if ndims == 3:
        return torch.nn.MaxPool3d(kernel_size=scale, stride=scale, padding=0)
    elif ndims == 2:
        return torch.nn.MaxPool2d(kernel_size=scale, stride=scale, padding=0)

    raise NotImplementedError()


def resize_up_func(in_channels, out_channels, scale=2, ndims=2):
    """Return function to resize the input to upsample

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        scale (int, optional): The scale used to upsize. Defaults to 2.
        ndims (int, optional): Number of dimensions (2 or 3). Defaults to 2.

    Raises:
        NotImplementedError: Only supports 2d or 3d

    Returns:
        function: The upsize function
    """
    if ndims == 3:
        return torch.nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=scale,
            stride=scale,
        )
    elif ndims == 2:
        return torch.nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=scale,
            stride=scale,
        )
    raise NotImplementedError()


class Conv(torch.nn.Module):
    def __init__(
        self, input_channels, output_channels, up_down_sample=0, dropout_probability=None, ndims=2
    ):

        super(Conv, self).__init__()

        self.pre_op = None
        size_and_stride = abs(up_down_sample)
        if up_down_sample < 0:
            self.pre_op = resize_down_func(size_and_stride, ndims=ndims)
        elif up_down_sample > 0:
            self.pre_op = resize_up_func(
                input_channels, output_channels, size_and_stride, ndims=ndims
            )

        layers = []
        layers.append(
            conv_nd(
                ndims=ndims,
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=3,
                padding=1,
            )
        )
        layers.append(nn.ReLU(inplace=True))
        if dropout_probability:
            layers.append(dropout_nd(ndims=ndims, p=dropout_probability))
        layers.append(
            conv_nd(
                ndims=ndims,
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=3,
                padding=1,
            )
        )
        if dropout_probability:
            layers.append(dropout_nd(ndims=ndims, p=dropout_probability))
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
        ndims=2,
        dropout_probability=None
    ):

        super(UNet, self).__init__()

        self.encoder = nn.ModuleList()
        for idx, layer_filters in enumerate(filters_per_layer):
            input_filters = input_channels if idx == 0 else output_filters
            output_filters = layer_filters
            down_sample = 0 if idx == 0 else -2

            self.encoder.append(
                Conv(input_filters, output_filters, up_down_sample=down_sample, dropout_probability=dropout_probability, ndims=ndims)
            )

        reversed_filters = list(reversed(filters_per_layer))
        self.decoder = nn.ModuleList()
        for idx, layer_filters in enumerate(reversed_filters):

            if idx == len(reversed_filters) - 1:
                continue

            input_filters = layer_filters
            output_filters = reversed_filters[idx + 1]

            self.decoder.append(Conv(input_filters, output_filters, up_down_sample=2, dropout_probability=dropout_probability, ndims=ndims))

        self.final = None
        if final_layer:
            self.final = conv_nd(
                ndims=ndims,
                in_channels=filters_per_layer[0],
                out_channels=output_classes,
                kernel_size=1,
            )

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
