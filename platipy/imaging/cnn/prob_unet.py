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
from torch.distributions import Normal, Independent, kl

import numpy as np

from platipy.imaging.cnn.unet import UNet, Conv, init_weights


class Encoder(torch.nn.Module):
    def __init__(self, input_channels, filters_per_layer=[64 * (2 ** x) for x in range(5)]):
        super(Encoder, self).__init__()

        layers = []
        for idx, layer_filters in enumerate(filters_per_layer):

            input_filters = input_channels if idx == 0 else output_filters
            output_filters = layer_filters

            down_sample = 0 if idx == 0 else -2

            layers.append(Conv(input_filters, output_filters, up_down_sample=down_sample))

        self.layers = torch.nn.Sequential(*layers)

        self.layers.apply(init_weights)

    def forward(self, x):

        return self.layers(x)


class AxisAlignedConvGaussian(torch.nn.Module):
    def __init__(
        self, input_channels, filters_per_layer=[64 * (2 ** x) for x in range(5)], latent_dim=2
    ):

        super(AxisAlignedConvGaussian, self).__init__()

        self.latent_dim = latent_dim

        self.encoder = Encoder(input_channels, filters_per_layer)
        self.final = torch.nn.Conv2d(filters_per_layer[-1], 2 * self.latent_dim, (1, 1), stride=1)

        self.final.apply(init_weights)

    def forward(self, img, seg=None):

        x = img
        if seg is not None:
            seg = torch.unsqueeze(seg, dim=1)
            x = torch.cat((img, seg), dim=1)

        encoding = self.encoder(x)

        # We only want the mean of the resulting hxw image
        encoding = torch.mean(encoding, dim=2, keepdim=True)
        encoding = torch.mean(encoding, dim=3, keepdim=True)

        # Convert encoding to 2 x latent dim and split up for mu and log_sigma
        mu_log_sigma = self.final(encoding)

        # We squeeze the second dimension twice, since otherwise it won't work when batch size is
        # equal to 1
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)

        mu = mu_log_sigma[:, : self.latent_dim]
        log_sigma = mu_log_sigma[:, self.latent_dim :]

        # This is a multivariate normal with diagonal covariance matrix sigma
        # https://github.com/pytorch/pytorch/pull/11178
        dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)), 1)
        return dist


class Fcomb(torch.nn.Module):
    """
    A function composed of no_convs_fcomb times a 1x1 convolution that combines the sample taken
    from the latent space, and output of the UNet (the feature map) by concatenating them along
    their channel axis.
    """

    def __init__(self, filters_per_layer, latent_dim, num_classes, no_convs_fcomb):
        super(Fcomb, self).__init__()

        layers = []

        # Decoder of N x a 1x1 convolution followed by a ReLU activation function except for the
        # last layer
        layers.append(
            torch.nn.Conv2d(filters_per_layer[0] + latent_dim, filters_per_layer[0], kernel_size=1)
        )
        layers.append(torch.nn.ReLU(inplace=True))

        for _ in range(no_convs_fcomb - 2):
            layers.append(
                torch.nn.Conv2d(filters_per_layer[0], filters_per_layer[0], kernel_size=1)
            )
            layers.append(torch.nn.ReLU(inplace=True))

        self.layers = torch.nn.Sequential(*layers)

        self.last_layer = torch.nn.Conv2d(filters_per_layer[0], num_classes, kernel_size=1)

        self.layers.apply(init_weights)
        self.last_layer.apply(init_weights)

    def forward(self, feature_map, z):

        z = torch.unsqueeze(z, 2).expand(-1, -1, feature_map.shape[2], -1)
        z = torch.unsqueeze(z, 3).expand(-1, -1, -1, feature_map.shape[3], -1)

        # Concatenate the feature map (output of the UNet) and the sample taken from the latent
        # space
        feature_map = torch.cat((feature_map, z), dim=1)
        output = self.layers(feature_map)
        return self.last_layer(output)


class ProbabilisticUnet(torch.nn.Module):
    """
    A probabilistic UNet implementation
    (https://papers.nips.cc/paper/2018/file/473447ac58e1cd7e96172575f48dca3b-Paper.pdf)

    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: is a list consisint of the amount of filters layer
    latent_dim: dimension of the latent space
    no_cons_per_block: no convs per block in the (convolutional) encoder of prior and posterior
    """

    def __init__(
        self,
        input_channels=1,
        num_classes=2,
        filters_per_layer=[64 * (2 ** x) for x in range(5)],
        latent_dim=6,
        no_convs_fcomb=4,
        beta=1.0,
    ):
        super(ProbabilisticUnet, self).__init__()

        self.no_convs_per_block = 3
        self.no_convs_fcomb = no_convs_fcomb
        self.initializers = {"w": "he_normal", "b": "normal"}
        self.beta = beta
        self.z_prior_sample = 0

        self.unet = UNet(input_channels, num_classes, filters_per_layer, final_layer=False)
        self.prior = AxisAlignedConvGaussian(input_channels, filters_per_layer, latent_dim)
        self.posterior = AxisAlignedConvGaussian(input_channels + 1, filters_per_layer, latent_dim)
        self.fcomb = Fcomb(filters_per_layer, latent_dim, num_classes, no_convs_fcomb)

        self.posterior_latent_space = None
        self.prior_latent_space = None
        self.unet_features = None

    def forward(self, img, seg, training=True):
        """
        Construct prior latent space for patch and run patch through UNet,
        in case training is True also construct posterior latent space
        """
        if training:
            self.posterior_latent_space = self.posterior.forward(img, seg=seg)
        self.prior_latent_space = self.prior.forward(img)
        self.unet_features = self.unet.forward(img)

    def sample(self, testing=False, use_mean=False, sample_x_stddev_from_mean=None):
        """
        Sample a segmentation by reconstructing from a prior sample
        and combining this with UNet features
        """

        if testing:
            if use_mean:
                z_prior = self.prior_latent_space.base_dist.loc
            elif not sample_x_stddev_from_mean is None:
                z_prior = self.prior_latent_space.base_dist.loc + (
                    self.prior_latent_space.base_dist.scale * sample_x_stddev_from_mean
                )
            else:
                z_prior = self.prior_latent_space.sample()
            self.z_prior_sample = z_prior
        else:
            z_prior = self.prior_latent_space.rsample()
            self.z_prior_sample = z_prior

        return self.fcomb.forward(self.unet_features, z_prior)

    def reconstruct(self, use_posterior_mean=False, z_posterior=None):
        """
        Reconstruct a segmentation from a posterior sample (decoding a posterior sample) and UNet
        feature map

        use_posterior_mean: use posterior_mean instead of sampling z_q
        """
        if use_posterior_mean:
            z_posterior = self.posterior_latent_space.mean
        else:
            if z_posterior is None:
                z_posterior = self.posterior_latent_space.rsample()
        return self.fcomb.forward(self.unet_features, z_posterior)

    def kl_divergence(self, analytic=True, z_posterior=None):
        """
        Calculate the KL divergence between the posterior and prior KL(Q||P)
        analytic: calculate KL analytically or via sampling from the posterior
        """
        if analytic:
            kl_div = kl.kl_divergence(self.posterior_latent_space, self.prior_latent_space)
        else:
            if z_posterior is None:
                z_posterior = self.posterior_latent_space.rsample()
            log_posterior_prob = self.posterior_latent_space.log_prob(z_posterior)
            log_prior_prob = self.prior_latent_space.log_prob(z_posterior)
            kl_div = log_posterior_prob - log_prior_prob
        return kl_div

    def elbo(self, segm, analytic_kl=True, reconstruct_posterior_mean=False):
        """
        Calculate the evidence lower bound of the log-likelihood of P(Y|X)
        """

        criterion = torch.nn.BCEWithLogitsLoss(size_average=False, reduce=False, reduction=None)
        z_posterior = self.posterior_latent_space.rsample()

        kl_div = torch.mean(self.kl_divergence(analytic=analytic_kl, z_posterior=z_posterior))

        # Here we use the posterior sample sampled above
        reconstruction = self.reconstruct(
            use_posterior_mean=reconstruct_posterior_mean, z_posterior=z_posterior
        )

        segm = torch.unsqueeze(segm, dim=1)
        not_seg = segm.logical_not()
        segm = torch.cat((not_seg, segm), dim=1).float()
        reconstruction_loss = criterion(input=reconstruction, target=segm)
        reconstruction_loss = torch.sum(reconstruction_loss)
        # mean_reconstruction_loss = torch.mean(reconstruction_loss)

        return -(reconstruction_loss + self.beta * kl_div)
