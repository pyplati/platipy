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

import torch
from torch.distributions import Normal, Independent, kl

from platipy.imaging.cnn.unet import UNet, Conv, init_weights, conv_nd


class Encoder(torch.nn.Module):
    """Encoder part of the probabilistic UNet"""

    def __init__(
        self, input_channels, filters_per_layer=[64 * (2 ** x) for x in range(5)], ndims=2
    ):
        super(Encoder, self).__init__()

        layers = []
        for idx, layer_filters in enumerate(filters_per_layer):

            input_filters = input_channels if idx == 0 else output_filters
            output_filters = layer_filters

            down_sample = 0 if idx == 0 else -2

            layers.append(
                Conv(input_filters, output_filters, up_down_sample=down_sample, ndims=ndims)
            )

        self.layers = torch.nn.Sequential(*layers)

        self.layers.apply(init_weights)

    def forward(self, x):

        return self.layers(x)


class AxisAlignedConvGaussian(torch.nn.Module):
    def __init__(
        self,
        input_channels,
        filters_per_layer=[64 * (2 ** x) for x in range(5)],
        latent_dim=2,
        ndims=2,
    ):

        super(AxisAlignedConvGaussian, self).__init__()

        self.latent_dim = latent_dim

        self.encoder = Encoder(input_channels, filters_per_layer, ndims=ndims)

        self.final = conv_nd(
            in_channels=filters_per_layer[-1],
            out_channels=2 * self.latent_dim,
            kernel_size=1,
            stride=1,
            ndims=ndims,
        )

        self.ndims = ndims

        self.final.apply(init_weights)

    def forward(self, img, seg=None):
        """Forward pass through the network

        Args:
            img (torch.Tensor): The image to be passed through.
            seg (torch.Tensor, optional): The segmentation mask to use in the case of the prior
                network. Defaults to None.

        Returns:
            torch.distributions.distribution.Distribution: The distribution output
        """

        x = img
        if seg is not None:
            # seg = torch.unsqueeze(seg, dim=1)
            x = torch.cat((img, seg), dim=1)

        encoding = self.encoder(x)

        # We only want the mean of the resulting hxw image
        encoding = torch.mean(encoding, dim=2, keepdim=True)
        encoding = torch.mean(encoding, dim=3, keepdim=True)
        if self.ndims == 3:
            encoding = torch.mean(encoding, dim=4, keepdim=True)

        # Convert encoding to 2 x latent dim and split up for mu and log_sigma
        mu_log_sigma = self.final(encoding)

        # We squeeze the second dimension twice, since otherwise it won't work when batch size is
        # equal to 1
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        if self.ndims == 3:
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

    def __init__(self, filters_per_layer, latent_dim, num_classes, no_convs_fcomb, ndims=2):
        super(Fcomb, self).__init__()

        layers = []

        # Decoder of N x a 1x1 convolution followed by a ReLU activation function except for the
        # last layer
        layers.append(
            conv_nd(
                in_channels=filters_per_layer[0] + latent_dim,
                out_channels=filters_per_layer[0],
                kernel_size=1,
                ndims=ndims,
            )
        )
        layers.append(torch.nn.ReLU(inplace=True))

        for _ in range(no_convs_fcomb - 2):
            layers.append(
                conv_nd(
                    in_channels=filters_per_layer[0],
                    out_channels=filters_per_layer[0],
                    kernel_size=1,
                    ndims=ndims,
                )
            )
            layers.append(torch.nn.ReLU(inplace=True))

        self.layers = torch.nn.Sequential(*layers)

        self.last_layer = conv_nd(
            in_channels=filters_per_layer[0], out_channels=num_classes, kernel_size=1, ndims=ndims
        )

        self.layers.apply(init_weights)
        self.last_layer.apply(init_weights)

        self.ndims = ndims

    def forward(self, feature_map, z):

        z = torch.unsqueeze(z, 2).expand(-1, -1, feature_map.shape[2])
        z = torch.unsqueeze(z, 3).expand(-1, -1, -1, feature_map.shape[3])
        if self.ndims == 3:
            z = torch.unsqueeze(z, 4).expand(-1, -1, -1, -1, feature_map.shape[4])

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
        loss_type="elbo",
        loss_params={"beta": 1},
        ndims=2,
    ):
        super(ProbabilisticUnet, self).__init__()

        self.num_classes = num_classes
        self.no_convs_per_block = 3
        self.no_convs_fcomb = no_convs_fcomb
        self.initializers = {"w": "he_normal", "b": "normal"}
        self.z_prior_sample = 0

        self.unet = UNet(
            input_channels, num_classes, filters_per_layer, final_layer=False, ndims=ndims
        )
        self.prior = AxisAlignedConvGaussian(
            input_channels, filters_per_layer, latent_dim, ndims=ndims
        )
        self.posterior = AxisAlignedConvGaussian(
            input_channels + num_classes, filters_per_layer, latent_dim, ndims=ndims
        )
        self.fcomb = Fcomb(filters_per_layer, latent_dim, num_classes, no_convs_fcomb, ndims=ndims)

        self.loss_type = loss_type
        self.loss_params = loss_params

        self.posterior_latent_space = None
        self.prior_latent_space = None
        self.unet_features = None

        if self.loss_type == "geco":
            self._rec_moving_avg = None
            self._contour_moving_avg = None
            self.register_buffer("_lambda", torch.zeros(2, requires_grad=False))

    def forward(self, img, seg=None, training=False):
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
                if isinstance(sample_x_stddev_from_mean, list):
                    sample_x_stddev_from_mean = torch.Tensor(sample_x_stddev_from_mean)
                    sample_x_stddev_from_mean = sample_x_stddev_from_mean.to(
                        self.prior_latent_space.base_dist.stddev.device
                    )
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

    def kl_divergence(self):
        """
        Calculate the KL divergence between the posterior and prior KL(Q||P)
        """

        kl_div = kl.kl_divergence(self.posterior_latent_space, self.prior_latent_space)

        return kl_div

    def topk_mask(self, score, k):
        """Returns a mask for the top-k elements in score."""

        values, _ = torch.topk(score, 1, axis=1)
        _, indices = torch.topk(values, k, axis=0)
        return torch.scatter_add(
            torch.zeros(score.shape[0]).to(score.device),
            0,
            indices.reshape(-1),
            torch.ones(score.shape[0]).to(score.device),
        )

    def prepare_mask(
        self,
        mask,
        top_k_percentage,
        deterministic,
        num_classes,
        device,
        batch_size,
        n_pixels_in_batch,
        xe,
    ):
        if mask is None or mask.sum() == 0:
            mask = torch.ones(n_pixels_in_batch)
        else:
            #            assert (
            #                mask.shape == segm.shape
            #            ), f"The loss mask shape differs from the target shape: {mask.shape} vs. {segm.shape}."
            mask = torch.reshape(mask, (-1,))
        mask = mask.to(device)

        if top_k_percentage is not None:

            assert 0.0 < top_k_percentage <= 1.0
            k_pixels = int(n_pixels_in_batch * top_k_percentage)

            with torch.no_grad():
                norm_xe = xe / torch.sum(xe)
                if deterministic:
                    score = torch.log(norm_xe)
                else:
                    # TODO Gumbel trick
                    raise NotImplementedError("Still need to implement Gumbel trick")

                score = score + torch.log(mask.unsqueeze(1).repeat((1, num_classes)))

                top_k_mask = self.topk_mask(score, k_pixels)
                top_k_mask = top_k_mask.to(device)
                mask = mask * top_k_mask

        mask = mask.unsqueeze(1).repeat((1, num_classes))
        mask = (
            mask.reshape((batch_size, -1, num_classes)).transpose(-1, 1).reshape((batch_size, -1))
        )

        return mask

    def reconstruction_loss(
        self,
        segm,
        z_posterior=None,
        mask=None,
        top_k_percentage=None,
        deterministic=True,
    ):

        criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

        if z_posterior is None:
            z_posterior = self.posterior_latent_space.rsample()

        reconstruction = self.reconstruct(use_posterior_mean=False, z_posterior=z_posterior)

        #####
        num_classes = reconstruction.shape[1]
        y_flat = torch.transpose(reconstruction, 1, -1).reshape((-1, num_classes))
        t_flat = torch.transpose(segm, 1, -1).reshape((-1, num_classes))
        n_pixels_in_batch = y_flat.shape[0]
        batch_size = segm.shape[0]

        xe = criterion(input=y_flat, target=t_flat)
        xe = xe.reshape((batch_size, -1, num_classes)).transpose(-1, 1).reshape((batch_size, -1))

        # If multiple masks supplied, compute a loss for each mask
        if hasattr(mask, "__iter__"):
            ce_sums = []
            ce_means = []
            masks = []
            for this_mask in mask:
                this_mask = self.prepare_mask(
                    this_mask,
                    top_k_percentage,
                    deterministic,
                    num_classes,
                    y_flat.device,
                    batch_size,
                    n_pixels_in_batch,
                    xe,
                )

                ce_sum_per_instance = torch.sum(this_mask * xe, axis=1)
                ce_sums.append(torch.mean(ce_sum_per_instance, axis=0))
                ce_means.append(torch.sum(this_mask * xe) / torch.sum(this_mask))
                masks.append(this_mask)

            return ce_sums, ce_means, masks

        mask = self.prepare_mask(
            mask,
            top_k_percentage,
            deterministic,
            num_classes,
            y_flat.device,
            batch_size,
            n_pixels_in_batch,
            xe,
        )

        ce_sum_per_instance = torch.sum(mask * xe, axis=1)
        ce_sum = torch.mean(ce_sum_per_instance, axis=0)
        ce_mean = torch.sum(mask * xe) / torch.sum(mask)

        return ce_sum, ce_mean, mask

    def loss(self, segm, mask=None):
        """
        Calculate the evidence lower bound of the log-likelihood of P(Y|X)
        """

        z_posterior = self.posterior_latent_space.rsample()

        kl_div = torch.mean(self.kl_divergence())
        kl_div = torch.clamp(kl_div,0.0,100.0)

        top_k_percentage = None
        if "top_k_percentage" in self.loss_params:
            top_k_percentage = self.loss_params["top_k_percentage"]

        loss_mask = None
        contour_threshold = None
        if self.loss_type == "geco":
            reconstruction_threshold = self.loss_params["kappa"]
            if (
                "kappa_contour" in self.loss_params
                and self.loss_params["kappa_contour"] is not None
            ):
                loss_mask = [None, mask]
                contour_threshold = self.loss_params["kappa_contour"]

        # Here we use the posterior sample sampled above
        _, rec_loss_mean, _ = self.reconstruction_loss(
            segm,
            z_posterior=z_posterior,
            top_k_percentage=top_k_percentage,
            mask=loss_mask,
        )

        # If using contour mask in loss, we get back those in a list. Unpack here.
        if contour_threshold:
            contour_loss = rec_loss_mean[1]
            contour_loss_mean = rec_loss_mean[1]
            reconstruction_loss = rec_loss_mean[0]
            rec_loss_mean = rec_loss_mean[0]
        else:
            reconstruction_loss = rec_loss_mean

        if self.loss_type == "elbo":

            return {
                "loss": reconstruction_loss + self.loss_params["beta"] * kl_div,
                "rec_loss": reconstruction_loss,
                "kl_div": kl_div,
            }
        elif self.loss_type == "geco":

            with torch.no_grad():

                moving_avg_factor = 0.8

                rl = rec_loss_mean.detach()
                if self._rec_moving_avg is None:
                    self._rec_moving_avg = rl
                else:
                    self._rec_moving_avg = self._rec_moving_avg * moving_avg_factor + rl * (
                        1 - moving_avg_factor
                    )

                rc = self._rec_moving_avg - reconstruction_threshold

                cc = 0
                if contour_threshold:
                    cl = contour_loss_mean.detach()
                    if self._contour_moving_avg is None:
                        self._contour_moving_avg = rl
                    else:
                        self._contour_moving_avg = (
                            self._contour_moving_avg * moving_avg_factor
                            + cl * (1 - moving_avg_factor)
                        )

                    cc = self._contour_moving_avg - contour_threshold

                lambda_lower = self.loss_params["clamp_rec"][0]
                lambda_upper = self.loss_params["clamp_rec"][1]
                lambda_lower_contour = self.loss_params["clamp_contour"][0]
                lambda_upper_contour = self.loss_params["clamp_contour"][1]

                self._lambda = (  # pylint: disable=attribute-defined-outside-init
                    torch.exp(torch.Tensor([rc, cc]).to(rc.device)) * self._lambda
                )

                self._lambda[0] = self._lambda[0].clamp(lambda_lower, lambda_upper)
                self._lambda[1] = self._lambda[1].clamp(lambda_lower_contour, lambda_upper_contour)

            # pylint: disable=access-member-before-definition
            loss = (self._lambda[0] * reconstruction_loss) + kl_div

            result = {
                "loss": loss,
                "rec_loss": reconstruction_loss,
                "kl_div": kl_div,
                "lambda_rec": self._lambda[0],
                "moving_avg": self._rec_moving_avg,
                "reconstruction_threshold": reconstruction_threshold,
                "rec_constraint": rc,
            }

            if contour_threshold is not None:
                result["loss"] = result["loss"] + (self._lambda[1] * contour_loss)
                result["contour_loss"] = contour_loss
                result["contour_threshold"] = contour_threshold
                result["contour_constraint"] = cc
                result["moving_avg_contour"] = self._contour_moving_avg
                result["lambda_contour"] = self._lambda[1]

            return result

        else:
            raise NotImplementedError("Loss must be 'elbo' or 'geco'")
