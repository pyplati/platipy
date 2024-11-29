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

# This code is adapted from
# https://github.com/deepmind/deepmind-research/tree/5cf55efe1f1748ebdd33cb69223b0df6bcc88e6a/hierarchical_probabilistic_unet
# which is released under the Apache Licence 2.0

# pylint: disable=invalid-name

import torch

from .unet import init_weights, init_zeros, conv_nd


class ResBlock(torch.nn.Module):
    """A residual block"""

    def __init__(
        self,
        input_channels,
        output_channels,
        n_down_channels=None,
        activation_fn=torch.nn.ReLU,
        convs_per_block=2,
        ndims=2,
    ):
        """Create a residual block

        Args:
            input_channels (int): The number of input channels to the block
            output_channels (int): The number of output channels from the block
            n_down_channels (int, optional): The number of intermediate cahnnels within the block.
                                             Defaults to the same as the number of output channels.
            activation_fn (torch.nn.Module, optional): The activation function to apply. Defaults
                                                       to torch.nn.ReLU.
            convs_per_block (int, optional): The number of convolutions to perform within the
                                             block. Defaults to 2.
            ndims (int,  optional): Specify whether to use 2 or 3 dimensions. Defaults to 2.
        """

        super(ResBlock, self).__init__()

        self._activation_fn = activation_fn()

        # Set the number of intermediate channels that we compress to.
        if n_down_channels is None:
            n_down_channels = output_channels

        layers = []
        in_channels = input_channels
        for c in range(convs_per_block):
            layers.append(
                conv_nd(
                    ndims=ndims,
                    in_channels=in_channels,
                    out_channels=n_down_channels,
                    kernel_size=3,
                    padding=1,
                )
            )

            if c < convs_per_block - 1:
                layers.append(activation_fn())

            in_channels = n_down_channels

        if not n_down_channels == output_channels:
            resize_outgoing = conv_nd(
                ndims=ndims,
                in_channels=n_down_channels,
                out_channels=output_channels,
                kernel_size=1,
                padding=0,
            )
            layers.append(resize_outgoing)

        self._layers = torch.nn.Sequential(*layers)
#        self._layers.apply(init_weights)

        self._resize_skip = None

        if not input_channels == output_channels:
            self._resize_skip = conv_nd(
                ndims=ndims,
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=1,
                padding=0,
            )
 #           self._resize_skip.apply(init_weights)

    def forward(self, input_features):

        # Pre-activate the inputs.
        skip = input_features
        residual = self._activation_fn(input_features)

        for layer in self._layers:
            residual = layer(residual)

        if not self._resize_skip is None:
            skip = self._resize_skip(skip)

        return skip + residual


def resize_up(input_features, scale=2):
    """Resize the the input to upsample

    Args:
        input_features (torch.Tensor): The Tensor to upsize
        scale (int, optional): The scale used to upsize. Defaults to 2.

    Returns:
        torch.Tensor: The upsized Tensor
    """

    input_shape = input_features.shape
    size_x = input_shape[2]
    size_y = input_shape[3]

    new_size = [int(round(size_x * scale)), int(round(size_y * scale))]

    if len(input_shape) == 5:
        size_z = input_shape[4]
        new_size = new_size + [int(round(size_z * scale))]

    return torch.nn.functional.interpolate(input_features, size=new_size)


def resize_down(input_features, scale=2):
    """Resize the the input to downsample

    Args:
        input_features (torch.Tensor): The Tensor to downsize
        scale (int, optional): The scale used to downsize. Defaults to 2.

    Returns:
        torch.Tensor: The downsized Tensor
    """
    if input_features.ndim == 5:
        return torch.nn.AvgPool3d(kernel_size=scale, stride=scale, padding=0)(input_features)
    else:
        return torch.nn.AvgPool2d(kernel_size=scale, stride=scale, padding=0)(input_features)


class _HierarchicalCore(torch.nn.Module):
    """A U-Net encoder-decoder with a full encoder and a truncated decoder.
    The truncated decoder is interleaved with the hierarchical latent space and
    has as many levels as there are levels in the hierarchy plus one additional
    level.
    """

    def __init__(
        self,
        latent_dims,
        input_channels,
        channels_per_block,
        down_channels_per_block=None,
        activation_fn=torch.nn.ReLU,
        convs_per_block=2,
        blocks_per_level=1,
        ndims=2,
    ):
        """Initializes a HierarchicalCore.

        Args:
            latent_dims (list): List of integers specifying the dimensions of the latents at
                                each scale. The length of the list indicates the number of U-Net
                                decoder scales that have latents.
            input_channels (int): The number of input channels.
            channels_per_block (list): A list of integers specifying the number of output
                                         channels for each encoder block.
            down_channels_per_block (list, optional): A list of integers specifying the number of
                                                      intermediate channels for each encoder block
                                                      or None. If None, the intermediate channels
                                                      are chosen equal to channels_per_block.
                                                      Defaults to None.
            activation_fn (torch.nn.Module, optional): A callable activation function. Defaults to
                                                       torch.nn.ReLU.
            convs_per_block (int, optional): An integer specifying the number of convolutional
                                             layers. Defaults to 2.
            blocks_per_level (int, optional): An integer specifying the number of residual blocks
                                              per level. Defaults to 1.
            ndims (int,  optional): Specify whether to use 2 or 3 dimensions. Defaults to 2.
        """

        super(_HierarchicalCore, self).__init__()

        self._latent_dims = latent_dims
        self._input_channels = input_channels
        self._channels_per_block = channels_per_block
        self._activation_fn = activation_fn
        self._convs_per_block = convs_per_block
        self._blocks_per_level = blocks_per_level
        if down_channels_per_block is None:
            self._down_channels_per_block = channels_per_block
        else:
            self._down_channels_per_block = down_channels_per_block

        num_levels = len(self._channels_per_block)
        self._num_latent_levels = len(self._latent_dims)

        # Iterate the descending levels in the U-Net encoder.
        self.encoder_layers = torch.nn.ModuleList()
        in_channels = input_channels
        for level in range(num_levels):
            # Iterate the residual blocks in each level.
            layer = []
            for _ in range(self._blocks_per_level):
                layer.append(
                    ResBlock(
                        in_channels,
                        channels_per_block[level],
                        n_down_channels=self._down_channels_per_block[level],
                        activation_fn=self._activation_fn,
                        convs_per_block=self._convs_per_block,
                        ndims=ndims,
                    )
                )
                in_channels = channels_per_block[level]

            self.encoder_layers.append(torch.nn.Sequential(*layer))

  #      self.encoder_layers.apply(init_weights)

        # Iterate the ascending levels in the (truncated) U-Net decoder.
        self.decoder_layers = torch.nn.ModuleList()
        self._mu_logsigma_blocks = torch.nn.ModuleList()

        for level in range(self._num_latent_levels):

            latent_dim = latent_dims[level]

            mu_logsigma_block = conv_nd(
                ndims=ndims,
                in_channels=channels_per_block[::-1][level],
                out_channels=2 * latent_dim,
                kernel_size=1,
                padding=0,
            )

            self._mu_logsigma_blocks.append(mu_logsigma_block)

            decoder_in_channels = (
                channels_per_block[::-1][level + 1] + channels_per_block[::-1][level]
            ) + latent_dim
            layer = []
            for _ in range(self._blocks_per_level):
                layer.append(
                    ResBlock(
                        decoder_in_channels,
                        channels_per_block[::-1][level + 1],
                        n_down_channels=self._down_channels_per_block[::-1][level + 1],
                        activation_fn=self._activation_fn,
                        convs_per_block=self._convs_per_block,
                        ndims=ndims,
                    )
                )
                decoder_in_channels = channels_per_block[::-1][level + 1]

            self.decoder_layers.append(torch.nn.Sequential(*layer))

     #   self._mu_logsigma_blocks.apply(init_zeros)
     #   self.decoder_layers.apply(init_weights)

    def forward(self, inputs, mean=False, std_devs_from_mean=0.0, z_q=None):
        """Forward pass to sample from the module as specified.

        Args:
            inputs (torch.Tensor): A tensor of shape (b,c,h,w). When using the module as a prior
                                   the `inputs` tensor should be a batch of images. When using it
                                   as a posterior the tensor should be a (batched) concatentation
                                   of images and segmentations.
            mean (bool|list, optional): A boolean or a list of booleans. If a boolean, it specifies
                                        whether or not to use the distributions' means in ALL
                                        latent scales. If a list, each bool therein specifies
                                        whether or not to use the scale's mean. If False, the
                                        latents of the scale are sampled. Defaults to False.
            std_devs_from_mean (float|list, optional): A float or list of floats describing how far
                                                       from the mean should be sampled. Only at
                                                       scales where mean is True. Defaults to 0.
            z_q (list, optional): None or a list of tensors. If not None, z_q provides external
                                  latents to be used instead of sampling them. This is used to
                                  employ posterior latents in the prior during training. Therefore,
                                  if z_q is not None, the value of `mean` is ignored. If z_q is
                                  None, either the distributions mean is used (in case `mean` for
                                  the respective scale is True) or else a sample from the
                                  distribution is drawn. Defaults to None.

        Returns:
            dict: A Dictionary holding the output feature map of the truncated U-Net decoder under
            key 'decoder_features', a list of the U-Net encoder features produced at the end of
            each encoder scale under key 'encoder_outputs', a list of the predicted distributions
            at each scale under key 'distributions', a list of the used latents at each scale under
            the key 'used_latents'.
        """

        encoder_features = inputs
        encoder_outputs = []
        num_levels = len(self._channels_per_block)
        num_latent_levels = len(self._latent_dims)

        if isinstance(mean, bool):
            mean = [mean] * self._num_latent_levels

        if isinstance(std_devs_from_mean, int):
            std_devs_from_mean = float(std_devs_from_mean)

        if isinstance(std_devs_from_mean, float):
            std_devs_from_mean = [std_devs_from_mean] * self._num_latent_levels

        distributions = []
        used_latents = []

        # Iterate the descending levels in the U-Net encoder.
        for level, encoder_layer in enumerate(self.encoder_layers):
            encoder_features = encoder_layer(encoder_features)
            encoder_outputs.append(encoder_features)
            if not level == num_levels - 1:
                encoder_features = resize_down(encoder_features, scale=2)

        # Iterate the ascending levels in the (truncated) U-Net decoder.
        decoder_features = encoder_outputs[-1]
        for level in range(num_latent_levels):

            # Predict a Gaussian distribution for each pixel in the feature map.
            latent_dim = self._latent_dims[level]
            mu_logsigma = self._mu_logsigma_blocks[level](decoder_features)

            mu = mu_logsigma[:, :latent_dim].clamp(-1000, 1000)
            log_sigma = mu_logsigma[:, latent_dim:].clamp(-10, 10)

            dist = torch.distributions.Independent(
                torch.distributions.Normal(loc=mu, scale=torch.exp(log_sigma)), 1
            )
            distributions.append(dist)

            # Get the latents to condition on.
            if z_q is not None:
                z = z_q[level]
            elif mean[level]:
                z = dist.mean + (dist.base_dist.stddev * std_devs_from_mean[level])
            else:
                z = dist.sample()

            used_latents.append(z)

            # Concat and upsample the latents with the previous features.
            decoder_output_lo = torch.cat([z, decoder_features], axis=1)
            decoder_output_hi = resize_up(decoder_output_lo, scale=2)
            decoder_features = torch.cat(
                [decoder_output_hi, encoder_outputs[::-1][level + 1]], axis=1
            )
            decoder_features = self.decoder_layers[level](decoder_features)

        return {
            "decoder_features": decoder_features,
            "encoder_features": encoder_outputs,
            "distributions": distributions,
            "used_latents": used_latents,
        }


class _StitchingDecoder(torch.nn.Module):
    """A module that completes the truncated U-Net decoder.
    Using the output of the HierarchicalCore this module fills in the missing
    decoder levels such that together the two form a symmetric U-Net.
    """

    def __init__(
        self,
        latent_dims,
        channels_per_block,
        num_classes,
        down_channels_per_block=None,
        activation_fn=torch.nn.ReLU,
        convs_per_block=2,
        blocks_per_level=1,
        ndims=2,
    ):
        """Initializes a StichtingDecoder.

        Args:
            latent_dims (list): List of integers specifying the dimensions of the latents at each
                                scale. The length of the list indicates the number of U-Net decoder
                                scales that have latents.
            channels_per_block (list): A list of integers specifying the number of output channels
                                       for each encoder block.
            num_classes (int): The number of segmentation classes.
            down_channels_per_block ([type], optional): A list of integers specifying the number of
                                                        intermediate channels for each encoder
                                                        block. If None, the intermediate channels
                                                        are chosen equal to channels_per_block.
                                                        Defaults to None.
            activation_fn (torch.nn.Module, optional): A callable activation function.Defaults to
                                                       torch.nn.ReLU.
            initializers ([type], optional): [description]. Defaults to None.
            regularizers ([type], optional): [description]. Defaults to None.
            convs_per_block (int, optional): An integer specifying the number of convolutional
                                             layers. Defaults to 2.
            blocks_per_level (int, optional): An integer specifying the number of residual blocks
                                              per level. Defaults to 1.
            ndims (int,  optional): Specify whether to use 2 or 3 dimensions. Defaults to 2.
        """
        super(_StitchingDecoder, self).__init__()
        self._latent_dims = latent_dims
        self._channels_per_block = channels_per_block
        self._num_classes = num_classes
        self._activation_fn = activation_fn
        self._convs_per_block = convs_per_block
        self._blocks_per_level = blocks_per_level
        if down_channels_per_block is None:
            down_channels_per_block = channels_per_block
        self._down_channels_per_block = down_channels_per_block

        num_latents = len(self._latent_dims)
        self._start_level = num_latents + 1
        self._num_levels = len(self._channels_per_block)

        self.layers = torch.nn.ModuleList()
        decoder_in_channels = None
        for level in range(self._start_level, self._num_levels, 1):

            decoder_in_channels = (
                channels_per_block[::-1][level - 1] + channels_per_block[::-1][level]
            )

            layer = []
            for _ in range(self._blocks_per_level):
                layer.append(
                    ResBlock(
                        decoder_in_channels,
                        channels_per_block[::-1][level],
                        n_down_channels=self._down_channels_per_block[::-1][level],
                        activation_fn=self._activation_fn,
                        convs_per_block=self._convs_per_block,
                        ndims=ndims,
                    )
                )
                decoder_in_channels = channels_per_block[::-1][level]

            self.layers.append(torch.nn.Sequential(*layer))
   #     self.layers.apply(init_weights)

        if decoder_in_channels is None:
            decoder_in_channels = channels_per_block[::-1][self._num_levels - 1]

        self.final_layer = conv_nd(
            ndims=ndims,
            in_channels=decoder_in_channels,
            out_channels=self._num_classes,
            kernel_size=1,
            padding=0,
        )
    #    self.final_layer.apply(init_weights)

    def forward(self, encoder_features, decoder_features):
        """Forward pass through the stiching decoder

        Args:
            encoder_features (torch.Tensor): Tensor of encoder features
            decoder_features (dict): Tensor of decoder features

        Returns:
            torch.Tensor: The stiched output
        """

        for level in range(len(self.layers)):
            enc_level = self._start_level + level
            decoder_features = resize_up(decoder_features, scale=2)
            decoder_features = torch.cat(
                [decoder_features, encoder_features[::-1][enc_level]], axis=1
            )
            decoder_features = self.layers[level](decoder_features)

        return self.final_layer(decoder_features)


class HierarchicalProbabilisticUnet(torch.nn.Module):
    """A hierarchical probabilistic UNet implementation: https://arxiv.org/abs/1905.13077"""

    def __init__(
        self,
        input_channels=1,
        num_classes=2,
        filters_per_layer=None,
        down_channels_per_block=None,
        latent_dims=(1, 1, 1, 1),
        convs_per_block=2,
        blocks_per_level=1,
        loss_type="elbo",
        loss_params={"beta": 1},
        ndims=2,
    ):
        """Initialize the Hierarchical Probabilistic UNet

        Args:
            input_channels (int, optional): The number of channels in the image (1 for
                                            greyscale and 3 for RGB). Defaults to 1.
            num_classes (int, optional): The number of classes to predict. Defaults to 2.
            filters_per_layer (list, optional): A list of channels to use in blocks of each
                                                 layer the amount of filters layer. Defaults
                                                 to None.
            down_channels_per_block (list, optional): [description]. Defaults to None.
            latent_dims (tuple, optional): The number of latent dimensions at each layer.
                                           Defaults to (1, 1, 1, 1).
            convs_per_block (int, optional): An integer specifying the number of convolutional
                                             layers. Defaults to 3. Defaults to 2.
            blocks_per_level (int, optional): An integer specifying the number of residual
                                              blocks per level. Defaults to 1.
            loss_kwargs (dict, optional): Dictionary of argument used by loss function.
                                          Defaults to None.
            ndims (int,  optional): Specify whether to use 2 or 3 dimensions. Defaults to 2.
        """
        super(HierarchicalProbabilisticUnet, self).__init__()

        base_channels = 24
        default_filters_per_layer = (
            base_channels,
            2 * base_channels,
            4 * base_channels,
            8 * base_channels,
            8 * base_channels,
            8 * base_channels,
            8 * base_channels,
            8 * base_channels,
        )
        if filters_per_layer is None:
            filters_per_layer = default_filters_per_layer
        if down_channels_per_block is None:
            down_channels_per_block = [int(i / 2) for i in filters_per_layer]

        self.prior = _HierarchicalCore(
            input_channels=input_channels,
            latent_dims=latent_dims,
            channels_per_block=filters_per_layer,
            down_channels_per_block=down_channels_per_block,
            convs_per_block=convs_per_block,
            blocks_per_level=blocks_per_level,
            ndims=ndims,
        )

        self.posterior = _HierarchicalCore(
            input_channels=input_channels + num_classes,
            latent_dims=latent_dims,
            channels_per_block=filters_per_layer,
            down_channels_per_block=down_channels_per_block,
            convs_per_block=convs_per_block,
            blocks_per_level=blocks_per_level,
            ndims=ndims,
        )

        self.fcomb = _StitchingDecoder(
            latent_dims=latent_dims,
            channels_per_block=filters_per_layer,
            num_classes=num_classes,
            down_channels_per_block=down_channels_per_block,
            convs_per_block=convs_per_block,
            blocks_per_level=blocks_per_level,
            ndims=ndims,
        )
        self.ndims = ndims

        self._cache = None

        self.loss_type = loss_type
        self.loss_params = loss_params

        if self.loss_type == "geco":
            self._rec_moving_avg = None
            self._contour_moving_avg = None
            self.register_buffer("_lambda", torch.zeros(2, requires_grad=False))

        self._q_sample = None
        self._q_sample_mean = None
        self._p_sample = None
        self._p_sample_z_q = None
        self._p_sample_z_q_mean = None

    def forward(self, img, seg):
        """Inserts all ops used during training into the graph exactly once. The first time this
        method is called given the input pair (img, seg) all ops relevant for training are inserted
        into the graph. Calling this method more than once does not re-insert the modules into the
        graph (memoization), thus preventing multiple forward passes of submodules for the same
        inputs.

        Args:
            img (torch.Tensor): A tensor of shape (b, c, h, w).
            seg (torch.Tensor): A tensor of shape (b, num_classes, h, w).
        """

        input_tensor = torch.cat([img, seg], axis=1)

        if not self._cache is None and torch.equal(self._cache, input_tensor):
            # No need to recompute
            return

        self._q_sample = self.posterior(input_tensor, mean=False)
        self._q_sample_mean = self.posterior(input_tensor, mean=True)
        self._p_sample = self.prior(img, mean=False, z_q=None)
        self._p_sample_z_q = self.prior(img, z_q=self._q_sample["used_latents"])
        self._p_sample_z_q_mean = self.prior(img, z_q=self._q_sample_mean["used_latents"])
        self._cache = input_tensor

    def sample(self, img, mean=False, std_devs_from_mean=0.0, z_q=None):
        """Sample a segmentation from the prior, given an input image.

        Args:
            img (torch.Tensor): A tensor of shape (b, c, h, w).
            mean (bool, optional): A boolean or a list of booleans. If a boolean, it specifies
                                   whether or not to use the distributions' means in ALL latent
                                   scales. If a list, each bool therein specifies whether or not to
                                   use the scale's mean. If False, the latents of the scale are
                                   sampled. Defaults to False.
            std_devs_from_mean (float|list, optional): A float or list of floats describing how far
                                                       from the mean should be sampled. Only at
                                                       scales where mean is True. Defaults to 0.
            z_q (list, optional): If not None, z_q provides external latents to be used instead of
                                  sampling them. This is used to employ posterior latents in the
                                  prior during training. Therefore, if z_q is not None, the value
                                  of `mean` is ignored. If z_q is None, either the distributions
                                  mean is used (in case `mean` for the respective scale is True) or
                                  else a sample from the distribution is drawn. Defaults to None.

        Returns:
            torch.Tensor: A segmentation tensor of shape (b, num_classes, h, w).
        """

        prior_out = self.prior(img, mean, std_devs_from_mean, z_q)
        encoder_features = prior_out["encoder_features"]
        decoder_features = prior_out["decoder_features"]
        return self.fcomb(encoder_features, decoder_features)

    def reconstruct(self, img, seg, mean=False):
        """Reconstruct a segmentation using the posterior.

        Args:
            img ([torch.Tensor): A tensor of shape (b, c, h, w).
            seg (torch.Tensor): A tensor of shape (b, num_classes, h, w).
            mean (bool, optional): A boolean, specifying whether to sample from the full hierarchy
                                   of the posterior or use the posterior means at each scale of the
                                   hierarchy. Defaults to False.

        Returns:
            torch.Tensor: A segmentation tensor of shape (b,num_classes,h,w).
        """

        # self.forward(img, seg)
        if mean:
            prior_out = self._p_sample_z_q_mean
        else:
            prior_out = self._p_sample_z_q
        encoder_features = prior_out["encoder_features"]
        decoder_features = prior_out["decoder_features"]
        return self.fcomb(encoder_features, decoder_features)

    def kl(self, img, seg):
        """Kullback-Leibler divergence between the posterior and the prior.

        Args:
            img (torch.Tensor): A tensor of shape (b, c, h, w).
            seg (torch.Tensor): A tensor of shape (b, num_classes, h, w).

        Returns:
            dict: A dictionary with keys indexing the hierarchy's levels and corresponding
                    values holding the KL-term for each level (per batch).
        """
        self.forward(img, seg)
        posterior_out = self._q_sample
        prior_out = self._p_sample_z_q

        q_dists = posterior_out["distributions"]
        p_dists = prior_out["distributions"]

        kl = {}
        for level, (p, q) in enumerate(zip(p_dists, q_dists)):
            kl_per_pixel = torch.distributions.kl.kl_divergence(p, q)

            if self.ndims == 2:
                kl_per_instance = torch.sum(kl_per_pixel, [1, 2])
            else:
                kl_per_instance = torch.sum(kl_per_pixel, [1, 2, 3])

            kl_clamp = img.shape[2:].numel() * 10
            kl_per_instance = kl_per_instance.clamp(0, kl_clamp)
            kl[level] = torch.mean(kl_per_instance)

        return kl

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
        img,
        segm,
        mask=None,
        top_k_percentage=None,
        deterministic=True,
    ):

        criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

        reconstruction = self.reconstruct(img, segm)

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

    def loss(self, img, seg, mask=None):
        """The full training objective, either ELBO or GECO.

        Args:
            img (torch.Tensor): A tensor of shape (b, c, h, w).
            seg (torch.Tensor): A tensor of shape (b, num_classes, h, w).

        Raises:
            NotImplementedError: Raised if loss function supplied isn't implemented yet.

        Returns:
            dict: A dictionary holding the loss (with key 'loss')
        """
        kl_summaries = {}
        kl_dict = self.kl(img, seg)
        kl_sum = torch.sum(torch.stack([kl for _, kl in kl_dict.items()], axis=-1))
        for level, kl in kl_dict.items():
            kl_summaries[f"kl_{level}"] = kl

        top_k_percentage = None
        if "top_k_percentage" in self.loss_params:
            top_k_percentage = self.loss_params["top_k_percentage"]

        loss_mask = None
        if "kappa" in self.loss_params:
            reconstruction_threshold = self.loss_params["kappa"]
        contour_threshold = None
        if "kappa_contour" in self.loss_params and self.loss_params["kappa_contour"] is not None:
            loss_mask = [None, mask]
            contour_threshold = self.loss_params["kappa_contour"]

        # Here we use the posterior sample sampled above
        _, rec_loss_mean, _ = self.reconstruction_loss(
            img,
            seg,
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
                "loss": reconstruction_loss + self.loss_params["beta"] * kl_sum,
                "rec_loss": reconstruction_loss,
                "kl_div": kl_sum,
            }
        elif self.loss_type == "geco":

            with torch.no_grad():

                moving_avg_factor = 0.5

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

                self._lambda[0] = (torch.exp(rc) * self._lambda[0]).clamp(lambda_lower, lambda_upper)
                if self._lambda[0].isnan(): self._lambda[0] = lambda_upper
                if contour_threshold:
                    lambda_lower_contour = self.loss_params["clamp_contour"][0]
                    lambda_upper_contour = self.loss_params["clamp_contour"][1]

                    self._lambda[1] = (torch.exp(cc) * self._lambda[1]).clamp(lambda_lower_contour, lambda_upper_contour)
                    if self._lambda[1].isnan(): self._lambda[1] = lambda_upper_contour

            # pylint: disable=access-member-before-definition
            loss = (self._lambda[0] * reconstruction_loss) + kl_sum

            result = {
                "loss": loss,
                "rec_loss": reconstruction_loss,
                "kl_div": kl_sum,
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
            result = {**result, **kl_summaries}

            return result

        else:
            raise NotImplementedError("Loss must be 'elbo' or 'geco'")
