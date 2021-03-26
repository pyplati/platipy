import torch

from platipy.imaging.cnn.hierarchical_prob_unet import (
    _HierarchicalCore,
    ResBlock,
    HierarchicalProbabilisticUnet,
)

base_channels = 24
default_channels_per_block = [
    base_channels,
    2 * base_channels,
    4 * base_channels,
    8 * base_channels,
    # 8 * base_channels,
    # 8 * base_channels,
    # 8 * base_channels,
    # 8 * base_channels,
]
# default_channels_per_block = [
#     base_channels,
#     2 * base_channels,
#     4 * base_channels,
#     8 * base_channels,
# ]
channels_per_block = default_channels_per_block
down_channels_per_block = [int(i / 2) for i in default_channels_per_block]
a = _HierarchicalCore([8, 6, 2], 1, channels_per_block, down_channels_per_block)
b = ResBlock(1, 3, base_channels)
d = ResBlock(3, 3, base_channels * 2)
c = torch.rand([1, 1, 256, 256])

fg = torch.ones(c.shape)
bg = torch.zeros(c.shape)
labels = torch.cat([fg, bg], axis=1)
# a(b)
# print(a)

hpunet = HierarchicalProbabilisticUnet()
output = hpunet.sample(c)
print(output.shape)
output = hpunet.reconstruct(c, labels)
print(output.shape)
loss = hpunet.loss(c, labels)
print(loss)


_NUM_CLASSES = 2
_BATCH_SIZE = 2
_SPATIAL_SHAPE = [32, 32]
_CHANNELS_PER_BLOCK = [5, 7, 9, 11, 13]
_IMAGE_SHAPE = [_BATCH_SIZE] + [1] + _SPATIAL_SHAPE
_BOTTLENECK_SIZE = _SPATIAL_SHAPE[0] // 2 ** (len(_CHANNELS_PER_BLOCK) - 1)
_SEGMENTATION_SHAPE = [_BATCH_SIZE] + [_NUM_CLASSES] + _SPATIAL_SHAPE
_LATENT_DIMS = [3, 2, 1]
# _INITIALIZERS = {
#     "w": tf.orthogonal_initializer(gain=1.0, seed=None),
#     "b": tf.truncated_normal_initializer(stddev=0.001),
# }


def _get_placeholders():
    """Returns placeholders for the image and segmentation."""
    img = torch.rand(_IMAGE_SHAPE)
    seg = torch.rand(_SEGMENTATION_SHAPE)
    return img, seg


def test_shape_of_sample():
    hpu_net = HierarchicalProbabilisticUnet(
        latent_dims=_LATENT_DIMS,
        channels_per_block=_CHANNELS_PER_BLOCK,
        num_classes=_NUM_CLASSES,
        # initializers=_INITIALIZERS,
    )
    img, _ = _get_placeholders()
    sample = hpu_net.sample(img)

    assert list(sample.shape) == _SEGMENTATION_SHAPE


def test_shape_of_reconstruction():
    hpu_net = HierarchicalProbabilisticUnet(
        latent_dims=_LATENT_DIMS,
        channels_per_block=_CHANNELS_PER_BLOCK,
        num_classes=_NUM_CLASSES,
        # initializers=_INITIALIZERS,
    )
    img, seg = _get_placeholders()
    reconstruction = hpu_net.reconstruct(img, seg)
    assert list(reconstruction.shape) == _SEGMENTATION_SHAPE


def test_shapes_in_prior():
    hpu_net = HierarchicalProbabilisticUnet(
        latent_dims=_LATENT_DIMS,
        channels_per_block=_CHANNELS_PER_BLOCK,
        num_classes=_NUM_CLASSES,
        # initializers=_INITIALIZERS,
    )
    img, _ = _get_placeholders()
    prior_out = hpu_net._prior(img)
    distributions = prior_out["distributions"]
    latents = prior_out["used_latents"]
    encoder_features = prior_out["encoder_features"]
    decoder_features = prior_out["decoder_features"]

    # Test number of latent disctributions.
    assert len(distributions) == len(_LATENT_DIMS)

    # Test shapes of latent scales.
    for level in range(len(_LATENT_DIMS)):
        latent_spatial_shape = _BOTTLENECK_SIZE * 2 ** level
        latent_shape = [
            _BATCH_SIZE,
            _LATENT_DIMS[level],
            latent_spatial_shape,
            latent_spatial_shape,
        ]
        assert list(latents[level].shape) == latent_shape

    # Test encoder shapes.
    for level in range(len(_CHANNELS_PER_BLOCK)):
        spatial_shape = _SPATIAL_SHAPE[0] // 2 ** level
        feature_shape = [_BATCH_SIZE, _CHANNELS_PER_BLOCK[level], spatial_shape, spatial_shape]

        assert list(encoder_features[level].shape) == feature_shape

    # Test decoder shape.
    start_level = len(_LATENT_DIMS)
    latent_spatial_shape = _BOTTLENECK_SIZE * 2 ** start_level
    latent_shape = [
        _BATCH_SIZE,
        _CHANNELS_PER_BLOCK[::-1][start_level],
        latent_spatial_shape,
        latent_spatial_shape,
    ]

    assert list(decoder_features.shape) == latent_shape


def test_shape_of_kl():
    hpu_net = HierarchicalProbabilisticUnet(
        latent_dims=_LATENT_DIMS,
        channels_per_block=_CHANNELS_PER_BLOCK,
        num_classes=_NUM_CLASSES,
        # initializers=_INITIALIZERS,
    )
    img, seg = _get_placeholders()
    kl_dict = hpu_net.kl(img, seg)
    assert len(kl_dict) == len(_LATENT_DIMS)


test_shape_of_sample()
test_shape_of_reconstruction()
test_shapes_in_prior()
test_shape_of_kl()
# if __name__ == "__main__":
