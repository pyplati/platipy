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
    # 4 * base_channels,
    # 8 * base_channels,
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

latent_dims = [8, 6, 2]
latent_dims = [2]

channels_per_block = default_channels_per_block
down_channels_per_block = [int(i / 2) for i in default_channels_per_block]
c = torch.rand([3, 1, 32, 32])

fg = torch.ones(c.shape)
bg = torch.zeros(c.shape)
labels = torch.cat([fg, bg], axis=1)

hpunet = HierarchicalProbabilisticUnet(
    filters_per_layer=channels_per_block,
    latent_dims=[1],
    loss_type="geco",
    loss_params={
        # "top_k_percentage": 0.02,
        "top_k_percentage": None,
        "deterministic_top_k": False,
        "kappa": 0.05,
        "decay": 0.99,
        "rate": 1e-2,
        "clamp_rec": [0.001, 10000],
        "beta": 5,
    },
)
output = hpunet.sample(c)
print(output.shape)
output = hpunet.reconstruct(c, labels)
print(output.shape)
loss = hpunet.loss(c, labels)
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


def _get_placeholders():
    """Returns placeholders for the image and segmentation."""
    img = torch.rand(_IMAGE_SHAPE)
    seg = torch.rand(_SEGMENTATION_SHAPE)
    return img, seg


def test_shape_of_sample():
    hpu_net = HierarchicalProbabilisticUnet(
        latent_dims=_LATENT_DIMS,
        filters_per_layer=_CHANNELS_PER_BLOCK,
        num_classes=_NUM_CLASSES,
    )
    img, _ = _get_placeholders()
    sample = hpu_net.sample(img)

    assert list(sample.shape) == _SEGMENTATION_SHAPE


def test_shape_of_reconstruction():
    hpu_net = HierarchicalProbabilisticUnet(
        latent_dims=_LATENT_DIMS,
        filters_per_layer=_CHANNELS_PER_BLOCK,
        num_classes=_NUM_CLASSES,
    )
    img, seg = _get_placeholders()
    reconstruction = hpu_net.reconstruct(img, seg)
    assert list(reconstruction.shape) == _SEGMENTATION_SHAPE


def test_shapes_in_prior():
    hpu_net = HierarchicalProbabilisticUnet(
        latent_dims=_LATENT_DIMS,
        filters_per_layer=_CHANNELS_PER_BLOCK,
        num_classes=_NUM_CLASSES,
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
        filters_per_layer=_CHANNELS_PER_BLOCK,
        num_classes=_NUM_CLASSES,
    )
    img, seg = _get_placeholders()
    kl_dict = hpu_net.kl(img, seg)
    assert len(kl_dict) == len(_LATENT_DIMS)


test_shape_of_sample()
test_shape_of_reconstruction()
test_shapes_in_prior()
test_shape_of_kl()
# if __name__ == "__main__":
