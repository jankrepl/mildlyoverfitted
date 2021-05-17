import jax
import numpy as np
import pytest
import torch

from official import MlpMixer as OfficialMixer
from ours import MlpMixer as OurMixer


@pytest.mark.parametrize("image_size", [6, 12])
@pytest.mark.parametrize("patch_size", [2, 3])
@pytest.mark.parametrize("hidden_dim", [4, 5])
@pytest.mark.parametrize("n_blocks", [1, 2])
@pytest.mark.parametrize("n_classes", [4, 8])
@pytest.mark.parametrize("tokens_mlp_dim", [2, 4])
@pytest.mark.parametrize("channels_mlp_dim", [3, 6])
def test_compare(
    image_size,
    patch_size,
    hidden_dim,
    n_blocks,
    n_classes,
    tokens_mlp_dim,
    channels_mlp_dim,
):
    # Create Flax model
    model_flax = OfficialMixer(
        num_classes=n_classes,
        num_blocks=n_blocks,
        patch_size=patch_size,
        hidden_dim=hidden_dim,
        tokens_mlp_dim=tokens_mlp_dim,
        channels_mlp_dim=channels_mlp_dim,
    )
    key1, key2 = jax.random.split(jax.random.PRNGKey(0))
    x = jax.random.normal(key1, (11, image_size, image_size, 3))  # Dummy input
    params = model_flax.init(key2, x)  # initialization call

    n_params_flax = sum(
        jax.tree_leaves(jax.tree_map(lambda x: np.prod(x.shape), params))
    )
    shape_flax = model_flax.apply(params, x).shape

    # Create Torch model
    model_torch = OurMixer(
        image_size=image_size,
        patch_size=patch_size,
        hidden_dim=hidden_dim,
        n_blocks=n_blocks,
        n_classes=n_classes,
        tokens_mlp_dim=tokens_mlp_dim,
        channels_mlp_dim=channels_mlp_dim,
    )

    n_params_torch = sum(
        p.numel() for p in model_torch.parameters() if p.requires_grad
    )
    shape_torch = model_torch(torch.rand(11, 3, image_size, image_size)).shape

    assert n_params_flax == n_params_torch
    assert shape_flax == shape_torch == (11, n_classes)
