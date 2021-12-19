import math

import torch
import torch.nn as nn
from torch.nn.utils.prune import l1_unstructured, random_unstructured


class MLP(nn.Module):
    """Multilayer perceptron.

    The bias is included in all linear layers.

    Parameters
    ----------
    n_features : int
        Number of input features (pixels inside of MNIST images).

    hidden_layer_sizes : tuple
        Tuple of ints representing sizes of the hidden layers.

    n_targets : int
        Number of target classes (10 for MNIST).

    Attributes
    ----------
    module_list : nn.ModuleList
        List holding all the linear layers in the right order.
    """

    def __init__(self, n_features, hidden_layer_sizes, n_targets):
        super().__init__()

        layer_sizes = (n_features,) + hidden_layer_sizes + (n_targets,)
        layer_list = []

        for i in range(len(layer_sizes) - 1):
            layer_list.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

        self.module_list = nn.ModuleList(layer_list)

    def forward(self, x):
        """Run the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Batch of features of shape `(batch_size, n_features)`.

        Returns
        -------
        torch.Tensor
            Batch of predictions (logits) of shape `(batch_size, n_targets)`.
        """
        n_layers = len(self.module_list)

        for i, layer in enumerate(self.module_list):
            x = layer(x)

            if i < n_layers - 1:
                x = nn.functional.relu(x)

        return x


def prune_linear(linear, prune_ratio=0.3, method="l1"):
    """Prune a linear layer.

    Modifies the module in-place. We make an assumption that the bias
    is included.

    Parameters
    ----------
    linear : nn.Linear
        Linear module containing a bias.

    prune_ratio : float
        Number between 0 and 1 representing the percentage of weights
        to prune.

    method : str, {"l1", "random"}
        Pruning method to use.
    """
    if method == "l1":
        prune_func = l1_unstructured
    elif method == "random":
        prune_func = random_unstructured
    else:
        raise ValueError

    prune_func(linear, "weight", prune_ratio)
    prune_func(linear, "bias", prune_ratio)


def prune_mlp(mlp, prune_ratio=0.3, method="l1"):
    """Prune each layer of the multilayer perceptron.

    Modifies the module in-place. We make an assumption that each
    linear layer has the bias included.

    Parameters
    ----------
    mlp : MLP
        Multilayer perceptron instance.

    prune_ratio : float or list
        Number between 0 and 1 representing the percentage of weights
        to prune. If `list` then different ratio for each
        layer.

    method : str, {"l1", "random"}
        Pruning method to use.
    """
    if isinstance(prune_ratio, float):
        prune_ratios = [prune_ratio] * len(mlp.module_list)
    elif isinstance(prune_ratio, list):
        if len(prune_ratio) != len(mlp.module_list):
            raise ValueError("Incompatible number of prune ratios provided")

        prune_ratios = prune_ratio
    else:
        raise TypeError

    for prune_ratio, linear in zip(prune_ratios, mlp.module_list):
        prune_linear(linear, prune_ratio=prune_ratio, method=method)


def check_pruned_linear(linear):
    """Check if a Linear module was pruned.

    We require both the bias and the weight to be pruned.

    Parameters
    ----------
    linear : nn.Linear
        Linear module containing a bias.

    Returns
    -------
    bool
        True if the model has been pruned.
    """
    params = {param_name for param_name, _ in linear.named_parameters()}
    expected_params = {"weight_orig", "bias_orig"}

    return params == expected_params


def reinit_linear(linear):
    """Reinitialize a linear layer.

    This is an in-place operation.
    If the module has some pruning logic we are not going to remove it
    and we only initialize the underlying tensors - `weight_orig` and
    `bias_orig`.

    Parameters
    ----------
    linear : nn.Linear
        Linear model containing a bias.
    """
    is_pruned = check_pruned_linear(linear)

    # Get parameters of interest
    if is_pruned:
        weight = linear.weight_orig
        bias = linear.bias_orig
    else:
        weight = linear.weight
        bias = linear.bias

    # Initialize weight
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

    # Initialize bias
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    nn.init.uniform_(bias, -bound, bound)


def reinit_mlp(mlp):
    """Reinitialize all layers of the MLP.

    Parameters
    ----------
    mlp : MLP
        Multi-layer perceptron.
    """
    for linear in mlp.module_list:
        reinit_linear(linear)


def copy_weights_linear(linear_unpruned, linear_pruned):
    """Copy weights from an unpruned model to a pruned model.

    Modifies `linear_pruned` in place.

    Parameters
    ----------
    linear_unpruned : nn.Linear
        Linear model with a bias that was not pruned.

    linear_pruned : nn.Linear
        Linear model with a bias that was pruned.
    """
    assert check_pruned_linear(linear_pruned)
    assert not check_pruned_linear(linear_unpruned)

    with torch.no_grad():
        linear_pruned.weight_orig.copy_(linear_unpruned.weight)
        linear_pruned.bias_orig.copy_(linear_unpruned.bias)


def copy_weights_mlp(mlp_unpruned, mlp_pruned):
    """Copy weights of an unpruned network to a pruned network.

    Modifies `mlp_pruned` in place.

    Parameters
    ----------
    mlp_unpruned : MLP
        MLP model that was not pruned.

    mlp_pruned : MLP
        MLP model that was pruned.
    """
    zipped = zip(mlp_unpruned.module_list, mlp_pruned.module_list)

    for linear_unpruned, linear_pruned in zipped:
        copy_weights_linear(linear_unpruned, linear_pruned)


def compute_stats(mlp):
    """Compute important statistics related to pruning.

    Parameters
    ----------
    mlp : MLP
        Multilayer perceptron.

    Returns
    -------
    dict
        Statistics.
    """
    stats = {}
    total_params = 0
    total_pruned_params = 0

    for layer_ix, linear in enumerate(mlp.module_list):
        assert check_pruned_linear(linear)

        weight_mask = linear.weight_mask
        bias_mask = linear.bias_mask

        params = weight_mask.numel() + bias_mask.numel()
        pruned_params = (weight_mask == 0).sum() + (bias_mask == 0).sum()

        total_params += params
        total_pruned_params += pruned_params

        stats[f"layer{layer_ix}_total_params"] = params
        stats[f"layer{layer_ix}_pruned_params"] = pruned_params
        stats[f"layer{layer_ix}_actual_prune_ratio"] = pruned_params / params

    stats["total_params"] = total_params
    stats["total_pruned_params"] = total_pruned_params
    stats["actual_prune_ratio"] = total_pruned_params / total_params

    return stats
