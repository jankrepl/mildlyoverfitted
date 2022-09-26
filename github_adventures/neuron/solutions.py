import abc

import numpy as np
import torch

from torch_utils import PermutationInvariantNetwork, MLP


class Solution(abc.ABC):
    """Solution abstract class.

    Attributes
    ----------
    policy : torch.nn.Module
        Network that holds all the learnable parameters.
    """

    @abc.abstractmethod
    def clone(self, obs):
        """Create a copy of the current solution without any links to self."""

    @abc.abstractmethod
    def get_action(self, obs):
        """Determine the next action given the observation array."""

    @abc.abstractmethod
    def get_n_features(self):
        """Get the number of features expected by the model.

        If None then the model can process variable-sized feature
        vectors.
        """

    @abc.abstractmethod
    def reset(self):
        """Reset solution.

        Will be called at the beginning of each rollout.

        Does not mean we will "reinitialize" the weights of `policy`.
        """

    def get_params(self):
        """Get learnable parameters of the solution.

        Returns
        -------
        params : np.ndarray
            1D array containing all parameters.
        """
        params_l = []

        for p in self.policy.parameters():
            params_l.append(p.numpy().ravel())

        params = np.concatenate(params_l)

        return params

    def set_params(self, params):
        """Set the learnable parameters.

        Parameters
        ----------
        params : np.ndarray
            1D array containing all parameters.

        Returns
        -------
        self : Solution
        """
        start_ix, end_ix = 0, 0

        for p in self.policy.parameters():
            end_ix = start_ix + np.prod(p.shape)
            p.data = torch.from_numpy(
                params[start_ix:end_ix].reshape(p.shape)
            ).float()
            start_ix = end_ix

        return self

    def get_n_params(self):
        return len(self.get_params())


class MLPSolution(Solution):
    """Multilayer perceptron solution.

    Parameters
    ----------
    n_features : int
        Number of input features.

    hidden_layer_sizes : tuple
        Tuple of int that defines the sizes of all hidden layers.

    Attributes
    ----------
    kwargs : dict
        All parameters necessary to instantiate the class.

    policy : MLP
        Policy network - multilayer perceptron.
    """

    def __init__(self, n_features=5, hidden_layer_sizes=(16,)):
        self.kwargs = {
            "n_features": n_features,
            "hidden_layer_sizes": hidden_layer_sizes,
        }
        self.dtype = torch.float32

        self.policy = MLP(n_features, hidden_layer_sizes)
        self.policy.to(self.dtype)
        self.policy.eval()

    def clone(self):
        old_policy = self.policy
        new_solution = self.__class__(**self.kwargs)

        new_solution.policy.load_state_dict(
            old_policy.state_dict(),
        )

        return new_solution

    def get_action(self, obs):
        y = self.policy(torch.from_numpy(obs).to(self.dtype))

        action = y.item()
        return action

    def get_n_features(self):
        return self.kwargs["n_features"]

    def reset(self):
        pass


class PermutationInvariantSolution(Solution):
    """Permutation invariant solution.

    Parameters
    ----------
    n_embeddings : int
        Number of rows in the Q tensor.

    proj_dim : int
        Size of the space to which we project the K and Q tensors.

    hidden_size : int
        Dimensionality of the Q and K tensors before linear projections.

    Attributes
    ----------
    kwargs : dict
        All parameters necessary to instantiate the class

    dtype : torch.dtype
        Dtype of both the network weights and input features.

    policy : PermutationInvariantNetwork
        Policy network.

    prev_action : float
        Stores the previous action. Automatically updated each time we call
        `get_action`.
    """

    def __init__(
            self,
            n_embeddings=16,
            proj_dim=32,
            hidden_size=8,
    ):
        self.kwargs = {
            "n_embeddings": n_embeddings,
            "proj_dim": proj_dim,
            "hidden_size": hidden_size,
        }
        self.policy = PermutationInvariantNetwork(
            n_embeddings=n_embeddings,
            proj_dim=proj_dim,
            hidden_size=hidden_size,
        )
        self.dtype = torch.float32

        self.policy.to(self.dtype)
        self.policy.eval()

        self.prev_action = 0  # will be continuously updated

    def clone(self):
        old_policy = self.policy
        new_solution = self.__class__(**self.kwargs)

        new_solution.policy.load_state_dict(
            old_policy.state_dict(),
        )

        return new_solution

    def get_action(self, obs):
        y = self.policy(torch.from_numpy(obs).to(self.dtype), self.prev_action)

        action = y.item()
        self.prev_action = action

        return action

    def reset(self):
        self.policy.attention_neuron.hx = None
        self.previous_action = 0

    def get_n_features(self):
        return None
