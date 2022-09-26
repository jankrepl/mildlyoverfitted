import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multilayer perceptron policy network.

    Parameters
    ----------
    n_features : int
        Number of input features.

    hidden_layer_sizes : tuple
        Tuple of int that defines the sizes of all hidden layers.

    Attributes
    ----------
    net : nn.Sequential
        The actual network.
    """

    def __init__(self, n_features, hidden_layer_sizes):
        super().__init__()

        layer_sizes = (n_features,) + hidden_layer_sizes + (1,)

        layers = []

        for i in range(len(layer_sizes) - 1):
            in_features = layer_sizes[i]
            out_features = layer_sizes[i + 1]
            layers.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.Tanh(),
                ]
            )

        self.net = nn.Sequential(*layers)

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, obs):
        """Run forward pass.

        Parameters
        ----------
        obs : torch.Tensor
            1D tensor representing the input observation of shape
            `(n_features,)`.

        Returns
        -------
        torch.Tensor
            Scalar between -1 and 1 representing the action.
        """

        return self.net(obs[None, :])[0]


def pos_table(n_embeddings, hidden_size):
    """Create a table of positional encodings.

    Parameters
    ----------
    n_embeddings : int
        Number of rows of the table.

    hidden_size : int
        Number of columns of the table.

    Returns
    -------
    tab : np.ndarray
        2D array holding the positional encodings.
    """

    def get_angle(x, h):
        return x / np.power(10000, 2 * (h // 2) / hidden_size)

    def get_angle_vec(x):
        return [get_angle(x, j) for j in range(hidden_size)]

    tab = np.array([get_angle_vec(i) for i in range(n_embeddings)]).astype(
        float
    )
    tab[:, 0::2] = np.sin(tab[:, 0::2])
    tab[:, 1::2] = np.cos(tab[:, 1::2])

    return tab


class AttentionMatrix(nn.Module):
    """Generates attention matrix using the key and query tensors.

    Parameters
    ----------
    proj_dim : int
        Size of the space to which we project the K and Q tensors.

    hidden_size : int
        Dimensionality of the Q and K tensors before linear projections.

    scale : bool
        If True, then the attention matrix will be divided by
        `proj_dim ** (1 / 2)` elementwise.

    Attributes
    ----------
    proj_q, proj_k : torch.nn.Linear
        Linear models projecting the Q and K tensors.

    scalar : float
        Number used for scaling the attention matrix elementwise.
    """

    def __init__(self, hidden_size, proj_dim, scale=True):
        super().__init__()

        self.proj_q = nn.Linear(
            in_features=hidden_size, out_features=proj_dim, bias=False
        )
        self.proj_k = nn.Linear(
            in_features=hidden_size, out_features=proj_dim, bias=False
        )
        if scale:
            self.scalar = np.sqrt(proj_dim)
        else:
            self.scalar = 1

    def forward(self, data_q, data_k):
        """Run the forward pass.

        Parameters
        ----------
        data_q : torch.Tensor
            Query tensor of shape `(n_embeddings, hidden_size)`.

        data_k : torch.Tensor
            Key tensor of shape `(n_features, hidden_size)`.

        Returns
        -------
        attention_weights : torch.Tensor
            Attention weights (don't sum up to 1 in general) of shape
            `(n_embeddings, n_features)`.
        """
        q = self.proj_q(data_q)  # (n_embeddings, proj_dim)
        k = self.proj_k(data_k)  # (n_features, proj_dim)
        dot = q @ k.T  # (n_embeddings, n_features)
        dot_scaled = torch.div(dot, self.scalar)  # (n_embeddings, n_features)
        attention_weights = torch.tanh(
            dot_scaled
        )  # (n_embeddings, n_features)

        return attention_weights


class AttentionNeuron(nn.Module):
    """Permutation invariant layer.

    Parameters
    ----------
    n_embeddings : int
        Number of rows in the Q tensor. In our case it is equal to the length
        of the latent code `m`.

    proj_dim : int
        Size of the space to which we project the K and Q tensors.

    hidden_size : int
        The dimensionality of the Q and K tensors before linear projections.

    Attributes
    ----------
    hx : tuple or None
        If not None then a tuple of 2 hidden state tensors (LSTM specific)

    lstm : nn.LSTMCell
        LSTM cell that inputs a hidden state and an observation and
        outputs a new hidden state.

    attention_matrix : AttentionMatrix
        Attention matrix (only needs Q and K tensors).

    Q : torch.Tensor
        Query tensor that is not learnable since it is populated with
        positional encodings.
    """

    def __init__(
            self,
            n_embeddings=16,
            proj_dim=32,
            hidden_size=8,
    ):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.proj_dim = proj_dim
        self.hidden_size = hidden_size

        # Modules
        self.hx = None
        self.lstm = nn.LSTMCell(input_size=2, hidden_size=hidden_size)

        self.attention_matrix = AttentionMatrix(
            hidden_size=hidden_size,
            proj_dim=proj_dim,
            scale=False,
        )

        self.register_buffer(
            "Q",
            torch.from_numpy(
                pos_table(
                    n_embeddings,
                    hidden_size,
                )
            ).float(),
        )

    def forward(self, obs, prev_action):
        """Run forward pass.

        Parameters
        ----------
        obs : torch.Tensor
            1D tensor representing the input observations of shape
            `(n_features,)`.

        prev_action : float
            Number between -1 and 1 based on what the previous action was.

        Returns
        -------
        latent_code : torch.Tensor
            1D tensor representing the latent code of shape `(n_embeddings,)`.

        attn_weights : torch.Tensor
            2D tensor of shape `(n_embeddings, n_features)` representing
            attention weights.
        """
        n_features = len(obs)
        prev_action = float(prev_action)

        obs_and_act = torch.cat(
            [
                obs[:, None],
                torch.ones(n_features, 1) * prev_action,
            ],
            dim=-1,
        )  # (n_features, 2)

        if self.hx is None:
            self.hx = (
                torch.zeros(n_features, self.hidden_size),
                torch.zeros(n_features, self.hidden_size),
            )

        self.hx = self.lstm(
            obs_and_act, self.hx
        )  # Tuple[(n_features, hidden_size)]

        data_q = self.Q  # (n_embeddings, hidden_size)
        data_k = self.hx[0]  # (n_features, hidden_size)
        data_v = obs[:, None]  # (n_features, 1)

        attn_weights = self.attention_matrix(
            data_q=data_q, data_k=data_k
        )  # (n_embeddings, n_features)

        latent_code_ = torch.tanh(attn_weights @ data_v)  # (n_embeddings, 1)
        latent_code = latent_code_.squeeze()  # (n_embeddings,)

        return latent_code, attn_weights


class PermutationInvariantNetwork(nn.Module):
    """Permutation invariant policy network.

    Parameters
    ----------
    n_embeddings : int
        Number of rows in the Q tensor.

    proj_dim : int
        Size of the space to which we project the K and Q tensors.

    hidden_size : int
        Dimensionality of the Q and K matrices before linear projections.

    Attributes
    ----------
    attention_neuron : AttentionNeuron
        Permutation invariant layer that generates latent codes.

    linear : nn.Linear
        Maps the latent code into a single number.
    """

    def __init__(
            self,
            n_embeddings=16,
            proj_dim=32,
            hidden_size=8,
    ):
        super().__init__()

        self.attention_neuron = AttentionNeuron(
            n_embeddings=n_embeddings,
            proj_dim=proj_dim,
            hidden_size=hidden_size,
        )

        self.linear = nn.Linear(n_embeddings, 1)

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, obs, prev_action):
        """Run forward pass.

        Parameters
        ----------
        obs : torch.Tensor
            1D tensor representing the input observations of shape
            `(n_features,)`.

        prev_action : float
            Number between -1 and 1 based on what the previous action was.

        Returns
        -------
        y : torch.Tensor
            Scalar tensor with a value in range (-1, 1) representing the
            next action.
        """

        latent_code, _ = self.attention_neuron(
            obs, prev_action
        )  # (n_embeddings,)

        y_ = torch.tanh(self.linear(latent_code[None, :]))  # (1, 1)
        y = y_[0]  # (1,)

        return y
