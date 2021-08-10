import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.colors import ListedColormap
from torch.utils.data import Dataset


class MLPClassifierMixup(nn.Module):
    """Multilayer perceptron with inbuilt mixup logic.

    Assuming binary classification.

    Parameters
    ----------
    n_features : int
        Number of features.

    hidden_dims : tuple
        The sizes of the hidden layers.

    p : float
        Dropout probability.

    Attributes
    ----------
    hidden_layers : nn.ModuleList
        List of hidden layers that are each composed of a `Linear`,
        `LeakyReLU` and `Dropout` modules.

    n_hidden : int
        Number of hidden layers.

    clf : nn.Linear
        The classifier at the end of the pipeline.
    """

    def __init__(self, n_features, hidden_dims, p=0):
        super().__init__()
        dims = (n_features,) + hidden_dims

        self.n_hidden = len(hidden_dims)
        self.hidden_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dims[i], dims[i + 1]),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(p),
                )
                for i in range(self.n_hidden)
            ]
        )
        self.clf = nn.Linear(dims[-1], 1)

    def forward(self, x, start=0, end=None):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape `(n_samples, dim)`. Note that the dim
            will depend on `start`.

        start : int
            The hidden layer where the forward pass starts (inclusive). We
            use a convention of `start=0` and `end=0` as a noop and the input
            tensor is returned. Useful for implementing input mixing.

        end : int or None
            The ending hidden layer (exclusive). If None, then always run until
            the last hidden layer and then we also apply the classifier.
        """
        for module in self.hidden_layers[start:end]:
            x = module(x)

        if end is None:
            x = self.clf(x)

        return x


class CustomDataset(Dataset):
    """Custom classification dataset assuming we have X and y loaded in memory.

    Parameters
    ----------
    X : np.ndarray
        Features of shape `(n_samples, n_features)`.

    y : np.ndarray
        Targets of shape `(n_samples,)`.
    """

    def __init__(self, X, y):
        if len(X) != len(y):
            raise ValueError("Inconsistent number of samples")

        classes = np.unique(y)
        if not np.array_equal(np.sort(classes), np.array([0, 1])):
            raise ValueError

        self.X = X
        self.y = y

    def __len__(self):
        """Compute the length of the dataset."""
        return len(self.X)

    def __getitem__(self, ix):
        """Return a single sample."""
        return self.X[ix], self.y[ix]


def generate_spirals(
    n_samples,
    noise_std=0.05,
    n_cycles=2,
    random_state=None,
):
    """Generate two spirals dataset.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate. For simplicity, an even number
        is required. The targets (2 spirals) are perfectly balanced.

    noise_std : float
        Standard deviation of the noise added to the spirals.

    n_cycles : int
        Number of revolutions the spirals make.

    random_state : int or None
        Controls randomness.

    Returns
    -------
    X : np.ndarray
        Features of shape `(n_samples, n_features)`.

    y : np.ndarray
        Targets of shape `(n_samples,)`. There are two
        classes 0 and 1 representing the two spirals.
    """
    if n_samples % 2 != 0:
        raise ValueError("The number of samples needs to be even")

    n_samples_per_class = int(n_samples // 2)

    angle_1 = np.linspace(0, n_cycles * 2 * np.pi, n_samples_per_class)
    angle_2 = np.pi + angle_1
    radius = np.linspace(0.2, 2, n_samples_per_class)

    x_1 = radius * np.cos(angle_1)
    y_1 = radius * np.sin(angle_1)

    x_2 = radius * np.cos(angle_2)
    y_2 = radius * np.sin(angle_2)

    X = np.concatenate(
        [
            np.stack([x_1, y_1], axis=1),
            np.stack([x_2, y_2], axis=1),
        ],
        axis=0,
    )
    y = np.zeros((n_samples,))
    y[n_samples_per_class:] = 1.0

    if random_state is not None:
        np.random.seed(random_state)

    new_ixs = np.random.permutation(n_samples)

    X = X[new_ixs] + np.random.normal(
        loc=0, scale=noise_std, size=(n_samples, 2)
    )
    y = y[new_ixs]

    return X, y


def generate_prediction_img(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
):
    """Generate contour and scatter plots with predictions.

    Parameters
    ----------
    model : MLPClassifierMixup
        Instance of a multilayer-perceptron.

    X_train, X_test : np.ndarray
        Trand and test features of shape `(n_samples, n_features)`.

    y_train, y_test : np.ndarray
        Train and test targets of shape `(n_samples,)`.

    Yields
    ------
    matplotlib.Figure
        Different figures.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])

    delta = 0.5

    xlim = (X_test[:, 0].min() - delta, X_test[:, 0].max() + delta)
    ylim = (X_test[:, 1].min() - delta, X_test[:, 1].max() + delta)

    n = 50
    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], n),
        np.linspace(ylim[0], ylim[1], n),
    )
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)

    with torch.no_grad():
        logits = model(torch.from_numpy(grid).to(device, dtype))

    probs = torch.sigmoid(logits)[:, 0].detach().cpu().numpy()

    probs = probs.reshape(xx.shape)

    fig, ax = plt.subplots(1, 1, dpi=170)

    ax.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors="k"
    )
    ax.set_title("Test data")

    yield fig
    ax.cla()

    ax.contourf(xx, yy, probs, cmap=cm, alpha=0.8)
    ax.set_title("Prediction contours")

    yield fig

    ax.scatter(
        X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
    )
    ax.set_title("Train data + prediction contours")

    yield fig
