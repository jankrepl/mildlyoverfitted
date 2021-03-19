import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import laplace, sobel
from torch.utils.data import Dataset


def paper_init_(weight, is_first=False, omega=1):
    """Initialize the weigth of the Linear layer.

    Parameters
    ----------
    weight : torch.Tensor
        The learnable 2D weight matrix.

    is_first : bool
        If True, this Linear layer is the very first one in the network.

    omega : float
        Hyperparamter.
    """
    in_features = weight.shape[1]

    with torch.no_grad():
        if is_first:
            bound = 1 / in_features
        else:
            bound = np.sqrt(6 / in_features) / omega

        weight.uniform_(-bound, bound)


class SineLayer(nn.Module):
    """Linear layer followed by the sine activation.

    Parameters
    ----------
    in_features : int
        Number of input features.

    out_features : int
        Number of output features.

    bias : bool
        If True, the bias is included.

    is_first : bool
        If True, then it represents the first layer of the network. Note that
        it influences the initialization scheme.

    omega : int
        Hyperparameter. Determines scaling.

    custom_init_function_ : None or callable
        If None, then we are going to use the `paper_init_` defined above.
        Otherwise, any callable that modifies the `weight` parameter in place.

    Attributes
    ----------
    linear : nn.Linear
        Linear layer.
    """
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            is_first=False,
            omega=30,
            custom_init_function_=None,
    ):
        super().__init__()
        self.omega = omega
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        if custom_init_function_ is None:
            paper_init_(self.linear.weight, is_first=is_first, omega=omega)
        else:
            custom_init_function_(self.linear.weight)

    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape `(n_samples, in_features)`.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(n_samples, out_features).
        """
        return torch.sin(self.omega * self.linear(x))

class ImageSiren(nn.Module):
    """Network composed of SineLayers.

    Parameters
    ----------
    hidden_features : int
        Number of hidden features (each hidden layer the same).

    hidden_layers : int
        Number of hidden layers.

    first_omega, hidden_omega : float
        Hyperparameter influencing scaling.

    custom_init_function_ : None or callable
        If None, then we are going to use the `paper_init_` defined above.
        Otherwise any callable that modifies the `weight` parameter in place.

    Attributes
    ----------
    net : nn.Sequential
        Sequential collection of `SineLayer` and `nn.Linear` at the end.
    """
    def __init__(
            self,
            hidden_features,
            hidden_layers=1,
            first_omega=30,
            hidden_omega=30,
            custom_init_function_=None,
            ):
        super().__init__()
        in_features = 2
        out_features = 1

        net = []
        net.append(
                SineLayer(
                    in_features,
                    hidden_features,
                    is_first=True,
                    custom_init_function_=custom_init_function_,
                    omega=first_omega,
            )
        )

        for _ in range(hidden_layers):
            net.append(
                    SineLayer(
                        hidden_features,
                        hidden_features,
                        is_first=False,
                        custom_init_function_=custom_init_function_,
                        omega=hidden_omega,
                )
            )

        final_linear = nn.Linear(hidden_features, out_features)
        if custom_init_function_ is None:
            paper_init_(final_linear.weight, is_first=False, omega=hidden_omega)
        else:
            custom_init_function_(final_linear.weight)

        net.append(final_linear)
        self.net = nn.Sequential(*net)


    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape `(n_samples, 2)` representing the 2D pixel coordinates.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(n_samples, 1)` representing the predicted
            intensities.
        """
        return self.net(x)


def generate_coordinates(n):
    """Generate regular grid of 2D coordinates on [0, n] x [0, n].

    Parameters
    ----------
    n : int
        Number of points per dimension.

    Returns
    -------
    coords_abs : np.ndarray
        Array of row and column coordinates of shape `(n ** 2, 2)`.
    """
    rows, cols = np.meshgrid(range(n), range(n), indexing="ij")
    coords_abs = np.stack([rows.ravel(), cols.ravel()], axis=-1)

    return coords_abs

class PixelDataset(Dataset):
    """Dataset yielding coordinates, intensitives and (higher) derivatives.

    Parameters
    ----------
    img : np.ndarray
        2D image representing a grayscale image.

    Attributes
    ----------
    size : int
        Height and width of the square image.

    coords_abs : np.ndarray
        Array of shape `(size ** 2, 2)` representing all coordinates of the
        `img`.

    grad : np.ndarray
        Array of shape `(size, size, 2)` representing the approximate
        gradient in the two directions.

    grad_norm : np.ndarray
        Array of shape `(size, size)` representing the approximate gradient
        norm of `img`.

    laplace : np.ndarray
        Array of shape `(size, size)` representing the approximate laplace operator.
    """
    def __init__(self, img):
        if not (img.ndim == 2 and img.shape[0] == img.shape[1]):
            raise ValueError("Only 2D square images are supported.")

        self.img = img
        self.size = img.shape[0]
        self.coords_abs = generate_coordinates(self.size)
        self.grad = np.stack([sobel(img, axis=0), sobel(img, axis=1)], axis=-1)
        self.grad_norm = np.linalg.norm(self.grad, axis=-1)
        self.laplace = laplace(img)

    def __len__(self):
        """Determine the number of samples (pixels)."""
        return self.size ** 2

    def __getitem__(self, idx):
        """Get all relevant data for a single coordinate."""
        coords_abs = self.coords_abs[idx]
        r, c = coords_abs

        coords = 2 * ((coords_abs / self.size) - 0.5)

        return {
            "coords": coords,
            "coords_abs": coords_abs,
            "intensity": self.img[r, c],
            "grad_norm": self.grad_norm[r, c],
            "grad": self.grad[r, c],
            "laplace": self.laplace[r, c],
        }


class GradientUtils:
    @staticmethod
    def gradient(target, coords):
        """Compute the gradient with respect to input.

        Parameters
        ----------
        target : torch.Tensor
            2D tensor of shape `(n_coords, ?)` representing the targets.

        coords : torch.Tensor
            2D tensor fo shape `(n_coords, 2)` representing the coordinates.

        Returns
        -------
        grad : torch.Tensor
            2D tensor of shape `(n_coords, 2)` representing the gradient.
        """
        return torch.autograd.grad(
            target, coords, grad_outputs=torch.ones_like(target), create_graph=True
        )[0]

    @staticmethod
    def divergence(grad, coords):
        """Compute divergence.

        Parameters
        ----------
        grad : torch.Tensor
            2D tensor of shape `(n_coords, 2)` representing the gradient wrt
            x and y.

        coords : torch.Tensor
            2D tensor of shape `(n_coords, 2)` representing the coordinates.

        Returns
        -------
        div : torch.Tensor
            2D tensor of shape `(n_coords, 1)` representing the divergence.

        Notes
        -----
        In a 2D case this will give us f_{xx} + f_{yy}.
        """
        div = 0.0
        for i in range(coords.shape[1]):
            div += torch.autograd.grad(
                grad[..., i], coords, torch.ones_like(grad[..., i]), create_graph=True,
            )[0][..., i : i + 1]
        return div

    @staticmethod
    def laplace(target, coords):
        """Compute laplace operator.

        Parameters
        ----------
        target : torch.Tensor
            2D tesnor of shape `(n_coords, 1)` representing the targets.

        coords : torch.Tensor
            2D tensor of shape `(n_coords, 2)` representing the coordinates.

        Returns
        -------
        torch.Tensor
            2D tensor of shape `(n_coords, 1)` representing the laplace.
        """
        grad = GradientUtils.gradient(target, coords)
        return GradientUtils.divergence(grad, coords)
