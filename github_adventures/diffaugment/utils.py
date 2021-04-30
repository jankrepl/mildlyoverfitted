import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset


class DatasetImages(Dataset):
    """Dataset loading photos on the hard drive.

    Parameters
    ----------
    path : pathlib.Path
        Path to the folder containing all the images.

    transform : None or callable
        The transform to be applied when yielding the image.

    Attributes
    ----------
    all_paths : list
        List of all paths to the `.jpg` images.
    """
    def __init__(self, path, transform=None):
        super().__init__()

        self.all_paths = sorted([p for p in path.iterdir() if p.suffix == ".jpg"])
        self.transform = transform

    def __len__(self):
        """Compute length of the dataset."""
        return len(self.all_paths)

    def __getitem__(self, ix):
        """Get a single item."""
        img = Image.open(self.all_paths[ix])

        if self.transform is not None:
            img = self.transform(img)

        return img



class Generator(nn.Module):
    """Generator network.

    Parameters
    ----------
    latent_dim : int
        The dimensionality of the input noise.

    ngf : int
        Number of generator filters. Note that the actual number of filters
        will be a multiple of this number and is going to be divided by two in
        each consecutive block of the network.

    Attributes
    ----------
    main : torch.Sequential
        The actual network that is composed of `ConvTranspose2d`, `BatchNorm2d`
        and `ReLU` blocks.
    """

    def __init__(self, latent_dim, ngf=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # (ngf * 16) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # (ngf * 8) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # (ngf * 4) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # (ngf * 2) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # ngf x 64 x 64
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
            # 3 x 128 x 128
        )

    def forward(self, x):
        """Run the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input noise of shape `(n_samples, latent_dim)`.

        Returns
        -------
        torch.Tensor
            Generated images of shape `(n_samples, 3, 128, 128)`.
        """
        x = x.reshape(*x.shape, 1, 1)  # (n_samples, latent_dim, 1, 1)
        return self.main(x)


class Discriminator(nn.Module):
    """Discriminator netowrk.

    Parameters
    ----------
    ndf : int
        Number of discriminator filters. It represents the number of filters
        after the first convolution block. Each consecutive block will double
        the number.

    augment_module : nn.Module or None
        If provided it represents the Kornia module that performs
        differentiable augmentation of the images.

    Attributes
    ----------
    augment_module : nn.Module
        If the input parameter `augment_module` provided then this is the
        same thing. If not, then this is just an identity mapping.
    """
    def __init__(self, ndf=16, augment_module=None):
        super().__init__()
        self.main = nn.Sequential(
            # 3 x 128 x 128
            nn.Conv2d(3, ndf, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # ndf x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf * 2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf * 4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf * 8) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf * 16) x 4 x 4
            nn.Conv2d(ndf * 16, 1, 4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # 1 x 1 x 1
        )
        if augment_module is not None:
            self.augment_module = augment_module
        else:
            self.augment_module = nn.Identity()


    def forward(self, x):
        """Run the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input images of shape `(n_samples, 3, 128, 128)`.

        Returns
        -------
        torch.Tensor
            Classification outputs of shape `(n_samples, 1)`.
        """
        if self.training:
            x = self.augment_module(x)

        x = self.main(x)  # (n_samples, 1, 1, 1)
        x = x.reshape(len(x), -1)  # (n_samples, 1)
        return x


def init_weights_(module):
    """Initialize weights by sampling from a normal distribution.

    Note that this operation is modifying the weights in place.

    Parameters
    ----------
    module : nn.Module
        Module with trainable weights.
    """
    cls_name = module.__class__.__name__

    if cls_name in {"Conv2d", "ConvTranspose2d"}:
        nn.init.normal_(module.weight.data, 0.0, 0.02)

    elif cls_name == "BatchNorm2d":
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)
