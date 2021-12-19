from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Lambda, ToTensor


class MNISTDataset(Dataset):
    """MNIST dataset.

    Feature images are automatically flattened.

    Parameters
    ----------
    root : str
        Directory where the actual data is located (or downloaded to).

    train : bool
        If True the training set is returned (60_000 samples). Otherwise
        the validation set is returned (10_000 samples).

    Attributes
    ----------
    tv_dataset : MNIST
        Instance of the torchvision `MNIST` dataset class.
    """

    def __init__(self, root, train=True, download=True):
        transform = Compose(
            [
                ToTensor(),
                Lambda(lambda x: x.ravel()),
            ]
        )

        self.tv_dataset = MNIST(
            root,
            train=train,
            download=download,
            transform=transform,
        )

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.tv_dataset)

    def __getitem__(self, ix):
        """Get a selected sample.

        Parameters
        ----------
        ix : int
            Index of the sample to get.

        Returns
        -------
        x : torch.Tensor
            Flattened feature tensor of shape `(784,)`.

        y : torch.Tensor
            Scalar representing the ground truth label. Number between 0 and 9.
        """
        return self.tv_dataset[ix]
