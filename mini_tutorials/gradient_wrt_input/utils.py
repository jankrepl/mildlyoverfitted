import torch
from PIL import Image
from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize,
                                    ToTensor)


def compute_gradient(func, inp, **kwargs):
    """Compute the gradient with respect to `inp`.

    Parameters
    ----------
    func : callable
        Function that takes in `inp` and `kwargs` and returns a single element
        tensor.

    inp : torch.Tensor
        The tensor that we want to get the gradients for. Needs to be a leaf
        node.

    **kwargs : dict
        Additional keyword arguments passed into `func`.

    Returns
    -------
    grad : torch.Tensor
        Tensor of the same shape as `inp` that is representing the gradient.
    """
    inp.requires_grad = True

    loss = func(inp, **kwargs)
    loss.backward()

    inp.requires_grad = False

    return inp.grad.data


def read_image(path):
    """Load image from disk and convert to torch.Tensor.

    Parameters
    ----------
    path : str
        Path to the image.

    Returns
    -------
    tensor : torch.Tensor
        Single sample batch containing our image (ready to be used with
        pretrained networks). The shape is `(1, 3, 224, 224)`.
    """
    img = Image.open(path)

    transform = Compose([Resize(256),
                         CenterCrop(224),
                         ToTensor(),
                         Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])])

    tensor_ = transform(img)
    tensor = tensor_.unsqueeze(0)

    return tensor


def to_array(tensor):
    """Convert torch.Tensor to np.ndarray.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor of shape `(1, 3, *, *)` representing one sample batch of images.

    Returns
    -------
    arr : np.ndarray
        Array of shape `(*, *, 3)` representing an image that can be plotted
        directly.
    """
    tensor_ = tensor.squeeze()

    unnormalize_transform = Compose([Normalize(mean=[0, 0, 0],
                                               std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                     Normalize(mean=[-0.485, -0.456, -0.406],
                                               std=[1, 1, 1])])
    arr_ = unnormalize_transform(tensor_)
    arr = arr_.permute(1, 2, 0).detach().numpy()

    return arr


def scale_grad(grad):
    """Scale gradient tensor.

    Parameters
    ----------
    grad : torch.Tensor
        Gradient of shape `(1, 3, *, *)`.

    Returns
    -------
    grad_arr : np.ndarray
        Array of shape `(*, *, 1)`.
    """
    grad_arr = torch.abs(grad).mean(dim=1).detach().permute(1, 2, 0)
    grad_arr /= grad_arr.quantile(0.98)
    grad_arr = torch.clamp(grad_arr, 0, 1)

    return grad_arr.numpy()
