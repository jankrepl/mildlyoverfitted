import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.models as models

from utils import compute_gradient, read_image, scale_grad, to_array


def func(inp, net=None, target=None):
    """Get logit of a target class.

    Parameters
    ----------
    inp : torch.Tensor
        Input image (single image batch).

    net : torch.nn.Module
        Classifier network.

    target : int
        Imagenet ground truth label id.

    Returns
    -------
    logit : torch.Tensor
        Logit of the `target` class.
    """
    out = net(inp)
    logit = out[0, target]

    return logit

def compute_integrated_gradients(inp, baseline, net, target, n_steps=100):
    """Compute integrated gradients.

    Parameters
    ----------
    inp : torch.Tensor
        Input image (single image batch) of shape `(1, 3, *, *)`.

    baseline : torch.Tensor
        Basline image of the same shape as the `inp`.

    net : torch.nn.Module
        Classifier network.

    target : int
        Imagenet ground truth label id.

    n_steps : int
        Number of steps between the `inp` and `baseline` tensors.

    Returns
    -------
    ig : torch.Tensor
        Integrated gradients with the same shape as the `inp`.

    inp_grad : torch.Tensor
        Gradient with respect to the `inp` tensor. Same shape as `inp`.
    """
    path = [baseline + a * (inp - baseline) for a in np.linspace(0, 1, n_steps)]
    grads = [compute_gradient(func, x, net=net, target=target) for x in path]

    ig = (inp - baseline) * torch.cat(grads[:-1]).mean(dim=0, keepdims=True)

    return ig, grads[-1]

if __name__ == "__main__":
    net = models.resnet18(pretrained=True)
    net.eval()

    tensor = read_image("img.jpg")
    arr = to_array(tensor)

    n_steps = 100
    baseline = -1.5 * torch.ones_like(tensor)

    ig, inp_grad = compute_integrated_gradients(
            tensor, baseline, net, 291, n_steps=n_steps
    )

    ig_scaled = scale_grad(ig)
    inp_grad_scaled = scale_grad(inp_grad)

    _, (ax_baseline, ax_img, ax_inp_grad, ax_ig) = plt.subplots(1, 4, figsize=(19.20,10.80))

    ax_baseline.imshow(to_array(baseline))
    ax_img.imshow(arr)
    ax_inp_grad.imshow(arr * inp_grad_scaled)
    ax_ig.imshow(arr * ig_scaled)

    ax_baseline.set_title("Baseline")
    ax_img.set_title("Input")
    ax_inp_grad.set_title("Gradient input")
    ax_ig.set_title("Integrated gradients")

    ax_baseline.axis("off")
    ax_img.axis("off")
    ax_inp_grad.axis("off")
    ax_ig.axis("off")

    plt.savefig("res_2.png")
