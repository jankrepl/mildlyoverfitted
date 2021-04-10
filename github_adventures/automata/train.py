import argparse
import pathlib

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import CAModel


def load_image(path, size=40):
    """Load an image.

    Parameters
    ----------
    path : pathlib.Path
        Path to where the image is located. Note that the image needs to be
        RGBA.

    size : int
        The image will be resized to a square wit ha side length of `size`.

    Returns
    -------
    torch.Tensor
        4D float image of shape `(1, 4, size, size)`. The RGB channels
        are premultiplied by the alpha channel.
    """
    img = Image.open(path)
    img = img.resize((size, size), Image.ANTIALIAS)
    img = np.float32(img) / 255.0
    img[..., :3] *= img[..., 3:]

    return torch.from_numpy(img).permute(2, 0, 1)[None, ...]


def to_rgb(img_rgba):
    """Convert RGBA image to RGB image.

    Parameters
    ----------
    img_rgba : torch.Tensor
        4D tensor of shape `(1, 4, size, size)` where the RGB channels
        were already multiplied by the alpha.

    Returns
    -------
    img_rgb : torch.Tensor
        4D tensor of shape `(1, 3, size, size)`.
    """
    rgb, a = img_rgba[:, :3, ...], torch.clamp(img_rgba[:, 3:, ...], 0, 1)
    return torch.clamp(1.0 - a + rgb, 0, 1)


def make_seed(size, n_channels):
    """Create a starting tensor for training.

    The only active pixels are going to be in the middle.

    Parameters
    ----------
    size : int
        The height and the width of the tensor.

    n_channels : int
        Overall number of channels. Note that it needs to be higher than 4
        since the first 4 channels represent RGBA.

    Returns
    -------
    torch.Tensor
        4D float tensor of shape `(1, n_chanels, size, size)`.
    """
    x = torch.zeros((1, n_channels, size, size), dtype=torch.float32)
    x[:, 3:, size // 2, size // 2] = 1
    return x


def main(argv=None):
    parser = argparse.ArgumentParser(
            description="Training script for the Celluar Automata"
    )
    parser.add_argument("img", type=str, help="Path to the image we want to reproduce")

    parser.add_argument(
            "-b",
            "--batch-size",
            type=int,
            default=8,
            help="Batch size. Samples will always be taken randomly from the pool."
    )
    parser.add_argument(
            "-d",
            "--device",
            type=str,
            default="cpu",
            help="Device to use",
            choices=("cpu", "cuda"),
    )
    parser.add_argument(
            "-e",
            "--eval-frequency",
            type=int,
            default=500,
            help="Evaluation frequency.",
    )
    parser.add_argument(
            "-i",
            "--eval-iterations",
            type=int,
            default=300,
            help="Number of iterations when evaluating.",
    )
    parser.add_argument(
            "-n",
            "--n-batches",
            type=int,
            default=5000,
            help="Number of batches to train for.",
    )
    parser.add_argument(
            "-c",
            "--n-channels",
            type=int,
            default=16,
            help="Number of channels of the input tensor",
    )
    parser.add_argument(
            "-l",
            "--logdir",
            type=str,
            default="logs",
            help="Folder where all the logs and outputs are saved.",
    )
    parser.add_argument(
            "-p",
            "--padding",
            type=int,
            default=16,
            help="Padding. The shape after padding is (h + 2 * p, w + 2 * p).",
    )
    parser.add_argument(
            "--pool-size",
            type=int,
            default=1024,
            help="Size of the training pool",
    )
    parser.add_argument(
            "-s",
            "--size",
            type=int,
            default=40,
            help="Image size",
    )
    # Parse arguments
    args = parser.parse_args()
    print(vars(args))

    # Misc
    device = torch.device(args.device)

    log_path = pathlib.Path(args.logdir)
    log_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_path)

    # Target image
    target_img_ = load_image(args.img, size=args.size)
    p = args.padding
    target_img_ = nn.functional.pad(target_img_, (p, p, p, p), "constant", 0)
    target_img = target_img_.to(device)
    target_img = target_img.repeat(args.batch_size, 1, 1, 1)

    writer.add_image("ground truth", to_rgb(target_img_)[0])

    # Model and optimizer
    model = CAModel(n_channels=args.n_channels, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    # Pool initialization
    seed = make_seed(args.size, args.n_channels).to(device)
    seed = nn.functional.pad(seed, (p, p, p, p), "constant", 0)
    pool = seed.clone().repeat(args.pool_size, 1, 1, 1)

    for it in tqdm(range(args.n_batches)):
        batch_ixs = np.random.choice(
                args.pool_size, args.batch_size, replace=False
        ).tolist()

        x = pool[batch_ixs]
        for i in range(np.random.randint(64, 96)):
            x = model(x)

        loss_batch = ((target_img - x[:, :4, ...]) ** 2).mean(dim=[1, 2, 3])
        loss = loss_batch.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar("train/loss", loss, it)

        argmax_batch = loss_batch.argmax().item()
        argmax_pool = batch_ixs[argmax_batch]
        remaining_batch = [i for i in range(args.batch_size) if i != argmax_batch]
        remaining_pool = [i for i in batch_ixs if i != argmax_pool]

        pool[argmax_pool] = seed.clone()
        pool[remaining_pool] = x[remaining_batch].detach()

        if it % args.eval_frequency == 0:
            x_eval = seed.clone()  # (1, n_channels, size, size)

            eval_video = torch.empty(1, args.eval_iterations, 3, *x_eval.shape[2:])

            for it_eval in range(args.eval_iterations):
                x_eval = model(x_eval)
                x_eval_out = to_rgb(x_eval[:, :4].detach().cpu())
                eval_video[0, it_eval] = x_eval_out

            writer.add_video("eval", eval_video, it, fps=60)


if __name__ == "__main__":
    main()
