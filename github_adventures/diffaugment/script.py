import argparse
import pathlib
import pprint
from datetime import datetime

import kornia.augmentation as K
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from utils import DatasetImages, Discriminator, Generator, init_weights_


def main(argv=None):
    # CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Name of the experiment")
    parser.add_argument(
        "-a",
        "--augment",
        action="store_true",
        help="If True, we apply augmentations",
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=16, help="Batch size"
    )
    parser.add_argument(
        "--b1",
        type=float,
        default=0.5,
        help="Adam optimizer hyperparamter",
    )
    parser.add_argument(
        "--b2",
        type=float,
        default=0.999,
        help="Adam optimizer hyperparamter",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use",
    )
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=400,
        help="Generate generator images every `eval_frequency` epochs",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=100,
        help="Dimensionality of the random noise",
    )
    parser.add_argument(
        "--lr", type=float, default=0.0002, help="Learning rate"
    )
    parser.add_argument(
        "--ndf",
        type=int,
        default=32,
        help="Number of discriminator feature maps (after first convolution)",
    )
    parser.add_argument(
        "--ngf",
        type=int,
        default=32,
        help="Number of generator feature maps (before last transposed convolution)",
    )
    parser.add_argument(
        "-n",
        "--n-epochs",
        type=int,
        default=200,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--mosaic-size",
        type=int,
        default=10,
        help="Size of the side of the rectangular mosaic",
    )
    parser.add_argument(
        "-p",
        "--prob",
        type=float,
        default=0.9,
        help="Probability of applying an augmentation",
    )

    args = parser.parse_args(argv)
    args_d = vars(args)
    print(args)

    img_size = 128

    # Additional parameters
    device = torch.device(args.device)
    mosaic_kwargs = {"nrow": args.mosaic_size, "normalize": True}
    n_mosaic_cells = args.mosaic_size * args.mosaic_size
    sample_showcase_ix = (
        0  # this one will be used to demonstrate the augmentations
    )

    augment_module = torch.nn.Sequential(
        K.RandomAffine(degrees=0, translate=(1 / 8, 1 / 8), p=args.prob),
        K.RandomErasing((0.0, 0.5), p=args.prob),
    )

    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator
    generator = Generator(latent_dim=args.latent_dim, ngf=args.ngf)
    discriminator = Discriminator(
        ndf=args.ndf, augment_module=augment_module if args.augment else None
    )

    generator.to(device)
    discriminator.to(device)

    # Initialize weights
    generator.apply(init_weights_)
    discriminator.apply(init_weights_)

    # Configure data loader
    data_path = pathlib.Path("data")
    tform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    dataset = DatasetImages(
        data_path,
        transform=tform,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Optimizers
    optimizer_G = torch.optim.Adam(
        generator.parameters(), lr=args.lr, betas=(args.b1, args.b2)
    )
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2)
    )

    # Output path and metadata
    output_path = pathlib.Path("outputs") / args.name
    output_path.mkdir(exist_ok=True, parents=True)

    # Add other parameters (not included in CLI)
    args_d["time"] = datetime.now()
    args_d["kornia"] = str(augment_module)

    # Prepare tensorboard writer
    writer = SummaryWriter(output_path)

    # Log hyperparameters as text
    writer.add_text(
        "hyperparameter",
        pprint.pformat(args_d).replace(
            "\n", "  \n"
        ),  # markdown needs 2 spaces before newline
        0,
    )
    # Log true data
    writer.add_image(
        "true_data",
        make_grid(
            torch.stack([dataset[i] for i in range(n_mosaic_cells)]),
            **mosaic_kwargs
        ),
        0,
    )
    # Log augmented data
    batch_showcase = dataset[sample_showcase_ix][None, ...].repeat(
        n_mosaic_cells, 1, 1, 1
    )
    batch_showcase_aug = discriminator.augment_module(batch_showcase)
    writer.add_image(
        "augmentations", make_grid(batch_showcase_aug, **mosaic_kwargs), 0
    )

    # Prepate evaluation noise
    z_eval = torch.randn(n_mosaic_cells, args.latent_dim).to(device)

    for epoch in tqdm(range(args.n_epochs)):
        for i, imgs in enumerate(dataloader):
            n_samples, *_ = imgs.shape
            batches_done = epoch * len(dataloader) + i

            # Adversarial ground truths
            valid = 0.9 * torch.ones(
                n_samples, 1, device=device, dtype=torch.float32
            )
            fake = torch.zeros(n_samples, 1, device=device, dtype=torch.float32)

            # D preparation
            optimizer_D.zero_grad()

            # D loss on reals
            real_imgs = imgs.to(device)
            d_x = discriminator(real_imgs)
            real_loss = adversarial_loss(d_x, valid)
            real_loss.backward()

            # D loss on fakes
            z = torch.randn(n_samples, args.latent_dim).to(device)
            gen_imgs = generator(z)
            d_g_z1 = discriminator(gen_imgs.detach())

            fake_loss = adversarial_loss(d_g_z1, fake)
            fake_loss.backward()

            optimizer_D.step()  # we called backward twice, the result is a sum

            # G preparation
            optimizer_G.zero_grad()

            # G loss
            d_g_z2 = discriminator(gen_imgs)
            g_loss = adversarial_loss(d_g_z2, valid)

            g_loss.backward()
            optimizer_G.step()

            # Logging
            if batches_done % 50 == 0:
                writer.add_scalar("d_x", d_x.mean().item(), batches_done)
                writer.add_scalar("d_g_z1", d_g_z1.mean().item(), batches_done)
                writer.add_scalar("d_g_z2", d_g_z2.mean().item(), batches_done)
                writer.add_scalar(
                    "D_loss", (real_loss + fake_loss).item(), batches_done
                )
                writer.add_scalar("G_loss", g_loss.item(), batches_done)

            if epoch % args.eval_frequency == 0 and i == 0:
                generator.eval()
                discriminator.eval()

                # Generate fake images
                gen_imgs_eval = generator(z_eval)

                # Generate nice mosaic
                writer.add_image(
                    "fake",
                    make_grid(gen_imgs_eval.data, **mosaic_kwargs),
                    batches_done,
                )

                # Save checkpoint (and potentially overwrite an existing one)
                torch.save(generator, output_path / "model.pt")

                # Make sure generator and discriminator in the training mode
                generator.train()
                discriminator.train()


if __name__ == "__main__":
    main()
