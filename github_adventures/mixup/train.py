import argparse
import json

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import (
    CustomDataset,
    MLPClassifierMixup,
    generate_prediction_img,
    generate_spirals,
)


def main(argv=None):
    parser = argparse.ArgumentParser("Training")

    # Parameters
    parser.add_argument(
        "logpath",
        type=str,
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--mixup",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--dropout-probability",
        type=float,
        default=0,
        help="The probability of dropout",
    )
    parser.add_argument(
        "--hidden-dims",
        nargs="+",
        type=int,
        default=(32, 32, 32),
        help="Hidden dimensions of the MLP",
    )
    parser.add_argument(
        "-c",
        "--n-cycles",
        type=float,
        default=2,
        help="Number of cycles when creating the spiral dataset",
    )
    parser.add_argument(
        "-n",
        "--n-epochs",
        type=int,
        default=100,
        help="Number of epochs",
    )
    parser.add_argument(
        "-k",
        "--mixing-layer",
        type=int,
        nargs=2,
        default=(None, None),
        help="The range of k to sample from",
    )
    parser.add_argument(
        "-s",
        "--n-samples",
        type=int,
        default=1000,
        help="Number of samples",
    )
    parser.add_argument(
        "-r",
        "--random-state",
        type=int,
        default=5,
        help="Random state",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay",
    )

    args = parser.parse_args(argv)

    device = torch.device("cpu")
    dtype = torch.float32

    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)

    # Dataset preparation
    X, y = generate_spirals(
        args.n_samples,
        noise_std=0,
        n_cycles=args.n_cycles,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.9,
        shuffle=True,
        stratify=y,
    )

    X_test_t = torch.from_numpy(X_test).to(device, dtype)

    dataset_train = CustomDataset(X_train, y_train)

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=2 * args.batch_size,
        drop_last=True,
        shuffle=True,
    )

    # Model and loss definition
    model = MLPClassifierMixup(
        n_features=2,
        hidden_dims=tuple(args.hidden_dims),
        p=args.dropout_probability,
    )
    model.to(device, dtype)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        weight_decay=args.weight_decay,
    )

    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Summary
    writer = SummaryWriter(args.logpath)
    writer.add_text("hparams", json.dumps(vars(args)))

    # Training + evaluation loop
    bs = args.batch_size
    n_steps = 0
    for e in range(args.n_epochs):
        for X_batch, y_batch in dataloader_train:
            X_batch, y_batch = X_batch.to(device, dtype), y_batch.to(
                device, dtype
            )
            if args.mixup:
                k_min, k_max = args.mixing_layer
                k_min = k_min or 0
                k_max = k_max or model.n_hidden + 1

                k = np.random.randint(k_min, k_max)
                lam = np.random.beta(2, 2)
                writer.add_scalar("k", k, n_steps)
                writer.add_scalar("lambda", lam, n_steps)

                h = model(X_batch, start=0, end=k)  # (2 * batch_size, *)

                h_mixed = lam * h[:bs] + (1 - lam) * h[bs:]  # (batch_size, *)
                y_mixed = lam * y_batch[:bs] + (1 - lam) * y_batch[bs:]  # (batch_size,)

                logits = model(h_mixed, start=k, end=None)  # (batch_size, 1)
                loss = loss_fn(logits.squeeze(), y_mixed)

            else:
                logits = model(X_batch[:bs])  # (batch_size, 1)
                loss = loss_fn(logits.squeeze(), y_batch[:bs])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            writer.add_scalar("loss_train", loss, n_steps)

            if n_steps % 2500 == 0:
                model.eval()
                fig_gen = generate_prediction_img(
                    model,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                )
                writer.add_figure("test", next(fig_gen))
                writer.add_figure("contour", next(fig_gen), n_steps)
                writer.add_figure("contour_train", next(fig_gen), n_steps)

                with torch.no_grad():
                    logits_test = model(X_test_t).squeeze().detach().cpu()

                acc_test = (
                    torch.sigmoid(logits_test).round().numpy() == y_test
                ).sum() / len(y_test)
                loss_test = loss_fn(logits_test, torch.from_numpy(y_test))

                writer.add_scalar("loss_test", loss_test, n_steps)
                writer.add_scalar("accuracy_test", acc_test, n_steps)

                model.train()

            n_steps += 1


if __name__ == "__main__":
    main()
