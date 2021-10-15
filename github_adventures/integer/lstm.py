import argparse
import json
import pathlib
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import (
    CustomDataset,
    Network,
    create_classification_targets,
    train_classifier,
)


def main(argv=None):
    parser = argparse.ArgumentParser("Embedding integers using LSTM")

    parser.add_argument(
        "data_path", type=str, help="Path to the pickled sequences"
    )

    parser.add_argument(
        "log_folder", type=str, help="Folder where to log results"
    )

    parser.add_argument(
        "-b", "--batch-size", type=int, default=128, help="Batch size"
    )

    parser.add_argument(
        "-d", "--dense-dim", type=int, default=256, help="Dense dimension"
    )

    parser.add_argument("--device", type=str, default="cpu", help="Device")

    parser.add_argument(
        "-e",
        "--embedding-dim",
        type=int,
        default=128,
        help="Embedding dimension",
    )

    parser.add_argument(
        "--hidden-dim", type=int, default=256, help="Hidden dimension"
    )
    parser.add_argument(
        "--max-value-eval",
        type=int,
        default=500,
        help="Evaluation limit",
    )

    parser.add_argument(
        "-m",
        "--max-value",
        type=int,
        default=20000,
        help="The maximum allowed value (non inclusive)",
    )

    parser.add_argument(
        "-n", "--n-epochs", type=int, default=100, help="Number of epochs"
    )

    parser.add_argument(
        "-l",
        "--sequence-len",
        type=int,
        default=100,
        help="The maximum length of a sequence",
    )

    args = parser.parse_args(argv)

    # Preparations
    device = torch.device(args.device)
    eval_frequency = 500

    log_folder = pathlib.Path(args.log_folder)
    model_path = log_folder / "checkpoint.pth"

    writer = SummaryWriter(log_folder)
    writer.add_text("parameters", json.dumps(vars(args)))

    # Dataset related
    data_path = pathlib.Path(args.data_path)
    with data_path.open("rb") as f:
        raw_sequences = pickle.load(f)

    dataset = CustomDataset(
        raw_sequences,
        max_value=args.max_value,
        sequence_len=args.sequence_len,
    )

    fig, ax = plt.subplots()
    ax.hist(dataset.normalized_sequences.ravel(), bins=100)
    ax.set_title(
        f"Number distribution (numbers={dataset.normalized_sequences.shape})"
    )
    writer.add_figure("number distribution", fig)

    dataloader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
    )

    # Newtork, loss and the optimizer
    net = Network(
        max_value=args.max_value,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        dense_dim=args.dense_dim,
    )

    net.to(device)

    loss_inst = nn.CrossEntropyLoss(
        ignore_index=args.max_value,
    )

    optimizer = torch.optim.Adam(net.parameters())

    # Validation preparation
    max_value_eval = args.max_value_eval or args.max_value
    arange = np.arange(max_value_eval)
    ys_clf = create_classification_targets(arange)

    keys = sorted(ys_clf.keys())
    metadata = np.array([arange] + [ys_clf[k] for k in keys]).T.tolist()
    metadata_header = ["value"] + keys

    step = 0
    for _ in range(args.n_epochs):
        for x in tqdm.tqdm(dataloader):
            x = x.to(device)
            logits_ = net(x)  # (batch_size, sequence_len, max_value)

            logits = logits_[:, :-1].permute(
                0, 2, 1
            )  # (batch_size, max_value, sequence_len - 1)
            target = x[:, 1:]  # (batch_size, sequence_len - 1)
            loss = loss_inst(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("loss", loss, step)

            if step % eval_frequency == 0:
                X = (
                    net.embedding.weight.detach()
                    .cpu()
                    .numpy()[:max_value_eval]
                )

                writer.add_embedding(
                    X,
                    global_step=step,
                    tag="Integer embeddings",
                    metadata=metadata,
                    metadata_header=metadata_header,
                )

                for name, y in ys_clf.items():
                    metrics = train_classifier(X, y)
                    for metric_name, value in metrics.items():
                        writer.add_scalar(
                            f"{name}/{metric_name}",
                            value,
                            step,
                        )
                torch.save(net, model_path)

            step += 1



if __name__ == "__main__":
    main()
