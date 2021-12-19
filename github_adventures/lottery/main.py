import argparse

import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader

import wandb
from data import MNISTDataset
from utils import MLP, compute_stats, copy_weights_mlp, prune_mlp, reinit_mlp


def loop_dataloader(dataloader):
    """Loop infinitely over a dataloader.

    Parameters
    ----------
    dataloader : DataLoader
        DataLoader streaming batches of samples.

    Yields
    ------
    X_batch : torch.Tensor
        Batch of features.

    y_batch : torch.Tensor
        Batch of predictions.
    """
    while True:
        for x in iter(dataloader):
            yield x


def train(
    model,
    dataloader_train,
    loss_inst,
    optimizer,
    max_iter=10_000,
    dataloader_val=None,
    val_freq=500,
):
    """Run the training loop.

    Parameters
    ----------
    model : nn.Module
        Neural network (in our case MLP).

    dataloader_train : DataLoader
        Dataloader yielding training samples.

    loss_inst : callable
        Computes the loss when called.

    optimizer : torch.optim.Optimizer
        Instance of an optimizer.

    max_iter : int
        The number of iterations we run the training for
        (= number of graident descent steps).

    dataloader_val : None or DataLoader
        Dataloader yielding validation samples. If provided it will
        also single to us that we want to track metrics.

    val_freq : int
        How often evaluation run.
    """
    iterable = loop_dataloader(dataloader_train)
    iterable = tqdm.tqdm(iterable, total=max_iter)

    it = 0
    for X_batch, y_batch in iterable:
        if it == max_iter:
            break

        logit_batch = model(X_batch)

        loss = loss_inst(logit_batch, y_batch)
        if dataloader_val is not None:
            wandb.log({"loss": loss}, step=it)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % val_freq == 0 and dataloader_val is not None:
            is_equal = []

            for X_batch_val, y_batch_val in dataloader_val:
                is_equal.append(
                    model(X_batch_val).argmax(dim=-1) == y_batch_val
                )

            is_equal_t = torch.cat(is_equal)
            acc = is_equal_t.sum() / len(is_equal_t)
            wandb.log({"accuracy_val": acc}, step=it)

        it += 1


def main(argv=None):
    """Create CLI and run experiments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-i",
        "--max-iter",
        help="Number of iterations",
        type=int,
        default=50000,
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        help="Batch size",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--prune-iter",
        help="Number of prune iterations",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-m",
        "--prune-method",
        help="Pruning method to employ",
        type=str,
        choices=("l1", "random"),
        default="l1",
    )
    parser.add_argument(
        "-p",
        "--prune-ratio",
        help="Percentage of weights to remove",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--val-freq",
        help="How often to compute the validation accuracy",
        type=int,
        default=250,
    )
    parser.add_argument(
        "-r",
        "--reinitialize",
        help="If true, reinitializes randomly all weights after pruning",
        type=str,
        choices=("true", "false"),  # easy for hyperparameter search
        default="false",
    )
    parser.add_argument(
        "-s",
        "--random-state",
        help="Random state",
        type=int,
    )
    parser.add_argument(
        "--wandb-entity",
        help="W&B entity",
        type=str,
        default="mildlyoverfitted",
    )
    parser.add_argument(
        "--wandb-project",
        help="W&B project",
        type=str,
    )
    args = parser.parse_args(argv)

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args),
    )
    wandb.define_metric("accuracy_val", summary="max")

    dataset_train = MNISTDataset(
        "data",
        train=True,
        download=True,
    )
    dataset_val = MNISTDataset(
        "data",
        train=False,
        download=True,
    )

    if args.random_state is not None:
        torch.manual_seed(args.random_state)

    dataloader_train = DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True
    )
    dataloader_val = DataLoader(
        dataset_val, batch_size=args.batch_size, shuffle=True
    )

    kwargs = dict(
        n_features=28 * 28,
        hidden_layer_sizes=(300, 100),
        n_targets=10,
    )

    mlp = MLP(**kwargs)

    mlp_copy = MLP(**kwargs)
    mlp_copy.load_state_dict(mlp.state_dict())

    loss_inst = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1.2 * 1e-3)

    # Train and prune loop
    if args.prune_ratio > 0:
        per_round_prune_ratio = 1 - (1 - args.prune_ratio) ** (
            1 / args.prune_iter
        )

        per_round_prune_ratios = [per_round_prune_ratio] * len(mlp.module_list)
        per_round_prune_ratios[-1] /= 2

        per_round_max_iter = int(args.max_iter / args.prune_iter)

        for prune_it in range(args.prune_iter):
            train(
                mlp,
                dataloader_train,
                loss_inst,
                optimizer,
                max_iter=per_round_max_iter,
            )
            prune_mlp(mlp, per_round_prune_ratios, method=args.prune_method)

            copy_weights_mlp(mlp_copy, mlp)

            stats = compute_stats(mlp)
            for name, stat in stats.items():
                summary_name = f"{name}_pruneiter={prune_it}"
                wandb.run.summary[summary_name] = stat

    if args.reinitialize == "true":
        reinit_mlp(mlp)

    # Run actual training with a final pruned network
    train(
        mlp,
        dataloader_train,
        loss_inst,
        optimizer,
        max_iter=args.max_iter,
        dataloader_val=dataloader_val,
        val_freq=args.val_freq,
    )


if __name__ == "__main__":
    main()
