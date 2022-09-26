"""Assumes you have already trained your model and you have a checkpoint."""
import argparse
import pathlib
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tasks import Task


def main(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-e",
        "--n-episodes",
        type=int,
        default=200,
    )
    args = parser.parse_args(argv)

    # Prepare solutions and tasks
    checkpoints = {}

    checkpoint_folder = pathlib.Path("pretrained")
    assert checkpoint_folder.exists()

    checkpoint_paths = [
        checkpoint_folder / "linear.pkl",
        checkpoint_folder / "linear_augment.pkl",
        checkpoint_folder / "MLP.pkl",
        checkpoint_folder / "MLP_augment.pkl",
        checkpoint_folder / "invariant_ours.pkl",
        checkpoint_folder / "invariant_official.pkl",
    ]

    for path in checkpoint_paths:
        with path.open("rb") as f:
            obj = pickle.load(f)

            if len(obj) == 1:
                solution_inst = obj[0]
            elif len(obj) == 2:
                solver, solution_inst = obj
                solution_inst.set_params(solver.result.xfavorite)
            else:
                raise ValueError

        checkpoints[path.stem] = solution_inst

    results = []

    for model_name, solution_inst in checkpoints.items():
        for shuffle in [True, False]:
            print(f"{model_name=}, {shuffle=}")
            task = Task(
                render=False,
                n_noise_features=0,
                shuffle_on_reset=shuffle,
                env_seed=None,
                feature_seed=None,
            )
            for episode_ix in range(args.n_episodes):
                reward = task.rollout(solution_inst)
                results.append(
                    {
                        "model": model_name,
                        "shuffle": shuffle,
                        "episode_ix": episode_ix,
                        "reward": reward,
                    }
                )

    results_df = pd.DataFrame(results)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=300)

    sns.violinplot(
        data=results_df,
        x="model",
        y="reward",
        hue="shuffle",
        split=True,
        inner="quart",
        linewidth=1,
        palette="muted",
        ax=ax,
        scale="count",
        order=sorted(checkpoints.keys()),
    )
    sns.despine(left=True)
    ax.set_ylim(0, 1000)
    ax.grid(True)

    fig.tight_layout()
    fig.savefig("all_models_shuffling.png")


if __name__ == "__main__":
    main()
