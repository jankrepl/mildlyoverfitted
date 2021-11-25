"""Assumes you have already trained your model and you have a checkpoint."""
import argparse
import pathlib
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tasks import IncompatibleNFeatures, Task


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
    checkpoint_path = pathlib.Path("pretrained") / "invariant_official.pkl"
    assert checkpoint_path.exists()

    with checkpoint_path.open("rb") as f:
        obj = pickle.load(f)

        if len(obj) == 1:
            solution_inst = obj[0]
        elif len(obj) == 2:
            solver, solution_inst = obj
            solution_inst.set_params(solver.result.xfavorite)
        else:
            raise ValueError

    results = []

    for n_noise_features in range(0, 30, 5):
        for shuffle in [True, False]:
            print(f"{n_noise_features=}, {shuffle=}")
            task = Task(
                render=False,
                n_noise_features=n_noise_features,
                shuffle_on_reset=shuffle,
                env_seed=None,
                feature_seed=None,
            )
            for episode_ix in range(args.n_episodes):
                reward = task.rollout(solution_inst)
                results.append(
                    {
                        "n_noise_features": n_noise_features,
                        "shuffle": shuffle,
                        "episode_ix": episode_ix,
                        "reward": reward,
                    }
                )

    results_df = pd.DataFrame(results)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=300)

    sns.violinplot(
        data=results_df,
        x="n_noise_features",
        y="reward",
        hue="shuffle",
        split=True,
        inner="quart",
        linewidth=1,
        palette="muted",
        ax=ax,
        scale="count",
    )
    sns.despine(left=True)
    ax.set_ylim(0, 1000)
    ax.grid(True)

    fig.tight_layout()
    fig.savefig("invariant_model_noise.png")


if __name__ == "__main__":
    main()
