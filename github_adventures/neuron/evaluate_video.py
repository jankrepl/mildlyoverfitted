"""Assumes you have already trained your model and you have a checkpoint."""
import argparse
import pathlib
import pickle

from gym.wrappers import Monitor

from tasks import Task


def main(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-e",
        "--n-episodes",
        type=int,
        default=2,
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
    checkpoint_paths = checkpoint_paths

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

    for model_name, solution_inst in checkpoints.items():
        for shuffle in [True, False]:
            for episode_ix in range(args.n_episodes):
                print(f"{model_name=}, {shuffle=}")
                task = Task(
                    render=False,
                    n_noise_features=0,
                    shuffle_on_reset=shuffle,
                    env_seed=None,
                    feature_seed=None,
                )

                task.env = Monitor(
                    task.env,
                    f"videos/{model_name}/{shuffle}/{episode_ix}/",
                )
                reward = task.rollout(solution_inst)


if __name__ == "__main__":
    main()
