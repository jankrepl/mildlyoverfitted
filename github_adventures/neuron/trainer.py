import argparse
import json
import multiprocessing as mp
import pathlib
import pickle
from functools import partial

import cma
import numpy as np
import tqdm
from torch.utils.tensorboard import SummaryWriter

from solutions import (
    MLPSolution,
    PermutationInvariantSolution,
)
from tasks import Task, N_ORIGINAL_FEATURES


def save(folder, n_iter, solver, solution_inst):
    """Save checkpoint.

    Parameters
    ----------
    folder : str
        Output folder.

    n_iter : int
        Iteration that corresponds to the checkpoint.

    solver : cma.CMAEvolutionStrategy
        Solver instance.

    solution_inst : Solution
        Solution instance.
    """
    folder = pathlib.Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    path = folder / f"{n_iter}.pkl"

    with path.open("wb") as f:
        obj = (solver, solution_inst)
        pickle.dump(obj, f)


def get_fitness(
    solution_inst,
    *,
    shuffle_on_reset,
    n_episodes,
    n_noise_features,
    env_seed,
    feature_seed,
):
    """Get fitness function used by the CMA optimizer/solver.

    Can be run independently on a single worker.


    Returns
    -------
    fitness : list
        List of floats of length `n_episodes` holding the per episode reward.
    """
    task = Task(
        render=False,
        shuffle_on_reset=shuffle_on_reset,
        n_noise_features=n_noise_features,
        env_seed=env_seed,
        feature_seed=feature_seed,
    )
    fitness = [task.rollout(solution_inst) for _ in range(n_episodes)]

    return fitness


def main(argv=None):
    parser = argparse.ArgumentParser(
        "Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "solution",
        type=str,
        choices=(
            "linear",
            "MLP",
            "invariant",
        ),
    )
    parser.add_argument(
        "log_dir",
        type=str,
        help="Logging folder",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Pickled solver and solution",
    )
    parser.add_argument(
        "--env-seed",
        type=int,
    )
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--feature-seed",
        type=int,
    )
    parser.add_argument(
        "-m",
        "--max-iter",
        type=int,
        default=10000,
        help="Maximum number of iterations",
    )
    parser.add_argument(
        "-e",
        "--n-episodes",
        type=int,
        default=16,
        help="Number of rollouts for fitness evaluation",
    )
    parser.add_argument(
        "-j",
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of processes",
    )
    parser.add_argument(
        "-n",
        "--n-noise-features",
        type=int,
        default=0,
        help="Number of noise features",
    )
    parser.add_argument(
        "-p",
        "--population-size",
        type=int,
        default=256,
        help="Number of solutions per generation",
    )
    parser.add_argument(
        "-s",
        "--shuffle-on-reset",
        action="store_true",
        help="Shuffle features before each rollout",
    )

    args = parser.parse_args(argv)

    writer = SummaryWriter(args.log_dir)
    writer.add_text("parameters", json.dumps(vars(args)))

    # Solution map
    if args.solution == "linear":
        solution_inst = MLPSolution(
            n_features=N_ORIGINAL_FEATURES + args.n_noise_features,
            hidden_layer_sizes=tuple(),
        )

    elif args.solution == "MLP":
        solution_inst = MLPSolution(
            n_features=N_ORIGINAL_FEATURES + args.n_noise_features,
            hidden_layer_sizes=(16,),
        )

    elif args.solution == "invariant":
        solution_inst = PermutationInvariantSolution(
            n_embeddings=16,
            proj_dim=32,
            hidden_size=8,
        )

    else:
        raise ValueError

    # Prepare solver
    if args.checkpoint is None:
        x0 = np.zeros(solution_inst.get_n_params())
        solver = cma.CMAEvolutionStrategy(
            x0=x0,
            sigma0=0.1,
            inopts={
                "popsize": args.population_size,
                "seed": 42,
                "randn": np.random.randn,
            },
        )
    else:
        with open(args.checkpoint, "rb") as f:
            solver, solution_inst_ = pickle.load(f)

            assert isinstance(solution_inst, solution_inst_.__class__)

            solution_inst = solution_inst_

    get_fitness_partial = partial(
        get_fitness,
        n_episodes=args.n_episodes,
        shuffle_on_reset=args.shuffle_on_reset,
        n_noise_features=args.n_noise_features,
        env_seed=args.env_seed,
        feature_seed=args.feature_seed,
    )

    if args.n_jobs == -1:
        n_jobs = mp.cpu_count()
    else:
        n_jobs = args.n_jobs


    with mp.Pool(processes=n_jobs) as pool:
        for n_iter in tqdm.tqdm(range(args.max_iter)):
            try:
                params_set = solver.ask()
                iterable = [
                    solution_inst.clone().set_params(p) for p in params_set
                ]
                rewards = pool.map(get_fitness_partial, iterable)
                pos_fitnesses = [np.mean(r) for r in rewards]

                neg_fitnesses = [-x for x in pos_fitnesses]

                all_parameters = np.concatenate(params_set)
                metrics = {
                    "parameter_mean": all_parameters.mean(),
                    "parameter_std": all_parameters.std(),
                    "mean": np.mean(pos_fitnesses),
                    "max (generation)": np.max(pos_fitnesses),
                    "max (overall)": -solver.result.fbest,
                }

                for metric_name, metric in metrics.items():
                    writer.add_scalar(metric_name, metric, global_step=n_iter)

                if (n_iter % args.eval_frequency == 0) or (
                    n_iter == (args.max_iter - 1)
                ):
                    save(args.log_dir, n_iter, solver, solution_inst)

                solver.tell(params_set, neg_fitnesses)

            except KeyboardInterrupt:
                save(
                    args.log_dir,
                    n_iter,
                    solver,
                    solution_inst,
                )
                break


if __name__ == "__main__":
    main()
