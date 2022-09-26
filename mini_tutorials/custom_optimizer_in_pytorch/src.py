import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from torch.optim import Adam, SGD
from tqdm import tqdm

from custom import WeirdDescent


def rosenbrock(xy):
    """Evaluate Rosenbrock function.

    Parameters
    ----------
    xy : tuple
        Two element tuple of floats representing the x resp. y coordinates.

    Returns
    -------
    float
        The Rosenbrock function evaluated at the point `xy`.
    """
    x, y = xy

    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def run_optimization(xy_init, optimizer_class, n_iter, **optimizer_kwargs):
    """Run optimization finding the minimum of the Rosenbrock function.

    Parameters
    ----------
    xy_init : tuple
        Two floats representing the x resp. y coordinates.

    optimizer_class : object
        Optimizer class.

    n_iter : int
        Number of iterations to run the optimization for.

    optimizer_kwargs : dict
        Additional parameters to be passed into the optimizer.

    Returns
    -------
    path : np.ndarray
        2D array of shape `(n_iter + 1, 2)`. Where the rows represent the
        iteration and the columns represent the x resp. y coordinates.
    """
    xy_t = torch.tensor(xy_init, requires_grad=True)
    optimizer = optimizer_class([xy_t], **optimizer_kwargs)

    path = np.empty((n_iter + 1, 2))
    path[0, :] = xy_init

    for i in tqdm(range(1, n_iter + 1)):
        optimizer.zero_grad()
        loss = rosenbrock(xy_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(xy_t, 1.0)
        optimizer.step()

        path[i, :] = xy_t.detach().numpy()

    return path


def create_animation(paths,
                     colors,
                     names,
                     figsize=(12, 12),
                     x_lim=(-2, 2),
                     y_lim=(-1, 3),
                     n_seconds=5):
    """Create an animation.

    Parameters
    ----------
    paths : list
        List of arrays representing the paths (history of x,y coordinates) the
        optimizer went through.

    colors :  list
        List of strings representing colors for each path.

    names : list
        List of strings representing names for each path.

    figsize : tuple
        Size of the figure.

    x_lim, y_lim : tuple
        Range of the x resp. y axis.

    n_seconds : int
        Number of seconds the animation should last.

    Returns
    -------
    anim : FuncAnimation
        Animation of the paths of all the optimizers.
    """
    if not (len(paths) == len(colors) == len(names)):
        raise ValueError

    path_length = max(len(path) for path in paths)

    n_points = 300
    x = np.linspace(*x_lim, n_points)
    y = np.linspace(*y_lim, n_points)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock([X, Y])

    minimum = (1.0, 1.0)

    fig, ax = plt.subplots(figsize=figsize)
    ax.contour(X, Y, Z, 90, cmap="jet")

    scatters = [ax.scatter(None,
                           None,
                           label=label,
                           c=c) for c, label in zip(colors, names)]

    ax.legend(prop={"size": 25})
    ax.plot(*minimum, "rD")

    def animate(i):
        for path, scatter in zip(paths, scatters):
            scatter.set_offsets(path[:i, :])

        ax.set_title(str(i))

    ms_per_frame = 1000 * n_seconds / path_length

    anim = FuncAnimation(fig, animate, frames=path_length, interval=ms_per_frame)

    return anim


if __name__ == "__main__":
    xy_init = (.3, .8)
    n_iter = 1500

    path_adam = run_optimization(xy_init, Adam, n_iter)
    path_sgd = run_optimization(xy_init, SGD, n_iter, lr=1e-3)
    path_weird = run_optimization(xy_init, WeirdDescent, n_iter, lr=1e-3)

    freq = 10

    paths = [path_adam[::freq], path_sgd[::freq], path_weird[::freq]]
    colors = ["green", "blue", "black"]
    names = ["Adam", "SGD", "Weird"]

    anim = create_animation(paths,
                            colors,
                            names,
                            figsize=(12, 7),
                            x_lim=(-.1, 1.1),
                            y_lim=(-.1, 1.1),
                            n_seconds=7)

    anim.save("result.gif")
    print(path_weird[-15:])
