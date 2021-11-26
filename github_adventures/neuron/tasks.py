import gym
import gym_cartpole_swingup  # noqa has a sideffect
import numpy as np

N_ORIGINAL_FEATURES = 5


class IncompatibleNFeatures(Exception):
    """Raised when observation and model number of features does not match."""


class Task:
    """Cartpoleswingup task.

    Parameters
    ----------
    render : bool
        If True, we render each step into a video frame.

    shuffle_on_reset : bool
        If True, the features are randomly shuffled before each rollout.

    n_noise_features : int
        Number of noise features added to the observation vector.

    env_seed : None or int
        Random state controling the underlying `gym.Env`.

    feature_seed : None or int
        Random state controling the shuffling and noise features.

    max_episode_steps : int
        Maximum number of steps per episode (=rollout). After his number
        `done=True` automatically.

    Attributes
    ----------
    n_features : int
        Overall number of features (original + noise).

    perm_ix : np.ndarray
        1D array storing a permutation indices of the features.

    env : gym.Env
        Environment.

    rnd : RandomState
        Random state.
    """

    def __init__(
        self,
        render=False,
        shuffle_on_reset=False,
        n_noise_features=0,
        env_seed=None,
        feature_seed=None,
        max_episode_steps=1000,
    ):

        self.env = gym.make("CartPoleSwingUp-v1")
        self.env._max_episode_steps = max_episode_steps
        self.shuffle_on_reset = shuffle_on_reset
        self.render = render
        self.n_noise_features = n_noise_features

        self.n_features = N_ORIGINAL_FEATURES + n_noise_features

        self.perm_ix = np.arange(self.n_features)
        self.noise_std = 0.1

        # Set seeds
        self.env.seed(env_seed)
        self.rnd = np.random.RandomState(seed=feature_seed)

    def reset_for_rollout(self):
        """Generate a new permutation of the features.

        It is going to be called at the beginning of each episode.
        Note that the permutation stays constant throughout the episode.
        """
        self.perm_ix = np.arange(self.n_features)

        if self.shuffle_on_reset:
            self.rnd.shuffle(self.perm_ix)

    def modify_obs(self, obs):
        """Modify raw observations.

        Parameters
        ----------
        obs : np.ndarray
            Raw observation/feature array of shape `(5,)`.

        Returns
        -------
        obs_modified : np.ndarray
            Modified observation array of shape `(5 + n_noise_features,)`.
            If `shuffle_on_reset` then the order of the features is going
            to change.
        """
        noise = self.rnd.randn(self.n_noise_features) * self.noise_std
        obs_and_noise = np.concatenate([obs, noise], axis=0)
        obs_modified = obs_and_noise[self.perm_ix]

        return obs_modified

    def rollout(self, solution):
        """Run a single episode/rollout.

        Parameters
        ----------
        solution : solutions.Solution
            Instance of a solution that yields an action given an
            observation.

        Returns
        -------
        ep_reward : int
            Overall episode reward computed as a sum of per step rewards.
        """
        # sanity check
        n_features_solution = solution.get_n_features()
        n_features_task = self.n_features

        if (
            n_features_solution is not None
            and n_features_solution != n_features_task
        ):
            raise IncompatibleNFeatures

        self.reset_for_rollout()
        solution.reset()  # important for PermutationInvariantSolution

        obs = self.env.reset()
        if self.render:
            self.env.render()

        ep_reward = 0
        done = False

        while not done:
            obs_modified = self.modify_obs(obs)
            action = solution.get_action(obs_modified)
            obs, reward, done, _ = self.env.step(action)

            ep_reward += reward
            if self.render:
                self.env.render()

        return ep_reward
