import warnings

from inspect import signature

from typing import Union


import gymnasium


try:

    import gym

    gym_installed = True

except ImportError:

    gym_installed = False


def _patch_env(env: Union["gym.Env", gymnasium.Env]) -> gymnasium.Env:
    """
    Adapted from https://github.com/thu-ml/tianshou.

    Takes an environment and patches it to return Gymnasium env.
    This function takes the environment object and returns a patched
    env, using shimmy wrapper to convert it to Gymnasium,
    if necessary.

    :param env: A gym/gymnasium env
    :return: Patched env (gymnasium env)
    """

    if isinstance(env, gymnasium.Env):

        return env

    if not gym_installed or not isinstance(env, gym.Env):

        raise ValueError(
            f"The environment is of type {type(env)}, not a Gymnasium "
            f"environment. In this case, we expect OpenAI Gym to be "
            f"installed and the environment to be an OpenAI Gym environment."
        )

    try:

        import shimmy

    except ImportError:

        warnings.warn(
            "shimmy is not installed; proceeding with the raw Gym environment. "
            "Ensure `apply_api_compatibility=True` is used so the env follows the Gymnasium API."
        )

        return env

    warnings.warn(
        "You provided an OpenAI Gym environment. "
        "We strongly recommend transitioning to Gymnasium environments. "
        "Stable-Baselines3 is automatically wrapping your environments in a compatibility "
        "layer, which could potentially cause issues."
    )

    if "seed" in signature(env.unwrapped.reset).parameters:

        return shimmy.GymV26CompatibilityV0(env=env)

    return shimmy.GymV21CompatibilityV0(env=env)


def _convert_space(space: Union["gym.Space", gymnasium.Space]) -> gymnasium.Space:
    """
    Takes a space and patches it to return Gymnasium Space.
    This function takes the space object and returns a patched
    space, using shimmy wrapper to convert it to Gymnasium,
    if necessary.

    :param env: A gym/gymnasium Space
    :return: Patched space (gymnasium Space)
    """

    if isinstance(space, gymnasium.Space):

        return space

    if not gym_installed or not isinstance(space, gym.Space):

        raise ValueError(
            f"The space is of type {type(space)}, not a Gymnasium "
            f"space. In this case, we expect OpenAI Gym to be "
            f"installed and the space to be an OpenAI Gym space."
        )

    try:

        import shimmy

    except ImportError:

        import numpy as _np

        if isinstance(space, gym.spaces.Box):

            return gymnasium.spaces.Box(
                low=_np.asarray(space.low),
                high=_np.asarray(space.high),
                shape=space.shape,
                dtype=space.dtype,
            )

        if isinstance(space, gym.spaces.Discrete):

            return gymnasium.spaces.Discrete(int(space.n))

        if isinstance(space, gym.spaces.MultiDiscrete):

            return gymnasium.spaces.MultiDiscrete(_np.asarray(space.nvec))

        if isinstance(space, gym.spaces.MultiBinary):

            return gymnasium.spaces.MultiBinary(int(space.n))

        raise ImportError(
            "Missing shimmy installation. You provided an OpenAI Gym space "
            f"({type(space).__name__}) that the manual fallback can't convert. "
            "Install shimmy (`pip install 'shimmy>=0.2.1'`) or extend the fallback."
        )

    warnings.warn(
        "You loaded a model that was trained using OpenAI Gym. "
        "We strongly recommend transitioning to Gymnasium by saving that model again."
    )

    return shimmy.openai_gym_compatibility._convert_space(space)
