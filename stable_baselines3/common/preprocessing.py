from typing import Dict, Tuple, Union


import numpy as np

import torch as th

from gymnasium import spaces

from torch.nn import functional as F


try:

    import gym as classic_gym

except Exception:

    classic_gym = None


CLASSIC_SPACES = (
    getattr(classic_gym, "spaces", None) if classic_gym is not None else None
)

BOX_TYPES = (spaces.Box,) + (
    (CLASSIC_SPACES.Box,) if CLASSIC_SPACES is not None else ()
)

DISCRETE_TYPES = (spaces.Discrete,) + (
    (CLASSIC_SPACES.Discrete,) if CLASSIC_SPACES is not None else ()
)

MULTI_DISCRETE_TYPES = (spaces.MultiDiscrete,) + (
    (CLASSIC_SPACES.MultiDiscrete,) if CLASSIC_SPACES is not None else ()
)

MULTI_BINARY_TYPES = (spaces.MultiBinary,) + (
    (CLASSIC_SPACES.MultiBinary,) if CLASSIC_SPACES is not None else ()
)


def preprocess_obs(
    obs: Union[th.Tensor, Dict[str, th.Tensor]],
    observation_space: spaces.Space,
    normalize_images: bool = True,
) -> Union[th.Tensor, Dict[str, th.Tensor]]:
    """
    Preprocess observations for a neural network.

    :param obs: Observation
    :param observation_space:
        :param normalize_images: Kept for API compatibility.
    :return:
    """

    if isinstance(observation_space, spaces.Dict):

        assert isinstance(obs, Dict), f"Expected dict, got {type(obs)}"

        preprocessed_obs = {}

        for key, _obs in obs.items():

            preprocessed_obs[key] = preprocess_obs(
                _obs, observation_space[key], normalize_images=normalize_images
            )

        return preprocessed_obs

    assert isinstance(obs, th.Tensor), f"Expecting a torch Tensor, but got {type(obs)}"

    if isinstance(observation_space, BOX_TYPES):

        return obs.float()

    elif isinstance(observation_space, DISCRETE_TYPES):

        return F.one_hot(obs.long(), num_classes=int(observation_space.n)).float()

    elif isinstance(observation_space, MULTI_DISCRETE_TYPES):

        return th.cat(
            [
                F.one_hot(
                    obs_.long(), num_classes=int(observation_space.nvec[idx])
                ).float()
                for idx, obs_ in enumerate(th.split(obs.long(), 1, dim=1))
            ],
            dim=-1,
        ).view(obs.shape[0], sum(observation_space.nvec))

    elif isinstance(observation_space, MULTI_BINARY_TYPES):

        return obs.float()

    else:

        raise NotImplementedError(
            f"Preprocessing not implemented for {observation_space}"
        )


def get_obs_shape(
    observation_space: spaces.Space,
) -> Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]:
    """
    Get the shape of the observation (useful for the buffers).

    :param observation_space:
    :return:
    """

    if isinstance(observation_space, BOX_TYPES):

        return observation_space.shape

    elif isinstance(observation_space, DISCRETE_TYPES):

        return (1,)

    elif isinstance(observation_space, MULTI_DISCRETE_TYPES):

        return (int(len(observation_space.nvec)),)

    elif isinstance(observation_space, MULTI_BINARY_TYPES):

        return observation_space.shape

    elif isinstance(observation_space, spaces.Dict):

        return {
            key: get_obs_shape(subspace)
            for (key, subspace) in observation_space.spaces.items()
        }

    else:

        raise NotImplementedError(
            f"{observation_space} observation space is not supported"
        )


def get_flattened_obs_dim(observation_space: spaces.Space) -> int:
    """
    Get the dimension of the observation space when flattened.
    It does not apply to image observation space.

    Used by the ``FlattenExtractor`` to compute the input shape.

    :param observation_space:
    :return:
    """

    if isinstance(observation_space, MULTI_DISCRETE_TYPES):

        return sum(observation_space.nvec)

    else:

        try:

            return spaces.utils.flatdim(observation_space)

        except Exception:

            if CLASSIC_SPACES is not None:

                return CLASSIC_SPACES.utils.flatdim(observation_space)

            raise


def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.

    :param action_space:
    :return:
    """

    if isinstance(action_space, BOX_TYPES):

        return int(np.prod(action_space.shape))

    elif isinstance(action_space, DISCRETE_TYPES):

        return 1

    elif isinstance(action_space, MULTI_DISCRETE_TYPES):

        return int(len(action_space.nvec))

    elif isinstance(action_space, MULTI_BINARY_TYPES):

        assert isinstance(
            action_space.n, int
        ), f"Multi-dimensional MultiBinary({action_space.n}) action space is not supported. You can flatten it instead."

        return int(action_space.n)

    else:

        raise NotImplementedError(f"{action_space} action space is not supported")


def check_for_nested_spaces(obs_space: spaces.Space) -> None:
    """
    Make sure the observation space does not have nested spaces (Dicts/Tuples inside Dicts/Tuples).
    If so, raise an Exception informing that there is no support for this.

    :param obs_space: an observation space
    """

    if isinstance(obs_space, (spaces.Dict, spaces.Tuple)):

        sub_spaces = (
            obs_space.spaces.values()
            if isinstance(obs_space, spaces.Dict)
            else obs_space.spaces
        )

        for sub_space in sub_spaces:

            if isinstance(sub_space, (spaces.Dict, spaces.Tuple)):

                raise NotImplementedError(
                    "Nested observation spaces are not supported (Tuple/Dict space inside Tuple/Dict space)."
                )
