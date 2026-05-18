from typing import Dict, List, Optional, Tuple, Type, Union


import gymnasium as gym

import torch as th

from torch import nn


from stable_baselines3.common.preprocessing import get_flattened_obs_dim

from stable_baselines3.common.utils import get_device


class BaseFeaturesExtractor(nn.Module):
    """
    Base class that represents a features extractor.

    :param observation_space:
    :param features_dim: Number of features extracted.
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 0) -> None:

        super().__init__()

        assert features_dim > 0

        self._observation_space = observation_space

        self._features_dim = features_dim

    @property
    def features_dim(self) -> int:

        return self._features_dim


class FlattenExtractor(BaseFeaturesExtractor):
    """
    Feature extractor that flattens vector observations.

    :param observation_space:
    """

    def __init__(self, observation_space: gym.Space) -> None:

        super().__init__(observation_space, get_flattened_obs_dim(observation_space))

        self.flatten = nn.Flatten()

    def forward(self, observations: th.Tensor) -> th.Tensor:

        return self.flatten(observations)


class MlpExtractor(nn.Module):
    """
    Constructs separate MLPs for the policy and value function.

    :param feature_dim: Dimension of the feature vector.
    :param net_arch: Specification of the policy and value networks.
    :param activation_fn: Activation function.
    :param device: PyTorch device.
    :param feature_dim_vf: Optional value-network input dimension.
    """

    def __init__(
        self,
        feature_dim: int,
        net_arch: Union[List[int], Dict[str, List[int]]],
        activation_fn: Type[nn.Module],
        device: Union[th.device, str] = "auto",
        feature_dim_vf: Optional[int] = None,
    ) -> None:

        super().__init__()

        device = get_device(device)

        policy_net: List[nn.Module] = []

        value_net: List[nn.Module] = []

        last_layer_dim_pi = feature_dim

        last_layer_dim_vf = feature_dim if feature_dim_vf is None else feature_dim_vf

        if isinstance(net_arch, dict):

            pi_layers_dims = net_arch.get("pi", [])

            vf_layers_dims = net_arch.get("vf", [])

        else:

            pi_layers_dims = vf_layers_dims = net_arch

        for curr_layer_dim in pi_layers_dims:

            policy_net.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))

            policy_net.append(activation_fn())

            last_layer_dim_pi = curr_layer_dim

        for curr_layer_dim in vf_layers_dims:

            value_net.append(nn.Linear(last_layer_dim_vf, curr_layer_dim))

            value_net.append(activation_fn())

            last_layer_dim_vf = curr_layer_dim

        self.latent_dim_pi = last_layer_dim_pi

        self.latent_dim_vf = last_layer_dim_vf

        self.policy_net = nn.Sequential(*policy_net).to(device)

        self.value_net = nn.Sequential(*value_net).to(device)

    def forward(
        self, features: th.Tensor, features_vf: Optional[th.Tensor] = None
    ) -> Tuple[th.Tensor, th.Tensor]:

        if features_vf is None:

            features_vf = features

        return self.forward_actor(features), self.forward_critic(features_vf)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:

        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:

        return self.value_net(features)
