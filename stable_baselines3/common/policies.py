"""Policies: abstract base class and concrete implementations."""

import collections

import time

import warnings

from abc import ABC, abstractmethod

from functools import partial

from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union


import numpy as np

import torch as th

from gymnasium import spaces

from torch import nn


from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)

from stable_baselines3.common.preprocessing import get_action_dim, preprocess_obs

from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    MlpExtractor,
)

from stable_baselines3.common.type_aliases import PyTorchObs, Schedule

from stable_baselines3.common.utils import (
    get_device,
    is_vectorized_observation,
    obs_as_tensor,
)


SelfBaseModel = TypeVar("SelfBaseModel", bound="BaseModel")


class BaseModel(nn.Module):
    """
    The base model object: makes predictions in response to observations.

    In the case of policies, the prediction is an action. In the case of critics, it is the
    estimated value of the observation.

    :param observation_space: The observation space of the environment
    :param action_space: The action space of the environment
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    optimizer: th.optim.Optimizer

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        features_extractor: Optional[BaseFeaturesExtractor] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):

        super().__init__()

        if optimizer_kwargs is None:

            optimizer_kwargs = {}

        if features_extractor_kwargs is None:

            features_extractor_kwargs = {}

        self.observation_space = observation_space

        self.action_space = action_space

        self.features_extractor = features_extractor

        self.normalize_images = normalize_images

        self.optimizer_class = optimizer_class

        self.optimizer_kwargs = optimizer_kwargs

        self.features_extractor_class = features_extractor_class

        self.features_extractor_kwargs = features_extractor_kwargs

    def _update_features_extractor(
        self,
        net_kwargs: Dict[str, Any],
        features_extractor: Optional[BaseFeaturesExtractor] = None,
    ) -> Dict[str, Any]:
        """
        Update the network keyword arguments and create a new features extractor object if needed.
        If a ``features_extractor`` object is passed, then it will be shared.

        :param net_kwargs: the base network keyword arguments, without the ones
            related to features extractor
        :param features_extractor: a features extractor object.
            If None, a new object will be created.
        :return: The updated keyword arguments
        """

        net_kwargs = net_kwargs.copy()

        if features_extractor is None:

            features_extractor = self.make_features_extractor()

        net_kwargs.update(
            dict(
                features_extractor=features_extractor,
                features_dim=features_extractor.features_dim,
            )
        )

        return net_kwargs

    def make_features_extractor(self) -> BaseFeaturesExtractor:
        """Helper method to create a features extractor."""

        return self.features_extractor_class(
            self.observation_space, **self.features_extractor_kwargs
        )

    def extract_features(
        self, obs: PyTorchObs, features_extractor: BaseFeaturesExtractor
    ) -> th.Tensor:
        """
        Preprocess the observation if needed and extract features.

        :param obs: Observation
        :param features_extractor: The features extractor to use.
        :return: The extracted features
        """

        preprocessed_obs = preprocess_obs(
            obs, self.observation_space, normalize_images=self.normalize_images
        )

        return features_extractor(preprocessed_obs)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        """
        Get data that need to be saved in order to re-create the model when loading it from disk.

        :return: The dictionary to pass to the as kwargs constructor when reconstruction this model.
        """

        return dict(
            observation_space=self.observation_space,
            action_space=self.action_space,
            normalize_images=self.normalize_images,
        )

    @property
    def device(self) -> th.device:
        """Infer which device this policy lives on by inspecting its parameters.
        If it has no parameters, the 'cpu' device is used as a fallback.

        :return:"""

        for param in self.parameters():

            return param.device

        return get_device("cpu")

    def save(self, path: str) -> None:
        """
        Save model to a given location.

        :param path:
        """

        th.save(
            {
                "state_dict": self.state_dict(),
                "data": self._get_constructor_parameters(),
            },
            path,
        )

    @classmethod
    def load(
        cls: Type[SelfBaseModel], path: str, device: Union[th.device, str] = "auto"
    ) -> SelfBaseModel:
        """
        Load model from path.

        :param path:
        :param device: Device on which the policy should be loaded.
        :return:
        """

        device = get_device(device)

        try:

            saved_variables = th.load(path, map_location=device, weights_only=False)

        except TypeError:

            saved_variables = th.load(path, map_location=device)

        model = cls(**saved_variables["data"])

        model.load_state_dict(saved_variables["state_dict"])

        model.to(device)

        return model

    def load_from_vector(self, vector: np.ndarray) -> None:
        """
        Load parameters from a 1D vector.

        :param vector:
        """

        th.nn.utils.vector_to_parameters(
            th.as_tensor(vector, dtype=th.float, device=self.device), self.parameters()
        )

    def parameters_to_vector(self) -> np.ndarray:
        """
        Convert the parameters to a 1D vector.

        :return:
        """

        return (
            th.nn.utils.parameters_to_vector(self.parameters()).detach().cpu().numpy()
        )

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """

        self.train(mode)

    def is_vectorized_observation(
        self, observation: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> bool:
        """
        Check whether or not the vector observation is batched.

        :param observation: the input observation to check
        :return: whether the given observation is vectorized or not
        """

        if isinstance(observation, dict):

            raise ValueError("Use vector observations.")

        return is_vectorized_observation(np.array(observation), self.observation_space)

    def obs_to_tensor(
        self, observation: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> Tuple[PyTorchObs, bool]:
        """
        Convert an input observation to a PyTorch tensor that can be fed to a model.

        :param observation: the input observation
        :return: The observation as PyTorch tensor
            and whether the observation is vectorized or not
        """

        if isinstance(observation, dict):

            raise ValueError("Use vector observations.")

        observation = np.array(observation)

        vectorized_env = is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1, *self.observation_space.shape))

        obs_tensor = obs_as_tensor(observation, self.device)

        return obs_tensor, vectorized_env


class BasePolicy(BaseModel, ABC):
    """The base policy object.

    Parameters are mostly the same as `BaseModel`; additions are documented below.

    :param args: positional arguments passed through to `BaseModel`.
    :param kwargs: keyword arguments passed through to `BaseModel`.
    :param squash_output: For continuous actions, whether the output is squashed
        or not using a ``tanh()`` function.
    """

    features_extractor: BaseFeaturesExtractor

    def __init__(self, *args, squash_output: bool = False, **kwargs):

        super().__init__(*args, **kwargs)

        self._squash_output = squash_output

    @staticmethod
    def _dummy_schedule(progress_remaining: float) -> float:
        """(float) Useful for pickling policy."""

        del progress_remaining

        return 0.0

    @property
    def squash_output(self) -> bool:
        """(bool) Getter for squash_output."""

        return self._squash_output

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """

        if isinstance(module, (nn.Linear, nn.Conv2d)):

            nn.init.orthogonal_(module.weight, gain=gain)

            if module.bias is not None:

                module.bias.data.fill_(0.0)

    @abstractmethod
    def _predict(
        self, observation: PyTorchObs, deterministic: bool = False
    ) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        By default provides a dummy implementation -- not all BasePolicy classes
        implement this, e.g. if they are a Critic in an Actor-Critic method.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """

        self.set_training_mode(False)

        if (
            isinstance(observation, tuple)
            and len(observation) == 2
            and isinstance(observation[1], dict)
        ):

            raise ValueError(
                "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
                "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
                "vs `obs = vec_env.reset()` (SB3 VecEnv). "
                "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
                "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
            )

        obs_tensor, vectorized_env = self.obs_to_tensor(observation)

        with th.no_grad():

            actions = self._predict(obs_tensor, deterministic=deterministic)

        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))

        if isinstance(self.action_space, spaces.Box):

            if self.squash_output:

                actions = self.unscale_action(actions)

            else:

                actions = np.clip(
                    actions, self.action_space.low, self.action_space.high
                )

        if not vectorized_env:

            assert isinstance(actions, np.ndarray)

            actions = actions.squeeze(axis=0)

        return actions, state

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)

        :param action: Action to scale
        :return: Scaled action
        """

        assert isinstance(
            self.action_space, spaces.Box
        ), f"Trying to scale an action using an action space that is not a Box(): {self.action_space}"

        low, high = self.action_space.low, self.action_space.high

        return 2.0 * ((action - low) / (high - low)) - 1.0

    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)

        :param scaled_action: Action to un-scale
        """

        assert isinstance(
            self.action_space, spaces.Box
        ), f"Trying to unscale an action using an action space that is not a Box(): {self.action_space}"

        low, high = self.action_space.low, self.action_space.high

        return low + (0.5 * (scaled_action + 1.0) * (high - low))


class ActorCriticPolicy(BasePolicy):
    """
    Policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param min_log_std: Minimum log standard deviation for continuous actions.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        min_log_std: float = -3.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        use_programmatic_policy: bool = False,
        prog_graph_depth: int = 6,
        prog_beta: float = 0.25,
        prog_domain: str = "CartPole-v1",
        prog_equally_initialize: bool = False,
    ):

        if optimizer_kwargs is None:

            optimizer_kwargs = {}

            if optimizer_class == th.optim.Adam:

                optimizer_kwargs["eps"] = 1e-5

        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
            normalize_images=normalize_images,
        )

        if (
            isinstance(net_arch, list)
            and len(net_arch) > 0
            and isinstance(net_arch[0], dict)
        ):

            warnings.warn(
                (
                    "As shared layers in the mlp_extractor are removed since SB3 v1.8.0, "
                    "you should now pass directly a dictionary and not a list "
                    "(net_arch=dict(pi=..., vf=...) instead of net_arch=[dict(pi=..., vf=...)])"
                ),
            )

            net_arch = net_arch[0]

        if net_arch is None:

            net_arch = dict(pi=[64, 64], vf=[64, 64])

        self.net_arch = net_arch

        self.activation_fn = activation_fn

        self.ortho_init = ortho_init

        self.share_features_extractor = share_features_extractor

        self.features_extractor = self.make_features_extractor()

        self.features_dim = self.features_extractor.features_dim

        if self.share_features_extractor:

            self.pi_features_extractor = self.features_extractor

            self.vf_features_extractor = self.features_extractor

        else:

            self.pi_features_extractor = self.features_extractor

            self.vf_features_extractor = self.make_features_extractor()

        self.log_std_init = log_std_init

        self.min_log_std = min_log_std

        dist_kwargs = None

        assert not (
            squash_output and not use_sde
        ), "squash_output=True is only available when using gSDE (use_sde=True)"

        if use_sde:

            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": False,
            }

        self.use_sde = use_sde

        self.dist_kwargs = dist_kwargs

        self.use_programmatic_policy = use_programmatic_policy

        self.prog_graph_depth = prog_graph_depth

        self.prog_beta = prog_beta

        self.prog_domain = prog_domain

        self.programmatic_policy = None

        self.prog_equally_initialize = prog_equally_initialize

        self.action_dist = make_proba_distribution(
            action_space, use_sde=use_sde, dist_kwargs=dist_kwargs
        )

        self._build(lr_schedule)

    def _get_constructor_parameters(self) -> Dict[str, Any]:

        data = super()._get_constructor_parameters()

        default_none_kwargs = self.dist_kwargs or collections.defaultdict(lambda: None)

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                min_log_std=self.min_log_std,
                squash_output=default_none_kwargs["squash_output"],
                full_std=default_none_kwargs["full_std"],
                use_expln=default_none_kwargs["use_expln"],
                lr_schedule=self._dummy_schedule,
                ortho_init=self.ortho_init,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )

        return data

    def reset_noise(self, n_envs: int = 1) -> None:
        """
        Sample new weights for the exploration matrix.

        :param n_envs:
        """

        assert isinstance(
            self.action_dist, StateDependentNoiseDistribution
        ), "reset_noise() is only available when using gSDE"

        self.action_dist.sample_weights(self.log_std, batch_size=n_envs)

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """

        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
            feature_dim_vf=self.features_dim,
        )

    def _clip_log_std(self) -> None:

        if self.min_log_std is None:

            return

        if isinstance(self.action_dist, DiagGaussianDistribution) and hasattr(
            self, "log_std"
        ):

            with th.no_grad():

                self.log_std.clamp_(min=self.min_log_std)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """

        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if self.use_programmatic_policy and isinstance(
            self.action_dist,
            (
                CategoricalDistribution,
                MultiCategoricalDistribution,
                BernoulliDistribution,
            ),
        ):

            from diprl.dprogram import SearchFusionProgram

            from diprl.prl_classic_env import parse_domain

            self.programmatic_policy = SearchFusionProgram(
                graph_depth=int(self.prog_graph_depth),
                domain=self.prog_domain,
                parse_domain=parse_domain,
                beta=self.prog_beta,
                search_map_type="architecture",
                equally_initialize=self.prog_equally_initialize,
            )

            self.action_net = None

        else:

            if isinstance(self.action_dist, DiagGaussianDistribution):

                self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                    latent_dim=latent_dim_pi, log_std_init=self.log_std_init
                )

            elif isinstance(self.action_dist, StateDependentNoiseDistribution):

                self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                    latent_dim=latent_dim_pi,
                    latent_sde_dim=latent_dim_pi,
                    log_std_init=self.log_std_init,
                )

            elif isinstance(
                self.action_dist,
                (
                    CategoricalDistribution,
                    MultiCategoricalDistribution,
                    BernoulliDistribution,
                ),
            ):

                self.action_net = self.action_dist.proba_distribution_net(
                    latent_dim=latent_dim_pi
                )

            else:

                raise NotImplementedError(
                    f"Unsupported distribution '{self.action_dist}'."
                )

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:

            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }

            if not self.share_features_extractor:

                del module_gains[self.features_extractor]

                module_gains[self.pi_features_extractor] = np.sqrt(2)

                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():

                if module is not None:

                    module.apply(partial(self.init_weights, gain=gain))

        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def forward(
        self, obs: th.Tensor, deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """

        features = self.extract_features(obs)

        if self.share_features_extractor:

            pi_features = features

            vf_features = features

        else:

            pi_features, vf_features = features

        if self.programmatic_policy is not None:

            latent_pi = pi_features

            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        else:

            latent_pi, latent_vf = self.mlp_extractor(pi_features, vf_features)

        distribution = self._get_action_dist_from_latent(latent_pi)

        actions = distribution.get_actions(deterministic=deterministic)

        log_prob = distribution.log_prob(actions)

        actions = actions.reshape((-1, *self.action_space.shape))

        values = self.value_net(latent_vf)

        return actions, values, log_prob

    def extract_features(
        self,
        obs: PyTorchObs,
        features_extractor: Optional[BaseFeaturesExtractor] = None,
    ) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        """
        Preprocess the observation if needed and extract features.

        :param obs: Observation
        :param features_extractor: The features extractor to use. If None, then ``self.features_extractor`` is used.
        :return: The extracted features. If features extractor is not shared, returns a tuple with the
            features for the actor and the features for the critic.
        """

        if self.share_features_extractor:

            return super().extract_features(
                obs,
                (
                    self.features_extractor
                    if features_extractor is None
                    else features_extractor
                ),
            )

        else:

            if features_extractor is not None:

                warnings.warn(
                    "Provided features_extractor will be ignored because the features extractor is not shared.",
                    UserWarning,
                )

            pi_features = super().extract_features(obs, self.pi_features_extractor)

            vf_features = super().extract_features(obs, self.vf_features_extractor)

            return pi_features, vf_features

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """

        if self.programmatic_policy is not None:

            if isinstance(
                self.action_dist,
                (
                    CategoricalDistribution,
                    MultiCategoricalDistribution,
                    BernoulliDistribution,
                ),
            ):

                mean_actions = self.programmatic_policy.discrete_forward(latent_pi)

            else:

                mean_actions = self.programmatic_policy(latent_pi)

        else:

            mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):

            if self.programmatic_policy is not None:

                self._clip_log_std()

            return self.action_dist.proba_distribution(mean_actions, self.log_std)

        elif isinstance(self.action_dist, CategoricalDistribution):

            return self.action_dist.proba_distribution(action_logits=mean_actions)

        elif isinstance(self.action_dist, MultiCategoricalDistribution):

            return self.action_dist.proba_distribution(action_logits=mean_actions)

        elif isinstance(self.action_dist, BernoulliDistribution):

            return self.action_dist.proba_distribution(action_logits=mean_actions)

        elif isinstance(self.action_dist, StateDependentNoiseDistribution):

            return self.action_dist.proba_distribution(
                mean_actions, self.log_std, latent_pi
            )

        else:

            raise ValueError("Invalid action distribution")

    def _predict(
        self, observation: PyTorchObs, deterministic: bool = False
    ) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """

        return self.get_distribution(observation).get_actions(
            deterministic=deterministic
        )

    def evaluate_actions(
        self, obs: PyTorchObs, actions: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """

        profile = getattr(self, "profile_evaluate_actions", False)

        if profile:

            t0 = time.perf_counter()

            t_latent = 0.0

            t_dist = 0.0

            t_value = 0.0

        features = self.extract_features(obs)

        if self.share_features_extractor:

            pi_features = features

            vf_features = features

        else:

            pi_features, vf_features = features

        if profile:

            t_features = time.perf_counter() - t0

        if profile:

            t_latent_start = time.perf_counter()

        if self.programmatic_policy is not None:

            latent_pi = pi_features

            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        else:

            latent_pi, latent_vf = self.mlp_extractor(pi_features, vf_features)

        if profile:

            t_latent = time.perf_counter() - t_latent_start

        if profile:

            t_dist_start = time.perf_counter()

        distribution = self._get_action_dist_from_latent(latent_pi)

        log_prob = distribution.log_prob(actions)

        entropy = distribution.entropy()

        if profile:

            t_dist = time.perf_counter() - t_dist_start

        if profile:

            t_value_start = time.perf_counter()

        values = self.value_net(latent_vf)

        if profile:

            t_value = time.perf_counter() - t_value_start

            t_total = time.perf_counter() - t0

            print(
                "[evaluate_actions][time] total={:.4f}s features={:.4f}s "
                "latent={:.4f}s dist={:.4f}s value={:.4f}s".format(
                    t_total, t_features, t_latent, t_dist, t_value
                )
            )

        return values, log_prob, entropy

    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """

        pi_features = super().extract_features(obs, self.pi_features_extractor)

        if self.programmatic_policy is not None:

            latent_pi = pi_features

        else:

            latent_pi = self.mlp_extractor.forward_actor(pi_features)

        return self._get_action_dist_from_latent(latent_pi)

    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """

        vf_features = super().extract_features(obs, self.vf_features_extractor)

        latent_vf = self.mlp_extractor.forward_critic(vf_features)

        return self.value_net(latent_vf)

    def flip_programmatic_policy(self) -> None:
        """
        Toggle between architecture search and program optimization phases.

        This method alternates the training mode of the programmatic policy:
        - Phase 0 (Architecture Search): search_map trainable, programs frozen
        - Phase 1 (Program Optimization): programs trainable, search_map frozen

        Note: Only applies to SearchFusionProgram. Extracted Program objects
        don't have flip() method and are ignored.
        """

        if self.programmatic_policy is not None and hasattr(
            self.programmatic_policy, "flip"
        ):

            self.programmatic_policy.flip()

    def extract_programmatic_policy(self, parse_domain):
        """
        Extract discrete program from the program derivation graph.

        After training with alternating optimization, this method extracts
        the most probable program structure based on the learned architecture
        distribution and returns it as a standalone Program object.

        :param parse_domain: Function to parse domain-specific configuration
        :return: Extracted Program object, or None if not using programmatic policy
        """

        if self.programmatic_policy is not None:

            return self.programmatic_policy.extract(parse_domain)

        return None
