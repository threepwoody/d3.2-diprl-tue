import sys

import time

import warnings

from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union


import torch as th

from gymnasium import spaces

from torch.nn import functional as F


from stable_baselines3.common.buffers import RolloutBuffer

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy

from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule

from stable_baselines3.common.utils import explained_variance, get_schedule_fn


def _time_ns() -> int:

    return time.time_ns() if hasattr(time, "time_ns") else int(time.time() * 1e9)


from stable_baselines3.ppo.ppo_logging import (
    log_programmatic_policy_metrics,
    compute_and_debug_gradient_norms,
    log_gradient_norms,
)

from stable_baselines3.ppo.entropy import ProgramArchitectureEntropy


SelfPPO = TypeVar("SelfPPO", bound="PPO")


class PPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        prog_arch_iter: int = 1,
        prog_prog_iter: int = 10,
        prog_auto_tune_entropy: bool = False,
    ):

        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        if normalize_advantage:

            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:

            buffer_size = self.env.num_envs * self.n_steps

            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"

            untruncated_batches = buffer_size // batch_size

            if buffer_size % batch_size > 0:

                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )

        self.batch_size = batch_size

        self.n_epochs = n_epochs

        self.clip_range = clip_range

        self.clip_range_vf = clip_range_vf

        self.normalize_advantage = normalize_advantage

        self.target_kl = target_kl

        self.train_step = 0

        self.prog_arch_iter = int(prog_arch_iter)

        self.prog_prog_iter = int(prog_prog_iter)

        self.prog_phase_counter = 0

        self.prog_current_phase = 0

        self.prog_architecture_frozen = False

        self.prog_arch_entropy = ProgramArchitectureEntropy(
            auto_tune=prog_auto_tune_entropy,
            device=device,
        )

        if _init_setup_model:

            self._setup_model()

    def _setup_model(self) -> None:

        super()._setup_model()

        self.clip_range = get_schedule_fn(self.clip_range)

        if self.clip_range_vf is not None:

            if isinstance(self.clip_range_vf, (float, int)):

                assert self.clip_range_vf > 0, (
                    "`clip_range_vf` must be positive, "
                    "pass `None` to deactivate vf clipping"
                )

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

        if (
            self.prog_arch_entropy.is_auto_tune
            and hasattr(self.policy, "programmatic_policy")
            and self.policy.programmatic_policy is not None
        ):

            max_depth = self.policy.prog_graph_depth - 1

            self.prog_arch_entropy.setup(max_depth)

    def train(self, debugging: bool = True) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """

        self.policy.set_training_mode(True)

        self._update_learning_rate([self.policy.optimizer])

        clip_range = self.clip_range(self._current_progress_remaining)

        if self.clip_range_vf is not None:

            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []

        pg_losses, value_losses = [], []

        clip_fractions = []

        architecture_entropy_losses = []

        prog_search_grad_norm = 0.0

        prog_fusion_grad_norm = 0.0

        should_flip = False

        if (
            hasattr(self.policy, "programmatic_policy")
            and self.policy.programmatic_policy is not None
        ):

            if hasattr(self.policy.programmatic_policy, "search_map"):

                if self.prog_architecture_frozen:

                    self.prog_current_phase = 1

                    self.policy.programmatic_policy.pointer = 1

                    self.policy.programmatic_policy.search_map.freeze()

                    self.policy.programmatic_policy.fusion_programs.unfreeze()

                elif self.prog_current_phase == 0:

                    self.policy.programmatic_policy.search_map.unfreeze()

                    self.policy.programmatic_policy.fusion_programs.freeze()

                else:

                    self.policy.programmatic_policy.search_map.freeze()

                    self.policy.programmatic_policy.fusion_programs.unfreeze()

        continue_training = True

        for epoch in range(self.n_epochs):

            approx_kl_divs = []

            if not self.rollout_buffer.full:

                print(
                    f"ERROR: Buffer not full before training! buffer_size={self.rollout_buffer.buffer_size}, pos={self.rollout_buffer.pos}, full={self.rollout_buffer.full}"
                )

                raise RuntimeError(
                    f"Cannot train: buffer is not full! "
                    f"buffer_size={self.rollout_buffer.buffer_size}, pos={self.rollout_buffer.pos}, "
                    f"full={self.rollout_buffer.full}. This should not happen."
                )

            print(
                f"[Train] Getting data from buffer: buffer_size={self.rollout_buffer.buffer_size}, n_envs={self.rollout_buffer.n_envs}, batch_size={self.batch_size}, full={self.rollout_buffer.full}"
            )

            try:

                rollout_data_list = list(self.rollout_buffer.get(self.batch_size))

                print(f"[Train] Got {len(rollout_data_list)} batches from buffer")

            except AssertionError as e:

                print(f"ERROR: AssertionError in buffer.get(): {e}")

                print(
                    f"Buffer state: buffer_size={self.rollout_buffer.buffer_size}, pos={self.rollout_buffer.pos}, full={self.rollout_buffer.full}"
                )

                raise

            except Exception as e:

                print(f"ERROR: Exception in buffer.get(): {type(e).__name__}: {e}")

                raise

            for batch_idx, rollout_data in enumerate(rollout_data_list):

                print(
                    f"[Train] Processing batch {batch_idx} of {len(rollout_data_list)}"
                )

                is_last_batch = batch_idx == len(rollout_data_list) - 1

                actions = rollout_data.actions

                if isinstance(self.action_space, spaces.Discrete):

                    actions = rollout_data.actions.long().flatten()

                if self.use_sde:

                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )

                values = values.flatten()

                advantages = rollout_data.advantages

                if self.normalize_advantage and len(advantages) > 1:

                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                policy_loss_1 = advantages * ratio

                policy_loss_2 = advantages * th.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )

                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.detach())

                clip_fraction = th.mean(
                    (th.abs(ratio - 1) > clip_range).float()
                ).detach()

                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:

                    values_pred = values

                else:

                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )

                value_loss = F.mse_loss(rollout_data.returns, values_pred)

                value_losses.append(value_loss.detach())

                if entropy is None:

                    entropy_loss = -th.mean(-log_prob)

                else:

                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.detach())

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                )

                with th.no_grad():

                    log_ratio = log_prob - rollout_data.old_log_prob

                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio)

                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:

                    continue_training = False

                    if self.verbose >= 1:

                        print(
                            f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}"
                        )

                    break

                if self.policy.programmatic_policy is not None:

                    if hasattr(self.policy.programmatic_policy, "search_map"):

                        search_map = self.policy.programmatic_policy.search_map

                        max_depth = self.policy.prog_graph_depth - 1

                        if search_map.type == "simple":

                            v = F.softmax(search_map.v, dim=0)

                        elif search_map.type == "architecture":

                            v_list = []

                            for i in range(max_depth):

                                if i == 0:

                                    v_list.append(
                                        search_map.options[0].softmax(dim=0)[0]
                                    )

                                else:

                                    prev = 1.0

                                    for j in range(i):

                                        prev = (
                                            prev
                                            * search_map.options[j].softmax(dim=0)[1]
                                        )

                                    if i == max_depth - 1:

                                        v_list.append(prev)

                                    else:

                                        v_list.append(
                                            prev
                                            * search_map.options[i].softmax(dim=0)[0]
                                        )

                            v = th.stack(v_list)

                        eps = 1e-8

                        architecture_entropy_loss = -th.sum(v * th.log(v + eps))

                        architecture_entropy_losses.append(
                            architecture_entropy_loss.detach()
                        )

                        if self.prog_current_phase == 0:

                            if (
                                self.prog_arch_entropy.is_auto_tune
                                and self.prog_arch_entropy.is_setup
                            ):

                                H = architecture_entropy_loss

                                self.prog_arch_entropy.update_alpha(H)

                                alpha_arch = self.prog_arch_entropy.get_alpha_tensor()

                                loss += alpha_arch * architecture_entropy_loss

                self.policy.optimizer.zero_grad()

                loss.backward()

                if debugging:

                    prog_search_grad_norm, prog_fusion_grad_norm = (
                        compute_and_debug_gradient_norms(
                            self.policy,
                            self.device,
                            loss,
                            is_last_batch,
                            epoch,
                            self.n_epochs,
                        )
                    )

                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )

                self.policy.optimizer.step()

            self._n_updates += 1

            if not continue_training:

                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        self.train_step += 1

        self.logger.record("train/entropy_loss", th.stack(entropy_losses).mean().item())

        self.logger.record(
            "train/policy_gradient_loss", th.stack(pg_losses).mean().item()
        )

        self.logger.record("train/value_loss", th.stack(value_losses).mean().item())

        self.logger.record("train/approx_kl", th.stack(approx_kl_divs).mean().item())

        self.logger.record(
            "train/clip_fraction", th.stack(clip_fractions).mean().item()
        )

        self.logger.record("train/loss", loss.item())

        self.logger.record("train/explained_variance", explained_var)

        if hasattr(self.policy, "log_std"):

            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

        self.logger.record("train/clip_range", clip_range)

        if self.clip_range_vf is not None:

            self.logger.record("train/clip_range_vf", clip_range_vf)

        if len(architecture_entropy_losses) > 0:

            self.logger.record(
                "train/architecture_entropy",
                th.stack(architecture_entropy_losses).mean().item(),
            )

        entropy_metrics = self.prog_arch_entropy.get_metrics()

        if self.prog_arch_entropy.is_auto_tune and self.prog_arch_entropy.is_setup:

            self.logger.record(
                "train/log_alpha_arch", entropy_metrics.get("log_alpha_arch", 0.0)
            )

            self.logger.record(
                "train/alpha_arch", entropy_metrics.get("alpha_arch", 0.0)
            )

            if "h_ema" in entropy_metrics:

                self.logger.record("train/h_ema", entropy_metrics["h_ema"])

            if "h_max" in entropy_metrics:

                self.logger.record("train/h_max", entropy_metrics["h_max"])

            if "h_target" in entropy_metrics:

                self.logger.record("train/h_target", entropy_metrics["h_target"])

        if (
            hasattr(self.policy, "programmatic_policy")
            and self.policy.programmatic_policy is not None
        ):

            log_programmatic_policy_metrics(
                self.logger,
                self.policy,
                self.device,
                self.prog_current_phase,
                self.prog_phase_counter,
            )

            self.logger.record(
                "train/prog_architecture_frozen", int(self.prog_architecture_frozen)
            )

            if debugging:

                log_gradient_norms(
                    self.logger, prog_search_grad_norm, prog_fusion_grad_norm
                )

            arch_entropy_collapsed = (
                self.prog_current_phase == 0
                and len(architecture_entropy_losses) > 0
                and th.stack(architecture_entropy_losses).mean().item() < 1e-4
            )

            if arch_entropy_collapsed:

                print(
                    "Architecture entropy is below 1e-4, switching to program optimization phase"
                )

                should_flip = True

                self.prog_architecture_frozen = True

                self.prog_current_phase = 1

                self.policy.programmatic_policy.pointer = 1

                self.policy.programmatic_policy.search_map.freeze()

                self.policy.programmatic_policy.fusion_programs.unfreeze()

                self.prog_phase_counter = 0

                phase_name = "Program Optimization"

                print(f"[Programmatic Policy] Switched to {phase_name} phase")

            else:

                self.prog_phase_counter += 1

                should_flip = False

                if self.prog_architecture_frozen:

                    self.prog_current_phase = 1

                elif self.prog_current_phase == 0:

                    if self.prog_phase_counter >= self.prog_arch_iter:

                        should_flip = True

                        self.prog_current_phase = 1

                else:

                    if self.prog_phase_counter >= self.prog_prog_iter:

                        should_flip = True

                        self.prog_current_phase = 0

                if should_flip:

                    self.policy.flip_programmatic_policy()

                    self.prog_phase_counter = 0

                    phase_name = (
                        "Architecture Search"
                        if self.prog_current_phase == 0
                        else "Program Optimization"
                    )

                    print(f"[Programmatic Policy] Switched to {phase_name} phase")

    def learn(
        self: SelfPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfPPO:

        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
