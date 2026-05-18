#!/usr/bin/env python


import argparse

import sys

import types

from pathlib import Path


DIPRL_PACKAGE_DIR = Path(__file__).resolve().with_name("diprl")

diprl_package = types.ModuleType("diprl")

diprl_package.__path__ = [str(DIPRL_PACKAGE_DIR)]

sys.modules["diprl"] = diprl_package


import gymnasium as gym

import torch

import wandb

from stable_baselines3 import PPO

from stable_baselines3.common.callbacks import CallbackList, EvalCallback

from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.vec_env import SubprocVecEnv

from wandb.integration.sb3 import WandbCallback


from utils.helpers import format_number, set_random_seed

from wrappers.classic_control_wrappers import StateClassicControlEnv


SEEDS = [123, 456, 789, 1011, 1213, 1415, 321, 654]


def parse_args():

    parser = argparse.ArgumentParser(description="diprl CartPole training entrypoint.")

    parser.add_argument(
        "--env", type=str, default="CartPole-v1", choices=["CartPole-v1"]
    )

    parser.add_argument("--run_id", type=int, default=0)

    parser.add_argument("--save_dir", type=str, default="runs/models")

    parser.add_argument("--wandb_dir", type=str, default="runs/wandb")

    parser.add_argument("--project", type=str, default="diprl")

    parser.add_argument("--additional_name", type=str, default="")

    parser.add_argument("--total_timesteps", type=int, default=2_000_000)

    parser.add_argument("--learning_rate", type=float, default=1e-3)

    parser.add_argument("--constant_lr", action="store_true")

    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--num_envs", type=int, default=8)

    parser.add_argument("--n_steps", type=int, default=512)

    parser.add_argument("--ent_coef", type=float, default=0.01)

    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--use_programmatic_policy", action="store_true")

    parser.add_argument("--prog_graph_depth", type=int, default=6)

    parser.add_argument("--prog_arch_iter", type=int, default=1)

    parser.add_argument("--prog_prog_iter", type=int, default=4)

    parser.add_argument("--prog_beta", type=float, default=0.25)

    parser.add_argument("--prog_auto_tune_entropy", action="store_true")

    parser.add_argument("--prog_equally_initialize", action="store_true")

    return parser.parse_args()


def build_run_name(args):

    parts = [
        args.env.replace("-", "_"),
        format_number(args.total_timesteps),
    ]

    if args.use_programmatic_policy:

        parts.extend(
            [
                "prog",
                str(args.prog_graph_depth),
                str(args.prog_beta),
                str(args.prog_arch_iter),
                str(args.prog_prog_iter),
            ]
        )

    if args.prog_auto_tune_entropy:

        parts.append("autoH")

    if args.constant_lr:

        parts.append("constLR")

    if args.additional_name:

        parts.append(args.additional_name)

    return "-".join(parts)


def main():

    args = parse_args()

    device = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"

    seed = 0 if args.run_id == -1 else SEEDS[args.run_id]

    run_name = build_run_name(args)

    config = vars(args).copy()

    config.update(
        {
            "seed": seed,
            "device": device,
            "num_envs": args.num_envs,
            "n_steps": args.n_steps,
            "n_epochs": 10,
            "vf_coef": 0.5,
            "run_name": run_name,
        }
    )

    print("\nTRAINING CONFIGURATION")

    for key, value in sorted(config.items()):

        print(f"  {key}: {value}")

    print()

    if args.run_id == -1:

        run = None

        save_folder = Path(args.save_dir) / f"test_{run_name}"

    else:

        run = wandb.init(
            project=args.project,
            group=run_name[:120],
            name=f"{run_name}-{seed}",
            config=config,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
            dir=args.wandb_dir,
        )

        save_folder = Path(args.save_dir) / f"{run.id}-{run_name}_seed{seed}"

    save_folder.mkdir(parents=True, exist_ok=True)

    set_random_seed(seed)

    env = make_vec_env(
        args.env,
        n_envs=config["num_envs"],
        wrapper_class=StateClassicControlEnv,
        vec_env_cls=SubprocVecEnv if args.num_envs > 1 else None,
    )

    eval_env = gym.make(args.env)

    eval_env = StateClassicControlEnv(eval_env)

    lr_schedule = (
        args.learning_rate
        if args.constant_lr
        else (lambda progress_remaining: args.learning_rate * progress_remaining)
    )

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=config["n_steps"],
        n_epochs=config["n_epochs"],
        learning_rate=lr_schedule,
        ent_coef=args.ent_coef,
        clip_range=lambda progress_remaining: 0.1 * progress_remaining,
        batch_size=args.batch_size,
        verbose=1 if args.run_id == -1 else 0,
        seed=seed,
        tensorboard_log=str(save_folder),
        device=device,
        policy_kwargs={
            "net_arch": dict(pi=[64, 64], vf=[64, 64]),
            "use_programmatic_policy": args.use_programmatic_policy,
            "prog_graph_depth": args.prog_graph_depth,
            "prog_beta": args.prog_beta,
            "prog_domain": args.env,
            "prog_equally_initialize": args.prog_equally_initialize,
        },
        prog_arch_iter=args.prog_arch_iter,
        prog_prog_iter=args.prog_prog_iter,
        prog_auto_tune_entropy=args.prog_auto_tune_entropy,
    )

    if args.run_id == -1:

        callback = None

    else:

        callback = CallbackList(
            [
                EvalCallback(
                    eval_env,
                    n_eval_episodes=20,
                    best_model_save_path=str(save_folder),
                    log_path=str(save_folder),
                    eval_freq=max(args.total_timesteps // 10 // config["num_envs"], 1),
                    deterministic=True,
                    render=False,
                ),
                WandbCallback(),
            ]
        )

    model.learn(total_timesteps=args.total_timesteps, callback=callback)

    model.save(save_folder / "model.zip")

    if args.run_id != -1:

        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)

        wandb.log({"final/mean_reward": mean_reward, "final/std_reward": std_reward})

        wandb.finish()

        print(f"Final evaluation: {mean_reward:.2f} +/- {std_reward:.2f}")

    env.close()

    eval_env.close()

    print(f"Artifacts saved to: {save_folder}")


if __name__ == "__main__":

    main()
