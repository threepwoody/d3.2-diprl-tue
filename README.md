# DiPRL

DiPRL is a compact implementation of Differentiable Discrete Programmatic Reinforcement Learning. The project trains interpretable programmatic policies: policies represented as small if/else-style programs over state features instead of only as opaque neural-network action heads.


## Setup

Create a Python environment, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The listed dependencies are intentionally minimal: Gymnasium, PyTorch, NumPy/Pandas utilities, tqdm, cloudpickle, and Weights & Biases.

## Training

Run a CartPole DiPRL training job with:

```bash
python diprl.py \
  --env CartPole-v1 \
  --run_id 0 \
  --total_timesteps 2000000 \
  --learning_rate 1e-3 \
  --batch_size 64 \
  --use_programmatic_policy \
  --prog_graph_depth 6 \
  --prog_beta 0.25 \
  --prog_arch_iter 1 \
  --prog_prog_iter 4 \
  --prog_auto_tune_entropy \
  --prog_equally_initialize \
  --constant_lr
```

Use `--run_id -1` for a local smoke run without Weights & Biases initialization. Logged runs are grouped by the generated run name and seed. Model artifacts are saved under `runs/models` by default, and W&B files are written under `runs/wandb`.

To train the neural PPO baseline path instead, omit `--use_programmatic_policy`; the same entrypoint then uses the regular Stable-Baselines3-style MLP actor head.

## Key Flags

| Flag | Default | Description |
| --- | --- | --- |
| `--env` | `CartPole-v1` | Supported Gymnasium environment. This repo currently restricts the value to `CartPole-v1`. |
| `--run_id` | `0` | Selects a seed from the built-in seed list. Use `-1` to skip W&B and save to a local test folder. |
| `--use_programmatic_policy` | off | Replaces the PPO actor action head with the DiPRL program derivation graph. |
| `--prog_graph_depth` | `6` | Maximum derivation-graph depth; program depths range up to `graph_depth - 1`. |
| `--prog_beta` | `0.25` | Temperature for softmax weighting over primitive actions. |
| `--prog_arch_iter` | `1` | Number of PPO training iterations spent in the architecture-search phase before switching phases. |
| `--prog_prog_iter` | `4` | Number of PPO training iterations spent optimizing program predicates and weights before switching back. |
| `--prog_auto_tune_entropy` | off | Enables automatic tuning of the architecture entropy coefficient. |
| `--prog_equally_initialize` | off | Initializes architecture-search probabilities to give program depths a more even starting distribution. |
| `--constant_lr` | off | Keeps the learning rate fixed instead of linearly decaying it with training progress. |

## Method Notes

The CartPole programmatic policy uses four symbolic state features: cart position, cart velocity, pole angle, and pole angular velocity. The terminal primitives are the discrete CartPole actions `left` and `right`.

During training, the modified PPO implementation alternates between:

1. architecture search, where the distribution over candidate program depths is optimized; and
2. program optimization, where symbolic predicates and primitive-action weights are optimized.

When architecture entropy auto-tuning is enabled, the implementation tracks the entropy of the depth distribution and adjusts the regularization coefficient so the relaxed program gradually concentrates toward a discrete architecture.

## Limitations

This repository is a focused CartPole implementation. The paper discusses additional discrete and continuous-control experiments, but those environments and experiment pipelines are not included here.
