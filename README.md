# DiPRL

## Setup

Create a Python environment and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Training

Primary command:

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

Use `--run_id -1` for local runs without Weights & Biases logging.
