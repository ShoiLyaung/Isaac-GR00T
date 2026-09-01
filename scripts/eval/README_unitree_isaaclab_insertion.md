# Unitree IsaacLab Insertion Evaluation (Server/Client)

This pipeline keeps the original `GR00T-WholeBodyControl` evaluation unchanged and adds a separate
client path for:

- task env: `Isaac-Insertion-Shaft-G129-Inspire-Joint-Play`
- env name in rollout: `unitree_g1_isaaclab/Isaac-Insertion-Shaft-G129-Inspire-Joint-Play`

Original GR00T-WholeBodyControl shortcuts are:

- `bash scripts/eval/run_gr00t_wholebody_server.sh`
- `bash scripts/eval/run_gr00t_wholebody_client.sh`

## 1) Optional readiness check

```bash
CHECK_UNITREE_ISAACLAB=1 \
UNITREE_SIM_ROOT=/home/sl/humanoid-lfd/unitree_sim_isaaclab \
UNITREE_SIM_PYTHON=/home/sl/miniconda3/envs/env_isaaclab/bin/python \
python scripts/eval/check_sim_eval_ready.py
```

## 2) Start server (terminal A)

```bash
MODEL_PATH=/path/to/checkpoint \
EMBODIMENT_TAG=UNITREE_G1 \
bash scripts/eval/run_insertion_shaft_server.sh
```

## 3) Start client (terminal B)

```bash
UNITREE_SIM_ROOT=/home/sl/humanoid-lfd/unitree_sim_isaaclab \
UNITREE_SIM_PYTHON=/home/sl/miniconda3/envs/env_isaaclab/bin/python \
UNITREE_SIM_TASK=Isaac-Insertion-Shaft-G129-Inspire-Joint-Play \
N_EPISODES=10 \
MAX_EPISODE_STEPS=800 \
N_ACTION_STEPS=20 \
N_ENVS=1 \
bash scripts/eval/run_insertion_shaft_client.sh
```

## Notes

- `N_ENVS` is currently required to be `1` for IsaacLab insertion.
- If your checkpoint was trained with another embodiment, set `EMBODIMENT_TAG` accordingly on server side.
- Existing `GR00T-WholeBodyControl` commands remain the same.
