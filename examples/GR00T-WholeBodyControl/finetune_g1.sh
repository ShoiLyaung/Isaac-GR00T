
set -x -e

export NUM_GPUS=2
# 强化 NCCL 可见性
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576

# 关键：禁用最容易在某些机器上“偶发挂”的 fast path
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1

torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 \
    gr00t/experiment/launch_finetune.py \
    --base_model_path  nvidia/GR00T-N1.6-3B \
    --dataset_path examples/GR00T-WholeBodyControl/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/unitree_g1.LMPnPAppleToPlateDC \
    --embodiment_tag UNITREE_G1 \
    --num_gpus $NUM_GPUS \
    --output_dir /tmp/g1_finetune \
    --save_total_limit 5 \
    --max_steps 10000 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --use_wandb \
    --global_batch_size 128 \
    --dataloader_num_workers 6 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    2>&1 | tee /tmp/gr00t_finetune_test.log
