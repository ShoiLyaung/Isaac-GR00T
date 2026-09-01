#!/usr/bin/env bash
set -euo pipefail
set -x

# Usage:
#   bash examples/GR00T-WholeBodyControl/finetune_g1_mamba.sh
# Optional env vars:
#   NUM_GPUS=2 MASTER_PORT=29550 USE_WANDB=1 OUTPUT_DIR=/abs/path/to/output
#   REQUIRE_REAL_MAMBA_BACKEND=1  # fail fast if mamba_ssm extension is unavailable
#   MAX_STEPS=100 DATALOADER_NUM_WORKERS=2  # convenient smoke-test overrides
#   FINITE_CHECK_EVERY_STEPS=1 STOP_ON_NON_FINITE_PARAMS=1  # numerical debug
#   SYSTEM1_DEBUG_GRADIENTS=1  # log non-finite gradients on key System1 intermediates
#   BF16=0  # disable bf16 and run full fp32 for numerical debugging
#   DEBUG_ROBOT_MAMBA_MODE=skip|norm_only  # isolate robot_mamba behavior
#   MAMBA_ENABLE_VIDEO_INJECTION=1  # enable Phase-2 Video->Robot injection

export NUM_GPUS="${NUM_GPUS:-2}"
export MASTER_PORT="${MASTER_PORT:-29550}"
export DATASET_PATH="${DATASET_PATH:-examples/GR00T-WholeBodyControl/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/unitree_g1.LMPnPAppleToPlateDC}"
export OUTPUT_DIR="${OUTPUT_DIR:-/home/sl/Isaac-GR00T/outputs/g1_mamba_fulltrain_$(date +%Y%m%d_%H%M%S)}"
export MAX_STEPS="${MAX_STEPS:-10000}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-6}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"

mkdir -p "${OUTPUT_DIR}"

# Improve distributed debug visibility and avoid flaky fast paths on some hosts.
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
# Force single-node rendezvous/collective traffic over localhost to avoid
# "NCCL bind failed: Cannot assign requested address" on multi-NIC hosts.
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export NO_ALBUMENTATIONS_UPDATE=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv_cache}"

WANDB_FLAG=()
if [[ "${USE_WANDB:-0}" == "1" ]]; then
    WANDB_FLAG+=(--use_wandb)
fi

MAMBA_BACKEND_FLAG=()
if [[ "${REQUIRE_REAL_MAMBA_BACKEND:-0}" == "1" ]]; then
    MAMBA_BACKEND_FLAG+=(--require-real-mamba-backend)
fi

FINITE_DEBUG_FLAG=()
if [[ "${FINITE_CHECK_EVERY_STEPS:-0}" != "0" ]]; then
    FINITE_DEBUG_FLAG+=(--finite-check-every-steps "${FINITE_CHECK_EVERY_STEPS}")
fi
if [[ "${STOP_ON_NON_FINITE_PARAMS:-0}" == "1" ]]; then
    FINITE_DEBUG_FLAG+=(--stop-on-non-finite-params)
fi

SYSTEM1_GRAD_FLAG=()
if [[ "${SYSTEM1_DEBUG_GRADIENTS:-0}" == "1" ]]; then
    SYSTEM1_GRAD_FLAG+=(--system1-debug-gradients)
fi

BF16_FLAG=()
if [[ "${BF16:-1}" == "0" ]]; then
    BF16_FLAG+=(--no-bf16)
fi

ANOMALY_FLAG=()
if [[ "${DETECT_ANOMALY:-0}" == "1" ]]; then
    ANOMALY_FLAG+=(--detect-anomaly)
fi

ROBOT_MAMBA_DEBUG_FLAG=()
if [[ "${DEBUG_ROBOT_MAMBA_MODE:-full}" != "full" ]]; then
    ROBOT_MAMBA_DEBUG_FLAG+=(--debug-robot-mamba-mode "${DEBUG_ROBOT_MAMBA_MODE}")
fi

VIDEO_INJECTION_FLAG=()
if [[ "${MAMBA_ENABLE_VIDEO_INJECTION:-0}" == "1" ]]; then
    VIDEO_INJECTION_FLAG+=(--mamba-enable-video-injection)
else
    VIDEO_INJECTION_FLAG+=(--no-mamba-enable-video-injection)
fi

uv --preview-features extra-build-dependencies run torchrun \
    --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" \
    gr00t/experiment/launch_finetune.py \
    --base_model_path nvidia/GR00T-N1.6-3B \
    --dataset_path "${DATASET_PATH}" \
    --embodiment_tag UNITREE_G1 \
    --num_gpus "${NUM_GPUS}" \
    --output_dir "${OUTPUT_DIR}" \
    --save_total_limit 5 \
    --max_steps "${MAX_STEPS}" \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    "${BF16_FLAG[@]}" \
    "${ROBOT_MAMBA_DEBUG_FLAG[@]}" \
    "${ANOMALY_FLAG[@]}" \
    --global_batch_size "${GLOBAL_BATCH_SIZE}" \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --system1_type mamba \
    --system1_debug_shapes \
    "${SYSTEM1_GRAD_FLAG[@]}" \
    "${VIDEO_INJECTION_FLAG[@]}" \
    --no-backbone-trainable-params-fp32 \
    "${MAMBA_BACKEND_FLAG[@]}" \
    "${FINITE_DEBUG_FLAG[@]}" \
    "${WANDB_FLAG[@]}" \
    2>&1 | tee "${OUTPUT_DIR}/train.log"
