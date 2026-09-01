from pathlib import Path
import shutil
import logging

from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
import torch


class CheckpointFormatCallback(TrainerCallback):
    """This callback format checkpoint to make them standalone. For now, it copies all config
    files to /checkpoint-{step}/experiment_cfg/:
    - conf.yaml
    - initial_actions.npz
    - metadata.json
    """

    def __init__(
        self, run_name: str, exp_cfg_dir: Path | None = None, processor_dir: Path | None = None
    ):
        """
        Args:
            run_name: Name of the experiment run
            exp_cfg_dir: Path to the directory containing all experiment metadata
        """
        self.exp_cfg_dir = exp_cfg_dir
        self.processor_dir = processor_dir

    def on_save(self, args, state, control, **kwargs):
        """Called after the trainer saves a checkpoint."""
        if state.is_world_process_zero:
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"

            # Copy experiment config directory if provided
            if self.exp_cfg_dir is not None:
                exp_cfg_dst = checkpoint_dir / self.exp_cfg_dir.name
                if self.exp_cfg_dir.exists():
                    print(
                        f"Copying experiment config directory {self.exp_cfg_dir} to {exp_cfg_dst}"
                    )
                    shutil.copytree(self.exp_cfg_dir, exp_cfg_dst, dirs_exist_ok=True)

            # Copy processor directory if provided
            if self.processor_dir is not None:
                if self.processor_dir.exists():
                    print(f"Copying processor directory {self.processor_dir} to {checkpoint_dir}")
                    shutil.copytree(self.processor_dir, checkpoint_dir, dirs_exist_ok=True)

            # Copy wandb_config.json if provided
            wandb_config_src = Path(args.output_dir) / "wandb_config.json"
            wandb_config_dst = checkpoint_dir / "wandb_config.json"
            if wandb_config_src.exists():
                print(f"Copying wandb_config.json from {wandb_config_src} to {wandb_config_dst}")
                shutil.copy2(wandb_config_src, wandb_config_dst)


class BestMetricCheckpointCallback(TrainerCallback):
    """This callback saves the best checkpoint based on the metric."""

    def __init__(
        self, metric_name: str, greater_is_better: bool = True, exp_cfg_dir: Path | None = None
    ):
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.best_metric = -float("inf") if greater_is_better else float("inf")
        self.exp_cfg_dir = exp_cfg_dir
        self._best_checkpoint_dir = None

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics,
        model,
        **kwargs,
    ):
        if state.is_world_process_zero and metrics is not None:
            current_metric = metrics.get(self.metric_name, None)
            if current_metric is not None:
                is_better = (
                    self.greater_is_better
                    if current_metric > self.best_metric
                    else not self.greater_is_better
                )
                if is_better:
                    self.best_metric = current_metric
                    best_checkpoint_dir = (
                        Path(args.output_dir)
                        / f"checkpoint-{state.global_step}-best-{self.metric_name}_{current_metric}"
                    )
                    best_checkpoint_dir.mkdir(exist_ok=True)
                    model.save_pretrained(best_checkpoint_dir)
                    # Copy experiment config directory if provided
                    if self.exp_cfg_dir is not None:
                        exp_cfg_dst = best_checkpoint_dir / self.exp_cfg_dir.name
                        if self.exp_cfg_dir.exists():
                            print(
                                f"Copying experiment config directory {self.exp_cfg_dir} to {exp_cfg_dst}"
                            )
                            shutil.copytree(self.exp_cfg_dir, exp_cfg_dst, dirs_exist_ok=True)

                    print(
                        f"Best checkpoint saved to {best_checkpoint_dir} with metric {self.metric_name} = {current_metric}"
                    )

                    if (
                        self._best_checkpoint_dir is not None
                        and Path(self._best_checkpoint_dir).exists()
                    ):
                        shutil.rmtree(self._best_checkpoint_dir)

                    self._best_checkpoint_dir = str(best_checkpoint_dir)


class FiniteParameterCallback(TrainerCallback):
    """Periodically inspect model parameters and report the first non-finite tensors."""

    def __init__(
        self,
        every_steps: int,
        module_prefix: str = "action_head",
        stop_on_non_finite: bool = False,
    ):
        self.every_steps = every_steps
        self.module_prefix = module_prefix
        self.stop_on_non_finite = stop_on_non_finite
        self._reported_bad_params = False
        self._reported_bad_grads = False

    def _collect_non_finite_names(self, model, check_grad: bool) -> list[str]:
        bad = []
        for name, param in model.named_parameters():
            if self.module_prefix and not name.startswith(self.module_prefix):
                continue
            tensor = param.grad if check_grad else param
            if tensor is None:
                continue
            if not torch.isfinite(tensor).all():
                bad.append(name)
                if len(bad) >= 20:
                    break
        return bad

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if model is None or self.every_steps <= 0:
            return control
        pending_step = state.global_step + 1
        if pending_step % self.every_steps != 0:
            return control

        bad_grads = self._collect_non_finite_names(model, check_grad=True)
        if bad_grads and not self._reported_bad_grads:
            logging.error(
                "Detected non-finite gradients before optimizer step=%s prefix=%s count=%d sample=%s",
                pending_step,
                self.module_prefix,
                len(bad_grads),
                bad_grads,
            )
            self._reported_bad_grads = True
        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None or self.every_steps <= 0:
            return control
        if state.global_step <= 0 or state.global_step % self.every_steps != 0:
            return control

        bad = self._collect_non_finite_names(model, check_grad=False)
        if not bad:
            return control

        if not self._reported_bad_params:
            logging.error(
                "Detected non-finite parameters at global_step=%s prefix=%s count=%d sample=%s",
                state.global_step,
                self.module_prefix,
                len(bad),
                bad,
            )
            self._reported_bad_params = True

        if self.stop_on_non_finite:
            logging.error(
                "Stopping training at global_step=%s because stop_on_non_finite_params=True",
                state.global_step,
            )
            control.should_training_stop = True

        return control
