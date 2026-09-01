# System1 I/O Contract (GR00T-N1.6)

This document summarizes the runtime contract used by both:
- `Gr00tN1d6ActionHead` (`system1_type="dit"`)
- `MambaSystem1ActionHead` (`system1_type="mamba"`)

The training/eval pipeline remains unchanged as long as this contract is preserved.

## 1) Forward Training Contract

Entry:
- `Gr00tN1d6.forward(inputs: dict)` calls
  - backbone: `self.backbone(backbone_inputs)`
  - system1: `self.action_head(backbone_outputs, action_inputs)`

### Required `backbone_output` keys
- `backbone_features`: `[B, L, C_backbone]`
- `backbone_attention_mask`: `[B, L]`
- `image_mask`: `[B, L]` (used by DiT alternate path; optional for Mamba if no video injection)

### Required `action_input` keys
- `state`: `[B, S, D_state]`
- `embodiment_id`: `[B]`
- `action`: `[B, T, D_action]`
- `action_mask`: `[B, T, D_action]`

### Forward output keys (must stay fixed)
- `loss`: scalar
- `action_loss`: `[B, T, D_action]`
- `action_mask`: `[B, T, D_action]`
- `backbone_features`: `[B, L, C_backbone]`
- `state_features`: `[B, S, D_state_embed]`

### Loss dependency
- Target is flow-matching velocity:
  - `velocity = action - noise`
- Loss:
  - `mse(pred_actions, velocity) * action_mask`
  - reduced by masked average.

## 2) Inference Contract (`get_action`)

Entry:
- `Gr00tN1d6.get_action(inputs: dict)` -> `self.action_head.get_action(...)`

Required `action_input` keys:
- `state`: `[B, S, D_state]`
- `embodiment_id`: `[B]`

Output keys:
- `action_pred`: `[B, T, D_action]`
- `backbone_features`: `[B, L, C_backbone]`
- `state_features`: `[B, S, D_state_embed]`

`gr00t/policy/gr00t_policy.py` only depends on `action_pred` for decoding.

## 3) Runtime Verification Hooks

Both System1 implementations now include:
- explicit key checks (`KeyError` for missing required keys),
- shape assertions for critical tensors,
- one-time shape logs (train/infer), controlled by:
  - `model.system1_debug_shapes` (default: `True`).

This avoids hardcoding assumptions without runtime confirmation.
