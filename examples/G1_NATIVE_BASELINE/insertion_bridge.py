"""Frozen R3 client compatibility entrypoint for the q-only action-head baseline."""

import importlib.util
from pathlib import Path


_SOURCE = Path(__file__).resolve().parents[1] / "G1_DIT_MAMBA_BASELINE/insertion_bridge.py"
_SPEC = importlib.util.spec_from_file_location("g1_dit_mamba_qonly_bridge", _SOURCE)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

PROTOCOL_ID = _MODULE.PROTOCOL_ID
ARM_NAMES = _MODULE.ARM_NAMES
HAND_NAMES = _MODULE.HAND_NAMES
HAND_MIN = _MODULE.HAND_MIN
HAND_MAX = _MODULE.HAND_MAX
HAND_DEPENDENCIES = _MODULE.HAND_DEPENDENCIES
hand_radians_to_u = _MODULE.hand_radians_to_u
hand_u_to_radians = _MODULE.hand_u_to_radians
combine_body_telemetry = _MODULE.combine_body_telemetry
observation_from_sim = _MODULE.observation_from_sim
action_to_sim = _MODULE.action_to_sim
