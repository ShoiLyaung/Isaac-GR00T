"""Smoke tests for UnitreeG1Env registration.

These tests deliberately avoid spinning up IsaacLab (which requires a GPU and
takes ~30 seconds per call). They verify:

1. The module imports cleanly.
2. ``register_unitree_g1_envs()`` registers the expected gym id.
3. The constants used to bridge obs/action streams are consistent.
4. ``EnvSpec`` for that id has the expected entry point and kwargs.
5. ``get_unitree_g1_env_fn()`` factory closure is built correctly.

We intentionally avoid importing ``gr00t.eval.rollout_policy`` from inside
these tests because that module imports ``gr00t.policy`` at the top level,
which transitively imports ``transformers`` and ``flash_attn``. Those imports
are broken on this machine due to a torch / flash-attn ABI mismatch that is
unrelated to our changes. The dispatch path is verified by a separate test
that loads ``rollout_policy.py`` via importlib with a stubbed
``gr00t.policy`` module.

Run with::

    python -m gr00t.eval.sim.UnitreeG1.test_smoke

or::

    pytest gr00t/eval/sim/UnitreeG1/test_smoke.py -v
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def test_module_imports() -> None:
    """E3 + E4: the new package and module are importable."""
    from gr00t.eval.sim import UnitreeG1  # noqa: F401
    from gr00t.eval.sim.UnitreeG1 import unitree_g1_env  # noqa: F401

    assert hasattr(unitree_g1_env, "UnitreeG1Env")
    assert hasattr(unitree_g1_env, "register_unitree_g1_envs")
    print("[ok] UnitreeG1 package + module import")


def test_constants_are_consistent() -> None:
    """Module-level invariants. The actual joint IDs are now resolved
    at runtime from physx's joint_names, so we only assert that the
    helper methods exist and that the obs chunk size is fixed."""
    from gr00t.eval.sim.UnitreeG1.unitree_g1_env import (
        BOY_NUM_JOINTS,
        UnitreeG1Env,
    )

    assert BOY_NUM_JOINTS == 29, f"expected 29 boy joints, got {BOY_NUM_JOINTS}"
    # The class must expose the runtime joint-id resolution helpers.
    assert hasattr(UnitreeG1Env, "_resolve_joint_ids")
    assert hasattr(UnitreeG1Env, "_ids_by_name")
    assert hasattr(UnitreeG1Env, "_map_action")
    # Hardcoded constants have been removed in favour of dynamic lookup.
    for removed in (
        "LEFT_ARM_ACTION_IDS",
        "RIGHT_ARM_ACTION_IDS",
        "LEFT_HAND_ACTION_IDS",
        "RIGHT_HAND_ACTION_IDS",
        "WAIST_ACTION_IDS",
        "INSPIRE_LEFT_HAND_IDS",
        "INSPIRE_RIGHT_HAND_IDS",
    ):
        from gr00t.eval.sim.UnitreeG1 import unitree_g1_env as mod

        assert not hasattr(mod, removed), (
            f"hardcoded constant {removed} should have been removed"
        )
    print("[ok] joint-id resolution is dynamic (no hardcoded indices)")


def test_register_unitree_g1_envs() -> None:
    """E4: register_unitree_g1_envs() registers the expected gym id."""
    import gymnasium as gym

    from gr00t.eval.sim.UnitreeG1.unitree_g1_env import register_unitree_g1_envs

    register_unitree_g1_envs()
    spec = gym.spec("unitree_g1_isaaclab/Isaac-Insertion-Shaft-G129-Inspire-Joint-Play")
    assert spec.id == "unitree_g1_isaaclab/Isaac-Insertion-Shaft-G129-Inspire-Joint-Play"
    assert spec.kwargs.get("task_name") == "Isaac-Insertion-Shaft-G129-Inspire-Joint-Play"
    # Entry point should resolve to our class.
    assert spec.entry_point.endswith("UnitreeG1Env"), spec.entry_point
    print(f"[ok] env registered: {spec.id}")


def test_env_utils_dispatch_for_unitree_g1() -> None:
    """E1: prefix -> tag mapping returns UNITREE_G1 for unitree_g1_isaaclab/*."""
    from gr00t.eval.sim.env_utils import (
        ENV_PREFIX_TO_EMBODIMENT_TAG,
        get_embodiment_tag_from_env_name,
    )

    assert "unitree_g1_isaaclab" in ENV_PREFIX_TO_EMBODIMENT_TAG
    tag = get_embodiment_tag_from_env_name(
        "unitree_g1_isaaclab/Isaac-Insertion-Shaft-G129-Inspire-Joint-Play"
    )
    assert tag.value == "unitree_g1_full_body_with_waist_height_nav_cmd", tag
    print("[ok] env_utils dispatches unitree_g1_isaaclab prefix to UNITREE_G1")


def _load_rollout_policy_with_stub() -> types.ModuleType:
    """Load gr00t/eval/rollout_policy.py without triggering the transformers
    import chain (which is broken on this host due to flash_attn ABI)."""
    # Pre-stub gr00t.policy so the top-level ``from gr00t.policy import
    # BasePolicy`` in rollout_policy.py resolves to a dummy class.
    if "gr00t.policy" not in sys.modules:
        stub = types.ModuleType("gr00t.policy")

        class _BasePolicy:  # noqa: D401 - simple placeholder
            pass

        stub.BasePolicy = _BasePolicy
        sys.modules["gr00t.policy"] = stub

    spec = importlib.util.spec_from_file_location(
        "_rollout_policy_under_test",
        os.path.join(_REPO_ROOT, "gr00t", "eval", "rollout_policy.py"),
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rollout_dispatch_for_unitree_g1() -> None:
    """E2: get_gym_env() should NOT raise for a unitree_g1_isaaclab env name.

    We patch ``gym.make`` so we don't actually instantiate the env (which
    would launch IsaacLab). The test verifies that the dispatch branches
    through get_unitree_g1_env_fn and not the LIBERO/SimplerEnv fall-through.
    """
    import gymnasium as gym

    mod = _load_rollout_policy_with_stub()

    # Sanity-check the new factory exists.
    assert hasattr(mod, "get_unitree_g1_env_fn")
    env_fn_factory = mod.get_unitree_g1_env_fn(
        "unitree_g1_isaaclab/Isaac-Insertion-Shaft-G129-Inspire-Joint-Play"
    )
    assert callable(env_fn_factory)

    # Now exercise the real get_gym_env() dispatch path with a stubbed gym.make.
    env_name = "unitree_g1_isaaclab/Isaac-Insertion-Shaft-G129-Inspire-Joint-Play"
    called = {"n": 0, "with": None}

    def fake_make(name, *args, **kwargs):
        called["n"] += 1
        called["with"] = name
        return object()

    original_make = gym.make
    gym.make = fake_make  # type: ignore[assignment]
    try:
        result = mod.get_gym_env(env_name, env_idx=0, total_n_envs=1)
    finally:
        gym.make = original_make  # type: ignore[assignment]

    assert called["n"] == 1, f"gym.make should be called exactly once, got {called}"
    assert called["with"] == env_name
    assert result is not None
    print("[ok] rollout_policy.get_gym_env dispatches to UnitreeG1 factory")


if __name__ == "__main__":
    test_module_imports()
    test_constants_are_consistent()
    test_register_unitree_g1_envs()
    test_env_utils_dispatch_for_unitree_g1()
    test_rollout_dispatch_for_unitree_g1()
    print("\nALL SMOKE TESTS PASSED")