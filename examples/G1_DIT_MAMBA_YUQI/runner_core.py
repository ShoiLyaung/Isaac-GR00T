"""Small deterministic contracts shared by the Yuqi full-vision runner and CPU tests."""

from contextlib import contextmanager
import hashlib
import random

import numpy as np
import torch


STEPS = 5000
GLOBAL_BATCH = 32
SEEN_EPISODES = (150, 147, 152, 112)
NOISE_SEEDS = (42, 43, 44, 45)
PROBE_STEPS = (0, 1000, 2500, 5000)


def deterministic_seed(*coordinates):
    """Stable independent streams; unrelated RNG draws cannot change a later sample."""
    digest = hashlib.sha256(": ".join(map(str, (42, *coordinates))).encode()).digest()
    return int.from_bytes(digest[:4], "little")


def set_seed(value, *, cuda=False):
    random.seed(value)
    np.random.seed(value % (2**32))
    # torch.manual_seed also touches CUDA; workers must remain strictly CPU-only.
    torch.random.default_generator.manual_seed(value)
    if cuda:
        torch.cuda.manual_seed_all(value)


def make_schedule(lengths, *, steps=STEPS):
    """Shuffle all legal H16 anchors in full epochs, never pad targets."""
    legal = np.asarray(
        [
            (episode, anchor)
            for episode, length in sorted(lengths.items())
            for anchor in range(length - 15)
        ],
        dtype=np.int32,
    )
    if not len(legal):
        raise ValueError("No legal H16 training anchors")
    rng = np.random.default_rng(42)
    needed = steps * GLOBAL_BATCH
    epochs = []
    while sum(len(epoch) for epoch in epochs) < needed:
        epochs.append(legal[rng.permutation(len(legal))])
    schedule = np.concatenate(epochs)[:needed].reshape(steps, GLOBAL_BATCH, 2)
    diag = []
    for episode in SEEN_EPISODES:
        if lengths[episode] - 15 < 32:
            raise ValueError("Each seen diagnostic episode needs 32 legal anchors")
        anchors = np.sort(rng.choice(lengths[episode] - 15, 32, replace=False))
        diag.extend([[episode, int(anchor)] for anchor in anchors])
    eligible = legal[~np.isin(legal[:, 0], SEEN_EPISODES)]
    train_probe = eligible[rng.choice(len(eligible), 16, replace=False)].tolist()
    seen_probe = [diag[episode * 32 + anchor] for episode in range(4) for anchor in (0, 8, 16, 24)]
    return schedule, {"diagnostic": diag, "train_probe": train_probe, "seen_probe": seen_probe}


def rank_slots(rank, microbatch, accumulation):
    if rank not in (0, 1) or microbatch * accumulation * 2 != GLOBAL_BATCH:
        raise ValueError("Require two ranks and global batch32")
    return [
        list(range((micro * 2 + rank) * microbatch, (micro * 2 + rank + 1) * microbatch))
        for micro in range(accumulation)
    ]


def assert_mask(mask):
    if tuple(mask.shape[-2:]) != (40, 13):
        raise ValueError("Expected model H40/compact13")
    if not torch.all(mask[..., :16, :] == 1) or not torch.all(mask[..., 16:, :] == 0):
        raise ValueError("Expected exactly 16x13 active entries per example")


def masked_flow(loss, mask):
    assert_mask(mask)
    if loss.shape != mask.shape:
        raise ValueError("Native elementwise loss/mask shape mismatch")
    if not torch.isfinite(loss).all():
        raise FloatingPointError("Nonfinite native elementwise flow loss")
    # Native action_loss already contains the mask. Multiplication is idempotent.
    return (loss * mask).sum(dim=(-2, -1)) / mask.sum(dim=(-2, -1))


@contextmanager
def preserve_eval_state(model, processor=None):
    """Restore each module's own mode plus all process RNG streams, even on error."""
    modes = [(module, module.training) for module in model.modules()]
    processor_mode = getattr(processor, "training", None)
    state_processor = getattr(processor, "state_action_processor", None)
    state_mode = getattr(state_processor, "training", None)
    py_rng, np_rng, cpu_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        model.eval()
        if processor is not None:
            processor.eval()
        yield
    finally:
        # module.train(flag) recursively overwrites heterogeneous frozen submodule modes.
        for module, mode in modes:
            module.training = mode
        if processor_mode is not None:
            processor.training = processor_mode
        if state_mode is not None:
            state_processor.training = state_mode
        random.setstate(py_rng)
        np.random.set_state(np_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def aggregate_metrics(rows, keys):
    """Noise seed -> anchor -> episode; episodes receive equal weight."""
    episode_rows = {}
    for episode in sorted({row["episode"] for row in rows}):
        subset = [row for row in rows if row["episode"] == episode]
        anchors = sorted({row["anchor"] for row in subset})
        episode_rows[str(episode)] = {
            key: float(
                np.mean(
                    [
                        np.mean([row[key] for row in subset if row["anchor"] == anchor])
                        for anchor in anchors
                    ]
                )
            )
            for key in keys
        }
    return episode_rows, {
        key: float(np.mean([row[key] for row in episode_rows.values()])) for key in keys
    }
