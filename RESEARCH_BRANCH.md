# Research branch scope

This branch is a GR00T N1.7 research prototype for causal history modeling on
the MIKASA history-hard diagnostic.

- Base: `feat/groot-n17-main-h40-r2` at
  `5615f267c990c66165b2b23867d6a1`.
- Source snapshot before this note:
  `5892d36be531490e9dfff87b4999b8914b061724`.
- Scope: frozen pretrained visual-feature audits, history-order controls,
  learned spatial resampling, causal Mamba training/evaluation runners, and
  focused tests.
- Scientific status: prototype only. Training-cohort memorization passed, but
  held-out cross-geometry ordered-history generalization failed and did not
  establish a Mamba advantage.
- Integration boundary: this is not yet a complete GR00T VLA policy. The
  history-hard task and expert/data generation belong to MIKASA-Robo.

Future method work should start from the pinned N1.7 baseline and import only
the final qualified adapter after its comparison protocol is frozen.

No model weights, checkpoints, datasets, run artifacts, or local Agent/Codex
configuration are included by this branch.
