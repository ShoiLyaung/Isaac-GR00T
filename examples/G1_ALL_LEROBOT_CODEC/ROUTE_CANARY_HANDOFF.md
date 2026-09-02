# Causal Codec route-canary handoff

This packet belongs to the ICRA control / humanoid-lfd task. The GR00T
implementation task does not edit simulator, evaluator, or control repositories.

## Source and retained artifacts

- current GR00T runtime code commit:
  `8eb902be16479d6efe944eca2b441c9d2935fdb5`
- pinned upstream base: `51d4c89f72fda44cbf77285c6a8114b52676b8a1`
- retained checkpoint source commit:
  `0696eaa28917766b0fb1c68298986d2e2b6741fe`
- retained checkpoint:
  `/home/sl/icra27-runs/codec-training/g1-all-lerobot-causal-c5-codec-only-20k-r1/train/checkpoint-20000`
- checkpoint index SHA-256:
  `26925948e090e0775039277bada6fdbb359de0a327030006e6555340d360fb17`
- retained fixture source commit:
  `32f70df98761391a60b012c54477e1fda5e38ea4`
- retained fixture:
  `/home/sl/icra27-runs/codec-fixtures/g1-all-lerobot-causal-full-r2`
- fixture manifest SHA-256:
  `5e4ec3590dd20fc767957bc1f7b74d8eaf60a766a45be6d3e4a1f8c55412b4b9`
- current prefix materializer SHA-256:
  `00d8759ce6221c05d75024429245ffd5230ae7e002692d6f8215bde8a9a1a69c`
- current production module SHA-256:
  `9d3266b7542b7a406ff98e5e7bdbbc49f76c654bb82fcb72851bcde1047fd857`

The wrapper must record `checkpoint_source_commit` and
`runtime_source_commit` separately. It must not require the historical training
summary's source commit to equal the new runtime source commit.

## GR00T policy contract

The external wrapper continues to own reset-bound RGB history, 50-to-30 Hz
sampling, raw-RGB RPC, request IDs, and simulator observations. After creating
one validated `EpisodeT8View`, it must invoke the stock policy in process:

```python
action, info = policy.get_action(
    observation,
    options={"codec_evidence_sample": episode_t8_view},
)
```

It must not reimplement the GR00T processor, collator, model forward, sampler,
or action decoder. `EpisodeT8View` is deliberately not a stock ZMQ wire type.

The current prefix-materializer contract is model-side only and contains
exactly:

```text
schema_version
online_prefix_equivalent
materializer_id
materializer_sha256
source_kind
camera_id
evidence_hz
history_length
future_frame_fallback_allowed
```

Simulator frequency, experiment/run IDs, checkpoint identity, Git lineage, and
runtime paths remain in the external control spec rather than the GR00T
materializer contract.

## One authorized diagnostic canary

- seed: `1028`
- maximum controller ticks: `1000`
- action execution chunk: `8`
- expected policy requests/responses: `125/125`
- expected Codec calls per request: `encode=1`, `condition=4`
- expected decoded active action: finite `[1,40,26]`
- require reset isolation, no cross-episode rows, exact request/receipt closure,
  owned-process exit, policy-port release, scratch cleanup, and GPU cleanup
- automatic retries: `0`
- native task success is diagnostic only and is not a technical pass gate

Historical route R8 remains at:

`/home/sl/icra27-runs/research-control/groot-causal-codec-route-canary-postfix-r8/experiments/groot-causal-codec-route-canary-postfix-r8/runs/causal-codec-route-canary-postfix-seed1028-attempt-1-r8.artifacts/stages/causal-codec-route-canary-c-1028-ego-view-1000/SUPERVISOR_RESULT.json`

The new canary must use a distinct no-clobber run root and report its own source
and artifact identities. It may reuse the retained checkpoint and fixture as
read-only inputs, but must not relabel the historical R8 result as new-runtime
evidence.
