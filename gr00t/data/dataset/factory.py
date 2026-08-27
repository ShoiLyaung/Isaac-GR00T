# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from tqdm import tqdm

from gr00t.configs.base_config import Config
from gr00t.data.dataset.sharded_mixture_dataset import ShardedMixtureDataset
from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.interfaces import BaseProcessor
from gr00t.data.stats import generate_rel_stats, generate_stats
from gr00t.experiment.dist_utils import run_or_wait_on_rank0


def _resolve_all_lerobot_codec_config(dataset_specs):
    """Require one exact optional Codec configuration across every joint-data leaf."""

    from gr00t.data.dataset.g1_all_lerobot_h40_adapter import (
        load_adapter_config,
        parse_adapter_codec_evidence_config,
    )

    resolved = []
    for dataset_spec in dataset_specs:
        if dataset_spec.dataset_adapter != "g1_all_lerobot_h40_v1":
            continue
        if dataset_spec.adapter_config_path is None:
            raise ValueError("G1 all-LeRobot H40 adapter requires adapter_config_path")
        adapter_config = load_adapter_config(dataset_spec.adapter_config_path)
        resolved.append(parse_adapter_codec_evidence_config(adapter_config))
    if not resolved:
        return None
    enabled = [config for config in resolved if config is not None]
    if not enabled:
        # Absent and every valid enabled=false surface are behavior-equivalent;
        # dead disabled fields must not create a mixture-level semantic drift.
        return None
    if len(enabled) != len(resolved):
        raise ValueError("Every G1 all-LeRobot leaf must enable Codec evidence together")
    expected = enabled[0]
    if any(config != expected for config in enabled[1:]):
        raise ValueError(
            "Every G1 all-LeRobot leaf must use one identical global codec_evidence config"
        )
    if expected.schema_version != "codec-evidence-episode-bundle-v1":
        raise ValueError(
            "G1 all-LeRobot enabled training requires codec-evidence-episode-bundle-v1"
        )
    return expected


def _validate_all_lerobot_codec_store_coverage(codec_config, identities_by_uid, store):
    """Require an exact selected-episode cohort and identity match before training."""

    from gr00t.data.codec_evidence.codec_contract import PRODUCTION_FIXTURE_MODE
    from gr00t.data.codec_evidence.episode_bundle import EpisodeBundleKey
    from gr00t.data.codec_evidence.identity import validate_episode_identity_preimage

    expected_keys = frozenset(
        EpisodeBundleKey(
            episode_uid=episode_uid,
            camera_id=codec_config.camera_id,
            source_kind=codec_config.source_kind,
        )
        for episode_uid in identities_by_uid
    )
    if store.keys != expected_keys:
        missing = expected_keys.difference(store.keys)
        extra = store.keys.difference(expected_keys)
        raise ValueError(
            "G1 all-LeRobot Codec index coverage differs from selected episodes: "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    for key in expected_keys:
        expected_identity = dict(identities_by_uid[key.episode_uid])
        expected_identity.pop("episode_uid")
        bundle_identity = store.identity(key)
        if bundle_identity["codec_contract"]["fixture_mode"] != PRODUCTION_FIXTURE_MODE:
            raise ValueError("G1 all-LeRobot training requires production Codec fixture mode")
        observed_identity = bundle_identity["episode_identity"]
        if validate_episode_identity_preimage(
            observed_identity
        ) != validate_episode_identity_preimage(expected_identity):
            raise ValueError("G1 all-LeRobot Codec bundle episode identity differs from adapter")


class DatasetFactory:
    """
    Factory class for building training datasets. Model-agnostic.
    """

    def __init__(self, config: Config):
        self.config = config

    def build(
        self, processor: BaseProcessor
    ) -> tuple[ShardedMixtureDataset, ShardedMixtureDataset | None]:
        """Build the dataset. Returns a tuple of (train_dataset, eval_dataset)."""
        assert self.config.training.eval_strategy == "no", (
            "Sharded dataset does not support evaluation sets"
        )
        all_lerobot_codec_config = _resolve_all_lerobot_codec_config(self.config.data.datasets)
        all_lerobot_codec_uids: set[str] = set()
        all_lerobot_codec_identities: dict[str, dict] = {}
        all_lerobot_codec_store = None
        if all_lerobot_codec_config is not None:
            from gr00t.data.codec_evidence.dataset import episode_bundle_store_from_index_manifest

            all_lerobot_codec_store = episode_bundle_store_from_index_manifest(
                all_lerobot_codec_config.fixture_root,
                fixture_manifest_sha256=all_lerobot_codec_config.fixture_manifest_sha256,
                camera_id=all_lerobot_codec_config.camera_id,
                source_kind=all_lerobot_codec_config.source_kind,
            )

        all_datasets = []
        all_weights = []
        for dataset_spec in tqdm(
            self.config.data.datasets,
            total=len(self.config.data.datasets),
            desc="Initializing datasets",
        ):
            datasets = []
            for dataset_path in dataset_spec.dataset_paths:
                embodiment_tag = dataset_spec.embodiment_tag
                assert embodiment_tag is not None, "Embodiment tag is required"
                assert self.config.data.mode == "single_turn", "Only single turn mode is supported"
                if dataset_spec.dataset_adapter is None:
                    with run_or_wait_on_rank0(label="dataset.generate_statistics") as is_rank0:
                        if is_rank0:
                            generate_stats(dataset_path)
                            generate_rel_stats(dataset_path, EmbodimentTag(embodiment_tag))
                    dataset = ShardedSingleStepDataset(
                        dataset_path=dataset_path,
                        embodiment_tag=EmbodimentTag(embodiment_tag),
                        modality_configs=self.config.data.modality_configs[embodiment_tag],
                        video_backend=self.config.data.video_backend,
                        shard_size=self.config.data.shard_size,
                        episode_sampling_rate=self.config.data.episode_sampling_rate,
                        seed=self.config.data.seed,
                        allow_padding=self.config.data.allow_padding,
                    )
                elif dataset_spec.dataset_adapter == "g1_id25_h50_synthetic_zoh_v1":
                    from gr00t.data.dataset.g1_id25_h50_adapter import G1Id25H50SingleStepDataset

                    if dataset_spec.adapter_config_path is None:
                        raise ValueError("G1 ID25/H50 adapter requires adapter_config_path")
                    if embodiment_tag != "unitree_g1_full_body_with_waist_height_nav_cmd":
                        raise ValueError("G1 ID25/H50 adapter requires the frozen UNITREE_G1 tag")
                    dataset = G1Id25H50SingleStepDataset(
                        dataset_path=dataset_path,
                        embodiment_tag=EmbodimentTag(embodiment_tag),
                        modality_configs=self.config.data.modality_configs[embodiment_tag],
                        video_backend=self.config.data.video_backend,
                        shard_size=self.config.data.shard_size,
                        episode_sampling_rate=self.config.data.episode_sampling_rate,
                        seed=self.config.data.seed,
                        allow_padding=self.config.data.allow_padding,
                        adapter_config_path=dataset_spec.adapter_config_path,
                    )
                elif dataset_spec.dataset_adapter == "g1_id10_h50_synthetic_zoh_v1":
                    from gr00t.data.dataset.g1_id10_h50_adapter import G1Id10H50SingleStepDataset

                    if dataset_spec.adapter_config_path is None:
                        raise ValueError("G1 ID10/H50 adapter requires adapter_config_path")
                    if embodiment_tag != "new_embodiment":
                        raise ValueError(
                            "G1 ID10/H50 adapter requires the official NEW_EMBODIMENT tag"
                        )
                    dataset = G1Id10H50SingleStepDataset(
                        dataset_path=dataset_path,
                        embodiment_tag=EmbodimentTag(embodiment_tag),
                        modality_configs=self.config.data.modality_configs[embodiment_tag],
                        video_backend=self.config.data.video_backend,
                        shard_size=self.config.data.shard_size,
                        episode_sampling_rate=self.config.data.episode_sampling_rate,
                        seed=self.config.data.seed,
                        allow_padding=self.config.data.allow_padding,
                        adapter_config_path=dataset_spec.adapter_config_path,
                    )
                elif dataset_spec.dataset_adapter == "g1_id10_h40_native30_v1":
                    from gr00t.data.dataset.g1_id10_h40_native30_adapter import (
                        G1Id10H40Native30SingleStepDataset,
                    )

                    if dataset_spec.adapter_config_path is None:
                        raise ValueError(
                            "G1 ID10/H40/native30 adapter requires adapter_config_path"
                        )
                    if embodiment_tag != "new_embodiment":
                        raise ValueError("G1 ID10/H40/native30 adapter requires NEW_EMBODIMENT")
                    dataset = G1Id10H40Native30SingleStepDataset(
                        dataset_path=dataset_path,
                        embodiment_tag=EmbodimentTag(embodiment_tag),
                        modality_configs=self.config.data.modality_configs[embodiment_tag],
                        video_backend=self.config.data.video_backend,
                        shard_size=self.config.data.shard_size,
                        episode_sampling_rate=self.config.data.episode_sampling_rate,
                        seed=self.config.data.seed,
                        allow_padding=self.config.data.allow_padding,
                        adapter_config_path=dataset_spec.adapter_config_path,
                    )
                elif dataset_spec.dataset_adapter == "g1_all_lerobot_h40_v1":
                    from gr00t.data.dataset.g1_all_lerobot_h40_adapter import (
                        G1AllLerobotH40SingleStepDataset,
                    )

                    if dataset_spec.adapter_config_path is None:
                        raise ValueError("G1 all-LeRobot H40 adapter requires adapter_config_path")
                    if embodiment_tag != "new_embodiment":
                        raise ValueError("G1 all-LeRobot H40 adapter requires NEW_EMBODIMENT")
                    dataset = G1AllLerobotH40SingleStepDataset(
                        dataset_path=dataset_path,
                        embodiment_tag=EmbodimentTag(embodiment_tag),
                        modality_configs=self.config.data.modality_configs[embodiment_tag],
                        video_backend=self.config.data.video_backend,
                        shard_size=self.config.data.shard_size,
                        episode_sampling_rate=self.config.data.episode_sampling_rate,
                        seed=self.config.data.seed,
                        allow_padding=self.config.data.allow_padding,
                        adapter_config_path=dataset_spec.adapter_config_path,
                    )
                    if all_lerobot_codec_config is not None and all_lerobot_codec_config.enabled:
                        episode_uids = tuple(dataset.episode_loader.selected_episode_uids)
                        episode_identities = tuple(dataset.episode_loader.codec_episode_identities)
                        if len(episode_identities) != len(episode_uids):
                            raise ValueError(
                                "G1 all-LeRobot Codec episode identity coverage is incomplete"
                            )
                        duplicate_uids = all_lerobot_codec_uids.intersection(episode_uids)
                        if duplicate_uids:
                            raise ValueError(
                                "G1 all-LeRobot Codec episode UIDs collide across sources"
                            )
                        all_lerobot_codec_uids.update(episode_uids)
                        for episode_uid, identity in zip(
                            episode_uids, episode_identities, strict=True
                        ):
                            if identity.get("episode_uid") != episode_uid:
                                raise ValueError(
                                    "G1 all-LeRobot Codec episode identity UID differs"
                                )
                            all_lerobot_codec_identities[episode_uid] = identity
                        from gr00t.data.codec_evidence.dataset import (
                            maybe_wrap_codec_evidence_dataset,
                        )

                        dataset = maybe_wrap_codec_evidence_dataset(
                            dataset,
                            dataset_spec.adapter_config_path,
                            store=all_lerobot_codec_store,
                            config=all_lerobot_codec_config,
                        )
                else:
                    raise ValueError(f"Unknown dataset adapter: {dataset_spec.dataset_adapter}")
                datasets.append(dataset)
            dataset_lengths = np.array([len(dataset) for dataset in datasets])
            dataset_relative_lengths = dataset_lengths / dataset_lengths.sum()
            for dataset, relative_length in zip(datasets, dataset_relative_lengths):
                weight = relative_length * dataset_spec.mix_ratio
                all_datasets.append(dataset)
                all_weights.append(weight)

        if all_lerobot_codec_config is not None:
            _validate_all_lerobot_codec_store_coverage(
                all_lerobot_codec_config,
                all_lerobot_codec_identities,
                all_lerobot_codec_store,
            )

        return (
            ShardedMixtureDataset(
                datasets=all_datasets,
                weights=all_weights,
                processor=processor,
                seed=self.config.data.seed,
                training=True,
                num_shards_per_epoch=self.config.data.num_shards_per_epoch,
                override_pretraining_statistics=self.config.data.override_pretraining_statistics,
            ),
            None,
        )
