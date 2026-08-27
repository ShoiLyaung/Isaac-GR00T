"""Deterministic framework-neutral preprocessing for codec evidence."""

from __future__ import annotations

import numpy as np

from .contract import (
    FRAME_TYPE_CODES,
    HISTORY,
    MAX_MV_PER_FRAME,
    MV_FIELD_DTYPES,
    REASON_CODES,
    SHA256_RE,
    TIME_REFERENCE_FIELDS,
    VALIDITY_FIELDS,
    CodecEvidenceRecord,
    CodecEvidenceSample,
    validate_sample,
)
from .episode_bundle import EpisodeT8View, validate_episode_view


_VALIDITY_MAP = {
    "frame_valid": "frame",
    "packet_valid": "packet",
    "reference_valid": "reference",
    "dependency_valid": "dependency",
    "alignment_valid": "alignment",
    "mv_valid": "mv",
    "rpe_valid": "rpe",
}

_INVALID_REASON_BY_VALIDITY = (
    ("packet", "PACKET_INVALID"),
    ("frame", "FRAME_INVALID"),
    ("reference", "REFERENCE_INVALID"),
    ("dependency", "REFERENCE_CHAIN_INVALID"),
    ("alignment", "ALIGNMENT_INVALID"),
    ("mv", "MV_OVERFLOW"),
    ("rpe", "RPE_INVALID"),
)


def _resize_appearance(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.dtype("uint8") or image.shape != (480, 640, 3):
        raise ValueError("I-frame appearance must be uint8 [480,640,3]")
    rows = np.rint(np.linspace(0, image.shape[0] - 1, 256)).astype(np.int64)
    columns = np.rint(np.linspace(0, image.shape[1] - 1, 256)).astype(np.int64)
    resized = image[rows[:, None], columns[None, :], :]
    return np.ascontiguousarray(resized.transpose(2, 0, 1), dtype=np.float32) / 255.0


class CodecEvidenceProcessorCore:
    """Validate a causal record and produce one exact T8 transport sample."""

    def process(
        self, record: CodecEvidenceRecord | EpisodeT8View
    ) -> CodecEvidenceSample | EpisodeT8View:
        if isinstance(record, EpisodeT8View):
            validate_episode_view(record)
            return record
        payload = record.payload
        if payload.get("schema_version") != "codec-evidence-record-v2":
            raise ValueError("Unsupported codec-evidence record schema")
        if int(payload.get("observation_row", -1)) != record.key.observation_row:
            raise ValueError("Record observation_row does not match lookup key")
        if payload.get("camera_id") != record.key.camera_id:
            raise ValueError("Record camera_id does not match lookup key")
        episode = payload.get("episode", {})
        if episode.get("episode_uid") != record.key.episode_uid:
            raise ValueError("Record episode_uid does not match lookup key")

        history = payload.get("history", {})
        frames = history.get("frames")
        if history.get("length") != HISTORY or not isinstance(frames, list):
            raise ValueError("Record must contain exactly eight history frames")
        if len(frames) != HISTORY:
            raise ValueError("Record must contain exactly eight history frames")

        time_and_reference = {
            name: np.full((HISTORY,), -1, dtype=np.int64) for name in TIME_REFERENCE_FIELDS
        }
        frame_type = np.zeros((HISTORY,), dtype=np.uint8)
        frame_validity = {name: np.zeros((HISTORY,), dtype=np.bool_) for name in VALIDITY_FIELDS}
        frame_reason = np.zeros((HISTORY,), dtype=np.int16)

        previous_nonpad = -1
        previous_pts = -1
        previous_dts = -1
        previous_gop_id = -1
        previous_gop_position = -1
        nonpad_count = 0
        nonpad_all_valid = True
        history_i_frames: dict[int, tuple[str, str]] = {}
        for slot, frame in enumerate(frames):
            if frame.get("slot") != slot:
                raise ValueError("History frame slots must be canonical 0..7")
            frame_name = frame.get("frame_type")
            if frame_name not in FRAME_TYPE_CODES:
                raise ValueError("Only PAD, I, and P frame types are allowed")
            frame_type[slot] = FRAME_TYPE_CODES[frame_name]
            reason_name = frame.get("reason_code")
            if reason_name not in REASON_CODES:
                raise ValueError("Unknown frame reason code")
            frame_reason[slot] = REASON_CODES[reason_name]
            validity = frame.get("validity")
            if not isinstance(validity, dict):
                raise ValueError("Frame validity must be an object")

            for name in TIME_REFERENCE_FIELDS:
                value = frame.get(name)
                if not isinstance(value, int):
                    raise TypeError(f"history.frames[{slot}].{name} must be integer")
                time_and_reference[name][slot] = value
            for output_name, input_name in _VALIDITY_MAP.items():
                value = validity.get(input_name)
                if not isinstance(value, bool):
                    raise TypeError(f"history.frames[{slot}].validity.{input_name} must be boolean")
                frame_validity[output_name][slot] = value

            frame_index = int(frame["frame_index"])
            reference_index = int(frame["reference_index"])
            if frame_name == "PAD":
                if any(time_and_reference[name][slot] != -1 for name in TIME_REFERENCE_FIELDS):
                    raise ValueError("PAD frames require -1 temporal/reference fields")
                if any(frame_validity[name][slot] for name in VALIDITY_FIELDS):
                    raise ValueError("PAD frames require every validity field to be false")
                if reason_name != "LEFT_PAD":
                    raise ValueError("PAD frames require LEFT_PAD reason")
                if (
                    frame.get("source_sha256") is not None
                    or frame.get("decoded_sha256") is not None
                ):
                    raise ValueError("PAD frames require null source and decoded hashes")
                if any(
                    int(frame.get(name, -1)) != 0
                    for name in ("mv_offset_begin", "mv_offset_end", "mv_count")
                ):
                    raise ValueError("PAD frames require zero MV descriptors")
                if previous_nonpad >= 0:
                    raise ValueError("Padding may only occur on the left")
                continue
            nonpad_count += 1
            if frame_index < 0 or frame_index > record.key.observation_row:
                raise ValueError("Codec history must not include a future frame")
            pts = int(frame["pts"])
            dts = int(frame["dts"])
            gop_id = int(frame["gop_id"])
            gop_position = int(frame["gop_position"])
            if pts < 0 or dts < 0 or gop_id < 0 or gop_position < 0:
                raise ValueError("Non-padding temporal and GOP fields must be non-negative")
            if previous_nonpad >= 0:
                if frame_index != previous_nonpad + 1:
                    raise ValueError("Non-padding frame_index values must be contiguous")
                if pts <= previous_pts or dts <= previous_dts:
                    raise ValueError("Non-padding PTS and DTS values must be strictly increasing")
            expected_gop_id, expected_gop_position = divmod(frame_index, 30)
            if gop_id != expected_gop_id or gop_position != expected_gop_position:
                raise ValueError("GOP identity must equal divmod(frame_index, 30)")
            expected_frame_type = "I" if expected_gop_position == 0 else "P"
            if frame_name != expected_frame_type:
                raise ValueError("GOP30 requires I only at position zero and P otherwise")
            source_sha256 = frame.get("source_sha256")
            decoded_sha256 = frame.get("decoded_sha256")
            if not isinstance(source_sha256, str) or not SHA256_RE.fullmatch(source_sha256):
                raise ValueError("Non-padding frames require a valid source SHA-256")
            if not isinstance(decoded_sha256, str) or not SHA256_RE.fullmatch(decoded_sha256):
                raise ValueError("Non-padding frames require a valid decoded SHA-256")
            if frame_name == "I":
                if reference_index != -1 or gop_position != 0:
                    raise ValueError("I frames require reference_index=-1 and gop_position=0")
                if previous_nonpad >= 0 and gop_id != previous_gop_id + 1:
                    raise ValueError("A new I frame must advance gop_id by exactly one")
                history_i_frames[frame_index] = (source_sha256, decoded_sha256)
            if frame_name == "P":
                if gop_position <= 0:
                    raise ValueError("P frames require a positive gop_position")
                if reference_index != frame_index - 1:
                    raise ValueError("P frames must reference the immediately preceding frame")
                if previous_nonpad >= 0 and reference_index != previous_nonpad:
                    raise ValueError(
                        "P frames must reference the immediately preceding history frame"
                    )
                if previous_nonpad >= 0:
                    if gop_id != previous_gop_id:
                        raise ValueError("Only an I frame may start a new GOP")
                    if gop_position != previous_gop_position + 1:
                        raise ValueError("P-frame GOP positions must be contiguous")
            invalid_reasons = [
                reason
                for validity_name, reason in _INVALID_REASON_BY_VALIDITY
                if not validity[validity_name]
            ]
            if invalid_reasons:
                nonpad_all_valid = False
                if reason_name != invalid_reasons[0]:
                    raise ValueError("Frame reason code disagrees with validity flags")
            elif reason_name != "NONE":
                raise ValueError("Fully valid frames require NONE reason code")
            previous_nonpad = frame_index
            previous_pts = pts
            previous_dts = dts
            previous_gop_id = gop_id
            previous_gop_position = gop_position

        if nonpad_count == 0:
            raise ValueError("Codec history must contain at least one non-padding frame")
        if previous_nonpad != record.key.observation_row:
            raise ValueError("Codec history must end at the observation row")

        offsets = np.asarray(record.mv_offsets)
        if offsets.dtype != np.dtype("int64") or offsets.shape != (HISTORY + 1,):
            raise ValueError("MV offsets must be int64 [9]")
        if offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1]):
            raise ValueError("MV offsets must start at zero and be monotonic")
        frame_counts = np.diff(offsets)
        if np.any(frame_counts > MAX_MV_PER_FRAME):
            raise ValueError("MV count exceeds maximum; truncation is forbidden")
        for slot, frame in enumerate(frames):
            if (
                int(frame["mv_offset_begin"]) != int(offsets[slot])
                or int(frame["mv_offset_end"]) != int(offsets[slot + 1])
                or int(frame["mv_count"]) != int(frame_counts[slot])
            ):
                raise ValueError("Frame MV descriptors disagree with offsets")

        total = int(offsets[-1])
        if set(record.mv_fields) != set(MV_FIELD_DTYPES):
            raise ValueError("Record does not contain the exact eleven MV fields")
        mv_fields: dict[str, np.ndarray] = {}
        for name, dtype in MV_FIELD_DTYPES.items():
            value = np.asarray(record.mv_fields[name])
            if value.dtype != dtype or value.shape != (total,):
                raise ValueError(f"MV field {name} has incorrect dtype or shape")
            mv_fields[name] = np.ascontiguousarray(value)

        rpe = np.asarray(record.rpe)
        if rpe.dtype != np.dtype("float32") or rpe.shape != (8, 2, 2, 4):
            raise ValueError("RPE must be float32 [8,2,2,4]")
        if not np.isfinite(rpe).all():
            raise ValueError("RPE must be finite")
        pad_mask = frame_type == FRAME_TYPE_CODES["PAD"]
        if np.any(rpe[pad_mask] != 0):
            raise ValueError("PAD frames require numerically zero RPE")
        rpe_mask = np.broadcast_to(
            (
                frame_validity["frame_valid"]
                & frame_validity["alignment_valid"]
                & frame_validity["rpe_valid"]
            )[:, None, None],
            (8, 2, 2),
        ).copy()

        camera = np.asarray(record.camera_motion_summary)
        if camera.dtype != np.dtype("float32") or camera.shape != (8, 4):
            raise ValueError("Camera motion summary must be float32 [8,4]")
        if not np.isfinite(camera).all():
            raise ValueError("Camera motion summary must be finite")
        if np.any(camera[pad_mask] != 0):
            raise ValueError("PAD frames require numerically zero camera motion")
        camera_mask = (frame_validity["frame_valid"] & frame_validity["mv_valid"]).astype(
            np.bool_, copy=False
        )

        sample_reason_name = payload.get("sample_reason_code")
        if sample_reason_name not in REASON_CODES:
            raise ValueError("Unknown sample reason code")
        sample_valid = payload.get("sample_valid")
        if not isinstance(sample_valid, bool):
            raise TypeError("sample_valid must be boolean")
        if sample_valid:
            if not nonpad_all_valid or sample_reason_name != "NONE":
                raise ValueError("A valid sample requires valid non-padding frames and NONE reason")
        elif sample_reason_name == "NONE":
            raise ValueError("An invalid sample requires a non-NONE reason code")

        i_frame = payload.get("i_frame", {})
        raw_i_frame_index = i_frame.get("frame_index", -1)
        if not isinstance(raw_i_frame_index, int) or isinstance(raw_i_frame_index, bool):
            raise TypeError("i_frame.frame_index must be integer")
        i_frame_index = np.int64(raw_i_frame_index)
        i_source_sha256 = i_frame.get("source_sha256")
        i_decoded_sha256 = i_frame.get("decoded_sha256")
        expected_i_frame_index = record.key.observation_row // 30 * 30
        if i_frame_index < 0:
            if i_frame_index != -1 or i_source_sha256 is not None or i_decoded_sha256 is not None:
                raise ValueError("Unavailable I-frame requires index -1 and null hashes")
            if sample_valid or sample_reason_name != "I_FRAME_UNAVAILABLE":
                raise ValueError(
                    "Missing I-frame requires invalid sample reason I_FRAME_UNAVAILABLE"
                )
            i_frame_mask = np.bool_(False)
        else:
            if i_frame_index != expected_i_frame_index:
                raise ValueError("I-frame must be the nearest legal past GOP30 I-frame")
            if not isinstance(i_source_sha256, str) or not SHA256_RE.fullmatch(i_source_sha256):
                raise ValueError("I-frame requires a valid source SHA-256")
            if not isinstance(i_decoded_sha256, str) or not SHA256_RE.fullmatch(i_decoded_sha256):
                raise ValueError("I-frame requires a valid decoded SHA-256")
            history_hashes = history_i_frames.get(int(i_frame_index))
            if history_hashes is None:
                raise ValueError(
                    "record-v2 cannot bind an I-frame outside T8; use an episode bundle"
                )
            if history_hashes != (i_source_sha256, i_decoded_sha256):
                raise ValueError("I-frame hashes disagree with the matching history I-frame")
            i_frame_mask = np.bool_(True)
        appearance = _resize_appearance(np.asarray(record.i_frame_appearance))
        if not i_frame_mask:
            appearance = np.zeros((3, 256, 256), dtype=np.float32)
        sample = CodecEvidenceSample(
            key=record.key,
            observation_row=np.int64(record.key.observation_row),
            time_and_reference=time_and_reference,
            frame_type=frame_type,
            frame_validity=frame_validity,
            frame_reason_code=frame_reason,
            mv_offsets=np.ascontiguousarray(offsets),
            motion_vector_fields=mv_fields,
            rpe=np.ascontiguousarray(rpe),
            rpe_mask=rpe_mask,
            i_frame_appearance=appearance,
            i_frame_mask=i_frame_mask,
            i_frame_index=i_frame_index,
            camera_motion_summary=np.ascontiguousarray(camera),
            camera_motion_mask=camera_mask,
            sample_valid=np.bool_(sample_valid),
            sample_reason_code=np.int16(REASON_CODES[sample_reason_name]),
        )
        validate_sample(sample)
        return sample
