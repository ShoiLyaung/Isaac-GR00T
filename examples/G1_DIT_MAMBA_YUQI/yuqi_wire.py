"""Legacy Yuqi response encoding; preserve the native safe request decoder."""

import io

from gr00t.data.types import ModalityConfig
from gr00t.data.utils import to_json_serializable
from gr00t.policy.server_client import MsgSerializer
import msgpack
import numpy as np


def _encode_legacy(value):
    if isinstance(value, ModalityConfig):
        return {"__ModalityConfig_class__": True, "as_json": to_json_serializable(value)}
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Object arrays are not supported by the Yuqi wire protocol")
        stream = io.BytesIO()
        np.save(stream, value, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": stream.getvalue()}
    raise TypeError(f"Unsupported Yuqi response type: {type(value).__name__}")


def legacy_to_bytes(value):
    return msgpack.packb(value, default=_encode_legacy)


def install_legacy_yuqi_response_encoding():
    # Scope is this dedicated server process. Model, action units and shapes are unchanged.
    # from_bytes (including its no-pickle checks and legacy request support) is untouched.
    MsgSerializer.to_bytes = staticmethod(legacy_to_bytes)
