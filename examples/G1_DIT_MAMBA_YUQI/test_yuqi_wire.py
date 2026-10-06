"""CPU-only real old-client/new-server RPC test; no model inference or simulation."""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
from types import SimpleNamespace

from contract import CAMERAS, sha
from gr00t.policy.server_client import MsgSerializer, PolicyServer
import msgpack
import numpy as np
from run_train_eval import processor_from
import torch
from yuqi_wire import install_legacy_yuqi_response_encoding, legacy_to_bytes


CLIENT = """
import json,sys,numpy as np
from gr00t.policy.server_client import PolicyClient
c=PolicyClient(host="127.0.0.1",port=int(sys.argv[1]),timeout_ms=10000)
m=c.get_modality_config()
assert m["video"].modality_keys==["cam_left_high","cam_left_wrist","cam_right_wrist"]
assert m["state"].modality_keys==m["action"].modality_keys==["left_arm","left_hand"]
assert m["action"].delta_indices==list(range(16))
obs={"video":{key:np.full((1,1,2,3,3),i,dtype=np.uint8) for i,key in enumerate(m["video"].modality_keys)},"state":{"left_arm":np.arange(7,dtype=np.float64).reshape(1,1,7),"left_hand":np.arange(6,dtype=np.float32).reshape(1,1,6)}}
a,info=c.get_action(obs)
assert np.array_equal(a["left_arm"],np.arange(112,dtype=np.float32).reshape(1,16,7)/100)
assert np.array_equal(a["left_hand"],np.arange(96,dtype=np.float32).reshape(1,16,6)/100)
assert a["left_arm"].dtype==a["left_hand"].dtype==np.float32
assert c.reset()=={"reset":True}
c.call_endpoint("kill",requires_input=False)
print(json.dumps({"modality_class":type(m["video"]).__name__,"camera_keys":m["video"].modality_keys,"action_shapes":{k:list(v.shape) for k,v in a.items()},"numeric_roundtrip_exact":True,"reset_endpoint":True}))
c.socket.close();c.context.term()
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    native_decoder = MsgSerializer.from_bytes
    install_legacy_yuqi_response_encoding()
    assert MsgSerializer.from_bytes is native_decoder
    try:
        legacy_to_bytes(np.array([object()], dtype=object))
    except TypeError:
        pass
    else:
        raise AssertionError("Object array unexpectedly encoded")
    try:
        MsgSerializer.from_bytes(
            msgpack.packb({b"nd": True, b"kind": b"O", b"data": b"not-pickle"})
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Object payload unexpectedly decoded")
    root = Path("/home/sl/research/icra27-artifacts/g1-dit-mamba-yuqi-5000step-r1")
    processor = processor_from(root / "input-candidate-r3/processor")
    modalities = processor.get_modality_configs()["new_embodiment"]

    def action(observation, options=None):
        assert list(observation["video"]) == list(CAMERAS)
        for index, array in enumerate(observation["video"].values()):
            assert array.dtype == np.uint8 and np.all(array == index)
        assert observation["state"]["left_arm"].dtype == np.float64
        assert observation["state"]["left_hand"].dtype == np.float32
        return {
            "left_arm": np.arange(112, dtype=np.float32).reshape(1, 16, 7) / 100,
            "left_hand": np.arange(96, dtype=np.float32).reshape(1, 16, 6) / 100,
        }, {}

    policy = SimpleNamespace(
        get_modality_config=lambda: modalities,
        get_action=action,
        reset=lambda options=None: {"reset": True},
    )
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    with PolicyServer(policy, host="127.0.0.1", port=port) as server:
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        environment = dict(os.environ, PYTHONPATH="/home/yuqi/Isaac-GR00T-N1.7")
        result = subprocess.run(
            ["/home/yuqi/Isaac-GR00T-N1.7/.venv/bin/python", "-c", CLIENT, str(port)],
            env=environment,
            text=True,
            capture_output=True,
            timeout=30,
        )
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert not torch.cuda.is_initialized()
    output = args.output
    output.mkdir(exist_ok=False)
    report = {
        "status": "passed",
        "scope": "actual legacy client and current server over loopback; deterministic CPU policy stub, no neural inference",
        "client": json.loads(result.stdout.splitlines()[-1]),
        "native_decoder_unchanged": True,
        "object_array_rejected": True,
        "pickle_bearing_payload_rejected": True,
        "gpu_initialized": False,
        "old_client_source_sha256": sha(
            "/home/yuqi/Isaac-GR00T-N1.7/gr00t/policy/server_client.py"
        ),
        "wire_encoder_sha256": sha(Path(__file__).with_name("yuqi_wire.py")),
        "test_source_sha256": sha(__file__),
    }
    (output / "RESULT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
