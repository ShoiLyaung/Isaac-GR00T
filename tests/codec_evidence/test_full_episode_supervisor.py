from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

import pytest


REPOSITORY = Path(__file__).resolve().parents[2]
SUPERVISOR = REPOSITORY / "examples/G1_ID10_H40_NATIVE30/supervise_codec_full_episode_canary.py"


def _run(
    tmp_path: Path,
    code: str,
    *,
    extra: list[str] | None = None,
    evidence_name: str = "supervisor.json",
) -> tuple[subprocess.CompletedProcess[str], dict]:
    warehouse = tmp_path / "warehouse"
    scratch = tmp_path / "scratch"
    warehouse.mkdir(exist_ok=True)
    scratch.mkdir(exist_ok=True)
    evidence = tmp_path / evidence_name
    command = [
        sys.executable,
        str(SUPERVISOR),
        "--evidence",
        str(evidence),
        "--warehouse-root",
        str(warehouse),
        "--scratch-root",
        str(scratch),
        "--poll-interval-seconds",
        "0.01",
        "--termination-grace-seconds",
        "0.1",
        *(extra or []),
        "--",
        sys.executable,
        "-c",
        code,
        str(warehouse),
        str(scratch),
    ]
    result = subprocess.run(command, check=False, text=True, capture_output=True, timeout=10)
    return result, json.loads(evidence.read_text(encoding="utf-8"))


def test_supervisor_passes_through_zero_and_nonzero_child_exit(tmp_path: Path):
    passed, pass_evidence = _run(tmp_path, "raise SystemExit(0)")
    assert passed.returncode == 0
    assert pass_evidence["status"] == "PASS"
    assert pass_evidence["termination_reason"] == "CHILD_EXITED"
    assert pass_evidence["propagated_exit_code"] == 0

    failed, fail_evidence = _run(
        tmp_path, "raise SystemExit(7)", evidence_name="supervisor-failed.json"
    )
    assert failed.returncode == 7
    assert fail_evidence["status"] == "FAIL"
    assert fail_evidence["termination_reason"] == "CHILD_EXITED"
    assert fail_evidence["propagated_exit_code"] == 7


def test_supervisor_wall_limit_kills_the_process_group(tmp_path: Path):
    pid_file = tmp_path / "grandchild.pid"
    code = (
        "import pathlib,subprocess,sys,time; "
        f"p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        f"pathlib.Path({str(pid_file)!r}).write_text(str(p.pid)); time.sleep(30)"
    )
    result, evidence = _run(
        tmp_path,
        code,
        extra=["--max-wall-seconds", "0.15"],
    )
    assert result.returncode == 124
    assert evidence["termination_reason"] == "WALL_LIMIT_EXCEEDED"
    assert evidence["kill_evidence"]["sigterm_sent"] is True
    grandchild = int(pid_file.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 2
    while Path(f"/proc/{grandchild}").exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not Path(f"/proc/{grandchild}").exists()


def test_supervisor_process_tree_rss_limit_kills_child(tmp_path: Path):
    code = "import time; payload=bytearray(64*1024*1024); time.sleep(30)"
    result, evidence = _run(
        tmp_path,
        code,
        extra=["--max-process-rss-bytes", str(32 * 1024**2)],
    )
    assert result.returncode == 125
    assert evidence["termination_reason"] in {
        "PROCESS_RSS_LIMIT_EXCEEDED",
        "TREE_RSS_LIMIT_EXCEEDED",
    }
    assert evidence["observed_peaks"]["tree_rss_bytes"] > 32 * 1024**2


@pytest.mark.parametrize(
    ("target_index", "limit_option", "reason", "peak"),
    [
        (1, "--max-warehouse-bytes", "WAREHOUSE_LIMIT_EXCEEDED", "warehouse_bytes"),
        (2, "--max-scratch-bytes", "SCRATCH_LIMIT_EXCEEDED", "scratch_bytes"),
    ],
)
def test_supervisor_output_byte_limits_kill_child(
    tmp_path: Path,
    target_index: int,
    limit_option: str,
    reason: str,
    peak: str,
):
    code = (
        "import pathlib,sys,time; "
        f"pathlib.Path(sys.argv[{target_index}],'large.bin').write_bytes(b'x'*8192); "
        "time.sleep(30)"
    )
    result, evidence = _run(tmp_path, code, extra=[limit_option, "1024"])
    assert result.returncode == 125
    assert evidence["termination_reason"] == reason
    assert evidence["observed_peaks"][peak] == 8192
    assert evidence["kill_evidence"]["sigterm_sent"] is True


def test_existing_evidence_fails_before_launch(tmp_path: Path):
    evidence = tmp_path / "supervisor.json"
    evidence.write_text("keep", encoding="utf-8")
    marker = tmp_path / "should-not-exist"
    warehouse = tmp_path / "warehouse"
    scratch = tmp_path / "scratch"
    warehouse.mkdir()
    scratch.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            str(SUPERVISOR),
            "--evidence",
            str(evidence),
            "--warehouse-root",
            str(warehouse),
            "--scratch-root",
            str(scratch),
            "--",
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).touch()",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=5,
    )
    assert result.returncode != 0
    assert evidence.read_text(encoding="utf-8") == "keep"
    assert not marker.exists()
