#!/usr/bin/env python3
"""Run the fixed three-episode C3-R2 live process-tree RSS canary."""

from __future__ import annotations

from build_c3_full_fixture import configure_memory_canary_r2, main


if __name__ == "__main__":
    configure_memory_canary_r2()
    raise SystemExit(main())
