"""Build the non-overwriting metadata-only loader view."""

import argparse
import json

from gr00t.data.dataset.g1_id25_h50_adapter import build_metadata_loader_view


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-root", required=True)
    parser.add_argument("--view-root", required=True)
    args = parser.parse_args()
    print(json.dumps(build_metadata_loader_view(args.canonical_root, args.view_root), indent=2))


if __name__ == "__main__":
    main()
