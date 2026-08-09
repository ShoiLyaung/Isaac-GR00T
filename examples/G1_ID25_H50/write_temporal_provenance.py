"""Write per-anchor/tick synthetic-ZOH lineage outside the canonical dataset."""

import argparse
import json

from gr00t.data.dataset.g1_id25_h50_adapter import write_h50_temporal_provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            write_h50_temporal_provenance(args.dataset_root, args.output),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
