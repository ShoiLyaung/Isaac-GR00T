"""Compute synthetic-ZOH H50 relative arm statistics outside the dataset."""

import argparse
import json

from gr00t.data.dataset.g1_id25_h50_adapter import compute_h50_relative_statistics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-episodes", type=int)
    args = parser.parse_args()
    result = compute_h50_relative_statistics(
        args.dataset_root, args.output, max_episodes=args.max_episodes
    )
    print(json.dumps({k: v for k, v in result.items() if k != "relative_action"}, indent=2))


if __name__ == "__main__":
    main()
