import argparse

from gr00t.data.dataset.g1_id10_h40_native30_adapter import (
    compute_h40_relative_statistics,
    compute_h40_train_statistics,
)


parser = argparse.ArgumentParser()
parser.add_argument("--dataset-root")
parser.add_argument("--split-manifest")
parser.add_argument("--output", required=True)
args = parser.parse_args()
if bool(args.dataset_root) == bool(args.split_manifest):
    parser.error("supply exactly one of --dataset-root or --split-manifest")
if args.split_manifest:
    compute_h40_train_statistics(args.split_manifest, args.output)
else:
    compute_h40_relative_statistics(args.dataset_root, args.output)
