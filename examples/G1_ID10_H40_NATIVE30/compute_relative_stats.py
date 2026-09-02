import argparse

from gr00t.data.dataset.g1_id10_h40_native30_adapter import compute_h40_relative_statistics


parser = argparse.ArgumentParser()
parser.add_argument("--dataset-root", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
compute_h40_relative_statistics(args.dataset_root, args.output)
