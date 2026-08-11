import argparse

from gr00t.data.dataset.g1_id10_h40_native30_adapter import build_metadata_loader_view


parser = argparse.ArgumentParser()
parser.add_argument("--canonical-root", required=True)
parser.add_argument("--view-root", required=True)
args = parser.parse_args()
build_metadata_loader_view(args.canonical_root, args.view_root)
