import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--template", required=True)
parser.add_argument("--relative-stats", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

output = Path(args.output)
if output.exists():
    raise FileExistsError(output)
with Path(args.template).open("r", encoding="utf-8") as stream:
    config = json.load(stream)
config["derived_relative_stats_path"] = str(Path(args.relative_stats).resolve())
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("x", encoding="utf-8") as stream:
    json.dump(config, stream, indent=2, sort_keys=True)
    stream.write("\n")
