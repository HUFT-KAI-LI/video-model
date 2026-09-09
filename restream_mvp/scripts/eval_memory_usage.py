"""Summarize existing R0 evaluation logs without running models."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from restream.reality_metrics import summarize_cases

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize_cases(json.loads(args.metrics.read_text())["cases"]), indent=2, allow_nan=False))
