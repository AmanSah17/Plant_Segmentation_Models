from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from plantseg_training.trainer import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PlantSeg segmentation models.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("plantseg_training/configs/unet_binary.json"),
        help="Path to an experiment config JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = json.load(f)
    run_training(config=config, config_path=args.config)


if __name__ == "__main__":
    main()
