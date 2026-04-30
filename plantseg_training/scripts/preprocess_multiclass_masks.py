from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


SPLIT_TO_DIR = {
    "Training": "train",
    "Validation": "val",
    "Test": "test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create PlantSeg multiclass masks from binary masks and metadata.")
    parser.add_argument("--root-dir", type=Path, default=Path("Data_exploration/data/archive/plantsegv2"))
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("Data_exploration/data/archive/plantsegv2/Metadatav2.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plantseg_training/processed/multiclass"),
    )
    parser.add_argument("--max-weight", type=float, default=25.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_class_map(metadata: pd.DataFrame) -> pd.DataFrame:
    classes = (
        metadata[["Index", "Plant", "Disease"]]
        .drop_duplicates(subset=["Index", "Disease"])
        .sort_values(["Index", "Disease"])
        .reset_index(drop=True)
    )
    classes.insert(0, "class_id", np.arange(1, len(classes) + 1, dtype=np.int64))
    background = pd.DataFrame([{"class_id": 0, "Index": -1, "Plant": "Background", "Disease": "background"}])
    return pd.concat([background, classes], ignore_index=True)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    metadata = pd.read_csv(args.metadata_csv)
    class_map = build_class_map(metadata)
    id_lookup = {
        (int(row.Index), str(row.Disease)): int(row.class_id)
        for row in class_map.loc[class_map["class_id"] > 0].itertuples(index=False)
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = args.output_dir / "masks"
    reports_dir = args.output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    class_map.to_csv(reports_dir / "class_map.csv", index=False)
    metadata.groupby(["Plant", "Disease"]).size().reset_index(name="images").sort_values(
        ["Plant", "Disease"]
    ).to_csv(reports_dir / "plant_disease_distribution.csv", index=False)
    metadata.groupby(["Split", "Plant", "Disease"]).size().reset_index(name="images").sort_values(
        ["Split", "Plant", "Disease"]
    ).to_csv(reports_dir / "split_plant_disease_distribution.csv", index=False)

    class_pixel_counts = np.zeros(len(class_map), dtype=np.int64)
    image_counts = np.zeros(len(class_map), dtype=np.int64)

    for split_name, split_dir in SPLIT_TO_DIR.items():
        split_frame = metadata.loc[metadata["Split"] == split_name].reset_index(drop=True)
        output_split_dir = masks_dir / split_dir
        output_split_dir.mkdir(parents=True, exist_ok=True)

        for _, row in tqdm(split_frame.iterrows(), total=len(split_frame), desc=f"building {split_dir} masks"):
            label_file = str(row["Label file"])
            source_mask = args.root_dir / "annotations" / split_dir / label_file
            output_mask = output_split_dir / label_file
            if output_mask.exists() and not args.overwrite:
                mask_array = np.array(Image.open(output_mask), dtype=np.uint8)
            else:
                binary = np.array(Image.open(source_mask).convert("L"), dtype=np.uint8)
                class_id = id_lookup[(int(row["Index"]), str(row["Disease"]))]
                mask_array = np.zeros(binary.shape, dtype=np.uint8)
                mask_array[binary > 0] = class_id
                Image.fromarray(mask_array).save(output_mask)

            counts = np.bincount(mask_array.reshape(-1), minlength=len(class_map))
            class_pixel_counts += counts
            image_counts[np.unique(mask_array)] += 1

    pixel_report = class_map.copy()
    pixel_report["pixel_count"] = class_pixel_counts
    pixel_report["image_count_with_class"] = image_counts
    pixel_report["pixel_ratio"] = class_pixel_counts / max(int(class_pixel_counts.sum()), 1)
    pixel_report.to_csv(reports_dir / "class_pixel_counts.csv", index=False)

    foreground_counts = np.maximum(class_pixel_counts.astype(np.float64), 1.0)
    median_foreground = np.median(foreground_counts[1:])
    weights = median_foreground / foreground_counts
    weights = np.clip(weights, 0.05, args.max_weight)
    weights[0] = min(weights[0], 1.0)

    save_json(
        reports_dir / "class_weights.json",
        {
            "num_classes": int(len(class_map)),
            "background_class_id": 0,
            "class_weights": [float(x) for x in weights],
            "max_weight": args.max_weight,
            "note": "Weights are median-frequency balancing from generated multiclass pixel counts.",
        },
    )
    save_json(
        reports_dir / "dataset_summary.json",
        {
            "rows": int(len(metadata)),
            "plants": int(metadata["Plant"].nunique()),
            "diseases": int(metadata["Disease"].nunique()),
            "num_classes_including_background": int(len(class_map)),
            "splits": {str(k): int(v) for k, v in metadata["Split"].value_counts().to_dict().items()},
            "raw_index_min": int(metadata["Index"].min()),
            "raw_index_max": int(metadata["Index"].max()),
        },
    )
    print(f"Saved multiclass masks to {masks_dir}")
    print(f"Saved reports to {reports_dir}")
    print(f"Classes including background: {len(class_map)}")


if __name__ == "__main__":
    main()
