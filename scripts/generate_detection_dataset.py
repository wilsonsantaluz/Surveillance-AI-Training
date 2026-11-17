import argparse
from pathlib import Path

from surveillance_ml.detection.synthetic import generate_synthetic_detection_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate geometric YOLO pipeline fixtures.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/detection-demo"))
    parser.add_argument("--source-count", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    dataset_yaml = generate_synthetic_detection_dataset(
        args.output_dir,
        source_count=args.source_count,
        image_size=args.image_size,
        seed=args.seed,
    )
    print(f"Generated and source-split synthetic data: {dataset_yaml}")


if __name__ == "__main__":
    main()
