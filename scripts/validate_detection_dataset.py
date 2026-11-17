import argparse
from pathlib import Path

from surveillance_ml.detection.dataset import validate_yolo_dataset, write_validation_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a YOLO detection dataset.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--imbalance-threshold", type=float, default=1.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_yolo_dataset(
        args.dataset_root,
        imbalance_threshold=args.imbalance_threshold,
    )
    if args.output is not None:
        write_validation_report(report, args.output)
    print(report.to_dict())


if __name__ == "__main__":
    main()
