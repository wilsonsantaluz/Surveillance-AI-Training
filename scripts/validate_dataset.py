import argparse
import json
from pathlib import Path

from surveillance_ml.core.reproducibility import dataset_fingerprint
from surveillance_ml.data.dataset import CLASS_NAMES, discover_images
from surveillance_ml.data.splitting import class_distribution


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an image-folder dataset")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/demo"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/dvc/dataset_validation.json"),
    )
    parser.add_argument("--imbalance-ratio", type=float, default=1.5)
    args = parser.parse_args()
    records = discover_images(args.dataset_root)
    report = {
        "dataset_fingerprint": dataset_fingerprint(args.dataset_root),
        "dataset_root_name": args.dataset_root.name,
        "image_count": len(records),
        "classes": list(CLASS_NAMES),
        "class_distribution": class_distribution(records, args.imbalance_ratio),
        "validated": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
