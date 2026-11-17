import argparse
import json
from pathlib import Path

from surveillance_ml.detection.voc import import_zenodo_pascal_voc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Zenodo record 11582054 Pascal VOC annotations to YOLO."
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--perceptual-distance", type=int, default=2)
    args = parser.parse_args()
    result = import_zenodo_pascal_voc(
        args.archive,
        args.output,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        perceptual_distance=args.perceptual_distance,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
