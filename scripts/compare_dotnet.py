"""Compare standalone .NET and Python ONNX predictions for the same image."""

import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image

from surveillance_ml.inference.onnx_classifier import OnnxClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/onnx/model.onnx"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/onnx/manifest.json"),
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=0.05)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1] / "dotnet" / "Inference" / "Inference.csproj"
    result = subprocess.run(
        [
            "dotnet",
            "run",
            "--project",
            str(project),
            "--configuration",
            "Release",
            "--",
            str(args.model.resolve()),
            str(args.manifest.resolve()),
            str(args.image.resolve()),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    dotnet_prediction = json.loads(result.stdout)
    classifier = OnnxClassifier(args.model, args.manifest)
    with Image.open(args.image) as image:
        python_prediction = classifier.predict_image(image)
    if dotnet_prediction["class_name"] != python_prediction.class_name:
        raise SystemExit("Python and .NET predicted different classes")
    differences = {
        name: abs(value - python_prediction.probabilities[name])
        for name, value in dotnet_prediction["probabilities"].items()
    }
    maximum_difference = max(differences.values())
    report = {
        "predicted_class": python_prediction.class_name,
        "max_probability_difference": maximum_difference,
        "tolerance": args.tolerance,
        "probabilities": {
            "python": python_prediction.probabilities,
            "dotnet": dotnet_prediction["probabilities"],
        },
    }
    print(json.dumps(report, indent=2))
    if maximum_difference > args.tolerance:
        raise SystemExit("Python and .NET probabilities exceed the configured tolerance")


if __name__ == "__main__":
    main()
