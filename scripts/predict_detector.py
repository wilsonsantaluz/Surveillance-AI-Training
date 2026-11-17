import argparse
import json
from pathlib import Path
from typing import Any

from surveillance_ml.detection.onnx import OnnxDetector
from surveillance_ml.detection.video import run_video_inference


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local YOLO ONNX image or video inference.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--source",
        required=True,
        help="Local image/video path, camera index, or RTSP URL",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--annotated-video", type=Path)
    parser.add_argument("--target-fps", type=float)
    args = parser.parse_args()
    detector = OnnxDetector(args.model, args.manifest)

    image_path = Path(args.source)
    result: dict[str, Any]
    if image_path.is_file() and _is_image(image_path):
        prediction = detector.predict_image(image_path)
        result = {
            "frame_width": prediction.image_width,
            "frame_height": prediction.image_height,
            "inference_seconds": prediction.elapsed_seconds,
            "detections": [
                {
                    "class_id": item.class_id,
                    "class_name": item.class_name,
                    "confidence": item.confidence,
                    "box": {
                        "x": item.x,
                        "y": item.y,
                        "width": item.width,
                        "height": item.height,
                    },
                }
                for item in prediction.detections
            ],
        }
    else:
        source: str | int = args.source
        if args.source.isdecimal():
            source = int(args.source)
        result = run_video_inference(
            detector,
            source,
            target_fps=args.target_fps,
            annotated_output=args.annotated_video,
        )
    serialized = json.dumps(result, indent=2)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
