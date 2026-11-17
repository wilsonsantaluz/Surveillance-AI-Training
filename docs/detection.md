# Object detection (Phase 4)

The detection pipeline is separate from the ResNet scene classifier. It uses
Ultralytics YOLO11 (`yolo11n` or `yolo11s`) for bounding-box detection of `firearm`
and `knife` classes, with a separate dataset contract, evaluator, ONNX manifest, and
inference runtime. It does not identify people, infer intent, or provide an alerting
service for live surveillance.

## Synthetic CPU demonstration

Install the training, MLOps, and detection extras:

```powershell
uv sync --extra training --extra mlops --extra detection
```

The opt-in DVC pipeline generates geometric fixtures, validates the YOLO labels and
source-aware split, trains for one CPU epoch, evaluates only the held-out test split,
and exports an ONNX bundle:

```powershell
uv run dvc repro pipelines\detection\dvc.yaml
```

This custom pipeline is intentionally not part of the default `dvc repro` command,
which remains the classifier workflow. Outputs, local MLflow tracking data, and the
DVC cache are generated locally and are not committed.

For manual steps, `configs\detection_demo.yaml` is the small, unpretrained CPU
configuration:

```powershell
uv run python scripts\generate_detection_dataset.py --output-dir data\detection-demo --source-count 12 --image-size 128 --seed 42
uv run python scripts\validate_detection_dataset.py --dataset-root data\detection-demo
uv run python scripts\train_detector.py --config configs\detection_demo.yaml
uv run python scripts\evaluate_detector.py --config configs\detection_demo.yaml --checkpoint artifacts\detection-demo\train\weights\best.pt --split test --output artifacts\detection-demo\test-metrics.json
uv run python scripts\export_detector.py --config configs\detection_demo.yaml --checkpoint artifacts\detection-demo\train\weights\best.pt --output-dir artifacts\detection-demo\onnx
```

## Dataset contract and training

Supply a YOLO detection dataset with `images/{train,val,test}` and corresponding
`labels/{train,val,test}` folders. Each label line uses
`class_id x_center y_center width height` with normalized coordinates. The dataset
YAML must declare the exact order `0: firearm`, `1: knife`; every image and label is
validated, and source-group overlap between splits is rejected. The loader reports
class distribution and image/object counts. It does not invent labels or download
third-party images.

Edit `configs\detection.yaml` to point to a properly sourced dataset and choose
architecture, pretrained initialization, device, image size, epochs, batch size,
optimizer, and evaluation thresholds. `pretrained: true` requests external YOLO11
weights and may download them; review the applicable weight terms before use. The
`auto` device setting uses CUDA when available and otherwise CPU. CUDA mixed precision
is enabled only when configured and on a CUDA device. Training writes the
validation-best checkpoint and epoch history, records the data/config fingerprints,
runtime versions, parameters, metrics, and artifacts in local MLflow, and fails
explicitly when the required data or checkpoint is missing.

Evaluation runs against `val` or `test` and reports precision, recall, mAP at IoU
0.50 and averaged over 0.50–0.95, per-class results, a background-aware confusion
matrix, and false-positive/false-negative sample IDs. Do not select thresholds or
models by repeatedly tuning against the held-out test split.

## ONNX and inference

The detector bundle is independent of the classifier bundle. Its manifest verifies
the model and checkpoint hashes, class order, shapes, opset, source license notice,
letterbox preprocessing, raw YOLO output layout, and class-aware NMS settings.
Preprocessing is RGB, OpenCV `INTER_LINEAR_EXACT` letterbox resize, padding value
114, NCHW float32, and division by 255. The manifest names this resize contract
`opencv_inter_linear_exact`; the .NET detector uses matching fixed-point
interpolation to avoid platform-dependent differences at confidence thresholds.
Export validates the ONNX graph and compares raw PyTorch/ONNX outputs plus
postprocessed boxes on deterministic synthetic inputs; export fails on a parity
mismatch.

Run image or video inference with ONNX Runtime:

```powershell
$image = Get-ChildItem data\detection-demo\images\test -File -Recurse | Select-Object -First 1 -ExpandProperty FullName
uv run python scripts\predict_detector.py --model artifacts\detection-demo\onnx\model.onnx --manifest artifacts\detection-demo\onnx\manifest.json --source $image --output-json artifacts\detection-demo\predictions.json
```

`--source` accepts one image, a local video, a camera index, or an RTSP/HTTP stream.
For a video source, `--annotated-video` writes an annotated copy and `--target-fps`
limits inference sampling. Live capture uses a bounded queue, drops stale frames
under load, and reports reconnects, dropped frames, throughput, and latency. The
benchmark script accepts a local video file:

```powershell
uv run python scripts\benchmark_detector.py --model artifacts\detection-demo\onnx\model.onnx --manifest artifacts\detection-demo\onnx\manifest.json --video path\to\your-video.mp4 --output artifacts\detection-demo\video-benchmark.json
```

The .NET 9 console consumer uses ONNX Runtime and the same bundle contract. It
accepts an ordered image sequence for one camera; the temporal confirmation emits a
versioned alert DTO only after enough consecutive, spatially associated detections:

```powershell
dotnet run --project dotnet\Detector\Detector.csproj --configuration Release -- artifacts\detection-demo\onnx\model.onnx artifacts\detection-demo\onnx\manifest.json camera-1 frame-001.png frame-002.png frame-003.png
```

Confirmation uses a configurable confidence threshold, IoU association, consecutive
frame count/window, miss-based clearing, and per-camera/class cooldown. This sample
does not capture video itself; a host application must supply timestamped frames to
the `TemporalConfirmation` component.

## Licensing, privacy, and limitations

The optional Ultralytics implementation is AGPL-3.0. That source license may impose
obligations when using or distributing software that incorporates it; private,
commercial, embedded, or network-service use may require a separate license.
Pretrained weights have terms that must be reviewed separately. This project is
independently licensed; its root license does not replace third-party license
obligations. Consult the [Ultralytics licensing information](https://www.ultralytics.com/license)
and the applicable licenses before redistribution or deployment; this is not legal
advice.

The generated dataset contains colored geometric fixtures rather than images of
firearms or knives. In the verified CPU demonstration, 36 generated images were
split by source into 21 training, 9 validation, and 6 test images. The one-epoch,
randomly initialized run achieved **0 precision, 0 recall, and 0 mAP50** on the
synthetic test split (four labeled objects). ONNX raw-output parity passed with
maximum absolute errors below `0.0001`, but both postprocessed parity cases
contained zero detections, so those cases do not demonstrate non-empty box parity.
An additional Python/.NET smoke check compared 46 boxes on a 128x128 synthetic frame
with the manifest threshold deliberately lowered to `1e-12`; those low-confidence
outputs verify serialization and coordinate agreement only, not model quality.
These results are pipeline smoke evidence only—not evidence of weapon detection
accuracy or safety.

No public real-world dataset is bundled or downloaded. Before using any real footage,
verify dataset provenance and usage rights, obtain required authorization, minimize
retention, protect access, and assess privacy-law requirements such as the LGPD. Do
not use this demo as the sole basis for security decisions or automated action.
