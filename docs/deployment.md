# ONNX export and inference

Phase 3 adds a hash-verified ONNX bundle for the ResNet scene classifier, a CPU ONNX
Runtime client, an authenticated FastAPI endpoint, a measured latency report, and a
standalone .NET example. Detection uses a separate bundle and runtime; see
[Object detection](detection.md). Neither path adds identity recognition or
deployment-ready accuracy.

## Export

Use a checkpoint produced by this repository. New checkpoints include the image size,
class order, data fingerprint, and MLflow run ID. For an older checkpoint without
preprocessing metadata, pass its training image size explicitly; the exporter does not
guess it or download pretrained weights.

```powershell
uv sync --extra training --extra mlops --extra inference
uv run surveillance-ml-export-onnx `
  --checkpoint artifacts\finalized\surveillance-resnet18-demo\best.pt `
  --output-dir artifacts\onnx `
  --image-size 64
```

For inference-only local serving without training and MLOps dependencies, install the
`inference` and `api` extras instead:

```powershell
uv sync --no-dev --extra inference --extra api
```

The export uses ResNet18, ONNX opset 17, and a dynamic batch dimension. It checks the
ONNX graph and compares PyTorch and ONNX Runtime logits on the same deterministic,
normalized tensors before publishing `model.onnx` and `manifest.json`. Export fails
on parity errors. The manifest records the class order, exact preprocessing contract,
input shape, checkpoint and model SHA-256 hashes, dataset fingerprint when available,
and source MLflow run ID. Inference refuses a hash or contract mismatch.

Preprocessing is RGB conversion, Pillow `ImageOps.fit` center-crop resize with bilinear
resampling, NCHW layout, and ImageNet mean/std normalization. Augmentation is not used
at inference. No class confidence threshold is applied; the returned confidence is
the largest softmax probability and is not a calibrated probability of correctness.

## Local API

The server binds to loopback by default and requires an API key of at least 32
characters. Generate and provide a key through the environment; never put it in a
tracked file:

```powershell
$env:SURVEILLANCE_API_KEY = [guid]::NewGuid().ToString("N") + [guid]::NewGuid().ToString("N")
$env:SURVEILLANCE_MODEL = "artifacts\onnx\model.onnx"
$env:SURVEILLANCE_MANIFEST = "artifacts\onnx\manifest.json"
uv run surveillance-ml-serve
```

`GET /health` reports whether the model loaded and its digest. `POST /predict` accepts
one JPEG, PNG, BMP, or WebP multipart image, requires the `X-API-Key` header, and
returns named class probabilities. The default upload cap is 10 MiB and the decoded
image cap is 40 megapixels; both oversized uploads and invalid images are rejected.
Uploaded bytes are not persisted. The API key is compared in constant time.

The service is intended for local development. Do not expose it directly to the
Internet; production deployments need TLS, secret management, network controls,
monitoring, and an independently reviewed threat model. The API key is not a user or
tenant identity system.

## Docker Compose

Build an ONNX bundle first, then set `SURVEILLANCE_API_KEY` in the shell or an
untracked `.env` file and run:

```powershell
docker compose build
docker compose up
```

Compose mounts `artifacts/onnx` read-only and publishes only to host loopback. The
container runs as a non-root user with a read-only root filesystem, no Linux
capabilities, and no-new-privileges. The image installs only inference and API extras;
PyTorch, MLflow, DVC, and Optuna are not runtime requirements.

## Latency report

The benchmark excludes image decoding and preprocessing and measures ONNX Runtime
inference only. Its output includes the model digest, CPU provider, input shape,
warmup/repetition counts, observed mean/median/p50/p95/min/max latency, and throughput.
Results are machine-, thread-, and model-dependent; no universal speedup or latency is
claimed.

```powershell
uv run surveillance-ml-benchmark-onnx `
  --image data\demo\empty_scene\sample_0000.png `
  --output artifacts\onnx\benchmark.json `
  --warmup 5 --iterations 50
```

Pass `--mlflow-run-id` to attach the measured JSON and latency metrics to an existing
MLflow run.

## .NET interoperability example

The standalone .NET console app loads the ONNX file and manifest, checks the model
digest, applies the documented image preprocessing, runs ONNX Runtime, and maps
probabilities to manifest class names. Install .NET 9 and run:

```powershell
dotnet run --project dotnet\Inference\Inference.csproj --configuration Release -- `
  artifacts\onnx\model.onnx artifacts\onnx\manifest.json `
  data\demo\empty_scene\sample_0000.png

uv run python scripts\compare_dotnet.py `
  --image data\demo\empty_scene\sample_0000.png
```

SkiaSharp and Pillow both implement bilinear resizing but may differ at fractional
crop/resampling boundaries. The comparison harness requires the same top predicted
class and reports probability differences against its default 0.05 tolerance. This is
an interoperability smoke check, not a proof of pixel-identical preprocessing or
equivalent accuracy on real-world data.
