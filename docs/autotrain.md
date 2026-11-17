# AutoTrain

AutoTrain is a local, resumable orchestration layer around the existing Phase 4
YOLO11 training, evaluation, MLflow, ONNX export, and .NET parity components. It
does not download datasets, introduce a second detector architecture, deploy
models, or modify the separate SurveillanceAI application.

## Requirements and data

Install the optional dependencies:

```powershell
uv sync --extra autotrain
```

Provide an already-downloaded YOLO dataset at the path configured in
`configs/autotrain.yaml`:

```text
data/weapon-detection/
├── dataset.yaml
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

`dataset.yaml` must use the existing class contract, `0: firearm`, `1: knife`.
All splits must be present. Images and labels are validated by the Phase 4
validator; AutoTrain additionally checks image hashes, perceptual duplicates,
source split integrity, suspicious annotations, class balance, and negative-image
fraction. It creates deterministic annotation contact sheets for manual review.
Cross-split exact or perceptual duplicates are audit failures.

AutoTrain records the dataset source, version, license, provenance, and license
review status from configuration. These are operator-provided declarations, not
automated provenance or legal verification. Review the dataset source and
annotation sheets before approving a run. Keep datasets, checkpoints, and MLflow
artifacts outside Git.

## Commands

Show the validated plan without accessing the dataset or starting training:

```powershell
uv run surveillance-ml-autotrain run --dry-run --config configs/autotrain.yaml
```

Audit data and generate review files without training:

```powershell
uv run surveillance-ml-autotrain audit --config configs/autotrain.yaml
```

Run the workflow:

```powershell
uv run surveillance-ml-autotrain run --config configs/autotrain.yaml
```

The default configuration requires human annotation review. A valid run pauses
in `WAITING_FOR_REVIEW`; inspect `artifacts/autotrain/<run-id>/annotation-review/`
and record an explicit decision before resuming:

```powershell
uv run surveillance-ml-autotrain review --run-id <run-id> --decision approved --reviewer "Name" --notes "Reviewed contact sheets"
uv run surveillance-ml-autotrain resume --run-id <run-id>
```

Other operations:

```powershell
uv run surveillance-ml-autotrain status --run-id <run-id>
uv run surveillance-ml-autotrain report --run-id <run-id>
uv run surveillance-ml-autotrain cancel --run-id <run-id>
```

Pass the same `--config` used to create a run when using a non-default
configuration or output directory. Resume rejects configuration or dataset
fingerprint changes. Cancellation is checked at epoch boundaries; interrupted
training resumes from Ultralytics' `last.pt` with the original cumulative epoch
target and optimizer state.

## Training and qualification

`configs/autotrain.yaml` starts with YOLO11n pretrained weights, 640-pixel images,
and cumulative smoke/pilot/maximum milestones of 1, 10, and 80 epochs. It inherits
the existing detector implementation and configuration contract. Set a smaller
`max_total_epochs`, batch size, or runtime limit when local resources require it.
The workflow records hardware, versions, DVC status and configuration-file hashes,
training duration, epoch history, checkpoints, evaluation metrics, error samples,
and MLflow run IDs.

Validation metrics include overall and per-class precision, recall, mAP50, and
mAP50-95. False-positive and missed-object samples are saved with review images.
The held-out test split is evaluated only after configured validation thresholds,
dataset integrity, export-manifest/hash validation, and required runtime parity
qualify a candidate. It is not used to select thresholds or models.

The default numeric quality thresholds are intentionally null. Therefore, the
default configuration cannot qualify or export a deployment candidate. Configure
measurable validation thresholds only after reviewing the data and evaluation
protocol. A false-positive-per-hour threshold cannot pass using still images;
representative timestamped video is required. Required Python/.NET parity also
blocks qualification if the .NET runtime is unavailable or validation data does
not produce non-empty detections for both classes.

Even a technically qualified candidate is only packaged locally for manual
review. License review and explicit external deployment approval remain separate
gates. `auto_deploy` is rejected when enabled; no workflow copies a model into
SurveillanceAI or deploys it automatically.

Artifacts are written under the configured output root, including `state.json`,
immutable configuration snapshot, audit reports/contact sheets, environment
details, staged training output, validation metrics, error analysis, optional ONNX
candidate, parity results, and JSON/Markdown report. The default local MLflow UI
can be started with:

```powershell
uv run mlflow ui --backend-store-uri sqlite:///artifacts/autotrain/mlflow.db --host 127.0.0.1 --port 5000
```

The implementation performs a DVC status/configuration snapshot; it does not
automatically add datasets to DVC or execute a DVC training pipeline. The default
workflow trains only from a local dataset and does not download data or weights
unless the existing Ultralytics pretrained-weight loader must fetch the configured
public model weights.

## Validation status

The configured default dataset path (`data/weapon-detection`) is not present in
this checkout. AutoTrain was validated with synthetic fixtures, local MLflow
tracking tests, and a dry-run plan; no weapon-dataset training, ONNX candidate
export, or Python/.NET model inference was performed.

## Limitations

AutoTrain does not certify annotation correctness, dataset provenance, license
compatibility, generalization, or real-time surveillance performance. Contact
sheets are a human review aid, not ground-truth verification. Image-based
validation does not measure false positives per hour or temporal stability. A
candidate should be treated as an experimental artifact requiring independent
review, representative video evaluation, privacy safeguards, and operator
verification—not as proof of intent or an autonomous security decision.
