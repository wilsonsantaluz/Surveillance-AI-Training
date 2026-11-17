# Weapon Detection Dataset Smoke Experiment

**Run date:** 2026-11-17
**Status:** Pipeline smoke only; not suitable for deployment or operational use.

## Dataset acquisition and audit

The requested [Roboflow dataset](https://universe.roboflow.com/wd-pigj0/train-weapon)
could not be verified or downloaded: its project page returned HTTP 403 and its API
returned HTTP 401 without credentials. No credentials were supplied or used. Its
downloadable version, annotations, provenance, and license therefore remain
unverified, so it was not used.

As a limited pipeline check, the experiment used the pinned
[Simuletic dataset](https://huggingface.co/datasets/Simuletic/Surveillance-VLM-Weapon-Knife-Detection-Dataset)
at revision `768a0ca69bf1b8f3a6f80f6f8a2ca7001a1241c7`. The dataset card declares
CC BY-SA 4.0 and describes the images as synthetic. These claims were taken from
the published card; the upstream provenance and license were not independently
audited. Attribution and share-alike requirements apply, and this report does not
clear commercial redistribution.

The source contained 146 records. Two records with contradictory box-related text
were excluded without relabeling; 144 records were retained. Source class names
were mapped explicitly to `0: firearm` and `1: knife`; no classes were merged.
The audited split was:

| Split | Images | Firearm boxes | Knife boxes | Negative images |
|---|---:|---:|---:|---:|
| Train | 101 | 44 | 41 | 16 |
| Validation | 30 | 10 | 12 | 8 |
| Test | 13 | 5 | 6 | 2 |

The validator reported 59 boxes in each class (object-count ratio 1.0), no
corrupt or missing images, no missing label files, no invalid coordinates, no
exact duplicate groups, and no source-group overlap between splits. Weapon
scenes 1–4 were assigned to train, scene 5 to validation, and scene 6 to test.
Knife scenario groups were stratified by positive/negative label using seed 42
and kept intact across splits. The complete audit metadata is in the ignored
local artifact `data/detection-real/simuletic-free-sample/yolo/source-manifest.json`;
the dataset itself is not committed.

## Training and evaluation

The real pretrained-model smoke run used the existing YOLO11 pipeline. It was
not a new architecture and did not modify `configs/detection.yaml` or the
synthetic demo configuration.

| Setting | Value |
|---|---|
| Architecture / weights | YOLO11n / pretrained |
| Input size | 640 × 640 |
| Epochs / batch | 1 / 1 |
| Optimizer / learning rate | AdamW / 0.001 |
| Seed | 42 |
| Device | CPU; mixed precision disabled |
| Training duration | 53.45 seconds |
| Prediction / operating confidence | 0.001 / 0.25 |
| NMS IoU | 0.7 |

The host has an Intel i7-1165G7, 4 physical and 8 logical cores, 23.73 GiB
memory, and an NVIDIA MX450 with 2 GiB. The installed PyTorch build is
`2.15.0+cpu`, so CUDA was unavailable to this run.

Recorded one-epoch losses were training box `2.1103`, classification `11.5473`,
and DFL `1.4878`; validation box `2.3803`, classification `7.0196`, and DFL
`1.8713`. These are actual run outputs, not estimates.

After correcting inference to use fixed-size letterboxing (`rect=False`) to match
the ONNX input contract, the existing evaluator was rerun and logged to MLflow:

| Split | Images | Precision | Recall | mAP50 | mAP50–95 | TP / FP / FN |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 30 | 0.0635 | 0.1818 | 0.0790 | 0.0344 | 4 / 59 / 18 |
| Held-out test | 13 | 0.2727 | 0.2727 | 0.1262 | 0.0615 | 3 / 8 / 8 |

On the held-out test split, firearm recall was zero (0/5 firearm boxes detected;
firearm AP50 `0.0008`). Knife recall was 0.5 (3/6 boxes; knife AP50 `0.2517`).
False negatives included all five held-out firearm images
(`weapons_Scene6_4.png`, `_6.png`, `_8.png`, `_12.png`, `_14.png`) and three
knife images. False positives included duplicate detections on knife images,
including `knifes_Knife_scenario_19.png` (2) and
`knifes_Knife_scenario_73.png` (2); one extra knife prediction also occurred on
`weapons_Scene6_4.png`.

The earlier evaluation used Ultralytics' automatic rectangular sizing and is
superseded by the fixed-shape metrics above. The evaluator and ONNX parity
reference now explicitly use `rect=False`; focused regression tests cover this
contract.

The 20-epoch schedule was not run. Validation results are weak, the dataset is
small and synthetic, and the environment is CPU-only. Those validation and
resource constraints did not justify extending this pipeline smoke. The
held-out test set was evaluated once for reporting only; it was not used to
choose a model or threshold.

## MLflow, export, and runtime checks

The local MLflow experiment is `weapon-detection-simuletic-smoke`:

| Run | Run ID |
|---|---|
| One-epoch training | `c0d304368d614c4a805a5de00e76b8c2` |
| Fixed-shape validation evaluation | `568d93af3c734c97b2c144e42c638014` |
| Fixed-shape held-out test evaluation | `d023b444ab2048d399fc1c4960bf121f` |
| Fixed-reference ONNX export | `921d6d94651541b7881d04d47c02dd96` |

The exported YOLO11n ONNX bundle uses opset 17, a fixed `[1, 3, 640, 640]`
input, and the explicit `firearm`, `knife` class order. The manifest contract
and model/checkpoint hashes were verified:

- ONNX SHA-256: `55f977ad630b5c2e937e7a47f99fca0cbda64d62f52a37aa5f5f6acd79f70124`
- Checkpoint SHA-256: `e27881acba01fa7a5afb8d5e8bb0734cfa3dec54152c82ce0d8d2503f2fe67d5`

The export's two deterministic raw-output parity cases passed (maximum absolute
errors `0.00501` and `0.00476`), but both produced zero postprocessed
detections. Therefore those cases alone do not demonstrate non-empty detection
parity.

On actual held-out images, fixed-shape PyTorch and Python ONNX Runtime agreed:
both returned three knife detections on `knifes_Knife_scenario_19.png`, and
neither returned detections on `weapons_Scene6_12.png` or the two negative
images. The independent .NET 9 detector returned the same three knife boxes on
the positive image, with small confidence/coordinate differences. It also
returned one knife detection at confidence `0.2527` on the negative
`knifes_safe_Knife_scenario_80.png`; Python PyTorch/ONNX returned none at the
manifest threshold `0.25`. This near-threshold disagreement means full
cross-runtime output agreement is **not verified**. It may be sensitive to
differences in image resizing/interpolation; that cause has not been confirmed.
The existing .NET test suite passed 5/5 tests, but those tests do not negate the
observed real-image mismatch.

## Limitations and next steps

- The repository's DVC detection demo generates its own synthetic fixtures, so
  it was not run for this audited external dataset. Dataset validation,
  training, evaluation, and export used the existing Phase 4 components
  directly; no new DVC stage or dataset artifacts were committed.
- The fallback images are synthetic and the dataset is tiny; they provide no
  evidence of real-world weapon detection accuracy.
- The model missed every firearm in the held-out set and produced many
  validation false positives. It is not a viable detector.
- Cross-runtime negative-image parity failed near the confidence threshold.
  Align .NET preprocessing with the ONNX/OpenCV preprocessing contract, then
  rerun the comparison; any operating threshold decision must use validation
  data only.
- CPU-only training and four one-off .NET image timings are not a real-time
  performance benchmark. Real-time throughput, camera variability, and
  robustness to occlusion, distance, lighting, and viewpoint are untested.
- The public Roboflow source remains inaccessible for verification. A future
  experiment needs an accessible, provenance-documented dataset with
  independently verified licenses and realistic, diverse positive and hard
  negative examples.
- Review the dataset's CC BY-SA terms, Ultralytics' AGPL-3.0 obligations, and
  separate pretrained-weight terms before redistribution or commercial use.
- Detections, if ever used, must be treated only as candidates for operator
  verification—not proof of intent or grounds for automated action.

Dataset files, checkpoints, ONNX bundles, and MLflow storage remain local and
gitignored. No dataset, private footage, credentials, or large model artifacts
were added to the repository.

The local run configuration is `artifacts/detection-real/simuletic-smoke.yaml`;
the fixed-shape evaluation JSON files, exported bundle, parity output, and .NET
predictions are under `artifacts/detection-real/`. These are intentionally not
tracked, so a fresh checkout must reacquire and re-audit the pinned dataset
before rerunning the experiment.

Verification performed after aligning fixed-size inference:

- `pytest tests/unit/test_detection_dataset.py tests/unit/test_detection_onnx.py tests/unit/test_detection_evaluation.py -q` — 9 passed.
- Ruff on the changed Python modules and tests — passed.
- Mypy on `evaluation.py` and `onnx.py` — passed.
- `dotnet test dotnet/Detector.Tests/Detector.Tests.csproj --configuration Release --no-restore` — 5 passed.
- ONNX manifest/hash verification and actual-image inference completed; the
  real-image .NET negative-sample discrepancy described above remains.
