# Zenodo Dangerous Items Dataset: Audit and Smoke Experiment

**Status:** One-epoch exploratory smoke complete. The checkpoint is not a
deployment candidate. No long training run or ONNX export was started.

## Source, access, and license

The [Zenodo record 11582054](https://zenodo.org/records/11582054) and its
[metadata API](https://zenodo.org/api/records/11582054) were accessible. The
published archive,
[`dangerous_items_dataset_ver.1.zip`](https://zenodo.org/api/records/11582054/files/dangerous_items_dataset_ver.1.zip/content),
was downloaded and audited. Its reported size was 358,059,123 bytes; the
downloaded archive matched the publisher's MD5
`d412181903d63fafe5e4f5b4e5c6d3bb`. The audited SHA-256 is
`5881e97aab643536eecca9658163fd07397de81514e3c27b9b4b56f2243b1f8c`.

The record declares **CC BY 4.0**. Its citation points to
[the dataset paper](https://doi.org/10.20944/preprints202406.1090.v1) and a
Google Drive source link. The archive's Pascal VOC metadata describes images
from COCO2017/Flickr, but all 4,000 Flickr IDs are `NULL`; per-image source URLs
and original license identifiers are therefore unavailable. The Zenodo-level
license declaration is not enough to independently clear the underlying
images for commercial redistribution. Preserve attribution and review source
rights before redistribution or commercial use.

## Audit and conversion

The archive contained 4,000 JPEGs and 4,000 Pascal VOC XML files, with 3,000
publisher-train and 1,000 publisher-test entries. Every image decoded and
matched the dimensions in its XML; there were no missing image/annotation
pairs. The raw annotations contained 160 boxes with `xmin` or `ymin` equal to
zero. The archive labels itself Pascal VOC but also identifies COCO2017/Flickr
and contains zero-based coordinates, so the importer retains zero-valued
coordinates and treats extents as zero-based half-open pixel coordinates. This
interpretation is documented but cannot be confirmed from per-image source
metadata; annotation convention remains a limitation.

Class mapping was explicit: `Gun` and `Rifle` map to `firearm`; `Knife` and
`Machete` map to `knife`. `Baseball Bat` is unsupported, so the entire
perceptual source group is excluded rather than relabelled as negative. Records
with exact duplicate pixels but conflicting mapped annotations are excluded.
Source groups are connected by 64-bit DCT perceptual hashes at Hamming distance
at most 2. Groups containing any publisher-test record stay in test; validation
is carved from the remaining publisher-train groups using seed 42. The
publisher's original lists were disjoint, but the audit found 167 exact
duplicate groups (186 redundant copies), 69 crossing the original train/test
boundary. It also found 228 multi-image perceptual groups, 94 crossing that
boundary.

The filtered converted dataset has 2,824 images and no cross-split source-group
overlap:

| Split | Images | Source groups | Firearm boxes | Knife boxes |
|---|---:|---:|---:|---:|
| Train | 1,779 | 1,754 | 959 | 1,059 |
| Validation | 315 | 310 | 153 | 194 |
| Test | 730 | 706 | 376 | 436 |
| **Total** | **2,824** | **2,770** | **1,488** | **1,689** |

Object-count imbalance is modest (knife/firearm ratio 1.135). The converter
excluded 862 records in groups containing unsupported baseball-bat objects,
312 in groups with duplicate annotation conflicts, and 2 in groups with both
conditions. The existing YOLO dataset validator passed: no corrupt images,
missing labels, invalid coordinates, or cross-split source groups. Every
retained image has at least one target box; **there are no negative/background
images**, so this dataset cannot measure false-positive rates on negative
scenes. Full conversion provenance and validator output remain in the
gitignored local `data/detection-real/zenodo-11582054/yolo/` directory; data
and archives were not added to Git.

## Training and validation

The existing Phase 4 YOLO11 pipeline was used without changing the synthetic
demonstration configuration. The smoke subset was source-group-safe, with 64
training, 32 validation, and 33 held-out test images. The test split was not
used to select the model or thresholds.

| Setting | Value |
|---|---|
| Architecture / weights | YOLO11n / pretrained `yolo11n.pt` |
| Image size | 640 |
| Epochs / batch | 1 / 4 |
| Optimizer / learning rate | AdamW / 0.001 |
| Seed | 42 |
| Device | CPU (`device:auto` resolved to CPU); 4 CPU threads |
| Mixed precision | Disabled |
| Training duration | 33.72 seconds |
| Prediction / operating confidence | 0.001 / 0.25 |
| NMS IoU | 0.7 |

The host had an Intel i7-1165G7 (4 cores / 8 logical processors), 23.73 GiB
RAM, and an NVIDIA MX450. Installed PyTorch `2.15.0+cpu` reported CUDA
unavailable. The run environment was Python 3.14.0, TorchVision 0.29.2, MLflow
3.18.0, and Ultralytics 8.4.215.

Actual final-epoch training losses were box `2.0251`, classification `6.84994`,
and DFL `2.25003`; validation losses were box `2.5999`, classification `6.1657`,
and DFL `2.91137`. At operating confidence 0.25 and IoU 0.50, validation
metrics were:

| Class | GT boxes | Precision | Recall | AP50 | AP50–95 | TP / FP / FN |
|---|---:|---:|---:|---:|---:|---:|
| Firearm | 20 | 0.0244 | 0.0500 | 0.0156 | 0.0041 | 1 / 40 / 19 |
| Knife | 19 | 0.1176 | 0.2105 | 0.0439 | 0.0152 | 4 / 30 / 15 |
| **Overall** | **39** | **0.0667** | **0.1282** | **0.0297** | **0.0097** | **5 / 70 / 34** |

The largest validation false-positive cluster was `7c0a1fdd-3268.jpg` (15
unmatched predictions); `d758c63e-1164.jpg` had the most false negatives (2).
These are source image names in the local smoke subset. Qualitative inference
at confidence 0.25 on held-out Zenodo test image `767771ca-1762.jpg` (firearm
ground truth) returned no detections. Image `8bdf7c3d-3734.jpg` (knife ground
truth) returned three firearm predictions at 0.68051, 0.63394, and 0.36052,
with no correct-class detection. The confirmed Simuletic synthetic negative
`knifes_safe_Knife_scenario_40.png` (empty label) returned no detections. One
negative sample is not a substitute for a representative negative test set.

MLflow recorded the training run (`3f2d2079196c4000b01c1ea6554e2957`) and
validation evaluation (`6524058239c7453aa5e45847b1b4e6d3`) in the local
`weapon-detection-zenodo-smoke` experiment. The best checkpoint is retained
only in ignored local artifacts. These results are very weak for both classes,
so no longer schedule was justified and no deployment candidate was exported.
No new Zenodo-model ONNX manifest, hash verification, Python/ONNX parity, or
final-model .NET parity is claimed.

## Python/.NET threshold discrepancy

The independent .NET detector previously differed from Python near the
confidence threshold because platform-optimized OpenCV `INTER_LINEAR`
interpolation did not exactly match the .NET resizer. Python now specifies
`INTER_LINEAR_EXACT`; the manifest declares `opencv_inter_linear_exact`, and
.NET implements matching fixed-point interpolation. Confidence-boundary and
NMS regression tests were added in Python and .NET.

After the change, the pre-existing detector bundle produced matching Python and
.NET results on the tested positive image (three knife detections) and negative
image (one knife false positive at confidence 0.25391 with threshold 0.25).
This resolves the runtime disagreement for those samples, not the underlying
false positive. It does not validate the unexported Zenodo smoke checkpoint.

## Limitations and recommendation

- One CPU epoch on a small subset is only a pipeline smoke test, not a useful
  accuracy estimate or deployment evaluation.
- Validation metrics are poor; the firearm AP50 is 0.0156 and knife AP50 is
  0.0439. The checkpoint is not suitable for deployment.
- The source archive has no negative images, incomplete per-image provenance,
  and unresolved coordinate-convention uncertainty.
- No long training run was attempted because the short run did not show
  meaningful validation performance. No ONNX export was made.
- The earlier Python/.NET parity check used a different, pre-existing model.
  Final-model export, manifest/hash validation, ONNX Runtime parity, and
  end-to-end .NET inference remain unverified for this experiment.
- License review for underlying COCO/Flickr content, any detector-weight
  redistribution, real-time throughput, and robustness to camera conditions
  remain open.

Any future detector output must be treated as a candidate for operator
verification, not proof of intent or grounds for automated action. Dataset
files, checkpoints, and MLflow storage remain outside Git; no commit or push
was made.

## Verification

- Converted-dataset validation — passed: 2,824 images; 1,488 firearm and 1,689
  knife boxes; no corrupt/missing images or labels, invalid coordinates, or
  source-group overlap.
- Focused Python detection tests — 17 passed.
- Ruff and mypy on changed Python code — passed.
- `dotnet test dotnet/Detector.Tests/Detector.Tests.csproj --configuration
  Release` — 8 passed. A NuGet restore was required because the local .NET 9
  targeting pack was not present for the initial `--no-restore` attempt.
- `git diff --check` — passed.
