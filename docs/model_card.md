# Model card: ResNet18 scene classifier

## Status

Phase 1 foundation. The model predicts one of `empty_scene`, `single_person`, or `multiple_people` for a single image. The repository does not ship a trained checkpoint.

## Intended use

Development and evaluation of a supervised image-classification pipeline. Do not use this model as a sole input to safety, employment, law-enforcement, access-control, or other consequential decisions.

## Training data

The included demo generator produces colored geometric backgrounds and silhouettes. Its labels are exact for the generated shapes, not human annotations. It is not representative of real people, camera viewpoints, lighting, occlusion, or operational environments. No third-party image dataset is bundled.

## Evaluation

Training runs produce a held-out test report and MLflow metrics. Demo metrics measure only the generated dataset and are not estimates of real-world performance. No pretrained-vs-finetuned or cross-domain results are claimed by this repository.

## Limitations

The classifier does not locate people, associate tracks, identify individuals, or understand scene context. Dataset selection, consent, label quality, bias, privacy, and performance validation remain the user's responsibility.
