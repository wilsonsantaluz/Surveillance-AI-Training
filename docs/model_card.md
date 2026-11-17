# Model card: ResNet18 scene classifier

## Status

Phases 1–3 provide training, evaluation, and ONNX inference support. The model predicts one of `empty_scene`, `single_person`, or `multiple_people` for a single image. The repository does not ship a trained checkpoint.

## Intended use

Development and evaluation of a supervised image-classification pipeline. Do not use this model as a sole input to safety, employment, law-enforcement, access-control, or other consequential decisions.

## Training data

The included demo generator produces colored geometric backgrounds and silhouettes. Its labels are exact for the generated shapes, not human annotations. It is not representative of real people, camera viewpoints, lighting, occlusion, or operational environments. No third-party image dataset is bundled.

## Evaluation

Training runs produce a held-out test report and MLflow metrics. Demo metrics measure only the generated dataset and are not estimates of real-world performance. No pretrained-vs-finetuned or cross-domain results are claimed by this repository.

## Limitations

The classifier does not locate people, associate tracks, identify individuals, or understand scene context. Dataset selection, consent, label quality, bias, privacy, and performance validation remain the user's responsibility.

## Deployment

The ONNX exporter validates the graph and checks PyTorch/ONNX Runtime logit parity
before writing a model bundle. The local FastAPI service requires an API key, limits
upload size and decoded image dimensions, and returns uncalibrated softmax scores.
The benchmark reports measured ONNX Runtime CPU timings for the selected machine and
input; it does not claim general performance or speedup. See
[`deployment.md`](deployment.md) for deployment commands and limitations.
