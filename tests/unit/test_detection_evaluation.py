from pathlib import Path
from types import SimpleNamespace

from surveillance_ml.detection.evaluation import _predict_image


def test_evaluation_uses_fixed_letterbox_geometry() -> None:
    call_arguments: dict[str, object] = {}

    class Model:
        def predict(self, **kwargs: object) -> list[SimpleNamespace]:
            call_arguments.update(kwargs)
            return [SimpleNamespace(boxes=None)]

    assert (
        _predict_image(
            Model(),  # type: ignore[arg-type]
            Path("image.png"),
            image_size=640,
            confidence_threshold=0.001,
            nms_iou_threshold=0.7,
            max_detections=300,
            device="cpu",
        )
        == ()
    )

    assert call_arguments["rect"] is False
