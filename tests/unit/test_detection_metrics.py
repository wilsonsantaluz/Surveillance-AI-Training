from surveillance_ml.detection.metrics import (
    Box,
    Detection,
    EvaluationSample,
    calculate_detection_metrics,
    intersection_over_union,
)


def test_perfect_boxes_produce_perfect_metrics() -> None:
    firearm_box = Box(10, 10, 30, 30)
    knife_box = Box(40, 10, 60, 30)
    sample = EvaluationSample(
        sample_id="sample-a",
        ground_truths=(
            Detection(0, 1.0, firearm_box),
            Detection(1, 1.0, knife_box),
        ),
        predictions=(
            Detection(0, 0.9, firearm_box),
            Detection(1, 0.8, knife_box),
        ),
    )

    metrics = calculate_detection_metrics(
        [sample],
        ("firearm", "knife"),
        confidence_threshold=0.25,
    )

    assert metrics["precision"] == 1
    assert metrics["recall"] == 1
    assert metrics["mAP50"] == 1
    assert metrics["mAP50_95"] == 1
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0
    assert metrics["confusion_matrix"]["rows_true_columns_predicted"][0][0] == 1


def test_metrics_count_false_positive_and_false_negative_separately() -> None:
    sample = EvaluationSample(
        sample_id="sample-b",
        ground_truths=(Detection(0, 1.0, Box(0, 0, 10, 10)),),
        predictions=(Detection(1, 0.8, Box(20, 20, 30, 30)),),
    )

    metrics = calculate_detection_metrics(
        [sample],
        ("firearm", "knife"),
        confidence_threshold=0.25,
    )

    assert metrics["true_positives"] == 0
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["confusion_matrix"]["rows_true_columns_predicted"][0][2] == 1
    assert metrics["confusion_matrix"]["rows_true_columns_predicted"][2][1] == 1


def test_iou_handles_non_overlapping_and_identical_boxes() -> None:
    assert intersection_over_union(Box(0, 0, 10, 10), Box(0, 0, 10, 10)) == 1
    assert intersection_over_union(Box(0, 0, 10, 10), Box(11, 11, 20, 20)) == 0


def test_equal_confidence_predictions_are_ordered_stably() -> None:
    firearm_boxes = (Box(0, 0, 10, 10), Box(20, 20, 30, 30))
    firearm_sample = EvaluationSample(
        sample_id="sample-ties",
        ground_truths=tuple(Detection(0, 1.0, box) for box in firearm_boxes),
        predictions=tuple(Detection(0, 0.8, box) for box in firearm_boxes),
    )
    knife_sample = EvaluationSample(
        sample_id="sample-knife",
        ground_truths=(Detection(1, 1.0, Box(40, 40, 50, 50)),),
        predictions=(Detection(1, 0.7, Box(40, 40, 50, 50)),),
    )

    metrics = calculate_detection_metrics(
        [firearm_sample, knife_sample],
        ("firearm", "knife"),
        confidence_threshold=0.25,
    )

    assert metrics["precision"] == 1
    assert metrics["recall"] == 1
    assert metrics["mAP50"] == 1
