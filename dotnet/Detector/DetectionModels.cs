using System.Text.Json.Serialization;

namespace SurveillanceDetection;

public sealed record BoundingBox(
    [property: JsonPropertyName("x")] float X,
    [property: JsonPropertyName("y")] float Y,
    [property: JsonPropertyName("width")] float Width,
    [property: JsonPropertyName("height")] float Height)
{
    public float Right => X + Width;
    public float Bottom => Y + Height;
}

public sealed record DetectionPrediction(
    [property: JsonPropertyName("class_id")] int ClassId,
    [property: JsonPropertyName("class_name")] string ClassName,
    [property: JsonPropertyName("confidence")] float Confidence,
    [property: JsonPropertyName("box")] BoundingBox Box);

public sealed record DetectionFrame(
    [property: JsonPropertyName("model_id")] string ModelId,
    [property: JsonPropertyName("model_version")] string ModelVersion,
    [property: JsonPropertyName("captured_at_utc")] DateTimeOffset CapturedAtUtc,
    [property: JsonPropertyName("frame_width")] int FrameWidth,
    [property: JsonPropertyName("frame_height")] int FrameHeight,
    [property: JsonPropertyName("inference_milliseconds")] double InferenceMilliseconds,
    [property: JsonPropertyName("detections")] IReadOnlyList<DetectionPrediction> Detections);

public sealed record DetectionAlertV1(
    [property: JsonPropertyName("schema_version")] int SchemaVersion,
    [property: JsonPropertyName("event_id")] Guid EventId,
    [property: JsonPropertyName("event_type")] string EventType,
    [property: JsonPropertyName("camera_id")] string CameraId,
    [property: JsonPropertyName("observed_at_utc")] DateTimeOffset ObservedAtUtc,
    [property: JsonPropertyName("model_id")] string ModelId,
    [property: JsonPropertyName("model_version")] string ModelVersion,
    [property: JsonPropertyName("rule_id")] string RuleId,
    [property: JsonPropertyName("detections")] IReadOnlyList<DetectionPrediction> Detections);

public sealed record DetectionFrameResult(
    [property: JsonPropertyName("frame")] DetectionFrame Frame,
    [property: JsonPropertyName("alerts")] IReadOnlyList<DetectionAlertV1> Alerts);
