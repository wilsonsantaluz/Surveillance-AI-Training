using System.Text.Json;
using SurveillanceDetection;

if (args.Length < 4)
{
    Console.Error.WriteLine(
        "Usage: Detector <model.onnx> <manifest.json> <camera-id> <image> [<image> ...]");
    return 2;
}

using var detector = new Detector(args[0], args[1]);
var confirmer = new TemporalConfirmation();
var results = new List<DetectionFrameResult>();
for (var index = 3; index < args.Length; index++)
{
    var frame = detector.Detect(File.ReadAllBytes(args[index]), DateTimeOffset.UtcNow);
    results.Add(new DetectionFrameResult(
        frame,
        confirmer.ProcessFrame(args[2], frame)));
}
Console.WriteLine(JsonSerializer.Serialize(
    results,
    new JsonSerializerOptions { WriteIndented = true }));
return 0;
