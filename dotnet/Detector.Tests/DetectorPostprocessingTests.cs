using SurveillanceDetection;
using SkiaSharp;
using Xunit;

namespace SurveillanceDetection.Tests;

public sealed class DetectorPostprocessingTests
{
    [Fact]
    public void ConfidenceThresholdIsInclusiveAndRejectsInvalidScores()
    {
        Assert.False(Detector.AcceptConfidence(0.249999f, 0.25f));
        Assert.True(Detector.AcceptConfidence(0.25f, 0.25f));
        Assert.True(Detector.AcceptConfidence(1f, 0.25f));
        Assert.False(Detector.AcceptConfidence(float.NaN, 0.25f));
        Assert.False(Detector.AcceptConfidence(1.001f, 0.25f));
    }

    [Fact]
    public void NmsIsClassAwareAndSuppressesOnlyAboveTheIouThreshold()
    {
        var candidates = new List<DetectionPrediction>
        {
            new(0, "firearm", 0.9f, new BoundingBox(0, 0, 15, 10)),
            new(0, "firearm", 0.85f, new BoundingBox(1, 0, 15, 10)),
            new(0, "firearm", 0.8f, new BoundingBox(5, 0, 15, 10)),
            new(1, "knife", 0.7f, new BoundingBox(0, 0, 15, 10)),
        };

        var predictions = Detector.ApplyNms(candidates, 0.5f, 10);

        Assert.Equal([0.9f, 0.8f, 0.7f], predictions.Select(item => item.Confidence));
    }

    [Fact]
    public void LinearResizeMatchesOpenCvBitExactLinearPixelRounding()
    {
        using var source = new SKBitmap(2, 2);
        source.SetPixel(0, 0, new SKColor(0, 0, 0));
        source.SetPixel(1, 0, new SKColor(100, 100, 100));
        source.SetPixel(0, 1, new SKColor(200, 200, 200));
        source.SetPixel(1, 1, new SKColor(254, 254, 254));

        var resized = Detector.ResizeRgbInterLinearExact(source, 3, 3);

        Assert.Equal(
            [
                0, 0, 0, 50, 50, 50, 100, 100, 100,
                100, 100, 100, 139, 139, 139, 177, 177, 177,
                200, 200, 200, 227, 227, 227, 254, 254, 254,
            ],
            resized);
    }

}
