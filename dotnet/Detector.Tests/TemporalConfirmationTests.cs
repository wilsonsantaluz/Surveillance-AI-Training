using SurveillanceDetection;
using Xunit;

namespace SurveillanceDetection.Tests;

public sealed class TemporalConfirmationTests
{
    private static readonly DateTimeOffset Start =
        new(2025, 1, 1, 0, 0, 0, TimeSpan.Zero);

    [Fact]
    public void EmitsOneTypedAlertAfterConsecutiveDetections()
    {
        var confirmer = new TemporalConfirmation(Options());

        Assert.Empty(confirmer.ProcessFrame("cam-1", Frame(0)));
        Assert.Empty(confirmer.ProcessFrame("cam-1", Frame(1)));
        var alerts = confirmer.ProcessFrame("cam-1", Frame(2));
        Assert.Single(alerts);
        Assert.Equal(1, alerts[0].SchemaVersion);
        Assert.Equal("object_detection.confirmed", alerts[0].EventType);
        Assert.Equal("cam-1", alerts[0].CameraId);
        Assert.Single(alerts[0].Detections);
        Assert.Empty(confirmer.ProcessFrame("cam-1", Frame(3)));
    }

    [Fact]
    public void AMissResetsConsecutiveCountAndClearsAfterConfiguredMisses()
    {
        var confirmer = new TemporalConfirmation(Options(clearAfterMisses: 2));
        Assert.Empty(confirmer.ProcessFrame("cam-1", Frame(0)));
        Assert.Empty(confirmer.ProcessFrame("cam-1", Frame(1)));
        Assert.Empty(confirmer.ProcessFrame("cam-1", Frame(2, detections: [])));
        Assert.Empty(confirmer.ProcessFrame("cam-1", Frame(3)));
        Assert.Empty(confirmer.ProcessFrame("cam-1", Frame(4)));
        Assert.Single(confirmer.ProcessFrame("cam-1", Frame(5)));
    }

    [Fact]
    public void CooldownAppliesEvenWhenABoxStartsANewTrack()
    {
        var confirmer = new TemporalConfirmation(Options(cooldown: TimeSpan.FromSeconds(10)));
        Assert.Empty(confirmer.ProcessFrame("cam-1", Frame(0)));
        Assert.Empty(confirmer.ProcessFrame("cam-1", Frame(1)));
        Assert.Single(confirmer.ProcessFrame("cam-1", Frame(2)));
        Assert.Empty(confirmer.ProcessFrame("cam-1", Frame(3, detections: [])));
        Assert.Empty(confirmer.ProcessFrame("cam-1", Frame(4, detections: [])));

        var shifted = Detection(new BoundingBox(100, 100, 20, 20));
        Assert.Empty(confirmer.ProcessFrame("cam-1", Frame(5, [shifted])));
        Assert.Empty(confirmer.ProcessFrame("cam-1", Frame(6, [shifted])));
        Assert.Empty(confirmer.ProcessFrame("cam-1", Frame(7, [shifted])));

        Assert.Empty(confirmer.ProcessFrame("cam-1", Frame(12, [shifted])));
        Assert.Empty(confirmer.ProcessFrame("cam-1", Frame(13, [shifted])));
        Assert.Single(confirmer.ProcessFrame("cam-1", Frame(14, [shifted])));
    }

    [Fact]
    public void CameraStateIsIndependent()
    {
        var confirmer = new TemporalConfirmation(Options());
        Assert.Empty(confirmer.ProcessFrame("cam-1", Frame(0)));
        Assert.Empty(confirmer.ProcessFrame("cam-2", Frame(0)));
        Assert.Empty(confirmer.ProcessFrame("cam-1", Frame(1)));
        Assert.Empty(confirmer.ProcessFrame("cam-2", Frame(1)));
        Assert.Single(confirmer.ProcessFrame("cam-1", Frame(2)));
        Assert.Single(confirmer.ProcessFrame("cam-2", Frame(2)));
    }

    [Fact]
    public void RejectsInvalidConfigurationAndDetections()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new TemporalConfirmation(new TemporalConfirmationOptions
            {
                MinimumConsecutiveDetections = 0,
            }));

        var confirmer = new TemporalConfirmation(Options());
        var invalid = Detection() with { ClassName = "unknown" };
        Assert.Throws<ArgumentException>(() =>
            confirmer.ProcessFrame("cam-1", Frame(0, [invalid])));
    }

    private static TemporalConfirmationOptions Options(
        int clearAfterMisses = 2,
        TimeSpan? cooldown = null) =>
        new()
        {
            MinimumConsecutiveDetections = 3,
            MinimumConfidence = 0.5f,
            ConfirmationWindow = TimeSpan.FromSeconds(2),
            MinimumMatchIoU = 0.1f,
            ClearAfterMisses = clearAfterMisses,
            Cooldown = cooldown ?? TimeSpan.Zero,
        };

    private static DetectionFrame Frame(
        int seconds,
        IReadOnlyList<DetectionPrediction>? detections = null) =>
        new(
            "demo-model",
            "1.0",
            Start.AddSeconds(seconds),
            320,
            240,
            1.5,
            detections ?? [Detection()]);

    private static DetectionPrediction Detection(BoundingBox? box = null) =>
        new(0, "firearm", 0.9f, box ?? new BoundingBox(10, 10, 20, 20));
}
