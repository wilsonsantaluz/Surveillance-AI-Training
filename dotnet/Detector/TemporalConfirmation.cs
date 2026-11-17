namespace SurveillanceDetection;

public sealed record TemporalConfirmationOptions
{
    public int MinimumConsecutiveDetections { get; init; } = 3;
    public float MinimumConfidence { get; init; } = 0.65f;
    public TimeSpan ConfirmationWindow { get; init; } = TimeSpan.FromSeconds(2);
    public float MinimumMatchIoU { get; init; } = 0.1f;
    public int ClearAfterMisses { get; init; } = 5;
    public TimeSpan Cooldown { get; init; } = TimeSpan.FromSeconds(15);
    public string RuleId { get; init; } = "temporal-confirmation-v1";

    public void Validate()
    {
        if (MinimumConsecutiveDetections < 1
            || !float.IsFinite(MinimumConfidence)
            || MinimumConfidence is < 0 or > 1
            || ConfirmationWindow <= TimeSpan.Zero
            || !float.IsFinite(MinimumMatchIoU)
            || MinimumMatchIoU is < 0 or > 1
            || ClearAfterMisses < 1
            || Cooldown < TimeSpan.Zero
            || string.IsNullOrWhiteSpace(RuleId))
        {
            throw new ArgumentOutOfRangeException(nameof(TemporalConfirmationOptions));
        }
    }
}

public sealed class TemporalConfirmation
{
    private static readonly string[] SupportedClassNames = ["firearm", "knife"];

    private sealed class Track(int classId, BoundingBox box, DateTimeOffset observedAt)
    {
        public Guid Id { get; } = Guid.NewGuid();
        public int ClassId { get; } = classId;
        public BoundingBox Box { get; set; } = box;
        public DateTimeOffset LastSeen { get; set; } = observedAt;
        public int ConsecutiveDetections { get; set; } = 1;
        public int Misses { get; set; }
        public bool AlertActive { get; set; }
    }

    private readonly TemporalConfirmationOptions _options;
    private readonly Dictionary<string, List<Track>> _tracks = new(StringComparer.Ordinal);
    private readonly Dictionary<(string CameraId, int ClassId), DateTimeOffset> _lastAlertAt = [];
    private readonly Dictionary<string, DateTimeOffset> _lastFrameAt = new(StringComparer.Ordinal);
    private readonly object _gate = new();

    public TemporalConfirmation(TemporalConfirmationOptions? options = null)
    {
        _options = options ?? new TemporalConfirmationOptions();
        _options.Validate();
    }

    public IReadOnlyList<DetectionAlertV1> ProcessFrame(string cameraId, DetectionFrame frame)
    {
        if (string.IsNullOrWhiteSpace(cameraId))
        {
            throw new ArgumentException("cameraId must not be empty.", nameof(cameraId));
        }
        ArgumentNullException.ThrowIfNull(frame);

        lock (_gate)
        {
            if (frame.CapturedAtUtc.Offset != TimeSpan.Zero
                || frame.FrameWidth < 1
                || frame.FrameHeight < 1
                || !double.IsFinite(frame.InferenceMilliseconds)
                || frame.InferenceMilliseconds < 0
                || string.IsNullOrWhiteSpace(frame.ModelId)
                || string.IsNullOrWhiteSpace(frame.ModelVersion)
                || frame.Detections is null)
            {
                throw new ArgumentException("Frame metadata is invalid.", nameof(frame));
            }
            if (_lastFrameAt.TryGetValue(cameraId, out var previousFrameAt)
                && frame.CapturedAtUtc < previousFrameAt)
            {
                throw new ArgumentException("Frame timestamps must be monotonic per camera.", nameof(frame));
            }

            foreach (var detection in frame.Detections)
            {
                if (detection is null
                    || detection.Box is null
                    || detection.ClassId is < 0 or > 1
                    || detection.ClassName != SupportedClassNames[detection.ClassId]
                    || !float.IsFinite(detection.Confidence)
                    || detection.Confidence is < 0 or > 1
                    || !float.IsFinite(detection.Box.X)
                    || !float.IsFinite(detection.Box.Y)
                    || !float.IsFinite(detection.Box.Width)
                    || !float.IsFinite(detection.Box.Height)
                    || detection.Box.Width <= 0
                    || detection.Box.Height <= 0
                    || !float.IsFinite(detection.Box.Right)
                    || !float.IsFinite(detection.Box.Bottom)
                    || detection.Box.X < 0
                    || detection.Box.Y < 0
                    || detection.Box.Right > frame.FrameWidth
                    || detection.Box.Bottom > frame.FrameHeight)
                {
                    throw new ArgumentException("Frame contains an invalid detection.", nameof(frame));
                }
            }

            if (!_tracks.TryGetValue(cameraId, out var tracks))
            {
                tracks = [];
                _tracks.Add(cameraId, tracks);
            }

            var matchedTracks = new HashSet<Guid>();
            var confirmed = new List<DetectionPrediction>();
            foreach (var detection in frame.Detections
                         .Where(item => item.Confidence >= _options.MinimumConfidence)
                         .OrderByDescending(item => item.Confidence))
            {
                var match = tracks
                    .Where(track =>
                        track.ClassId == detection.ClassId
                        && !matchedTracks.Contains(track.Id)
                        && frame.CapturedAtUtc >= track.LastSeen
                        && frame.CapturedAtUtc - track.LastSeen <= _options.ConfirmationWindow)
                    .Select(track => (Track: track, IoU: IoU(track.Box, detection.Box)))
                    .Where(candidate => candidate.IoU >= _options.MinimumMatchIoU)
                    .OrderByDescending(candidate => candidate.IoU)
                    .FirstOrDefault();

                Track track;
                if (match.Track is null)
                {
                    track = new Track(detection.ClassId, detection.Box, frame.CapturedAtUtc);
                    tracks.Add(track);
                }
                else
                {
                    track = match.Track;
                    track.ConsecutiveDetections = track.Misses == 0
                        ? track.ConsecutiveDetections + 1
                        : 1;
                    track.Box = detection.Box;
                    track.LastSeen = frame.CapturedAtUtc;
                    track.Misses = 0;
                }
                matchedTracks.Add(track.Id);

                var alertKey = (cameraId, detection.ClassId);
                var cooldownElapsed = !_lastAlertAt.TryGetValue(alertKey, out var lastAlertAt)
                    || frame.CapturedAtUtc - lastAlertAt >= _options.Cooldown;
                if (track.ConsecutiveDetections >= _options.MinimumConsecutiveDetections
                    && !track.AlertActive
                    && cooldownElapsed)
                {
                    track.AlertActive = true;
                    _lastAlertAt[alertKey] = frame.CapturedAtUtc;
                    confirmed.Add(detection);
                }
            }

            foreach (var track in tracks.Where(item => !matchedTracks.Contains(item.Id)))
            {
                track.Misses = Math.Min(track.Misses + 1, _options.ClearAfterMisses);
                if (track.Misses >= _options.ClearAfterMisses)
                {
                    track.AlertActive = false;
                    track.ConsecutiveDetections = 0;
                }
            }

            var retention = _options.ConfirmationWindow
                + _options.Cooldown
                + TimeSpan.FromMinutes(1);
            tracks.RemoveAll(track => frame.CapturedAtUtc - track.LastSeen > retention);
            _lastFrameAt[cameraId] = frame.CapturedAtUtc;
            if (confirmed.Count == 0)
            {
                return [];
            }
            return
            [
                new DetectionAlertV1(
                    1,
                    Guid.NewGuid(),
                    "object_detection.confirmed",
                    cameraId,
                    frame.CapturedAtUtc,
                    frame.ModelId,
                    frame.ModelVersion,
                    _options.RuleId,
                    confirmed)
            ];
        }
    }

    private static float IoU(BoundingBox left, BoundingBox right)
    {
        var width = Math.Max(0, Math.Min(left.Right, right.Right) - Math.Max(left.X, right.X));
        var height = Math.Max(0, Math.Min(left.Bottom, right.Bottom) - Math.Max(left.Y, right.Y));
        var intersection = width * height;
        var union = left.Width * left.Height + right.Width * right.Height - intersection;
        return union > 0 ? intersection / union : 0;
    }
}
