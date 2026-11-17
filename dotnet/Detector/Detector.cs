using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

[assembly: InternalsVisibleTo("Detector.Tests")]

namespace SurveillanceDetection;

public sealed class Detector : IDisposable
{
    private static readonly string[] SupportedClasses = ["firearm", "knife"];
    private readonly InferenceSession _session;
    private readonly int _height;
    private readonly int _width;
    private readonly int _candidateCount;
    private readonly string _inputName;
    private readonly string _outputName;
    private readonly float _confidenceThreshold;
    private readonly float _iouThreshold;
    private readonly int _maxDetections;
    private readonly string[] _classNames;

    public string ModelId { get; }
    public string ModelVersion { get; }

    public Detector(string modelPath, string manifestPath, int intraOpThreads = 0)
    {
        if (intraOpThreads < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(intraOpThreads));
        }

        modelPath = Path.GetFullPath(modelPath);
        manifestPath = Path.GetFullPath(manifestPath);
        using var document = JsonDocument.Parse(File.ReadAllText(manifestPath));
        var root = document.RootElement;
        if (root.GetProperty("schema_version").GetInt32() != 1
            || root.GetProperty("architecture").GetString() is not ("yolo11n" or "yolo11s")
            || root.GetProperty("source_license").GetString() != "AGPL-3.0")
        {
            throw new InvalidDataException("Unsupported detector manifest schema or architecture.");
        }

        var manifestModelFile = root.GetProperty("model_file").GetString()
            ?? throw new InvalidDataException("Manifest model_file is missing.");
        if (Path.GetFileName(manifestModelFile) != manifestModelFile
            || Path.GetFileName(modelPath) != manifestModelFile)
        {
            throw new InvalidDataException("Model filename does not match the manifest.");
        }

        var expectedHash = root.GetProperty("model_sha256").GetString()
            ?? throw new InvalidDataException("Manifest model_sha256 is missing.");
        var checkpointHash = root.GetProperty("checkpoint_sha256").GetString()
            ?? throw new InvalidDataException("Manifest checkpoint_sha256 is missing.");
        try
        {
            if (Convert.FromHexString(checkpointHash).Length != 32)
            {
                throw new InvalidDataException("Manifest checkpoint_sha256 is invalid.");
            }
        }
        catch (FormatException error)
        {
            throw new InvalidDataException("Manifest checkpoint_sha256 is invalid.", error);
        }
        using (var stream = File.OpenRead(modelPath))
        {
            var actualHash = SHA256.HashData(stream);
            byte[] expectedBytes;
            try
            {
                expectedBytes = Convert.FromHexString(expectedHash);
            }
            catch (FormatException error)
            {
                throw new InvalidDataException("Manifest model_sha256 is invalid.", error);
            }
            if (expectedBytes.Length != actualHash.Length
                || !CryptographicOperations.FixedTimeEquals(actualHash, expectedBytes))
            {
                throw new InvalidDataException("ONNX model hash does not match the manifest.");
            }
        }

        var names = root.GetProperty("class_names").EnumerateArray()
            .Select(item => item.GetString() ?? throw new InvalidDataException("Invalid class name."))
            .ToArray();
        if (!names.SequenceEqual(SupportedClasses))
        {
            throw new InvalidDataException("Manifest class order must be firearm, knife.");
        }
        _classNames = names;
        ModelId = root.GetProperty("model_id").GetString()
            ?? throw new InvalidDataException("Manifest model_id is missing.");
        ModelVersion = root.GetProperty("model_version").GetString()
            ?? throw new InvalidDataException("Manifest model_version is missing.");
        if (string.IsNullOrWhiteSpace(ModelId) || string.IsNullOrWhiteSpace(ModelVersion))
        {
            throw new InvalidDataException("Manifest model identifiers must not be empty.");
        }

        var input = root.GetProperty("input");
        var output = root.GetProperty("output");
        var inputShape = input.GetProperty("shape");
        var outputShape = output.GetProperty("shape");
        if (input.GetProperty("dtype").GetString() != "float32"
            || output.GetProperty("dtype").GetString() != "float32"
            || inputShape.GetArrayLength() != 4
            || outputShape.GetArrayLength() != 3
            || inputShape[0].GetInt32() != 1
            || inputShape[1].GetInt32() != 3
            || outputShape[0].GetInt32() != 1
            || outputShape[1].GetInt32() != 4 + names.Length)
        {
            throw new InvalidDataException("Model input/output shapes do not match the manifest.");
        }
        _height = inputShape[2].GetInt32();
        _width = inputShape[3].GetInt32();
        _candidateCount = outputShape[2].GetInt32();
        if (_height < 64 || _width < 64 || _height % 32 != 0 || _width % 32 != 0)
        {
            throw new InvalidDataException("Input dimensions must be multiples of 32 and at least 64.");
        }

        var preprocessing = root.GetProperty("preprocessing");
        if (preprocessing.GetProperty("color_space").GetString() != "RGB"
            || preprocessing.GetProperty("layout").GetString() != "NCHW"
            || preprocessing.GetProperty("resize_mode").GetString() != "letterbox"
            || preprocessing.GetProperty("resample").GetString() != "opencv_inter_linear_exact"
            || preprocessing.GetProperty("normalization").GetString() != "divide_by_255"
            || preprocessing.GetProperty("padding_value").GetInt32() != 114
            || preprocessing.GetProperty("width").GetInt32() != _width
            || preprocessing.GetProperty("height").GetInt32() != _height)
        {
            throw new InvalidDataException("Unsupported detection preprocessing contract.");
        }

        var outputContract = root.GetProperty("output_contract");
        if (outputContract.GetProperty("layout").GetString() != "batch_channels_candidates"
            || outputContract.GetProperty("box_format").GetString()
                != "cxcywh_pixels_in_letterboxed_input"
            || outputContract.GetProperty("class_scores").GetString()
                != "sigmoid_probabilities_without_objectness")
        {
            throw new InvalidDataException("Unsupported detector output contract.");
        }

        var postprocessing = root.GetProperty("postprocessing");
        if (postprocessing.GetProperty("nms").GetString() != "class_aware_greedy")
        {
            throw new InvalidDataException("Unsupported detector NMS contract.");
        }
        _confidenceThreshold = postprocessing.GetProperty("confidence_threshold").GetSingle();
        _iouThreshold = postprocessing.GetProperty("iou_threshold").GetSingle();
        _maxDetections = postprocessing.GetProperty("max_detections").GetInt32();
        if (_confidenceThreshold is < 0 or > 1
            || _iouThreshold is < 0 or > 1
            || _maxDetections < 1)
        {
            throw new InvalidDataException("Invalid detector postprocessing thresholds.");
        }

        using var options = new SessionOptions();
        if (intraOpThreads > 0)
        {
            options.IntraOpNumThreads = intraOpThreads;
        }
        _session = new InferenceSession(modelPath, options);
        var inputName = input.GetProperty("name").GetString()
            ?? throw new InvalidDataException("Manifest input name is missing.");
        var outputName = output.GetProperty("name").GetString()
            ?? throw new InvalidDataException("Manifest output name is missing.");
        if (!_session.InputMetadata.ContainsKey(inputName)
            || !_session.OutputMetadata.ContainsKey(outputName)
            || _session.InputMetadata.Count != 1
            || _session.OutputMetadata.Count != 1)
        {
            _session.Dispose();
            throw new InvalidDataException("ONNX graph does not match the manifest node names.");
        }
        if (!_session.InputMetadata[inputName].Dimensions.SequenceEqual([1, 3, _height, _width])
            || !_session.OutputMetadata[outputName].Dimensions.SequenceEqual(
                [1, 4 + _classNames.Length, _candidateCount]))
        {
            _session.Dispose();
            throw new InvalidDataException("ONNX graph shapes do not match the manifest.");
        }
        _inputName = inputName;
        _outputName = outputName;
    }

    public DetectionFrame Detect(
        ReadOnlyMemory<byte> encodedImage,
        DateTimeOffset? capturedAtUtc = null)
    {
        using var stream = new SKMemoryStream(encodedImage.ToArray());
        using var source = SKBitmap.Decode(stream)
            ?? throw new InvalidDataException("Unable to decode the input image.");
        return Detect(source, capturedAtUtc);
    }

    public DetectionFrame Detect(SKBitmap source, DateTimeOffset? capturedAtUtc = null)
    {
        if (source.Width < 1 || source.Height < 1)
        {
            throw new ArgumentException("Image dimensions must be positive.", nameof(source));
        }

        var timer = Stopwatch.StartNew();
        var tensorData = Preprocess(source, out var scale, out var padLeft, out var padTop);
        var inputTensor = new DenseTensor<float>(tensorData, [1, 3, _height, _width]);
        using var outputs = _session.Run(
        [
            NamedOnnxValue.CreateFromTensor(_inputName, inputTensor)
        ]);
        var values = outputs.Single(value => value.Name == _outputName)
            .AsEnumerable<float>()
            .ToArray();
        var candidates = Decode(
            values,
            scale,
            padLeft,
            padTop,
            source.Width,
            source.Height);
        timer.Stop();
        return new DetectionFrame(
            ModelId,
            ModelVersion,
            capturedAtUtc ?? DateTimeOffset.UtcNow,
            source.Width,
            source.Height,
            timer.Elapsed.TotalMilliseconds,
            candidates);
    }

    private float[] Preprocess(
        SKBitmap source,
        out float scale,
        out int padLeft,
        out int padTop)
    {
        scale = Math.Min((float)_height / source.Height, (float)_width / source.Width);
        var resizedWidth = (int)Math.Round(source.Width * scale, MidpointRounding.ToEven);
        var resizedHeight = (int)Math.Round(source.Height * scale, MidpointRounding.ToEven);
        var horizontalPadding = _width - resizedWidth;
        var verticalPadding = _height - resizedHeight;
        padLeft = (int)Math.Round(horizontalPadding / 2f - 0.1f, MidpointRounding.ToEven);
        var padRight = (int)Math.Round(horizontalPadding / 2f + 0.1f, MidpointRounding.ToEven);
        padTop = (int)Math.Round(verticalPadding / 2f - 0.1f, MidpointRounding.ToEven);
        var padBottom = (int)Math.Round(verticalPadding / 2f + 0.1f, MidpointRounding.ToEven);
        if (padLeft + resizedWidth + padRight != _width
            || padTop + resizedHeight + padBottom != _height)
        {
            throw new InvalidDataException("Letterbox dimensions do not match the model input.");
        }

        var resized = ResizeRgbInterLinearExact(source, resizedWidth, resizedHeight);
        var tensor = new float[3 * _height * _width];
        for (var y = 0; y < _height; y++)
        {
            for (var x = 0; x < _width; x++)
            {
                var index = y * _width + x;
                var resizedX = x - padLeft;
                var resizedY = y - padTop;
                if (resizedX < 0 || resizedX >= resizedWidth
                    || resizedY < 0 || resizedY >= resizedHeight)
                {
                    tensor[index] = 114 / 255f;
                    tensor[_height * _width + index] = 114 / 255f;
                    tensor[2 * _height * _width + index] = 114 / 255f;
                    continue;
                }

                var pixelIndex = (resizedY * resizedWidth + resizedX) * 3;
                tensor[index] = resized[pixelIndex] / 255f;
                tensor[_height * _width + index] = resized[pixelIndex + 1] / 255f;
                tensor[2 * _height * _width + index] = resized[pixelIndex + 2] / 255f;
            }
        }
        return tensor;
    }

    internal static byte[] ResizeRgbInterLinearExact(SKBitmap source, int width, int height)
    {
        if (width < 1 || height < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(width), "Resize dimensions must be positive.");
        }

        const int coefficientScale = 1 << 8;
        var scaleX = 1.0 / ((double)width / source.Width);
        var scaleY = 1.0 / ((double)height / source.Height);
        var sourcePixels = new SKColor[source.Width * source.Height];
        for (var y = 0; y < source.Height; y++)
        {
            for (var x = 0; x < source.Width; x++)
            {
                sourcePixels[y * source.Width + x] = source.GetPixel(x, y);
            }
        }

        var resized = new byte[width * height * 3];
        for (var y = 0; y < height; y++)
        {
            var sourceY = scaleY * (y + 0.5) - 0.5;
            var y0 = (int)Math.Floor(sourceY);
            var beta1 = sourceY - y0;
            if (y0 < 0)
            {
                y0 = 0;
                beta1 = 0;
            }
            if (y0 >= source.Height - 1)
            {
                y0 = source.Height - 1;
                beta1 = 0;
            }
            var y1 = Math.Min(y0 + 1, source.Height - 1);
            var beta1Fixed = Math.Clamp(
                (int)Math.Round(beta1 * coefficientScale, MidpointRounding.ToEven),
                0,
                coefficientScale);
            var beta0Fixed = Math.Clamp(
                (int)Math.Round((1 - beta1) * coefficientScale, MidpointRounding.ToEven),
                0,
                coefficientScale);

            for (var x = 0; x < width; x++)
            {
                var sourceX = scaleX * (x + 0.5) - 0.5;
                var x0 = (int)Math.Floor(sourceX);
                var alpha1 = sourceX - x0;
                if (x0 < 0)
                {
                    x0 = 0;
                    alpha1 = 0;
                }
                if (x0 >= source.Width - 1)
                {
                    x0 = source.Width - 1;
                    alpha1 = 0;
                }
                var x1 = Math.Min(x0 + 1, source.Width - 1);
                var alpha1Fixed = Math.Clamp(
                    (int)Math.Round(alpha1 * coefficientScale, MidpointRounding.ToEven),
                    0,
                    coefficientScale);
                var alpha0Fixed = Math.Clamp(
                    (int)Math.Round((1 - alpha1) * coefficientScale, MidpointRounding.ToEven),
                    0,
                    coefficientScale);

                var topLeft = sourcePixels[y0 * source.Width + x0];
                var topRight = sourcePixels[y0 * source.Width + x1];
                var bottomLeft = sourcePixels[y1 * source.Width + x0];
                var bottomRight = sourcePixels[y1 * source.Width + x1];
                var destination = (y * width + x) * 3;
                resized[destination] = Interpolate(
                    topLeft.Red, topRight.Red, bottomLeft.Red, bottomRight.Red,
                    alpha0Fixed, alpha1Fixed, beta0Fixed, beta1Fixed);
                resized[destination + 1] = Interpolate(
                    topLeft.Green, topRight.Green, bottomLeft.Green, bottomRight.Green,
                    alpha0Fixed, alpha1Fixed, beta0Fixed, beta1Fixed);
                resized[destination + 2] = Interpolate(
                    topLeft.Blue, topRight.Blue, bottomLeft.Blue, bottomRight.Blue,
                    alpha0Fixed, alpha1Fixed, beta0Fixed, beta1Fixed);
            }
        }
        return resized;
    }

    private static byte Interpolate(
        byte topLeft,
        byte topRight,
        byte bottomLeft,
        byte bottomRight,
        int alpha0,
        int alpha1,
        int beta0,
        int beta1)
    {
        const long rounding = 1L << 15;
        var value =
            (long)topLeft * alpha0 * beta0
            + (long)topRight * alpha1 * beta0
            + (long)bottomLeft * alpha0 * beta1
            + (long)bottomRight * alpha1 * beta1;
        return (byte)((value + rounding) >> 16);
    }

    private IReadOnlyList<DetectionPrediction> Decode(
        float[] values,
        float scale,
        int padLeft,
        int padTop,
        int originalWidth,
        int originalHeight)
    {
        var channels = 4 + _classNames.Length;
        if (values.Length % channels != 0)
        {
            throw new InvalidDataException("YOLO output length does not match the manifest.");
        }
        var candidateCount = values.Length / channels;
        if (candidateCount != _candidateCount)
        {
            throw new InvalidDataException("YOLO output candidate count does not match the manifest.");
        }
        var candidates = new List<DetectionPrediction>();
        for (var index = 0; index < candidateCount; index++)
        {
            var centerX = values[index];
            var centerY = values[candidateCount + index];
            var boxWidth = values[2 * candidateCount + index];
            var boxHeight = values[3 * candidateCount + index];
            if (!float.IsFinite(centerX)
                || !float.IsFinite(centerY)
                || !float.IsFinite(boxWidth)
                || !float.IsFinite(boxHeight)
                || boxWidth <= 0
                || boxHeight <= 0)
            {
                continue;
            }
            for (var classId = 0; classId < _classNames.Length; classId++)
            {
                var confidence = values[(4 + classId) * candidateCount + index];
                if (!AcceptConfidence(confidence, _confidenceThreshold))
                {
                    continue;
                }
                var left = Math.Clamp((centerX - boxWidth / 2 - padLeft) / scale, 0, originalWidth);
                var top = Math.Clamp((centerY - boxHeight / 2 - padTop) / scale, 0, originalHeight);
                var right = Math.Clamp((centerX + boxWidth / 2 - padLeft) / scale, 0, originalWidth);
                var bottom = Math.Clamp((centerY + boxHeight / 2 - padTop) / scale, 0, originalHeight);
                if (right <= left || bottom <= top)
                {
                    continue;
                }
                candidates.Add(new DetectionPrediction(
                    classId,
                    _classNames[classId],
                    confidence,
                    new BoundingBox(left, top, right - left, bottom - top)));
            }
        }
        return ApplyNms(candidates, _iouThreshold, _maxDetections);
    }

    internal static bool AcceptConfidence(float confidence, float threshold) =>
        float.IsFinite(confidence) && confidence >= threshold && confidence <= 1;

    internal static IReadOnlyList<DetectionPrediction> ApplyNms(
        List<DetectionPrediction> candidates,
        float iouThreshold,
        int maxDetections)
    {
        var remaining = candidates.OrderByDescending(item => item.Confidence).ToList();
        var selected = new List<DetectionPrediction>();
        while (remaining.Count > 0 && selected.Count < maxDetections)
        {
            var best = remaining[0];
            remaining.RemoveAt(0);
            selected.Add(best);
            remaining.RemoveAll(candidate =>
                candidate.ClassId == best.ClassId && IoU(candidate.Box, best.Box) > iouThreshold);
        }
        return selected;
    }

    private static float IoU(BoundingBox left, BoundingBox right)
    {
        var width = Math.Max(0, Math.Min(left.Right, right.Right) - Math.Max(left.X, right.X));
        var height = Math.Max(0, Math.Min(left.Bottom, right.Bottom) - Math.Max(left.Y, right.Y));
        var intersection = width * height;
        var union = left.Width * left.Height + right.Width * right.Height - intersection;
        return union > 0 ? intersection / union : 0;
    }

    public void Dispose() => _session.Dispose();
}
