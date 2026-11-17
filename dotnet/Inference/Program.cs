using System.Security.Cryptography;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

if (args.Length != 3)
{
    Console.Error.WriteLine("Usage: Inference <model.onnx> <manifest.json> <image>");
    return 2;
}

var modelPath = Path.GetFullPath(args[0]);
var manifestPath = Path.GetFullPath(args[1]);
var imagePath = Path.GetFullPath(args[2]);
using var manifestDocument = JsonDocument.Parse(File.ReadAllText(manifestPath));
var root = manifestDocument.RootElement;
if (root.GetProperty("schema_version").GetInt32() != 1
    || root.GetProperty("architecture").GetString() != "resnet18")
{
    throw new InvalidDataException("Unsupported model manifest schema or architecture.");
}
var modelFile = root.GetProperty("model_file").GetString()
    ?? throw new InvalidDataException("Manifest model_file is missing.");
if (Path.GetFileName(modelFile) != modelFile || Path.GetFileName(modelPath) != modelFile)
{
    throw new InvalidDataException("Model filename does not match the manifest.");
}

var expectedHash = root.GetProperty("model_sha256").GetString()
    ?? throw new InvalidDataException("Manifest model_sha256 is missing.");
using (var modelStream = File.OpenRead(modelPath))
{
    var actualHash = Convert.ToHexString(SHA256.HashData(modelStream)).ToLowerInvariant();
    if (!CryptographicOperations.FixedTimeEquals(
        Convert.FromHexString(actualHash),
        Convert.FromHexString(expectedHash)))
    {
        throw new InvalidDataException("ONNX model hash does not match the manifest.");
    }
}

var classNames = root.GetProperty("class_names")
    .EnumerateArray()
    .Select(item => item.GetString() ?? throw new InvalidDataException("Invalid class name."))
    .ToArray();
if (classNames.Length != 3 || classNames.Distinct().Count() != classNames.Length)
{
    throw new InvalidDataException("Manifest must contain three unique class names.");
}
var shape = root.GetProperty("input").GetProperty("shape");
if (shape[0].GetString() != "batch"
    || shape[1].GetInt32() != 3
    || root.GetProperty("output").GetProperty("shape")[0].GetString() != "batch"
    || root.GetProperty("output").GetProperty("shape")[1].GetInt32() != classNames.Length)
{
    throw new InvalidDataException("Model input/output shapes do not match the manifest.");
}
var height = shape[2].GetInt32();
var width = shape[3].GetInt32();
var preprocessing = root.GetProperty("preprocessing");
var mean = new[] { 0.485f, 0.456f, 0.406f };
var standardDeviation = new[] { 0.229f, 0.224f, 0.225f };
if (preprocessing.GetProperty("color_space").GetString() != "RGB"
    || preprocessing.GetProperty("layout").GetString() != "NCHW"
    || preprocessing.GetProperty("resize_mode").GetString() != "fit"
    || preprocessing.GetProperty("resample").GetString() != "bilinear"
    || preprocessing.GetProperty("normalization").GetString() != "imagenet"
    || preprocessing.GetProperty("width").GetInt32() != width
    || preprocessing.GetProperty("height").GetInt32() != height
    || !preprocessing.GetProperty("mean").EnumerateArray()
        .Select(item => item.GetSingle()).SequenceEqual(mean)
    || !preprocessing.GetProperty("std").EnumerateArray()
        .Select(item => item.GetSingle()).SequenceEqual(standardDeviation))
{
    throw new InvalidDataException("Unsupported preprocessing contract.");
}
var inputName = root.GetProperty("input").GetProperty("name").GetString()
    ?? throw new InvalidDataException("Manifest input name is missing.");
var outputName = root.GetProperty("output").GetProperty("name").GetString()
    ?? throw new InvalidDataException("Manifest output name is missing.");
var input = PreprocessImage(imagePath, width, height);
using var session = new InferenceSession(modelPath);
var tensor = new DenseTensor<float>(input, new[] { 1, 3, height, width });
using var results = session.Run(new[]
{
    NamedOnnxValue.CreateFromTensor(inputName, tensor)
});
var logits = results.Single(result => result.Name == outputName).AsEnumerable<float>().ToArray();
if (logits.Length != classNames.Length)
{
    throw new InvalidDataException("Model output length does not match class_names.");
}

var maximum = logits.Max();
var exponentials = logits.Select(value => Math.Exp(value - maximum)).ToArray();
var denominator = exponentials.Sum();
var probabilities = exponentials.Select(value => value / denominator).ToArray();
var bestIndex = Array.IndexOf(probabilities, probabilities.Max());
Console.WriteLine(JsonSerializer.Serialize(new
{
    class_name = classNames[bestIndex],
    confidence = probabilities[bestIndex],
    probabilities = classNames
        .Select((name, index) => new KeyValuePair<string, double>(name, probabilities[index]))
        .ToDictionary(item => item.Key, item => item.Value)
}));
return 0;

float[] PreprocessImage(string path, int width, int height)
{
    using var source = SKBitmap.Decode(path)
        ?? throw new InvalidDataException($"Unable to decode image: {path}");
    var targetAspect = (double)width / height;
    var sourceAspect = (double)source.Width / source.Height;
    SKRect crop;
    if (sourceAspect > targetAspect)
    {
        var cropWidth = source.Height * targetAspect;
        crop = new SKRect(
            (float)((source.Width - cropWidth) / 2),
            0,
            (float)((source.Width + cropWidth) / 2),
            source.Height);
    }
    else
    {
        var cropHeight = source.Width / targetAspect;
        crop = new SKRect(
            0,
            (float)((source.Height - cropHeight) / 2),
            source.Width,
            (float)((source.Height + cropHeight) / 2));
    }

    using var resized = new SKBitmap(new SKImageInfo(width, height, SKColorType.Rgba8888));
    using (var canvas = new SKCanvas(resized))
    {
        using var sourceImage = SKImage.FromBitmap(source);
        canvas.DrawImage(sourceImage, crop, new SKRect(0, 0, width, height),
            new SKSamplingOptions(SKFilterMode.Linear, SKMipmapMode.None));
    }

    var tensor = new float[3 * height * width];
    for (var y = 0; y < height; y++)
    {
        for (var x = 0; x < width; x++)
        {
            var pixel = resized.GetPixel(x, y);
            var pixelIndex = y * width + x;
            tensor[pixelIndex] = (pixel.Red / 255f - mean[0]) / standardDeviation[0];
            tensor[height * width + pixelIndex] =
                (pixel.Green / 255f - mean[1]) / standardDeviation[1];
            tensor[2 * height * width + pixelIndex] =
                (pixel.Blue / 255f - mean[2]) / standardDeviation[2];
        }
    }
    return tensor;
}
