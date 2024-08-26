using Microsoft.ML;
using Microsoft.ML.Data;

var _assetsPath = Path.Combine(Environment.CurrentDirectory, "assets");
var _imagesFolder = Path.Combine(_assetsPath, "images");
var _trainTagsTsv = Path.Combine(_imagesFolder, "tags.tsv");
var _testTagsTsv = Path.Combine(_imagesFolder, "test-tags.tsv");
var _predictSingleImage = Path.Combine(_imagesFolder, "toaster3.jpg");
var _inceptionTensorFlowModel = Path.Combine(_assetsPath, "inception", "tensorflow_inception_graph.pb");


/// <summary>
/// The MLContext instance for performing ML operations.
/// </summary>
var mlContext = new MLContext();

var model = GenerateModel(mlContext);

ClassifySingleImage(mlContext, model);


/// <summary>
/// Generates the ML model for image classification.
/// </summary>
/// <param name="mlContext">The MLContext instance.</param>
/// <returns>The trained ML model.</returns>
ITransformer GenerateModel(MLContext mlContext)
{
    // Define the ML pipeline
    IEstimator<ITransformer> pipeline = mlContext.Transforms
        .LoadImages("input", _imagesFolder, nameof(ImageData.ImagePath))
        // The image transforms transform the images into the model's expected format.
        .Append(mlContext.Transforms.ResizeImages("input", InceptionSettings.ImageWidth, InceptionSettings.ImageHeight,
            "input"))
        .Append(mlContext.Transforms.ExtractPixels("input", interleavePixelColors: InceptionSettings.ChannelsLast,
            offsetImage: InceptionSettings.Mean))
        .Append(mlContext.Model.LoadTensorFlowModel(_inceptionTensorFlowModel)
            .ScoreTensorFlowModel(new[] { "softmax2_pre_activation" }, new[] { "input" }, true))
        .Append(mlContext.Transforms.Conversion.MapValueToKey("LabelKey", "Label"))
        .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy("LabelKey", "softmax2_pre_activation"))
        .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabelValue", "PredictedLabel"))
        .AppendCacheCheckpoint(mlContext);

    // Load the training data
    var trainingData = mlContext.Data.LoadFromTextFile<ImageData>(_trainTagsTsv, hasHeader: false);

    Console.WriteLine("=============== Training classification model ===============");

    // Train the model
    var model = pipeline.Fit(trainingData);

    // Load the test data and make predictions
    var testData = mlContext.Data.LoadFromTextFile<ImageData>(_testTagsTsv, hasHeader: false);
    var predictions = model.Transform(testData);

    // Create an IEnumerable for the predictions for displaying results
    var imagePredictionData = mlContext.Data.CreateEnumerable<ImagePrediction>(predictions, true);
    DisplayResults(imagePredictionData);

    Console.WriteLine("=============== Classification metrics ===============");

    // Evaluate the model
    var metrics =
        mlContext.MulticlassClassification.Evaluate(predictions,
            "LabelKey",
            predictedLabelColumnName: "PredictedLabel");

    Console.WriteLine($"LogLoss is: {metrics.LogLoss}");
    Console.WriteLine($"PerClassLogLoss is: {string.Join(" , ", metrics.PerClassLogLoss.Select(c => c.ToString()))}");

    return model;
}

/// <summary>
/// Classifies a single image using the trained ML model.
/// </summary>
/// <param name="mlContext">The MLContext instance.</param>
/// <param name="model">The trained ML model.</param>
void ClassifySingleImage(MLContext mlContext, ITransformer model)
{
    // Create an instance of ImageData for the image to be classified
    var imageData = new ImageData
    {
        ImagePath = _predictSingleImage
    };

    // Make prediction function (input = ImageData, output = ImagePrediction)
    var predictor = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
    var prediction = predictor.Predict(imageData);

    Console.WriteLine("=============== Making single image classification ===============");
    Console.WriteLine(
        $"Image: {Path.GetFileName(imageData.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score?.Max()} ");
}

/// <summary>
/// Displays the results of image predictions.
/// </summary>
/// <param name="imagePredictionData">The collection of image predictions.</param>
void DisplayResults(IEnumerable<ImagePrediction> imagePredictionData)
{
    foreach (var prediction in imagePredictionData)
        Console.WriteLine(
            $"Image: {Path.GetFileName(prediction.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score?.Max()} ");
}

internal struct InceptionSettings
{
    public const int ImageHeight = 224;
    public const int ImageWidth = 224;
    public const float Mean = 117;
    public const float Scale = 1;
    public const bool ChannelsLast = true;
}

/// <summary>
///     Represents the data for an image.
/// </summary>
public class ImageData
{
    [LoadColumn(0)] public string? ImagePath;

    [LoadColumn(1)] public string? Label;
}

/// <summary>
///     Represents the prediction result for an image.
/// </summary>
public class ImagePrediction : ImageData
{
    public string? PredictedLabelValue;
    public float[]? Score;
}