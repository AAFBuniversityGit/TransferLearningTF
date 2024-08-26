

# TransferLearningTF

This repository demonstrates how to use Transfer Learning techniques with TensorFlow in ML.NET to perform image classification tasks. Transfer learning is a machine learning technique where a pre-trained model is used as the starting point to solve a new problem. This approach is especially useful when you have limited data for the new task.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Code Overview](#code-overview)
  - [1. Data Loading and Preprocessing](#1-data-loading-and-preprocessing)
  - [2. Model Setup and Transfer Learning](#2-model-setup-and-transfer-learning)
  - [3. Model Training](#3-model-training)
  - [4. Model Evaluation](#4-model-evaluation)
- [Running the Project](#running-the-project)
- [Results](#results)
- [Conclusion](#conclusion)
- [References](#references)

## Introduction

This project showcases how to leverage transfer learning in TensorFlow using ML.NET to classify images into different categories. Transfer learning allows you to use a model pre-trained on a large dataset (like ImageNet) and fine-tune it to your specific dataset, saving both time and computational resources.

## Requirements

- .NET SDK 6.0 or later
- ML.NET 2.0.0 or later
- TensorFlow 2.x
- Visual Studio 2022 or later
- A dataset of images organized by category

## Dataset

The dataset used for this project should be organized in a directory where each subdirectory contains images of a specific category. The directory structure should look something like this:

```
/dataset
    /category1
        image1.jpg
        image2.jpg
        ...
    /category2
        image1.jpg
        image2.jpg
        ...
    ...
```

## Model Architecture

In this project, we use a pre-trained TensorFlow model (e.g., MobileNetV2) as the base model. We then add a few layers on top of this model and fine-tune the entire architecture on our custom dataset.

## Code Overview

### 1. Data Loading and Preprocessing

The first step in the code involves loading the dataset and preprocessing it for training. The images are resized to a uniform size, normalized, and then split into training and validation sets.

```csharp
// Example: Data loading and preprocessing
var imageData = mlContext.Data.LoadFromEnumerable<ImageData>(imageDataList);
var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
    .Append(mlContext.Transforms.LoadImages("ImagePath", "ImagePath"))
    .Append(mlContext.Transforms.ResizeImages("ImagePath", inputColumnName: "ImagePath", imageWidth: 224, imageHeight: 224))
    .Append(mlContext.Transforms.ExtractPixels("ImagePixels"))
    .Append(mlContext.Model.LoadTensorFlowModel(modelLocation)
        .ScoreTensorName("dense_2/Sigmoid")
        .AddInput("input_1", imageHeight: 224, imageWidth: 224, 3)
        .AddOutput("output_1"))
    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
```

### 2. Model Setup and Transfer Learning

We set up the pre-trained model and add new layers on top to adapt it to our specific classification task.

```csharp
// Example: Transfer learning setup
var options = new ImageClassificationTrainer.Options()
{
    FeatureColumnName = "ImagePixels",
    LabelColumnName = "Label",
    Arch = ImageClassificationTrainer.Architecture.ResNetV2101,
    Epoch = 100,
    BatchSize = 10,
    LearningRate = 0.01f,
    ValidationSet = validationDataView
};

var trainingPipeline = mlContext.MulticlassClassification.Trainers.ImageClassification(options)
    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
```

### 3. Model Training

The model is trained on the dataset using the pipeline defined above. This includes fitting the model to the training data and validating it on the validation set.

```csharp
// Example: Model training
var trainedModel = trainingPipeline.Fit(trainDataView);
```

### 4. Model Evaluation

After training, the model's performance is evaluated using metrics such as accuracy, precision, and recall.

```
// Example: Model evaluation
var predictions = trainedModel.Transform(testDataView);
var metrics = mlContext.MulticlassClassification.Evaluate(predictions, "Label");
```

## Running the Project

To run the project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/TransferLearningTF.git
    ```
2. Open the solution file in Visual Studio.
3. Restore the NuGet packages:
    ```bash
    dotnet restore
    ```
4. Run the project:
    ```bash
    dotnet run
    ```

## Results

After running the project, you will see the model's performance metrics printed in the console. You can also visualize the training process, accuracy, and loss over epochs.

## Conclusion

This project demonstrates how to effectively use transfer learning with TensorFlow in ML.NET for image classification. Transfer learning allows you to leverage the power of pre-trained models and apply them to your specific tasks with limited data.

## References

- [ML.NET Documentation](https://docs.microsoft.com/en-us/dotnet/machine-learning/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
