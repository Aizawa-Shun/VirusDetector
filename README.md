# Virus Image Classification System

![Virus Detection Heatmap](./image/virus_detection_heatmap.png)

This repository contains a deep learning-based system for virus image classification with uncertainty analysis capabilities.

## Overview

The system uses a ResNet-50 based neural network to classify virus images into three categories:

- Positive
- Negative
- Uncertain

It offers both training and prediction functionalities, along with comprehensive uncertainty analysis and visualization tools.

## Project Structure

```
virus-classification/
├── data/               # Training and validation data
│   ├── train/          # Training images divided by class
│   └── val/            # Validation images divided by class
├── image/              # Images directory
├── input/              # Input images for prediction
├── model/              # Trained model files
├── output/             # Output results
│   ├── graph/          # Visualizations and plots
│   ├── image/          # Image outputs
│   ├── sheet/          # CSV results
│   └── uncertainty/    # Uncertainty analysis results
├── analyzer.py         # Model analysis and evaluation
├── classifier_evaluator.py # Additional evaluation functions
├── config.yaml         # Configuration file
├── data_utils.py       # Data loading and preprocessing
├── losses.py           # Loss functions
├── main.py             # Main execution script
├── model_utils.py      # Model-related functions
├── requirements.txt    # Required packages
├── trainer.py          # Model training
└── visualization.py    # Visualization utilities
```

## Installation

1. Clone the repository:

```
git clone https://github.com/Aizawa-Shun/VirusDetector.git
cd VirusDetector
```

2. Install the required packages:

```
pip install -r requirements.txt
```

## Configuration

The system is configured using the `config.yaml` file:

```yaml
# Main mode setting: train or predict
mode: "predict"

# Training settings
training:
  # Data directory (containing train/val subdirectories)
  data_dir: "./data"
  # Number of training epochs
  num_epochs: 10
  # Batch size
  batch_size: 16
  # Learning rate
  learning_rate: 0.001
  # Loss function: "cross_entropy" or "focal"
  loss_function: "focal"
  # Focal Loss parameters
  focal_alpha: 0.25
  focal_gamma: 2.0
  # Path to save trained model
  model_save_path: "./model/virus_model.pth"
  # Class names for Positive/Negative identification
  classes: ["Positive", "Negative", "Uncertain"]

# Prediction and analysis settings
prediction:
  # Path to trained model
  model_path: "./model/virus_model.pth"
  # Input images directory
  input_folder: "./input"
  # Output directory
  output_folder: "./output"
  # Threshold for determining "high" uncertainty
  uncertainty_threshold: 0.25
  # Whether to include uncertainty analysis
  include_uncertainty_analysis: true
  # Grad-CAM integration method ("average" or "max")
  integration_method: "average"
  # Class names for Positive/Negative identification
  classes: ["Positive", "Negative", "Uncertain"]

# Additional options
options:
  # Whether to run evaluation after training
  evaluate_after_training: true
  # Whether to output detailed logs
  verbose_logging: true
```

## Usage

### Training Mode

To train a new model:

```
python main.py --mode train
```

Training will:

1. Load images from the `data_dir` specified in the config
2. Train a ResNet-50 based model for the specified number of epochs
3. Save the trained model to the specified path
4. Generate training metrics plots in the output directory

### Prediction Mode

To run predictions and analysis on new images:

```
python main.py --mode predict
```

Prediction will:

1. Load images from the `input_folder` specified in the config
2. Run the model to predict each image's class
3. Generate comprehensive analysis results
4. Save all outputs to the specified output directory

## Model Architecture

The system uses a ResNet-50 architecture with a modified final layer for 3-class classification. The model is implemented to output softmax probabilities for each class.

## Virus Detection Visualization

The system's primary visualization is a heatmap overlay that highlights regions of interest in virus images:

![Virus Detection Heatmap](./image/virus_detection_heatmap.png)

The heatmap uses Grad-CAM (Gradient-weighted Class Activation Mapping) technology to visualize which regions of the image are most influential in the model's classification decision. Warmer colors (red) indicate areas that strongly contribute to the classification.

## Uncertainty Analysis

The system includes advanced uncertainty analysis capabilities:

- Uncertainty scoring based on model confidence
- Visualization of uncertain regions using Grad-CAM
- Clustering of uncertain samples to identify patterns
- Integrated heatmaps to analyze common patterns in uncertainty

## Adding New Images

To classify new images:

1. Place your images in the `input` directory (not in the `image` directory which is reserved for documentation images)
2. Set `mode: "predict"` in the config file
3. Run `python main.py`
4. Results will be available in the `output` directory

## Evaluation Metrics

The system produces several evaluation metrics:

- Classification accuracy
- Uncertainty distribution
- Feature clustering
- Confidence histograms

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- ResNet architecture implementation based on torchvision
- Focal Loss implementation based on the original paper: [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
