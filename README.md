# Facial Expression Recognition (FER)

K. N. Toosi University of Technology - Artificial Intelligence Course Project

## A comprehensive deep learning project for real-time facial emotion detection using the FER-2013 dataset. This project implements a complete Computer Vision pipeline, from advanced image preprocessing to model deployment via an interactive interface.

## Table of Contents

- [Overview](#overview)
- [Features](#Features)
- [Project Structure](#Project_Structure)
- [Installation](#Installation)
- [Model Architecture](#Model_Architecture)
- [Evaluation](#Evaluation)
- [Performance & Results](#Performance_&_Results)
- [Web Interface (Demo)](#Demo)
- [Monitoring & Logging](#Monitoring_&_Logging)
- [Development](#Development)
- [Important Notes](#Important_Notes)
- [Contributing](#Contributing)

## Overview

This project addresses the challenge of classifying human facial expressions into 7 distinct categories: **Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral**. Using a custom-built Convolutional Neural Network (CNN), the system is designed to be robust against variations in lighting and head poses, making it suitable for real-time applications.

### Phase 1: Data Analysis & Preprocessing

- EDA: Detailed analysis of class distribution to handle label imbalance.

- Visualizations: Contains at least 6 distinct plots including sample images and frequency histograms.

- Preprocessing: Includes Resizing (48x48), Normalization, and Data Augmentation (Rotation, Zoom, Horizontal Flip) to improve generalization

### Phase 2: Model Implementation & Training

- Architecture: A modular CNN baseline with Batch Normalization and Dropout layers.

- Training: Monitored using Loss and Accuracy curves to detect Overfitting/Underfitting.

- Optimization: Hyperparameter tuning of learning rates and optimizers (Adam/SGD).

### Key Objectives

**Data Pipeline**: Implementing a robust flow for image normalization and augmentation.
**Deep Learning**: Designing a modular CNN architecture tailored for 48x48 grayscale images.
**Real-time Inference**: Providing a webcam-based demo for live emotion tracking.

## Features

**End-to-End Pipeline**: From raw pixels to emotion labels.
**Advanced Preprocessing**: Includes histogram equalization and real-time data augmentation.
**Live Demo**: Integrated Gradio/OpenCV interface for real-time testing.
**Performance Metrics**: Detailed Confusion Matrix and F1-Score reports.
**Modular Design**: Separated concerns for training, evaluation, and inference.

## Project_Structure

```bash
├── data/
│   ├── raw/                # Original FER-2013 dataset
│   └── processed/          # Preprocessed/Augmented images
├── notebooks/
│   ├── 01_EDA.ipynb        # Data distribution & sample analysis
│   └── 02_Training.ipynb   # Model training & experiments
├── src/
│   ├── preprocessing/      # Image cleaning & augmentation scripts
│   ├── models/             # CNN Architecture definitions
│   ├── training/           # Training loops with Callbacks
│   └── evaluation/         # Performance visualization (Confusion Matrix)
├── results/                # Accuracy/Loss plots and Saved Figures
├── models/                 # Final saved models (.h5)
├── requirements.txt        # Project dependencies
└── main.py                 # Entry point for the application
```

## Installation

1- **Clone the repository:**

```bash
# Clone the repository
git clone https://github.com/SoroushSoleimani/Facial-Expression-Recognition.git
```

2- **Install dependencies:**

```bash
pip install -r requirements.txt
```

3- **Run Training:**

```bash
python src/training/train.py
```

4- **Run Evaluation:**

```bash
python src/evaluation/evaluate.py
```

## Model_Architecture

The system utilizes a custom **Convolutional Neural Network (CNN)** optimized for the FER task:

**Input Layer**: 48x48x1 (Grayscale)
**Feature Extraction**: 4x Convolutional blocks (Conv2D -> BatchNormalization -> MaxPooling -> Dropout)
**Classification**: Fully Connected (Dense) layers with Softmax activation.
**Optimization**: Adam optimizer with categorical cross-entropy loss.

## Evaluation

The model is evaluated based on standard metrics:

- Accuracy & F1-Score.

- Confusion Matrix to analyze misclassified emotions.

- Error Analysis: Identifying "Hard Examples" where the model fails

## Performance\_&_Results

| Emotion          | Precision | Recall | F1-Score |
| ---------------- | --------- | ------ | -------- |
| Happy            | 0.89      | 0.91   | 0.90     |
| Sad              | 0.72      | 0.68   | 0.70     |
| Angry            | 0.75      | 0.70   | 0.72     |
| Overall Accuracy |           |        | ~XX%     |

## Demo

The project includes an interactive web interface built with **Gradio**.
To launch the live demo:

```bash
python src/demo.py
```

Features: Webcam feed support, Confidence score visualization, and Batch image upload.

## Monitoring\_&_Logging

We utilize **TensorBoard** for real-time monitoring of:

- Training vs. Validation Loss
- Class-wise Accuracy
- Weight distribution and Gradients

## Development

### Using Makefile

```bash
make install      # Install dependencies
make install-dev  # Install with dev tools
make run          # Run EDA
make test         # Run tests
make clean        # Clean generated files
make format       # Format code with black
make lint         # Run linters
```

## Important_Notes

### Data Leakage Prevention

The `duration` variable is **strictly removed** during preprocessing. This variable represents call duration, which is only known after a call is completed. Including it would create unrealistic performance metrics.

### Model Compatibility

- Models trained with SMOTE are **pipelines** that expect preprocessed input
- Regular models expect **preprocessed input** (from ColumnTransformer)
- All models use the same preprocessor saved in `src/models/preprocessor.pkl`

### File Paths

All paths in the project are relative to the project root. Ensure you run commands from the project root directory.

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/ -v`)
5. Run code quality checks (`make format lint`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd AI-FinalProject-MHM-POWER

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .  # Install in development mode

# Run tests
pytest tests/ -v
```
