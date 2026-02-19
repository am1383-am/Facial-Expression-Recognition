# Facial Expression Recognition (FER)

K. N. Toosi University of Technology - Artificial Intelligence Course Project

### A comprehensive deep learning project for real-time facial emotion detection using the **FERPlus** dataset. This project implements a complete Computer Vision pipeline, from advanced image preprocessing to model deployment via an interactive interface.

## Table of Contents

- [ Overview](#-overview)
- [ Features](#-features)
- [ Project Structure](#-project-structure)
- [ Installation & Usage](#️-installation--usage)
- [ Data Acquisition & Preparation](#-data-acquisition--preparation)
- [ Model Architecture & Training Strategy](#-model-architecture--training-strategy)
- [ Evaluation & Results](#-evaluation--results)
- [ Interactive Web Interface (Demo)](#-interactive-web-interface-demo)
- [ Important Notes](#️-important-notes)
- [ Authors & Contact](#-authors--contact)

##  Overview

This project addresses the challenge of classifying human facial expressions into **8 distinct categories**: **Angry, Contempt, Disgust, Fear, Happy, Neutral, Sad, and Surprise**. Using a custom-built Convolutional Neural Network (CNN) trained on the **FERPlus** dataset, the system is designed to be robust against variations in lighting and head poses, making it highly suitable for real-time applications.

The project is structured into two main phases, simulating a real-world AI development lifecycle:

### Phase 1: Data Analysis & Preprocessing

- EDA: Detailed analysis of class distribution to handle label imbalance.

- Visualizations: Contains at least 6 distinct plots including sample images and frequency histograms.

- Preprocessing: Includes Resizing (48x48), Normalization, and Data Augmentation (Rotation, Zoom, Horizontal Flip) to improve generalization

### Phase 2: Model Implementation & Training

- Architecture: A modular CNN baseline with Batch Normalization and Dropout layers.

- Training: Monitored using Loss and Accuracy curves to detect Overfitting/Underfitting.

- Optimization: Hyperparameter tuning of learning rates and optimizers (Adam/SGD).

### Key Objectives

- **Data Pipeline**: Implementing a robust flow for image normalization and augmentation.

- **Deep Learning**: Designing a modular CNN architecture tailored for 48x48 grayscale images.

- **Real-time Inference**: Providing a webcam-based demo for live emotion tracking.

##  Features

- **End-to-End Pipeline:** A complete workflow converting raw pixel data into one of **8 emotion categories** (Angry, Contempt, Disgust, Fear, Happy, Neutral, Sad, Surprise).
- **Interactive Web Interface:** A user-friendly dashboard built with **Streamlit**, supporting both **Live Webcam** feed and **Image Upload**.
- **Explainable AI (XAI):** Integrated **Grad-CAM (Gradient-weighted Class Activation Mapping)** visualization. This allows users to see a "Heatmap" overlay, revealing exactly which parts of the face the model focuses on to make a decision.
- **Real-Time Analytics:** Live visualization of emotion probabilities using dynamic **Plotly** bar charts and confidence metrics.
- **Robust Preprocessing:** resizing (48x48), grayscale conversion, and normalization.
- **Modular Design:** Structured codebase separating training logic, model architecture, and the frontend interface.

##  Project Structure

```bash
├── data/
│   └── raw/                    # Contains dataset and CSV files (Content ignored by .gitignore)
├── models/                     # Stores trained .keras models and history logs (Content ignored by .gitignore)
├── notebooks/                  # Jupyter notebooks for step-by-step analysis
│   ├── data_preparation.ipynb
│   ├── EDA.ipynb
│   ├── evaluation.ipynb
│   └── Visualizing during training.ipynb
├── results/                    # Generated plots, metrics, training figures, and reports
├── src/                        # Core source code modules
│   ├── data_preparation/       # Scripts for data cleaning and preparation
│   │   └── data_preparation.py
│   ├── evaluation/             # Evaluation classes and metric calculations
│   │   └── evaluator.py
│   ├── models/                 # Deep Learning architecture definitions
│   │   ├── baseline_model.py
│   │   └── final_model.py
│   ├── preprocessing/          # Data loaders and augmentation logic
│   │   └── data_loader.py
│   ├── training/               # Training loop implementation
│   │   └── train.py
│   ├── utils/                  # Helper functions (e.g., plotting results)
│   │   └── plot_results.py
│   └── training_pipeline.py    # Orchestrator script to run the full pipeline
├── app.py                      # Streamlit web application (Demo)
├── main.py                     # Main entry point for console execution
├── requirements.txt            # Project dependencies
└── .gitignore                  # Git configuration
```

##  Installation & Usage

This project utilizes a **centralized CLI (Command Line Interface)** to manage the entire lifecycle of the project. You do not need to run individual scripts manually.

### 1. Setup Environment

First, clone the repository and install the required dependencies:

```bash
# Clone the repository
git clone https://github.com/SoroushSoleimani/Facial-Expression-Recognition.git

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application

Simply execute the `main.py` script to access the interactive project menu:

```bash
python main.py
```

### 3. Operation Modes

Once launched, you will see a menu to select the desired operation:

| Option | Mode            | Description                                                                                              |
| :----- | :-------------- | :------------------------------------------------------------------------------------------------------- |
| **1**  | **[Data Prep]** | Automatically downloads the **FERPlus** dataset and runs the preprocessing pipeline.                     |
| **2**  | **[Training]**  | Executes the full Deep Learning pipeline: trains the model, runs evaluation, and saves the best weights. |
| **3**  | **[Demo]**      | Launches the **Streamlit Web Interface** for real-time webcam testing and visualization.                 |
| **0**  | **Exit**        | Closes the application.                                                                                  |

##  Data Acquisition & Preparation

The project features a fully automated data pipeline managed by `src/data_preparation/data_preparation.py`. Instead of manual downloads, the script executes a robust four-step workflow:

1.  **Auto-Download:** Fetches the [FERPlus](https://www.kaggle.com/datasets/arnabkumarroy02/ferplus?select=train) dataset directly from Kaggle using the `kagglehub` library.
2.  **Organization:** Flattens the complex original directory structure and consolidates all images into a unified `data/raw` folder for easier access.
3.  **Noise Injection:** To simulate real-world "dirty data" challenges, the pipeline deliberately introduces synthetic errors such as label typos, missing values, and invalid IDs.
4.  **Data Cleaning:** Performs rigorous validation to remove corrupted entries and mismatched labels, ultimately generating a reliable `dataset_cleaned.csv` which serves as the ground truth for all subsequent training phases.

##  Model Architecture & Training Strategy

The project pipeline consists of three core stages implemented in the `src/` directory:

### 1. Data Pipeline (`data_loader.py`)

We utilize **ImageDataGenerator** for dynamic data loading and augmentation to ensure robustness.

- **Input:** Resized to **48x48 Grayscale** pixels.
- **Splitting:** Stratified split into **Train (80%)**, **Validation (10%)**, and **Test (10%)** sets.
- **Augmentation:** Applies real-time transformations (Rotation, Zoom, Shifts, Flips) to the training set to prevent overfitting.

### 2. Model Architectures

Two distinct architectures were designed to benchmark performance:

- **Baseline Model (`baseline_model.py`):** A lightweight CNN with 3 standard Convolutional blocks to establish a performance baseline.
- **Final Model (`final_model.py`):** A deeper, **VGG-style architecture** featuring **Double Convolutions**, **Batch Normalization**, **L2 Regularization**, and increasing **Dropout rates** (0.2 → 0.5) for maximum feature extraction and generalization.

### 3. Training Protocol (`train.py`)

The model is optimized using **Adam** with **Categorical Cross-Entropy** loss, utilizing advanced callbacks:

- **ModelCheckpoint:** Saves the best model based on validation accuracy.
- **EarlyStopping:** Halts training if loss stagnates (Patience: 8).
- **ReduceLROnPlateau:** Reduces learning rate by factor of 0.2 when validation improvement slows down.

##  Evaluation & Results

We employ a comprehensive set of metrics including **Accuracy, Weighted F1-Score, and Confusion Matrices** to rigorously assess model performance across all 8 emotion classes.

- **Training Analysis:** Loss and Accuracy curves are plotted to monitor convergence and detect overfitting or underfitting issues.
- **Detailed Reports:** For full visual analysis, including **ROC Curves and Heatmaps**, please refer to the Jupyter Notebooks in the `notebooks/` directory:
  - [ Training Viz](notebooks/Visualizing%20during%20training.ipynb)
  - [ Model Evaluation](notebooks/evaluation.ipynb)
  - [ Data Preparation](notebooks/data_preparation.ipynb)
  - [ EDA Analysis](notebooks/EDA.ipynb)

##  Interactive Web Interface (Demo)

The project features a polished, user-friendly web application built with **Streamlit**, designed to demonstrate the model's capabilities in real-time.

### How to Launch

You can start the demo directly via the command line:

```bash
streamlit run app.py
```

(Or select Option 3 in the `main.py` menu)

### Key Features

- **Dual Input Modes:** Supports both **Real-time Webcam** (with adjustable processing interval) and **Static Image Upload**.
- **Explainable AI (XAI):** Integrated **Grad-CAM Heatmaps** to visualize the model's focus areas (e.g., eyes/mouth) for better interpretability.
- **Dynamic Model Switching:** Instantly swap between different architectures (e.g., `baseline` vs `final`) via the sidebar without restarting.
- **Live Analytics:** Interactive **Plotly** probability charts for all 8 emotions.

![Web Interface Demo](results/demo_interface.png)

##  Important Notes

### 1. Face Detection Dependency

The real-time demo relies on **OpenCV Haar Cascades** for face detection.

- **Lighting:** Ensure the face is well-lit for accurate detection.
- **Orientation:** The current cascade (`haarcascade_frontalface_default.xml`) works best on frontal faces. Profile or tilted faces might not be detected.

### 2. Input Requirements

- **Grayscale Only:** The models are trained specifically on **Grayscale** images. The pipeline automatically converts any RGB input to grayscale before inference.
- **Resolution:** All inputs are resized to **48x48 pixels** internally. Providing high-resolution images is fine, but fine details might be lost during downsampling.

### 3. Execution Directory

All scripts (e.g., `main.py`, `app.py`) are designed to be run from the **project root directory**. Running them from inside subfolders (like `src/`) may cause `FileNotFoundError` for models or data config files.

##  Authors & Contact

This project was developed by a **3-member team** for the **Artificial Intelligence Course (Fall 2024)** at **K. N. Toosi University of Technology**.

| Name                     | Email                        |
| :----------------------- | :--------------------------- |
| **Parham Kootzari**      | pkootzari1383@gmail.com      |
| **AmirHossein Babaee**   | amirhoseinbabaee83@gmail.com |
| **Soroush Soleimani**    | Soroushsoleimani1s@gmail.com |

---

**Acknowledgment:** Special thanks to **Dr. Pishgoo** and **Eng. Alireza Ghorbani** for their guidance and supervision throughout this project.
