Facial Expression Recognition (FER) using Deep Learning
K. N. Toosi University of Technology - Artificial Intelligence Course Project

---

_Project Overview_
This project aims to design and implement a complete intelligent system for detecting human emotions from facial images. Using the FER2013 dataset and a Convolutional Neural Network (CNN), the model classifies facial expressions into 7 categories (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral).

---

_Modular Project Structure_
Following the industrial standards required by the course, the project is organized as follows:

├── data/
│ ├── raw/ # Original FER2013 dataset
│ └── processed/ # Preprocessed and normalized images
├── notebooks/
│ ├── EDA.ipynb # Exploratory Data Analysis & Visualizations
│ └── experiments.ipynb # Initial model testing and prototyping
├── src/
│ ├── preprocessing/ # Data augmentation and cleaning scripts
│ ├── models/ # CNN Architecture definitions
│ ├── training/ # Training loops and early stopping logic
│ └── evaluation/ # Confusion matrix and metric reports
├── results/
│ ├── charts/ # Accuracy/Loss curves
│ └── metrics/ # Evaluation reports (F1-score, Precision, Recall)
├── models/ # Saved model weights (.h5/.pt) - [Gitignored]
├── README.md # Full project documentation
└── requirements.txt # Environment dependencies

---

_Phase 1: Data Analysis & Preprocessing_

- EDA: Detailed analysis of class distribution to handle label imbalance.

- Visualizations: Contains at least 6 distinct plots including sample images and frequency histograms.

- Preprocessing: Includes Resizing (48x48), Normalization, and Data Augmentation (Rotation, Zoom, Horizontal Flip) to improve generalization

_Phase 2: Model Implementation & Training_

- Architecture: A modular CNN baseline with Batch Normalization and Dropout layers.

- Training: Monitored using Loss and Accuracy curves to detect Overfitting/Underfitting.

- Optimization: Hyperparameter tuning of learning rates and optimizers (Adam/SGD).

---

_Evaluation_
The model is evaluated based on standard metrics:

- Accuracy & F1-Score.

- Confusion Matrix to analyze misclassified emotions.

- Error Analysis: Identifying "Hard Examples" where the model fails

---

_Installation & Usage_
1- Clone the repository:

# " git clone https://github.com/SoroushSoleimani/Facial-Expression-Recognition.git "

2- Install dependencies:

# " pip install -r requirements.txt "

3- Run Training:

# " python src/training/train.py "

4- Run Evaluation:

## " python src/evaluation/evaluate.py "

_Interactive Demo_

A live demo is provided using Gradio/Streamlit, allowing users to upload images or use a webcam for real-time emotion detection.
