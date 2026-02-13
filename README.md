Facial Expression Recognition (FER) using Deep Learning
K. N. Toosi University of Technology - Artificial Intelligence Course Project

📝 Project Overview
This project aims to design and implement a complete intelligent system for detecting human emotions from facial images. Using the FER2013 dataset and a Convolutional Neural Network (CNN), the model classifies facial expressions into 7 categories (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral).
+2

🏗 Modular Project Structure
Following the industrial standards required by the course, the project is organized as follows:

├── data/
│ ├── raw/ # Original FER2013 dataset [cite: 248]
│ └── processed/ # Preprocessed and normalized images [cite: 248]
├── notebooks/
│ ├── EDA.ipynb # Exploratory Data Analysis & Visualizations [cite: 257]
│ └── experiments.ipynb # Initial model testing and prototyping [cite: 248]
├── src/
│ ├── preprocessing/ # Data augmentation and cleaning scripts [cite: 262]
│ ├── models/ # CNN Architecture definitions [cite: 262]
│ ├── training/ # Training loops and early stopping logic [cite: 262]
│ └── evaluation/ # Confusion matrix and metric reports [cite: 262]
├── results/
│ ├── charts/ # Accuracy/Loss curves [cite: 248]
│ └── metrics/ # Evaluation reports (F1-score, Precision, Recall) [cite: 248]
├── models/ # Saved model weights (.h5/.pt) - [Gitignored] [cite: 248, 249]
├── README.md # Full project documentation [cite: 242]
└── requirements.txt # Environment dependencies [cite: 248]
