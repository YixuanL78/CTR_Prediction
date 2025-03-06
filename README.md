# Click-Through Rate (CTR) Prediction Using Apache Spark & PyTorch

## Overview

This project focuses on predicting Click-Through Rates (CTR) for online advertisements by leveraging big data processing with Apache Spark and deep learning modeling with PyTorch. The model helps optimize ad targeting strategies by analyzing user interaction data and identifying key patterns influencing clicks.

## Skills Demonstrated

### 1. Big Data Processing & Feature Engineering

Used Apache Spark (PySpark) to process and clean 400K+ advertising click records efficiently.

Applied Spark SQL for data querying, filtering, and transformation.

Performed feature engineering using Label Encoding and Standardization to optimize categorical and numerical data.

### 2. Deep Learning Model Development

Designed and trained a deep neural network (DNN) using PyTorch to predict user clicks.

Implemented ReLU activations and sigmoid output layers for binary classification.

Optimized the model using Adam optimizer and Binary Cross-Entropy Loss (BCELoss).

### 3. Performance Evaluation & Optimization

Achieved 83.89% accuracy in predicting user click behavior.

Evaluated model performance using AUC-ROC scores to measure effectiveness.

Fine-tuned hyperparameters to improve generalization and reduce overfitting.

### 4. Scalability & Efficiency

Leveraged distributed computing with Apache Spark to scale data preprocessing for large datasets.

Used GPU acceleration for deep learning model training in Google Colab.

### Tools & Technologies Used

Big Data Processing: Apache Spark, PySpark, Spark SQL

Deep Learning Framework: PyTorch, Torch.nn, Torch.optim

Data Processing & Analysis: Pandas, NumPy, Scikit-learn

Feature Engineering: Label Encoding, StandardScaler

Evaluation Metrics: Accuracy Score, AUC-ROC
