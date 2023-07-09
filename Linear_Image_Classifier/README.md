# Linear Image Classifier

## Introduction
This project aims to implement a linear image classifier using numpy and demonstrates the benefits of vectorized operations in Python. Two models, a Perceptron and a Logistic Regression, are used to classify images from the CIFAR-10 dataset. The project involves implementing loss functions, calculating gradients, implementing gradient descent, training classifiers, and evaluating their performance.

## Dataset
The dataset used in this project is the CIFAR-10 dataset. The CIFAR-10 dataset is a well-known dataset used for image recognition tasks. In this project, we specifically use the "plane" and "car" classes for a binary classification task.

## Getting Started
To run the project, you need a Python environment with Jupyter Notebook installed. The project uses the following Python libraries:
- os
- numpy
- matplotlib
- pickle
- urllib.request
- tarfile
- zipfile
- itertools
- time

## Files Included
The main file in this project is the `Linear_Image_Classifier.ipynb` Jupyter notebook, which contains all the code for the project.

## Usage
To run the project, open the `Linear_Image_Classifier.ipynb` notebook in Jupyter Notebook and execute the cells in order.

## Overview of the Code
The notebook is divided into several sections:

1. **Data Download and Processing**: This section contains code to download and extract the CIFAR-10 dataset, load it into memory, and preprocess it. The data is split into training, validation, and testing sets.

2. **Perceptron Model**: The Perceptron model is implemented in this section as a subclass of the LinearClassifier class. The loss function for the Perceptron model is implemented in both naive and vectorized forms. The model is then trained, and the loss history is plotted over iterations.

3. **Hyperparameter Tuning for Perceptron Model**: This section includes a grid search over learning rates and batch sizes to find the optimal hyperparameters for the Perceptron model. The model with the highest validation accuracy is stored for evaluation.

4. **Logistic Regression Model**: In this section, the Logistic Regression model is implemented as a subclass of the LinearClassifier class. The sigmoid function and binary cross-entropy loss function are implemented. The model is then trained, and the loss history is plotted over iterations.

5. **Hyperparameter Tuning for Logistic Regression Model**: Similar to the Perceptron model, a grid search is conducted over learning rates and batch sizes to find the optimal hyperparameters for the Logistic Regression model. The model with the highest validation accuracy is stored for evaluation.

6. **Evaluation**: Both the Perceptron and Logistic Regression models are evaluated on the testing set, and the accuracy is printed.

## Results
The results of the project, including the accuracy of the trained models, are printed in the notebook. You can run the notebook to see the results for yourself.

## Note
This project is intended to be an exercise in implementing linear classifiers and does not necessarily aim to achieve state-of-the-art performance on the CIFAR-10 dataset. The main focus of the project is on the implementation of the models and the usage of vectorized operations in Python.
