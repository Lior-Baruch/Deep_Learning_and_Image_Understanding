# Three-Layer Neural Network for CIFAR-10 Image Classification

This project implements a three-layer fully connected neural network for image classification tasks, specifically using the CIFAR-10 dataset. The notebook contains various sections including data preprocessing, model implementation, model training, and performance evaluation.

## Dependencies

The project is implemented in Python, with the use of several libraries:

- `numpy`: Used for numerical operations.
- `matplotlib`: Used for data visualization, such as loss and accuracy curves.
- `os`: Used for OS-related operations, such as directory manipulation.
- `pickle`: Used to save and load Python objects.
- `urllib.request`: Used to download the CIFAR-10 dataset.
- `tarfile` and `zipfile`: Used to extract the downloaded CIFAR-10 dataset.
- `itertools`: Used for efficient looping.

## Dataset

The CIFAR-10 dataset is a well-known dataset used for image recognition tasks. It consists of 60,000 32x32 color images divided into 10 classes, with 6,000 images per class. The dataset is split into 50,000 training images and 10,000 testing images. The notebook includes code to automatically download and extract the CIFAR-10 dataset.

## Model Architecture

The model used is a fully connected three-layer neural network. The architecture of the network is as follows:

- Input layer
- Fully connected layer (FC1) followed by a ReLU activation function
- Fully connected layer (FC2) followed by a ReLU activation function
- Fully connected layer (FC3)
- Softmax function to output probabilities for each class

The model is initialized with small random weights and biases set to zero.

## Model Training

The model is trained using stochastic gradient descent with L2 regularization. The loss function used is the softmax loss function, and the gradients are calculated using backpropagation. The notebook provides functions for both forward and backward propagation through a fully connected layer followed by a ReLU. The training process also includes a grid search over several values of learning rates, hidden layer sizes, and regularization strengths to tune the hyperparameters. The model with the best performance on the validation set is selected as the final model.

## Model Evaluation

After training, the model's performance is evaluated on the training, validation, and test datasets. The accuracy of the model on these datasets is calculated and printed. The notebook also includes code to plot the loss history and the classification accuracy history during the training process.

## Usage

To use this notebook:

1. Clone this repository.
2. Ensure that all necessary Python libraries mentioned in the "Dependencies" section are installed.
3. Run the Jupyter notebook. It will automatically download and preprocess the CIFAR-10 dataset.
4. Follow along with the notebook to train the model on the training data and evaluate its performance on the validation set.
5. Observe the loss and accuracy curves plotted for a visual understanding of the model's performance during training.
6. The notebook also includes a section that performs a grid search over several hyperparameters, training a new model for each combination and evaluating its performance on the validation set.
7. The model with the best validation accuracy is selected, and its performance is evaluated on the test set.

## Results

The accuracy of the trained model on the training, validation, and test sets is printed in the notebook. Additionally, the notebook includes plots of the model's loss and classification accuracy over the course of training, providing a visual representation of the model's learning process.

## Note

This project is primarily intended as an exercise in implementing a three-layer neural network from scratch, with emphasis on understanding the forward and backward propagation process. Therefore, it does not aim to achieve state-of-the-art performance on the CIFAR-10 dataset.
