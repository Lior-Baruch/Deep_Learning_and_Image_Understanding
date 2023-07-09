# Convolutional_Neural_Network.ipynb

This Jupyter notebook, named "Convolutional_Neural_Network.ipynb", is a comprehensive guide to building, training, and evaluating Convolutional Neural Networks (CNNs) using PyTorch. The notebook primarily focuses on image classification and localization tasks using the CIFAR-10 dataset and a custom dataset featuring images of cats and dogs. 

## Notebook Structure

The notebook is organized into several sections, each focusing on specific aspects of the problem:

### 1. Introduction

In the introduction section, the exercise's general instructions are outlined. It emphasizes the requirement for the code to be compatible with both GPU and CPU environments and encourages the utilization of best practices learned in class.

### 2. Convolutional Neural Network - Classifying CIFAR-10

This section dives into the task of training a convolutional network using PyTorch for the CIFAR-10 dataset classification. The use of PyTorch's autograd package for automatic differentiation is highlighted. It provides a detailed explanation of forward and backward passes in the network using computational graphs.

### 3. Data Preprocessing

The section is dedicated to loading and normalizing the CIFAR-10 dataset using PyTorch's torchvision module. The images are transformed into tensors and normalized. Iterable objects for the training and test datasets are created using PyTorch's DataLoader utility.

### 4. Building a CNN in PyTorch

Here, a custom Convolutional Neural Network is constructed using PyTorch's `nn.Module`. The network comprises convolutional layers, max pooling layers, and fully connected layers. The forward pass of the network is also defined in this section.

### 5. Evaluating the Model

In this part, the trained model's evaluation is performed using a confusion matrix. The overall accuracy of the model on the test images is calculated, and a confusion matrix is plotted to visualize the performance of the model across different classes.

### 6. Further Evaluation

The section discusses the limitations of using accuracy as the only metric for model evaluation. It proposes additional metrics like precision, recall, and F1 score for a more comprehensive understanding of the model's performance.

### 7. Localization as Regression

The section introduces a new task—using a pre-trained network (ResNet18) for localizing and classifying images of cats and dogs. The concept of using pre-trained models for transfer learning is discussed in detail. 

### 8. Training Guidelines

This part provides guidelines for training the model and tracking its performance using classification accuracy and Intersection over Union (IoU) score. It also underlines the importance of visualizing these metrics over the training epochs.

### 9. Localization as Regression - Model Building

Here, a pre-trained ResNet18 model is used for feature extraction from the images. A new model is built using these features, including a classifier and a regressor for bounding box coordinates. The model is trained with a combined loss from both the classification and regression tasks.

### 10. Visualizing Model Performance

The final part of the notebook displays the model's performance visualizations, showing accuracy, IoU, and loss metrics over the training epochs. It also shows sample images with their predicted bounding boxes and labels.

## Dependencies

This notebook relies on several Python libraries:

- `PyTorch`: For building and training neural networks
- `torchvision`: For loading and transforming the CIFAR-10 dataset
- `NumPy`: For numerical operations
- `Matplotlib`: For visualizing model performance and confusion matrices
- `Scikit-learn`: For computing the confusion matrix
- `PIL (Python Imaging Library)`: For handling image data

## Usage

1. Clone the repository containing the notebook.
2. Make sure you have installed all the dependencies listed above.
3. Open the notebook in Jupyter.
4. Run the cells in the notebook to train and evaluate the model.

## Results

The notebook includes the results of the model's performance in terms of accuracy, loss, and IoU metrics. These metrics are visualized over the training epochs to understand the model's learning process. The notebook also provides visualization of the model's predictions on sample images, showing both the predicted class labels and the predicted bounding boxes.
