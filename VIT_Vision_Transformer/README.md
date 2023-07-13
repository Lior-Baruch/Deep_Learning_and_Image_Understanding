# Vision Transformer (ViT) Model on CIFAR10

## Overview
This project involves implementing the Vision Transformer (ViT) model for image classification using the CIFAR10 dataset. It covers a range of tasks including data loading, exploratory data analysis (EDA), model building, training, and evaluation.

## Dependencies
The project requires:
- Python 3.8 or above
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib

## Data Loading
The CIFAR10 dataset is loaded using torchvision's in-built dataset functionality. Image transformations such as normalizing the images are applied during the data loading stage.

## Exploratory Data Analysis (EDA)
The notebook includes code for performing EDA on the CIFAR10 dataset. This includes calculating the number of samples per label and plotting a bar chart to visualize these counts.

## Model Building
A Vision Transformer (ViT) model is built for image classification. The model consists of the following parts:
- Patch Embedding: A Conv2D layer that breaks the image into patches and embeds them into vectors.
- Positional Embedding: A Linear layer that gives positional information to the patch embeddings.
- Transformer Encoder: A pre-built Encoder from the PyTorch library.
- Multi-Layer Perceptron (MLP): A Fully Connected layer.
- Output Layer: A Fully Connected layer.

## Training
The model is trained using the Adam optimizer and CrossEntropyLoss as the loss function. The training process is run for a specified number of epochs, with metrics such as loss and accuracy calculated for each epoch. These metrics are saved for later visualization. 

## Evaluation
The trained model is evaluated on a validation set that is split from the original training set. The model's performance is also evaluated on the CIFAR10 test set. The evaluation metrics include accuracy and loss. A confusion matrix is plotted to visualize the performance of the model on the test set.

## Visualization
The notebook includes code for plotting the loss and accuracy of the model for each iteration of training. Additionally, it includes code for plotting images from the test set along with their predicted labels.

## Saving and Loading the Model
The trained model is saved and can be loaded for future use. The state dictionary of the model, which includes the model's parameters, is saved to a file named 'model.ckpt'.

## Instructions to Run the Notebook
1. Ensure that all the dependencies are installed.
2. Download the notebook and open it using Jupyter Notebook.
3. Run the cells in order from top to bottom.

## Note
The notebook is designed for learning purposes and assumes some prior experience in the field.

## References
- [Vision Transformer (ViT): Tutorial & Baseline](https://www.kaggle.com/code/abhinand05/vision-transformer-vit-tutorial-baseline)

---
