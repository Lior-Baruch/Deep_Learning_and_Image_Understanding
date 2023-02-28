#### Deep Learning and Image Understanding
This repository contains the homework assignments for the course "Deep Learning and Image Understanding" as part of my MS.c degree program at Reichman University. The course covers a wide range of topics related to deep learning, including neural networks, convolutional neural networks (CNNs), recurrent neural networks (RNNs), and transformers.

Files
The repository contains the following files:

1. Linear_Image_Classifier.ipynb
This Jupyter Notebook implements a linear image classifier using the PyTorch library. The classifier is trained on the CIFAR-10 dataset, which contains 60,000 32x32 color images in 10 different classes. The notebook demonstrates how to load and preprocess the data, create a linear classifier, train the model, and evaluate its performance.

2. Neural_Network.ipynb
This Jupyter Notebook implements a basic neural network with three layers (input, hidden, and output) using PyTorch. The notebook demonstrates how to load and preprocess data, define the network architecture, train the model, and evaluate its performance on a classification task.

3. Convolutional_Neural_Network.ipynb
This Jupyter Notebook implements a convolutional neural network (CNN) using PyTorch. The notebook demonstrates how to load and preprocess data, define the network architecture, train the model, and evaluate its performance on a classification task. The notebook uses the CIFAR-10 dataset, like the linear image classifier notebook.

4. RNN_Image_Captioning_ConvNet_LSTM.ipynb
This Jupyter Notebook implements an image captioning model using a convolutional neural network (CNN) to extract features from images and a recurrent neural network (RNN) with long short-term memory (LSTM) cells to generate captions. The notebook demonstrates how to load and preprocess data, define the network architecture, train the model, and evaluate its performance on a captioning task using the Flickr8k dataset.

5. VIT_Vision_Transformer.ipynb
This Jupyter Notebook implements a vision transformer model using PyTorch. The notebook demonstrates how to load and preprocess data, define the network architecture, train the model, and evaluate its performance on a classification task using the CIFAR-10 dataset.

Requirements
To run the Jupyter Notebooks, you will need to have the following Python libraries installed:

PyTorch
NumPy
Matplotlib
Pillow
NLTK (only for RNN_Image_Captioning_ConvNet_LSTM.ipynb)
You can install these libraries using pip by running the following command:

Copy code
pip install torch numpy matplotlib pillow nltk
Usage
To use the Jupyter Notebooks, you can either download the entire repository and run the notebooks locally on your computer, or you can open them using Google Colaboratory.

To open a notebook using Google Colaboratory, simply click on the notebook file in the repository and then click on the "Open in Colab" button at the top of the file. This will open the notebook in a new tab in your web browser, and you can run the cells by clicking on them and then pressing Shift + Enter.

License
The code in this repository is released under the MIT License. See the LICENSE file for more details.
