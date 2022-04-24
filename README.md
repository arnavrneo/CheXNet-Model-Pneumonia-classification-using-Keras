# Pneumonia classification using CheXNet model in Keras

ChexNet is a state-of-the-art deep learning algorithm that can detect pneumonia from chest X-rays at a level exceeding practicing radiologists.

The ChexNet algorithm is a 121-layer convolutional neural network which has been trained on ChestX-ray14 dataset (which is currently the largest publicly available chest X-ray dataset: and contains 112,120 frontal view X-ray images from 30,805 unique patients.). It is based on Dense Convolutional Network (DenseNet-121) with the exception that the final fully connected layer is replaced with one that has a single output with sigmoid activation.

<br />

# Model weights

CheXNet model is not pre-included in Keras. But luckily the weights are provided by <a href='https://github.com/brucechou1983'>brucechou1983</a> at this <a href='https://github.com/brucechou1983/CheXNet-Keras'>github page.</a>

The weights can be downloaded from there. The same model weights file has been added in this repository also.

<br />

# Example Dataset used

The dataset that has been used for testing CheXNet Model is the <a href='https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia'>Kaggle's Chest X-ray Pneumonia prediction.</a>

<br />

# How to use it?

The python file performing the CheXNet prediction on the same Kaggle dataset is provided above and can be run on local Jupyter Notebook or any other similar IDE. 
(Note that some directory paths may require change depending on the system).
