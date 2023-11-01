# Sustainable-Apparel-Classification

## Problem Statement 

Develop an AI solution using the Fashion MNIST dataset with a focus on identifying and classifying sustainable apparel products in alignment with the company's vision and the provided job description.

## Dataset Description 

**Dataset Link:**  <a href='https://www.kaggle.com/datasets/zalando-research/fashionmnist/data'>Fashion MNIST</a> <br> <br>
**Description:** Fashion-MNIST is a dataset comprising 70,000 grayscale images of clothing articles, divided into a training set of 60,000 examples and a test set of 10,000 examples. It features 10 distinct classes and is designed as a direct replacement for the original MNIST dataset, maintaining the same image size and training/testing data division, making it a valuable benchmark for machine learning algorithms. <br>

Each image in Fashion-MNIST is 28x28 pixels, totaling 784 pixels. Each pixel holds a single value representing its lightness or darkness, with a range of 0 to 255. Both training and test data sets consist of 785 columns. The first column contains class labels, identifying the clothing item, while the remaining columns store pixel values associated with the image. To locate a specific pixel in the image, it can be expressed as 'x = i * 28 + j,' with 'i' and 'j' as integers from 0 to 27, referring to the pixel's position in a 28x28 matrix

**Training Set Distribution:**

| Class Label | Class Name | Number of Samples| Distribution |
|:-:|:-:|:-:|:-:|
| 0 |  T-shirt/top  | 6000 |  10%  |
| 1 | Trouser | 6000 | 10% |
| 2 | Pullover | 6000 | 10% |
| 3 | Dress | 6000 | 10% |
| 4 | Coat | 6000 | 10% |
| 5 | Sandal | 6000 | 10% |
| 6 | Shirt | 6000 | 10% |
| 7 | Sneaker | 6000 | 10% |
| 8 | Bag | 6000 | 10% |
| 9 | Ankle boot | 6000 | 10% |


**Summary:**
<ul>
  <li> Training set Dimension : <i> 60000x785 </i>, Testing Set Dimension :  <i> 10000x785 </i> </li>
  <li>Each row is a separate image</li>
  <li>Column 1 is the class label.</li>
  <li>Remaining columns are pixel numbers (784 total).</li>
  <li>Each value is the darkness of the pixel (1 to 255)</li>
</ul>



## Solution Approach
**Approach:** In the pursuit of identifying and classifying sustainable apparel products, the chosen primary approach is the Convolutional Neural Network (CNN).


**Why CNN?** 

<p> A Convolutional Neural Network (CNN) is a superior choice for identifying sustainable apparel products compared to traditional Machine Learning (ML) and Artificial Intelligence (AI) approaches because it can automatically learn and generalize from large-scale image data, adapt to diverse visual conditions, and handle intricate patterns and complexities without the need for manual feature engineering. </p>

**How CNN works?**

A step-by-step explanation of how a Convolutional Neural Network (CNN) works:
<ul> 
<li> <i><b> Input Layer: </b></i> The process begins with the input layer, which receives the raw image data. </li>

<li> <i><b>  Convolution (Conv) Layer: </b></i> The Convolution layer applies a set of learnable filters (kernels) to the input image. Each filter scans the image using a small window and performs element-wise multiplication and summation. The result is a feature map that highlights specific patterns or features in the image. </li>

<li> <i><b>   Activation (ReLU) Layer: </b></i>  After each convolution operation, a Rectified Linear Unit (ReLU) layer is applied element-wise to introduce non-linearity. ReLU helps CNNs learn complex relationships within the data. </li>

<li> <i><b> Pooling (Subsampling) Layer: </b></i> The pooling layer reduces the spatial dimensions of the feature map, which helps manage computational complexity. Common pooling methods include Max-Pooling, which retains the maximum value in a local region, or Average-Pooling, which calculates the average. 

Steps 2 to 4 can be repeated with additional convolutional layers to extract increasingly complex features. </li>

<li> <i><b> Flatten Layer: </b></i>  The flattened layer reshapes the output from the convolutional layers into a 1D vector, preparing it for the fully connected layers. </li>

<li> <i><b> Fully Connected (Dense) Layer: </b></i>   The fully connected layers process the flattened vector, applying weights and biases to learn high-level patterns and make predictions. Common activation functions in these layers include ReLU or Softmax for classification tasks. </li>

<li> <i><b> Output Layer:  </b></i>  The output layer produces the final predictions. It might involve a single neuron for binary classification or multiple neurons for multi-class classification. </li>

<li> <i><b>  Loss Function: </b></i>  The loss function measures the difference between the predicted values and the actual labels. Common loss functions include Mean Squared Error (MSE) for regression and Cross-Entropy for classification. </li>

<li> <i><b>  Backpropagation and Optimization: </b></i>  The CNN uses backpropagation to update the model's parameters (weights and biases) based on the loss. Optimization algorithms like Stochastic Gradient Descent (SGD) adjust these parameters to minimize the loss.

Repeat Training: Steps 2 to 10 are repeated multiple times during training, adjusting the network's parameters to improve its accuracy. </li>

<li> <i><b>  Evaluation: </b></i>  The trained CNN is evaluated on a separate test dataset to measure its performance and generalization. </li>

<li> <i><b>  Prediction: </b></i>  Once trained, the CNN can make predictions on new, unseen data, which is especially valuable for image classification tasks. </li>
</ul>


![image](https://github.com/Sajjad-Hossain-Talukder/Sustainable-Apparel-Classification/assets/63524824/1722a96d-3567-4856-913f-a1b41040233e)


## CNN Model Description

Model architecture that used to train: 

|Layer (Type)|Output Shape	|
|:-:|:-:|
|Input|(28, 28, 1)|
|Conv2D(16 filters, 3x3) |(26, 26, 16)|
|MaxPooling2D (2x2)		|(13, 13, 16)|
|Dropout (0.5)	|	(13, 13, 16)|
|Flatten||
|Dense (128, ReLU)| 128|
|Dropout (0.5)	|128|
|Dense (10, Softmax)| 10 |




## Model training Parameters

11

## Execution Steps 

11


