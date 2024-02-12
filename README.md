Assignment-2: Neural Network Training

![](Aspose.Words.9c4e1301-bd91-42cb-ad7b-19f1e67daae5.001.png)

ELL784: Introduction To Machine Learning

Submitted By

Bitthal Bhai Patel (2023EET2184)

Submitted To Prof. Jayadeval

**Contents**

[**1 Problem statement](#_page2_x70.87_y83.82) **1 [2 Procedure](#_page2_x70.87_y400.12) 1**

1. [Creating DataSet . ](#_page2_x70.87_y430.51). . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1
1. [Spliting the Dataset ](#_page3_x70.87_y407.22). . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2
1. [Creating Neural Network ](#_page3_x70.87_y516.52). . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2
1. [Compiling Model ](#_page4_x70.87_y668.90). . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
1. [Fitting or Training The Model .](#_page5_x70.87_y83.82) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
1. [Plotting Graph . ](#_page5_x70.87_y181.30). . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
1. [Receptive Field Determination .](#_page5_x70.87_y543.54) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
1. [Regularization Term and Training . ](#_page5_x70.87_y738.92). . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
1. [Tuning Hyperparameters . ](#_page6_x70.87_y708.14). . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5

**IIT Delhi Software Fundamentals![](Aspose.Words.9c4e1301-bd91-42cb-ad7b-19f1e67daae5.002.png)**

<a name="_page2_x70.87_y83.82"></a>**1** | **Problem statement**

Using images from the web, create a dataset with at least 2500 samples, for training a neural network to recognize if an input image is a face / non-face. Split the dataset into training, validation, and test sets.

Train a neural network using

1) hinge loss function
1) cross entropy loss function.

Use at least 2 hidden layers.

Plot the training, validation, and test loss values across epochs. Determine the receptive field of any chosen neuron in each layer.

Explain your approach to determining this. The receptive field of a neuron is defined as the primary (first) layer input that excites this neuron the most.

Add a regularization term and determine how the training changes. Determine how to tune the hyperparameter that controls the emphasis on the mis-classification term. Repeat all other steps described above.

<a name="_page2_x70.87_y400.12"></a>**2** | **Procedure**

1. |<a name="_page2_x70.87_y430.51"></a> **Creating DataSet**
- Downloaded total 2590 images consisting of different categories like human face, aeroplanes, cat, dog, cars, motorcycle, flowers and,fruits. majorly human face (1512).
- Imported the dataset from local folder “images”.
- Resizing image (64x64).
- Now, for every image making a list of labels or the true class value i.e “1” for Human and “0” for Non-Human.![ref1]

Figure 1: Importing Dataset and Preprocessing

2. |<a name="_page3_x70.87_y407.22"></a> **Spliting the Dataset**
- Making use of sklearn to split dataset in three parts as:
- “70%” Training Data,“15%” Validation and Test each.
- Normalizing the pixel values to [0,1] by dividing with “255.0”.
3. |<a name="_page3_x70.87_y516.52"></a> **Creating Neural Network**
- Making use of Tensor flow to make sequential network.
- Making each image matrix as 1-D array using Flatten tf.keras.layers.Flatten(inputshape=(64, 64)),

  *#1st Hidden Layer consisting of 128 neurons and activation (ReLU).* tf.keras.layers.Dense(128, activation=‘relu’),

*#2nd Hidden Layer consisting of 64 neurons and activation (ReLU).* tf.keras.layers.Dense(64, activation=‘relu’),

*#Output Layer consisting of 1 neurons and activation (Sigmoid).* tf.keras.layers.Dense(1, activation=‘sigmoid’)![ref1]

![](Aspose.Words.9c4e1301-bd91-42cb-ad7b-19f1e67daae5.004.png)

Figure 2: Sequential Neural Network

- we can also add Convolution and Pooling layers to extract the features from image hence increasing accuracy with less dense network.

![](Aspose.Words.9c4e1301-bd91-42cb-ad7b-19f1e67daae5.005.png)

Figure 2: Sequential Neural Network

4. |<a name="_page4_x70.87_y668.90"></a> **Compiling Model**
- Compile the Model with optimizer (Optimizer is used to reduce the cost calculated by cross-entropy or Hinge).
- Used two loss functions Cross-entropy or Hinge Loss function one at a time.![ref1]
5. |<a name="_page5_x70.87_y83.82"></a> **Fitting or Training The Model**
- Now, training the model using train data and validation set for some value of epoch.
- After training, we evaluate the model on test set.
6. |<a name="_page5_x70.87_y181.30"></a> **Plotting Graph**
- Plotting the training, validation, and test loss values across epochs.

![](Aspose.Words.9c4e1301-bd91-42cb-ad7b-19f1e67daae5.006.png)

Figure 3:Loss graph across Epochs

7. |<a name="_page5_x70.87_y543.54"></a> **Receptive Field Determination**
- Select the neuron of interest in a specific layer.
- Trace the connections and weights back from the chosen neuron through the network to the input layer. This process involves examining the convolutional kernels and pooling layers.
- Identify the region in the input layer (image or feature map) that has the most significant impact on the chosen neuron’s activation. This is usually the region that aligns with the weights of the neuron’s receptive field.![ref1]
8. | **Regularization Term and Training**
- Choose a suitable regularization technique (e.g., L1, L2) and define the regularization strength (hyperparameter).
- Modify the loss function used for training by adding the regularization term. For example, in L2 regularization, you would add the sum of squared weights to the loss.
- Train the network with the modified loss function. The regularization term encourages the model to have smaller weights, which can help prevent overfitting.
- Monitor the training process and evaluate the model’s performance on a validation dataset.May need to adjust the regularization strength to find the right balance between fitting the data and preventing overfitting.
- **Result:** On adding Regularization(L2) it will not significantly affect the accuracy in binary cross entropy loss but it drastically increases the accuracy in case of hinge loss.

![](Aspose.Words.9c4e1301-bd91-42cb-ad7b-19f1e67daae5.007.png)

Figure 3:Loss graph across Epochs after adding regularization

9. |<a name="_page6_x70.87_y708.14"></a> **Tuning Hyperparameters**
- Select a range of possible hyperparameter values. For C, consider values ranging from very small (strong regularization) to large (weak regularization).![ref1]
- Use a search technique such as grid search, random search, or cross- validation to evaluate the model’s performance with different hyper- parameter values. Measure performance using an appropriate metric (e.g., accuracy, F1-score).
- Identify the hyperparameter value that results in the best performance on a validation dataset. This value balances the trade-off between fitting the data and regularization.
- Train a final model using the selected hyperparameter value on the combined training and validation data.![ref1]
Page 6

[ref1]: Aspose.Words.9c4e1301-bd91-42cb-ad7b-19f1e67daae5.003.png
