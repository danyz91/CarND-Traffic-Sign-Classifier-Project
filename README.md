## Project: Traffic Sign Recognition 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

<br/><br/>

# Overview

In this project, there is shown a deep convolutional neural networks classifying traffic signs. The network is trained and evaluated over the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./img/train_barchart.png "Training Set Barchart"
[image2]: ./img/train_piechart.png "Training Set Piechart"
[image3]: ./img/preproc_dst.png "Preprocessing Source"
[image4]: ./img/preproc_src.png "Preprocessing Output"
[image5]: ./img/train_valid_acc.png "Training/Validation Accuracy"
[image6]: ./img/loss.png "Loss over Epochs"
[image7]: ./img/confusion_matrix.png "Confusion Matrix"
[image8]: ./img/worst_example.png "Worst Examples"
[image9]: ./img/test_out_1.png "Test out 1"
[image10]: ./img/test_out_2.png "Test out 2"
[image11]: ./img/test_out_3.png "Test out 3"
[image12]: ./img/test_out_4.png "Test out 4"
[image13]: ./img/test_out_5.png "Test out 5"
[image14]: ./img/feature_map.png "Feature map visualization"
[image15]: ./img/lenet_arch.jpg "LeNet Architecture"
[image16]: ./img/test_orig.png "Test Image Loaded"
[image17]: ./img/normalization.png "Normalization effect"

<br/><br/>
### Submission Files

---
The project submission requested files:

- Ipython notebook with code
    :[ Project code](./Traffic_Sign_Classifier.ipynb)
- HTML output of the code
    :[ HTML output](./Traffic_Sign_Classifier.html)
- A writeup report 
    You are here! -> :[ Writeup](./README.md)

<br/><br/>
### Data Set Summary & Exploration

---

<br/><br/>
#### 1. Basic summary of the data set.



In **cell3** of th project notebook, it used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is **34799**
* The size of the validation set is **4410**
* The size of test set is **12630**
* The shape of a traffic sign image is **(32, 32, 3)**
* The number of unique classes/labels in the data set is **43**

<br/><br/>
#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 

In **cell 4** there are shown 43 random images, one per class, in order to give a general insight of the dataset.

In **cell 5** there are shown three bar plots, for Training Set, Validation Set and Test Set, respectively.

The barplot for the training set is showed in the image below:

![alt text][image1]

From the inspection of this image it emerges that the training set is unbalanced.

There are provided few examples of signs such as:

- Speed Limit (20km/h)
- Dangerous curve to the left
- Go Straight or Left

There are provided a lot of examples of signs such as:

- Speed Limit (50km/h)
- Yield
- Keep Right


In order to give a broader view of this unbalancing, a pie chart of the same set is computed in **cell 6**. The computed chart is showed in the image below:

![alt text][image2]

Here it emerges more clearly the unbalancing, and we can easily find most and least occurred classes:

```
Most occurred class :  Speed limit (50km/h)  Freq: 5.77 %
Least occurred class :  Speed limit (20km/h)  Freq: 0.52 %
```



Imbalanced Dataset should be taken into account for improving overall performance.

There could be tried some balancing technique such as

- Class Weights
- Oversampling

These techniques are commented and showed in TensorFlow at this [link](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#class_weights) 



<br/><br/> 


### Design and Test a Model Architecture

---
<br/><br/>
#### 1. Pre Processing

It has been decided to divide pre processing of each image in 3 steps:

* **Order shuffling**,
    * The result of this operation is the shuffling of the dataset in order to remove dependancies on set ordering.
    * The code for this step is contained in **cell 7**
    
* **Grayscale conversion**, 
    * Since LeNet traditionally was designed for grayscale conversion and since this led to better performance in current problem, a gray scale conversion of each image is performed   
    *  The code for this step is contained in **cell 8** 

* **Normalization**,
    * Normalization helps the stability of learning process
    * The code for this step is contained in **cell 8** 

The beneficial effect of the normalization is summarized in the image below.

![alt text][image17]

In the image on the right it is clear how normalizing dataset helps gradient descent to find minimum direction more efficiently. Gradient descent steps are represented by arrows.



Here is an example of a traffic sign image before and after applying preprocessing pipeline.

![alt text][image4]

![alt text][image3]



<br/><br/>
#### 2. Model Architecture 

The neural network architecture is heavily inspired by the original net described at [ this link](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)

The model architecture is showed in the image below.

![alt text][image15]


In **cell 11** there is the code for the implementation of this architecture with slight modifications commented below.
The implemented neural network architecture is summarized by the table:



| Layer                 |     Description  | Output Size
|:---------------------:|:------------------:| :------------------:| 
| Input                 | 32x32x1 image                                 | 
| Convolution 5x5       | 1x1 stride, valid padding |  28x28x6    |
| RELU                  |      Rectified Linear Units activation function                                         |
| Max pooling           | 2x2 stride | 14x14x6                  |
| Convolution 5x5       | 1x1 stride, valid padding | 10x10x16    |
| RELU                  |     Rectified Linear Units activation function                                              |
| Max pooling           | 2x2 stride | 5x5x16                  |
| Flatten               | | 400    |  |
| DROPOUT               |  training keep probability: 0.5        |  |
| Fully Connected       |  | 120            | 
| RELU                  |   Rectified Linear Units activation function    |  |
| Fully Connected       |  | 84            | 
| RELU                  |   Rectified Linear Units activation function    |    -                                         |
| Fully Connected       |  | 43            | 


<br/><br/>
##### Regularization


The major add-on w.r.t the traditional architecture is the add of dropout layer. This is a regularization technique for reducing overfitting .
During training, some number of layer outputs are randomly ignored or “dropped out”. Dropout has the effect of making the training process noisy, forcing nodes within a layer to probabilistically take on more or less responsibility for the inputs.

Dropout is here applied in training with a keep probability of **0.5**

<br/><br/><br/><br/>
#### 3. Hyperparameters

The parameters for training are contained in **cell 10** and **cell 12**. They are summarized below:

| Parameter                 |     Value  |
|:---------------------:|:------------------:|
| Epochs                 | 60 |
| Batch Size                 | 512 |
| Learning Rate                 | 0.00091 |


The optimizer chosen for the process is the Adam algorithm. Its usage is defined in **cell 12**, with the call to function 

``` 
tf.train.AdamOptimizer(learning_rate = rate)
```

<br/><br/><br/><br/>
#### 4. Training Procedure

The first design choice made was to used a well-known architecture, the LeNet architecture. This architecture seemed promising in multi-label classification of images and it was decided to start from here.

There was used an iterative approach whose steps are summarized below:

1. **Grayscale conversion**

    First trainings with color images showed *underfitting* with poor perfomance on both training and validation set. So it was decided to resemble what traditional architecture treated well: grayscaled image. 

2. **Dropout Add**

    Later trainings showed poor performance on validation set, under lighting an *overfitting* problem. In order to prevent this, there has been added a dropout layer between second convolutional layer and first fully connected layer. This prevented the overfitting problem.

3. **Learning Rate adjustment**

    In order to achieve more robust performance, learning rate has been tuned with few values, and *0.00091* showed the best performance


The result of training is showed in the images below.

![alt text][image5]

Here it could be appreciated how accuracy over training set reaches almost 100% while validation accuracy follows it as training proceeds.

![alt text][image6]

More formally, the model correctly minimizes the loss function. 
This is a certain measure of improvement over time. The neural network effectively learns best parameters for numerical optimization problem stated.


<br/><br/>
##### Final Results



| Set                 |     Accuracy  |
|:---------------------:|:------------------:|
| Training Set                 |  0.999 |
| Validation Set                | 0.956 |
| Test Set                 | 0.948 |

In **cell 13** is contained the code for evaluating the neural network.

This cell contains two evaluation routines:

* `evaluate` : runs the network on passed dataset and return its accuracy metric of neural network trained
* `evaluate_prediction` : runs the network on passed dataset and return two sets: `(predictions, label)` with the predictions over all dataset, linked with their correct labels


<br/><br/>
##### Performance Evaluation


<br/><br/>
###### Precision, Recall, F1-Score


These results are above target accuracy of *0.93* on validation set.

In order to have a deeper understanding on training result, some more detailed metrics are computed:

- **Precision** 
    
    This metric talks about how precise/accurate your model is out of those predicted positive, how many of them are actual positive.

    *Having an High Precision limit the cost of a False Positive*

- **Recall**
 
    This metric actually calculates how many of the Actual Positives our model capture through labeling it as Positive

    *Having High Recall limit the cost of False Negative*

- **F1 Score**

    This is a measure to use if we need to seek a balance between Precision and Recall

These values are shown *class by class* in **cell 15**.

Average performance of this metrics are showed below

Metric  | Value
---- | -----
Average Precision |      0.942
Average Recall |  0.926
Average F1 Score |  0.930


<br/><br/>
###### Confusion matrix


Confusion matrix is a summary of prediction results on a classification problem.

It shows the ways in which a classification model is confused when it makes predictions.

It gives insight not only into the errors being made by a classifier but more importantly the types of errors that are being made.

Confusion matrix is presented in image below.


![alt text][image7]


From its inspection it emerges that network:

- Has a good score in evaluating classes -> High values on the diagonal
- Is more confident on some classes, such as Speed Limit (50km/h) or Turn Right ahead than other, such as Speed Limit (20km/h)


Moreover, there have been inspected the class where the network showed worst *F1 score*, representing the classes on which it performs bad.

One of these classes is showed below.

![alt text][image8]


From the inspection of this and the other presented in **cell 16** , we can say that classification suffers in classifying *triangle-shape* images.

The neural network should enhance its geometrical recognition of such shapes 

<br/><br/><br/><br/>

### Test a Model on New Images

---
<br/><br/>
#### 1. German Traffic Signs Loading

The five German traffic signs found on the web:

![alt text][image16]

These images could be tricky for our network to classify since they show different lighting conditions and orientation w.r.t the ones present in the dataset

<br/><br/>
#### 2. Model Prediction

In the images below it is shown the top 5 softmax probabilities over these image:


![alt text][image9]

As it is showed in the image, this sign is **correctly** classified as *"Priority Road"*. The second more probable sign is a Yield sign. This is somewhat and index of how the net thinks these two signs are similar. This is in part true since the share some geometric properties.


![alt text][image10]

As it is showed in the image, this sign is **correctly** classified as *"Go Straight or right"*. Even if this image is not in the training set, it is a clean view of the sign. A good performance here is a index of neural network certainty on classifying correctly.The second more probable sign is a turn right. This is plausible since this sign has a right arrow too.


![alt text][image11]


As it is showed in the image, this sign is **wrongly** classified as *"End of all speed and passing limits"*. The reason of this misclassification could be found in unusual orientation of this sign. Data augmentation could help in treating image with this orientation.

![alt text][image12]

As it is showed in the image, this sign is **wrongly** classified as *"Beware of ice/snow"*. The reason are similar to the misclassification discussed above: bad orientation.

![alt text][image13]

As it is showed in the image, this sign is **correctly** classified as *"Children crossing"*. Even if this image is a triangle shaped image, the ones in which the net suffer the most, since this image is front-facing the camera the classification works well.

<br/><br/>
#### 3. Performance

The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. 

These under light the need of *robustness* of the network w.r.t. to orientation and more in general on different conditions of traffic signs.


<br/><br/>
### Visualizing the Neural Network 

In the image below there is shown the feature maps of the internal layer of the neural network:

![alt text][image14]


From this image it is clear that the more shallow layer is selecting edges while deeper layers are more excited by geometrical patterns that enlight the shape of input signal.

<br/><br/><br/><br/>
### Known Limitations and Open Points

Some known limitations of the neural networks are:

- Learning on Unbalanced Dataset
- Distinction between triangle-shaped signal must be enforced
- Robustness over orientation

Open points that can be addressed are:

* Data Augmentation: using methods such as class weighting or oversampling to work on a more balanced dataset -> [link](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#class_weights) 
* Deeper Networks: there could be an improvement in performance using 
    * Deeper network such as VGG16 ([ paper link](https://arxiv.org/abs/1409.1556)) 
    * Using ensembling technique such as inception module :[ paper link](https://arxiv.org/pdf/1409.4842.pdf)
